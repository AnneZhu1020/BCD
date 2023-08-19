import os
import time
import wandb

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

from minirl.algos.dqn.agent import DQNActor, DQNLearner, DQNWorker
from minirl.utils import get_callable


def wandb_log_actor(rref, step):
    ret = rref.local_value().env.callmethod("get_ep_stat_mean", "r")
    wandb.log({"return": ret}, step=step)


def wandb_close_actor():
    wandb.join()


def run_dqn_actor_learner(rank, config, env_fn):
    # Rank-agnostic setup
    os.environ["MASTER_ADDR"] = config["comm_cfg"]["master_address"]
    os.environ["MASTER_PORT"] = config["comm_cfg"]["master_port"]
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        # Rank-specific setup (learner on rank 0)
        wandb.init(
            name=f"learner-{rank}",
            job_type="learner",
            config=config,
            **config["run_cfg"]["wandb_kwargs"],
        )
        rpc.init_rpc("Learner", rank=rank, world_size=config["comm_cfg"]["world_size"])
        # Initialize actors remotely
        actor_ranks = range(1, config["comm_cfg"]["world_size"])
        actor_infos, actor_rrefs = [], []
        for actor_rank in actor_ranks:
            if (
                config["eval_cfg"]["enabled"]
                and config["eval_cfg"]["rank"] == actor_rank
            ):
                actor_kwargs = {
                    **config["actor_kwargs"],
                    **config["eval_cfg"]["eval_actor_kwargs"],
                }
            else:
                actor_kwargs = config["actor_kwargs"]
            actor_info = rpc.get_worker_info(f"Actor{actor_rank}")
            actor_rref = rpc.remote(
                actor_info,
                DQNActor,
                kwargs={
                    "env_fn": env_fn,
                    "device": config["run_cfg"]["devices"][actor_rank],
                    **actor_kwargs,
                },
            )
            actor_infos.append(actor_info)
            actor_rrefs.append(actor_rref)
        # Initialize learner
        learner = DQNLearner(
            device=config["run_cfg"]["devices"][rank], **config["learner_kwargs"]
        )
        learner_rref = RRef(learner)
        # Create buffer
        buffer_cfg = config["buffer_cfg"]
        buffer = get_callable(buffer_cfg["buffer_fn"])(**buffer_cfg["buffer_kwargs"])
        buffer_rref = RRef(buffer)
        # Training
        stats_dict = {}
        for i in range(config["run_cfg"]["n_timesteps"]):
            tstart = time.perf_counter()
            # Collect data using actors
            futures = []
            for actor_rank, actor_rref in zip(actor_ranks, actor_rrefs):
                if_sync_param = (i + 1) % config["run_cfg"]["train_freq"] == 1
                if (
                    config["eval_cfg"]["enabled"]
                    and config["eval_cfg"]["rank"] == actor_rank
                ):
                    future = actor_rref.rpc_async().eval_collect(
                        current_timestep=i,
                        learner_rref=learner_rref if if_sync_param else None,
                    )
                else:
                    future = actor_rref.rpc_async().collect(
                        current_timestep=i,
                        buffer=buffer_rref,
                        learner_rref=learner_rref if if_sync_param else None,
                    )
                futures.append(future)
            # Synchronize
            torch.futures.wait_all(futures)
            # Learning
            if i >= config["run_cfg"]["learning_starts"]:
                if (i + 1) % config["run_cfg"]["train_freq"] == 0:
                    stats_dict = learner.learn(current_timestep=i, buffer=buffer)
                if (i + 1) % config["run_cfg"]["target_update_freq"] == 0:
                    learner.policy.update_target_net()
            # Logging
            if (i + 1) % config["run_cfg"]["logging_freq"] == 0:
                wandb.log(stats_dict, step=i + 1)
                wandb.log({"time": time.perf_counter() - tstart}, step=i + 1)
                for actor_info, actor_rref in zip(actor_infos, actor_rrefs):
                    rpc.remote(
                        to=actor_info, func=wandb_log_actor, args=(actor_rref, i + 1)
                    )
        # Close wandb
        wandb.join()
        for actor_info in actor_infos:
            rpc.remote(to=actor_info, func=wandb_close_actor)
    else:
        # Rank-specific setup (actors on other ranks)
        if config["eval_cfg"]["enabled"] and config["eval_cfg"]["rank"] == rank:
            job_type = "eval_actor"
        else:
            job_type = "actor"
        wandb.init(
            name=f"actor-{rank}",
            job_type=job_type,
            config=config,
            **config["run_cfg"]["wandb_kwargs"],
        )
        rpc.init_rpc(
            f"Actor{rank}", rank=rank, world_size=config["comm_cfg"]["world_size"]
        )
    # Exit
    rpc.shutdown()


def run_dqn_worker(rank, config, env_fn):
    # Setup
    torch.backends.cudnn.benchmark = True
    if config["comm_cfg"]["enabled"]:
        os.environ["MASTER_ADDR"] = config["comm_cfg"]["master_address"]
        os.environ["MASTER_PORT"] = config["comm_cfg"]["master_port"]
        torch.distributed.init_process_group(
            "nccl", world_size=config["comm_cfg"]["world_size"], rank=rank
        )
    if config["eval_cfg"]["enabled"] and config["eval_cfg"]["rank"] == rank:
        worker_kwargs = {
            **config["worker_kwargs"],
            **config["eval_cfg"]["eval_worker_kwargs"],
        }
        job_type = "eval_worker"
    else:
        worker_kwargs = config["worker_kwargs"]
        job_type = "worker"
    wandb.init(
        name=f"worker-{rank}",
        job_type=job_type,
        config=config,
        **config["run_cfg"]["wandb_kwargs"],
    )
    # Initialize worker
    worker = DQNWorker(
        env_fn=env_fn, device=config["run_cfg"]["devices"][rank], **worker_kwargs,
    )
    # Create buffer
    buffer_cfg = config["buffer_cfg"]
    buffer = get_callable(buffer_cfg["buffer_fn"])(**buffer_cfg["buffer_kwargs"])
    # Training
    stats_dict = {}
    for i in range(config["run_cfg"]["n_timesteps"]):
        tstart = time.perf_counter()
        # Collect data
        worker.collect(current_timestep=i, buffer=buffer)
        # Learning
        if i >= config["run_cfg"]["learning_starts"]:
            if (i + 1) % config["run_cfg"]["train_freq"] == 0:
                stats_dict = worker.learn(current_timestep=i, buffer=buffer)
            if (i + 1) % config["run_cfg"]["target_update_freq"] == 0:
                worker.policy.update_target_net()
        # Logging
        if (i + 1) % config["run_cfg"]["logging_freq"] == 0:
            wandb.log(stats_dict, step=i + 1)
            ret = worker.env.callmethod("get_ep_stat_mean", "r")
            wandb.log({"return": ret, "time": time.perf_counter() - tstart}, step=i + 1)
