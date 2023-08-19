import os
import time
import wandb

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

from minirl.algos.ppo.agent import PPOActor, PPOLearner, PPOWorker
from minirl.buffer import Buffer


def wandb_log_actor(rref, step):
    ret = rref.local_value().env.callmethod("get_ep_stat_mean", "r")
    wandb.log({"return": ret}, step=step)


def wandb_close_actor():
    wandb.join()


def run_ppo_actor_learner(rank, config, env_fn):
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
        n_train_actor, n_eval_actor = 0, 0
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
                n_eval_actor += 1
            else:
                actor_kwargs = config["actor_kwargs"]
                n_train_actor += 1
            actor_info = rpc.get_worker_info(f"Actor{actor_rank}")
            actor_rref = rpc.remote(
                to=actor_info,
                func=PPOActor,
                kwargs={
                    "env_fn": env_fn,
                    "device": config["run_cfg"]["devices"][actor_rank],
                    **actor_kwargs,
                },
            )
            actor_infos.append(actor_info)
            actor_rrefs.append(actor_rref)
        # Initialize learner
        learner = PPOLearner(
            device=config["run_cfg"]["devices"][rank], **config["learner_kwargs"],
        )
        learner_rref = RRef(learner)
        # Create buffer
        n_envs = config["actor_kwargs"]["env_kwargs"]["num_envs"]
        n_steps = config["actor_kwargs"]["n_steps"]
        buffer_size = n_train_actor * n_envs * n_steps
        buffer = Buffer(max_size=buffer_size)
        buffer_rref = RRef(buffer)
        # Training
        for i in range(config["run_cfg"]["n_iters"]):
            tstart = time.perf_counter()
            # Collect data using actors
            futures = []
            for actor_rank, actor_rref in zip(actor_ranks, actor_rrefs):
                if (
                    config["eval_cfg"]["enabled"]
                    and config["eval_cfg"]["rank"] == actor_rank
                ):
                    future = actor_rref.rpc_async().eval_collect(learner_rref)
                else:
                    future = actor_rref.rpc_async().collect(buffer_rref, learner_rref)
                futures.append(future)
            # Synchronize
            torch.futures.wait_all(futures)
            # Learn on data
            stats_dict = learner.learn(current_step=i, buffer=buffer)
            # Logging
            if (i + 1) % config["run_cfg"]["logging_freq"] == 0:
                wandb.log(stats_dict, step=i + i)
                wandb.log({"time": time.perf_counter() - tstart}, step=i + i)
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


def run_ppo_worker(rank, config, env_fn):
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
    worker = PPOWorker(
        env_fn=env_fn, device=config["run_cfg"]["devices"][rank], **worker_kwargs,
    )
    # Create buffer
    buffer_size = worker.env.num * worker.n_steps
    buffer = Buffer(max_size=buffer_size)
    # Training
    for i in range(config["run_cfg"]["n_iters"]):
        tstart = time.perf_counter()
        # Collect data
        worker.collect(buffer)
        # Learn on data
        stats_dict = worker.learn(current_step=i, buffer=buffer)
        # Logging
        if (i + 1) % config["run_cfg"]["logging_freq"] == 0:
            wandb.log(stats_dict, step=i + 1)
            ret = worker.env.callmethod("get_ep_stat_mean", "r")
            wandb.log({"return": ret, "time": time.perf_counter() - tstart}, step=i + 1)
    wandb.join()
