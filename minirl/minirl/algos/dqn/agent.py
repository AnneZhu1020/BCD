from typing import Optional, Union

import torch
from torch.distributed import ReduceOp
from torch.distributed.rpc import RRef

from minirl.buffer import ReplayBuffer
from minirl.utils import get_callable, get_scheduler
from minirl.type_utils import Schedulable


class DQNActor:
    def __init__(
        self,
        env_fn,
        env_kwargs: dict,
        policy_fn: str,
        policy_kwargs: dict,
        n_step_return: int,
        exploration_eps: Schedulable,
        device: Union[torch.device, str] = "cpu",
    ):
        self.env = env_fn(**env_kwargs)
        self.policy = get_callable(policy_fn)(device=device, **policy_kwargs)
        self.policy.to(device)
        self.n_step_return = n_step_return
        self.exploration_eps = get_scheduler(exploration_eps)
        self.device = device

    def collect(
        self,
        current_timestep: int,
        buffer: Union[ReplayBuffer, RRef],
        learner_rref: Optional[RRef] = None,
    ):
        # Sync parameters if needed
        if learner_rref is not None:
            self.sync_params(learner_rref)
        # Collect samples
        batch = {}
        _, obs, _ = self.env.observe()
        eps = self.exploration_eps.value(step=current_timestep)
        action = self.policy.step(obs=obs, eps=eps)
        self.env.act(action)
        reward, new_obs, first = self.env.observe()
        # TODO: unnormalize obs / rew if needed (before add into buffer)
        batch["obs"] = obs
        batch["actions"] = action
        batch["rewards"] = reward
        batch["new_obs"] = new_obs
        batch["firsts"] = first
        # Send data to buffer
        if isinstance(buffer, ReplayBuffer):
            next_idx = buffer.update_next_idx(size=1)
            buffer.add(
                current_timestep=current_timestep, data=batch, idx=next_idx, size=1,
            )
        else:
            next_idx = buffer.rpc_sync().update_next_idx(size=1)
            buffer.rpc_sync().add(
                current_timestep=current_timestep, data=batch, idx=next_idx, size=1,
            )

    def eval_collect(
        self, current_timestep: int, learner_rref: Optional[RRef] = None,
    ):
        # Sync parameters if needed
        if learner_rref is not None:
            self.sync_params(learner_rref)
        # Collect samples
        eps = self.exploration_eps.value(step=current_timestep)
        _, obs, _ = self.env.observe()
        action = self.policy.step(obs=obs, eps=eps)
        self.env.act(action)

    def sync_params(self, learner_rref: RRef):
        if learner_rref.is_owner():
            params = learner_rref.local_value().broadcast_params()
        else:
            params = learner_rref.rpc_sync().broadcast_params()
        self.policy.set_params(params)
        self.policy = self.policy.to(self.device)


class DQNLearner:
    def __init__(
        self,
        policy_fn: str,
        policy_kwargs: dict,
        optimizer_fn: str,
        optimizer_kwargs: dict,
        batch_size: int,
        discount_gamma: float = 0.99,
        max_grad_norm: Optional[float] = 40.0,
        device: Union[torch.device, str] = "cpu",
    ):
        self.policy = get_callable(policy_fn)(device=device, **policy_kwargs)
        self.policy.to(device)
        self.optimizer = get_callable(optimizer_fn)(
            params=self.policy.online_net.parameters(), **optimizer_kwargs
        )
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma
        self.max_grad_norm = max_grad_norm
        self.device = device

    def learn(self, current_timestep: int, buffer: Union[ReplayBuffer, RRef]):
        # Retrieve data from buffer
        if isinstance(buffer, ReplayBuffer):
            batch = buffer.sample(
                current_timestep=current_timestep, batch_size=self.batch_size
            )
        else:
            batch = buffer.rpc_sync().sample(
                current_timestep=current_timestep, batch_size=self.batch_size
            )
        # Build a dict to save training statistics
        stats_dict = {}
        # Train
        self.optimizer.zero_grad()
        loss, extra_out = self.policy.loss(
            obs=batch["obs"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            new_obs=batch["new_obs"],
            firsts=batch["firsts"],
            gamma=self.discount_gamma,
            weights=batch.get("weights", None),
        )
        loss.backward()
        self.pre_optim_step_hook()
        self.optimizer.step()
        # Update priorities if needed
        if "weights" in batch:
            indices = batch["indices"]
            priorities = extra_out["td_error"].abs().cpu().numpy()
            extra_out["td_error"] = extra_out["td_error"].mean()
            if isinstance(buffer, ReplayBuffer):
                buffer.update_priorities(
                    current_timestep=current_timestep,
                    indices=indices,
                    priorities=priorities,
                )
            else:
                buffer.rpc_sync().update_priorities(
                    current_timestep=current_timestep,
                    indices=indices,
                    priorities=priorities,
                )
        # Saving statistics
        stats_dict["loss"] = loss.item()
        for key in extra_out:
            stats_dict[key] = extra_out[key].item()
        return stats_dict

    def broadcast_params(self):
        return self.policy.get_params()

    def pre_optim_step_hook(self):
        # Gradient clipping
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy.online_net.parameters(), max_norm=self.max_grad_norm
            )


class DQNWorker(DQNActor, DQNLearner):
    def __init__(
        self,
        env_fn,
        env_kwargs: dict,
        policy_fn: str,
        policy_kwargs: dict,
        optimizer_fn: str,
        optimizer_kwargs: dict,
        n_step_return: int,
        exploration_eps: Schedulable,
        batch_size: int,
        discount_gamma: float = 0.99,
        max_grad_norm: Optional[float] = 40.0,
        device: Union[torch.device, str] = "cpu",
        worker_weight: float = 1.0,
    ):
        self.env = env_fn(**env_kwargs)
        self.policy = get_callable(policy_fn)(device=device, **policy_kwargs)
        self.policy.to(device)
        self.optimizer = get_callable(optimizer_fn)(
            params=self.policy.online_net.parameters(), **optimizer_kwargs
        )
        self.n_step_return = n_step_return
        self.exploration_eps = get_scheduler(exploration_eps)
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.worker_weight = worker_weight
        # Sync parameters if needed
        self.distributed_on = torch.distributed.is_initialized()
        if self.distributed_on:
            for param in self.policy.parameters():
                torch.distributed.broadcast(param, 0)

    def pre_optim_step_hook(self):
        # All-reduce gradient if needed
        if self.distributed_on:
            total_weight = torch.tensor(self.worker_weight, device=self.device)
            torch.distributed.all_reduce(total_weight, op=ReduceOp.SUM)
            for param in self.policy.online_net.parameters():
                param.grad.mul_(self.worker_weight)
                torch.distributed.all_reduce(param.grad, op=ReduceOp.SUM)
                param.grad.div_(total_weight)
        # Call parent's hook (e.g. gradient clipping)
        super().pre_optim_step_hook()
