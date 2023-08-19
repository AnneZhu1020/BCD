from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Optional, Union

import gym3
import torch as th
from torch.distributed import ReduceOp
from torch.distributed.rpc import RRef

from minirl.buffer import Buffer
from minirl.utils import get_callable


class Learner(ABC):
    def __init__(
        self,
        policy_fn: str,
        policy_kwargs: dict,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        """
        Optimizer might be algorithm-specific
        """
        super().__init__()
        self.policy = get_callable(policy_fn)(device=device, **policy_kwargs)
        self.policy.to(device)
        self.device = device

    @abstractmethod
    def learn(self, scheduler_step: int, buffer: Union[Buffer, RRef]) -> dict:
        pass

    def broadcast_params(self) -> OrderedDict:
        return self.policy.get_params()

    def clip_gradient(
        self,
        max_norm: Optional[float] = None,
        clip_value: Optional[float] = None,
        norm_type: float = 2.0,
    ):
        if max_norm is not None:
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm, norm_type)
        elif clip_value is not None:
            th.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value)

    def pre_optim_step_hook(self) -> None:
        pass


class Actor(ABC):
    def __init__(
        self,
        env_fn: Callable[..., gym3.Env],
        env_kwargs: dict,
        policy_fn: str,
        policy_kwargs: dict,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        super().__init__()
        self.env = env_fn(**env_kwargs)
        self.policy = get_callable(policy_fn)(device=device, **policy_kwargs)
        self.policy.to(device)
        self.device = device

    @abstractmethod
    def collect(
        self,
        scheduler_step: int,
        buffer: Union[Buffer, RRef],
        learner: Optional[Union[Learner, RRef]] = None,
    ) -> None:
        """
        Collect transitions from interaction with the environment, and
        add collected data to the buffer.
        """
        pass

    def sync_params(self, learner: Union[Learner, RRef]) -> None:
        if isinstance(learner, RRef):
            params = learner.rpc_sync().broadcast_params()
        else:
            params = learner.broadcast_params()
        self.policy.set_params(params)

    def add_batch_to_buffer(
        self, scheduler_step: int, batch: dict, size: int, buffer: Union[Buffer, RRef]
    ) -> None:
        if isinstance(buffer, RRef):
            next_idx = buffer.rpc_sync().update_next_idx(size=size)
            buffer.rpc_sync().add(
                scheduler_step=scheduler_step, data=batch, idx=next_idx, size=size
            )
        else:
            next_idx = buffer.update_next_idx(size=size)
            buffer.add(
                scheduler_step=scheduler_step, data=batch, idx=next_idx, size=size
            )


def worker_class(Actor, Learner):
    class Worker(Actor, Learner):
        def __init__(
            self,
            env_fn: Callable[..., gym3.Env],
            env_kwargs: dict,
            policy_fn: str,
            policy_kwargs: dict,
            device: Union[str, th.device] = "cpu",
            worker_weight: float = 1.0,
        ) -> None:
            self.env = env_fn(**env_kwargs)
            self.policy = get_callable(policy_fn)(device=device, **policy_kwargs)
            self.policy.to(device)
            self.device = device
            self.worker_weight = worker_weight
            # Sync parameters if needed
            self.distributed_on = th.distributed.is_initialized()
            if self.distributed_on:
                for param in self.policy.parameters():
                    th.distributed.broadcast(param, 0)

        def pre_optim_step_hook(self) -> None:
            # All-reduce gradient if needed
            if self.distributed_on:
                total_weight = th.tensor(self.worker_weight, device=self.device)
                th.distributed.all_reduce(total_weight, op=ReduceOp.SUM)
                for param in self.policy.parameters():
                    param.grad.mul_(self.worker_weight)
                    th.distributed.all_reduce(param.grad, op=ReduceOp.SUM)
                    param.grad.div_(total_weight)
            # Call parent's hook (e.g. gradient clipping)
            super().pre_optim_step_hook()

    return Worker
