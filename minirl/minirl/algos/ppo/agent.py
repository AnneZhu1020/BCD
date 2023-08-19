from typing import Callable, Optional, Tuple, Union
from collections import defaultdict

import gym3
import numpy as np
import torch as th
from torch.distributed.rpc import RRef

from minirl.common.agent import Actor, Learner, worker_class
from minirl.buffer import Buffer
from minirl.type_utils import Schedulable
from minirl.utils import (
    calculate_gae,
    swap_flatten_01,
    get_callable,
    get_scheduler,
    explained_variance,
)


class PPOActor(Actor):
    def __init__(
        self,
        env_fn: Callable[..., gym3.Env],
        env_kwargs: dict,
        policy_fn: str,
        policy_kwargs: dict,
        n_steps: int,
        discount_gamma: float = 0.99,
        gae_lambda: float = 1.0,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(env_fn, env_kwargs, policy_fn, policy_kwargs, device)
        self.n_steps = n_steps
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda

    def collect(
        self,
        scheduler_step: int,
        buffer: Union[Buffer, RRef],
        learner: Optional[Union[Learner, RRef]] = None,
    ) -> None:
        # Sync parameters if needed
        if learner is not None:
            self.sync_params(learner)
        # Collect a batch of samples
        batch, last_obs, last_first = self.collect_batch()
        # Compute advantage
        batch = self.process_batch(batch, last_obs, last_first, scheduler_step)
        # Organize axes
        for key in batch:
            batch[key] = batch[key].swapaxes(0, 1)
        # Send data to buffer
        self.add_batch_to_buffer(
            scheduler_step=scheduler_step,
            batch=batch,
            size=len(batch["obs"]),
            buffer=buffer,
        )

    def collect_batch(self) -> Tuple[dict, np.ndarray, np.ndarray]:
        """
        Collect a batch of trajectories
        """
        batch = defaultdict(list)
        if self.policy.is_recurrent:
            batch["rnn_states"] = self.policy.rnn_states
        for i in range(self.n_steps):
            reward, obs, first = self.env.observe()
            action, value, logpacs = self.policy.step(obs[None, ...], first[None, ...])
            batch["obs"].append(obs)
            batch["first"].append(first)
            batch["action"].append(action.squeeze(0))
            batch["value"].append(value.squeeze(0))
            batch["logpac"].append(logpacs.squeeze(0))
            self.env.act(action.squeeze(0))
            reward, obs, first = self.env.observe()
            batch["reward"].append(reward)
        if self.policy.is_recurrent:
            if batch["rnn_states"] is None:
                rnn_states = tuple(self.policy.rnn_states)
                batch["rnn_states"] = tuple(th.zeros_like(s) for s in rnn_states)
            batch["rnn_states"] = th.stack(batch["rnn_states"], dim=-1).cpu().numpy()
        # Concatenate
        batch["reward"] = np.asarray(batch["reward"], dtype=np.float32)
        batch["obs"] = np.asarray(batch["obs"], dtype=obs.dtype)
        batch["first"] = np.asarray(batch["first"], dtype=np.bool)
        batch["action"] = np.asarray(batch["action"])
        batch["value"] = np.asarray(batch["value"], dtype=np.float32)
        batch["logpac"] = np.asarray(batch["logpac"], dtype=np.float32)
        return batch, obs, first

    def process_batch(self, batch: dict, last_obs: np.ndarray, last_first: np.ndarray, scheduler_step: int):
        """
        Process the collected batch, e.g. computing advantages
        """
        last_value = self.policy.value(last_obs[None, ...], last_first[None, ...])
        advs = calculate_gae(
            rewards=batch["reward"],
            values=batch["value"],
            firsts=batch["first"],
            last_value=last_value,
            last_first=last_first,
            discount_gamma=self.discount_gamma,
            gae_lambda=self.gae_lambda,
        )
        batch["adv"] = advs
        return batch

    def eval_collect(self, learner_rref: Optional[RRef] = None):
        # Sync parameters if needed
        if learner_rref is not None:
            self.sync_params(learner_rref)
        # Collect samples
        for i in range(self.n_steps):
            reward, obs, first = self.env.observe()
            action, value, logpacs = self.policy.step(obs)
            self.env.act(action)


class PPOLearner(Learner):
    def __init__(
        self,
        policy_fn: str,
        policy_kwargs: dict,
        optimizer_fn: str,
        optimizer_kwargs: dict,
        n_epochs: int,
        n_minibatches: int,
        normalize_adv: bool = False,
        clip_range: Schedulable = 0.2,
        vf_clip_range: Schedulable = 0.2,
        vf_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(policy_fn, policy_kwargs, device)
        self.optimizer = get_callable(optimizer_fn)(
            params=self.policy.parameters(), **optimizer_kwargs
        )
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.normalize_adv = normalize_adv
        self.clip_range = get_scheduler(clip_range)
        self.vf_clip_range = get_scheduler(vf_clip_range)
        self.vf_loss_coef = vf_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def learn(self, scheduler_step: int, buffer: Union[Buffer, RRef]):
        # Retrieve data from buffer
        if isinstance(buffer, RRef):
            batch = buffer.rpc_sync().get_all()
        else:
            batch = buffer.get_all()
        # Build a dict to save training statistics
        stats_dict = defaultdict(list)
        # Minibatch training
        B, T = batch["obs"].shape[:2]
        if self.policy.is_recurrent:
            batch_size = B
            indices = np.arange(B)
        else:
            batch_size = B * T
            indices = np.mgrid[0:B, 0:T].reshape(2, batch_size).T
        minibatch_size = batch_size // self.n_minibatches
        assert minibatch_size > 1
        # Get current clip range
        cur_clip_range = self.clip_range.value(step=scheduler_step)
        cur_vf_clip_range = self.vf_clip_range.value(step=scheduler_step)
        # Train for n_epochs
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                if self.policy.is_recurrent:
                    sub_indices = indices[start:end]
                    rnn_states = batch["rnn_states"][sub_indices].swapaxes(0, 1)
                else:
                    sub_indices = indices[start:end]
                    sub_indices = tuple(sub_indices.T) + (None,)
                    rnn_states = None
                self.optimizer.zero_grad()
                pg_loss, vf_loss, entropy, extra_out = self.policy.loss(
                    obs=batch["obs"][sub_indices].swapaxes(0, 1),
                    advs=batch["adv"][sub_indices].swapaxes(0, 1),
                    firsts=batch["first"][sub_indices].swapaxes(0, 1),
                    actions=batch["action"][sub_indices].swapaxes(0, 1),
                    old_values=batch["value"][sub_indices].swapaxes(0, 1),
                    old_logpacs=batch["logpac"][sub_indices].swapaxes(0, 1),
                    rnn_states=rnn_states,
                    clip_range=cur_clip_range,
                    vf_clip_range=cur_vf_clip_range,
                    normalize_adv=self.normalize_adv,
                )
                total_loss = (
                    pg_loss + self.vf_loss_coef * vf_loss - self.entropy_coef * entropy
                )
                total_loss.backward()
                self.pre_optim_step_hook()
                self.optimizer.step()
                # Saving statistics
                stats_dict["policy_loss"].append(pg_loss.item())
                stats_dict["value_loss"].append(vf_loss.item())
                stats_dict["entropy"].append(entropy.item())
                stats_dict["total_loss"].append(total_loss.item())
                for key in extra_out:
                    stats_dict[key].append(extra_out[key].item())
        # Compute mean
        for key in stats_dict:
            stats_dict[key] = np.mean(stats_dict[key])
        # Compute explained variance
        stats_dict["explained_variance"] = explained_variance(
            y_pred=batch["value"], y_true=batch["value"] + batch["adv"]
        )
        return stats_dict

    def pre_optim_step_hook(self):
        self.clip_gradient(max_norm=self.max_grad_norm)


class PPOWorker(worker_class(PPOActor, PPOLearner)):
    def __init__(
        self,
        env_fn: Callable[..., gym3.Env],
        env_kwargs: dict,
        policy_fn: str,
        policy_kwargs: dict,
        optimizer_fn: str,
        optimizer_kwargs: dict,
        n_steps: int,
        n_epochs: int,
        n_minibatches: int,
        discount_gamma: float = 0.99,
        gae_lambda: float = 1.0,
        normalize_adv: bool = False,
        clip_range: Schedulable = 0.2,
        vf_clip_range: Schedulable = 0.2,
        vf_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        device: Union[str, th.device] = "cpu",
        worker_weight: float = 1.0,
    ):
        super().__init__(
            env_fn, env_kwargs, policy_fn, policy_kwargs, device, worker_weight
        )
        self.optimizer = get_callable(optimizer_fn)(
            params=self.policy.parameters(), **optimizer_kwargs
        )
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.clip_range = get_scheduler(clip_range)
        self.vf_clip_range = get_scheduler(vf_clip_range)
        self.vf_loss_coef = vf_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
