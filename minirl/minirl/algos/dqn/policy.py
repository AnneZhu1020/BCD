from collections import OrderedDict
from copy import deepcopy
from more_itertools import pairwise
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from minirl.utils import get_callable


class DQNDiscretePolicy(nn.Module):
    def __init__(
        self,
        extractor_fn,
        extractor_kwargs: dict,
        n_actions: int,
        device,
        hiddens: Tuple[int, ...] = (),
        double_q: bool = False,
        shared_action_bias: bool = False,
        dueling: bool = False,
        init_weight_fn: Optional[str] = None,
    ):
        super().__init__()
        extractor = get_callable(extractor_fn)(**extractor_kwargs)
        hiddens = (extractor.output_dim, *hiddens)
        self.online_net = nn.ModuleDict(
            {
                "extractor": extractor,
                "action_vf": nn.Sequential(
                    *(nn.Linear(n_in, n_out) for n_in, n_out in pairwise(hiddens)),
                    nn.Linear(hiddens[-1], n_actions, bias=not shared_action_bias)
                ),
            }
        )
        if shared_action_bias:
            self.online_net["action_vf"].register_parameter(
                "bias", nn.Parameter(torch.zeros(1))
            )
        if dueling:
            state_vf = nn.Sequential(
                *(nn.Linear(n_in, n_out) for n_in, n_out in pairwise(hiddens)),
                nn.Linear(hiddens[-1], 1)
            )
            self.online_net.update([["state_vf", state_vf]])
        self.target_net = deepcopy(self.online_net)
        self.n_actions = n_actions
        self.device = device
        self.double_q = double_q
        self.shared_action_bias = shared_action_bias
        self.dueling = dueling
        if init_weight_fn is not None:
            get_callable(init_weight_fn)(self)
        self.update_target_net()

    @torch.no_grad()
    def step(self, obs, eps: float) -> Tuple[Any, Any, Any]:
        obs = torch.from_numpy(obs).to(self.device)
        fts = self.online_net["extractor"](self.preprocess(obs))
        determ_actions = torch.argmax(self.online_net["action_vf"](fts), axis=1)
        random_actions = torch.randint_like(determ_actions, high=self.n_actions)
        choose_random = torch.rand_like(determ_actions, dtype=torch.float) < eps
        actions = torch.where(choose_random, random_actions, determ_actions)
        return actions.cpu().numpy()

    def get_params(self):
        params = OrderedDict(
            {name: weight.cpu() for name, weight in self.state_dict().items()}
        )
        return params

    def set_params(self, params):
        self.load_state_dict(params)

    def preprocess(self, obs):
        return obs / 255.0

    def loss(self, obs, actions, rewards, new_obs, firsts, gamma, weights=None):
        # Convert from numpy array to torch tensor
        obs = torch.from_numpy(obs).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        new_obs = torch.from_numpy(new_obs).to(self.device)
        firsts = torch.from_numpy(firsts).to(self.device)
        # Q-network evaluation
        q_t = self.compute_q_values(net=self.online_net, obs=obs)  # Q(s_t)
        q_t_selected = q_t[range(len(actions)), actions]
        # Target Q-network evaluation
        with torch.no_grad():
            tq_tp1 = self.compute_q_values(net=self.target_net, obs=new_obs)  # Q'(s_{t+1})
            if self.double_q:
                q_tp1 = self.compute_q_values(net=self.online_net, obs=new_obs)  # Q(s_{t+1})
                actions_selected = q_tp1.max(dim=1).indices
                tq_tp1_selected = tq_tp1[range(len(actions)), actions_selected]
            else:
                tq_tp1_selected = tq_tp1.max(dim=1).values
            tq_tp1_masked = (1.0 - firsts.float()) * tq_tp1_selected
            td_target = rewards + gamma * tq_tp1_masked
        # Compute loss
        losses = F.smooth_l1_loss(
            input=q_t_selected, target=td_target, reduction="none"
        )
        # Average loss
        if weights is not None:
            weights = torch.from_numpy(weights).to(self.device)
            losses = losses * weights
        loss = losses.mean()
        # Calculate additional quantities
        extra_out = {}
        with torch.no_grad():
            if weights is not None:
                extra_out["td_error"] = q_t_selected - td_target
        return loss, extra_out

    def compute_q_values(self, net, obs):
        fts = net["extractor"](self.preprocess(obs))
        action_scores = net["action_vf"](fts)
        if self.shared_action_bias:
            action_scores += net["action_vf"].bias
        if self.dueling:
            state_score = net["state_vf"](fts)
            action_scores = action_scores - action_scores.mean(dim=-1, keepdim=True)
            q_values = state_score + action_scores
        else:
            q_values = action_scores
        return q_values

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
