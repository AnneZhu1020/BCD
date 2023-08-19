import os
import json
import time
from collections import defaultdict
from typing import Tuple, Union

import gym
import gym3
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.distributed.rpc import RRef
from gym_minigrid.wrappers import ImgObsWrapper

from minirl.algos.ppo.agent import PPOWorker
from minirl.algos.ppo.policy import PPODiscretePolicy
from minirl.buffer import Buffer
from minirl.envs.gym3_wrapper import ObsTransposeWrapper
from minirl.utils import explained_variance

import logger
from network import StateEmbeddingNet, ForwardDynamicNet, InverseDynamicNet, ForwardDynamicUncertaintyNet, SR_rep, VQEncoder, VQDecoder, BackDynamicNet
from wrapper import ModifiedEpisodeStatsWrapper

from vq_embedding import VQEmbedding
from codebook import EuclideanCodebook

def make_gym_env(**env_kwargs):
    env = gym.make(**env_kwargs)
    env = ImgObsWrapper(env)
    return env


def make_gym3_env(**kwargs):
    env = gym3.vectorize_gym(**kwargs)
    env = ObsTransposeWrapper(env, axes=(2, 0, 1))
    env = ModifiedEpisodeStatsWrapper(env)
    return env


class BCDPPODiscretePolicy(PPODiscretePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = 128

        self.state_emb_net = StateEmbeddingNet(
            input_shape=kwargs["extractor_kwargs"]["input_shape"], embedding_size=self.embedding_size
        )
        
        self.bisim = kwargs["bisim"]
        self.lap = kwargs["lap"]
        self.contrastive_batch_wise = kwargs["contrastive_batch_wise"]
        self.bisim_delta = kwargs["bisim_delta"]
        self.uncert = kwargs["uncert"]
        self.permute = kwargs["permute"]
        self.uniform = kwargs["uniform"]
        self.sr = kwargs["sr"]
        self.hinge = kwargs["hinge"]
        self.vq_vae = kwargs["vq_vae"]
        self.backward = kwargs["backward"]



        if self.bisim:
            self.coef_bisim = kwargs["coef_bisim"]
        if self.lap:
            self.coef_lap = kwargs["coef_lap"]
        if self.contrastive_batch_wise:
            self.coef_contrastive = kwargs["coef_contrastive"]
        if self.bisim_delta:
            self.coef_bisim_delta = kwargs["coef_bisim_delta"]
            self.lat_sim = kwargs["lat_sim"]


        if self.uncert:
            self.forward_uncert_pred_net = ForwardDynamicUncertaintyNet(num_actions=kwargs["n_actions"])
            self.uncertainty_budget = 0.05
        else:
            self.forward_pred_net = ForwardDynamicNet(num_actions=kwargs["n_actions"])
            if self.backward:
                self.coef_back = kwargs["coef_back"]
                self.backward_pred_net = BackDynamicNet(num_actions=kwargs["n_actions"], embedding_dim = self.embedding_size)

        if self.uniform:
            self.coef_uniform = kwargs["coef_uniform"]
        if self.sr:
            self.sr = SR_rep(input_shape=kwargs["extractor_kwargs"]["input_shape"], num_actions=kwargs["n_actions"], embedding_dim = self.embedding_size)
            self.discount = 0.99
            self.sr_sim = kwargs["sr_sim"]

        if self.hinge:
            self.coef_hinge = kwargs["coef_hinge"]

        self.tran_sim = kwargs["tran_sim"]

        if self.vq_vae:
            self.codebook_size = 8
            self.code_dim = 16
            self.beta = 0.25          
            self.vq = VQEmbedding(self.codebook_size, self.code_dim, self.beta)
            self.vq_encoder = VQEncoder(num_actions=kwargs["n_actions"], embedding_size=self.embedding_size, code_dim=self.code_dim)
            self.vq_decoder = VQDecoder(embedding_size=self.embedding_size, code_dim=self.code_dim)
            self.mse_loss = nn.MSELoss(reduction='none')
            self.coef_vq = kwargs["coef_vq"]
            # self.vq = VectorQuantizerEMA(self.codebook_size, self.code_dim, self.beta)
            self.codebook = EuclideanCodebook(self.code_dim, self.codebook_size)

        self.inverse_pred_net = InverseDynamicNet(num_actions=kwargs["n_actions"], embedding_size=self.embedding_size)


    def mahalanobis_dist(self, tensor1, tensor2, epsilon=1e-8):
        covariance_matrix = th.cov(tensor1.T)
        if th.allclose(covariance_matrix, th.zeros_like(covariance_matrix)):
            covariance_matrix += epsilon * th.eye(covariance_matrix.size(0))
        else:
            covariance_matrix += epsilon * th.diag(th.diagonal(covariance_matrix))
        inv_covariance = th.inverse(covariance_matrix)
        diff = tensor1 - tensor2
        dist = th.sqrt(th.einsum('ij,jk,ik->i', diff, inv_covariance, diff.T))
        return dist

    def lunif(self, x, t=2):
        sq_pdist = th.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def sample_uniform_ball(self, n, eps=1e-10):
        gaussian_rdv = th.FloatTensor(n, self.embedding_dim).normal_(mean=0, std=1)
        gaussian_rdv /= th.norm(gaussian_rdv, dim=-1, keepdim=True) + eps
        uniform_rdv = th.FloatTensor(n, self.embedding_dim).uniform_()
        w = np.sqrt(self.embedding_dim) * gaussian_rdv * uniform_rdv
        w = w.cuda()
        return w

    def compute_logprob(self, state_emb, next_state_emb, actions):
        zex = self.vq_encoder(state_emb, actions, next_state_emb)  # size(1, 2048, 128)
        # zqx: quantized latent vectors, inputs for the decoder, detach codebook
        # selected_codes: selected codes from the codebook (for the VQ objective), codebook with gradients
        zqx, selected_codes = self.vq.straight_through(zex) # size(1, 2048, 128), selected_codes: size(2048, 128)
        pred_next_emb = self.vq_decoder(zqx) # size(1, 2048, 128)
        logprob = -1. * th.norm(pred_next_emb - next_state_emb, dim=-1, p=2) # size(1, 2048)
        # logprob = -1. * self.mse_loss(pred_next_emb, next_state_emb).sum(dim=-1)  # size(1, 2048)
        return logprob, zex, selected_codes, pred_next_emb, zqx


    def compute_vq_loss(self, state_emb, next_state_emb, actions):
        logprob, zex, selected_codes, pred_next_emb, zqx = self.compute_logprob(state_emb, next_state_emb, actions) # logprob: size(1, 2048), zex: (1, 2048, 128), code: size(2048, 128)
        loss = self.vq(zex, selected_codes) - logprob
        return loss.mean(), selected_codes.unsqueeze(0), pred_next_emb, zqx


    def compute_vq_loss2(self, state_emb, next_state_emb, actions):
        zex = self.vq_encoder(state_emb, actions, next_state_emb)  # size(1, 2048, 128)
        vq_loss, quantized, encodings = self.vq(zex)
        pred_next_emb = self.vq_decoder(quantized)
        logprob = -1. * th.norm(pred_next_emb - next_state_emb, dim=-1, p=2) # size(1, 2048)
        loss = vq_loss - logprob
        # loss = vq_loss + F.mse_loss(pred_next_emb, next_state_emb)
        return loss.mean(), quantized, pred_next_emb, encodings

    def compute_vq_loss3(self, state_emb, next_state_emb, actions):
        zex = self.vq_encoder(state_emb, actions, next_state_emb)  # size(1, 2048, 128)
        quantize, embed_ind, dist = self.codebook(zex) # size(1, bs, 16), (1, bs), 
        pred_next_emb = self.vq_decoder(quantize)

        if self.training:
            quantize = zex + (quantize - zex).detach()
        loss = th.tensor([0.], device=state_emb.device, requires_grad=self.training)
        # commit_loss = F.mse_loss(quantize.detach(), zex)
        commit_loss = th.norm(quantize.detach() - zex, dim=-1, p=2)
        vq_loss = loss + commit_loss * self.beta


        logprob = -1. * th.norm(pred_next_emb - next_state_emb, dim=-1, p=2) # size(1, 2048)
        loss = vq_loss - logprob
        return loss.mean(), embed_ind.unsqueeze(0), pred_next_emb, quantize

        # return quantize, embed_ind, loss

    def loss(self, *args, next_obs, **kwargs):
        pg_loss, vf_loss, entropy, extra_out = super().loss(*args, **kwargs)

        obs = th.as_tensor(kwargs["obs"]).to(self.device).float()
        next_obs = th.as_tensor(next_obs).to(self.device).float()
        firsts = th.as_tensor(kwargs["firsts"]).to(self.device)
        next_firsts = th.as_tensor(kwargs["next_firsts"]).to(self.device)
        actions = th.as_tensor(kwargs["actions"]).to(self.device)

        state_emb, _ = self.state_emb_net.extract_features(obs, firsts)
        next_state_emb, _ = self.state_emb_net.extract_features(next_obs, next_firsts)
        pred_actions = self.inverse_pred_net(state_emb, next_state_emb)  # size([1, 2048, 7])
        
        if self.vq_vae:
            # use vq-vae_based to forward prediction
            # forward_loss, selected_codes, pred_next_state_emb, zqx = self.compute_vq_loss(state_emb, next_state_emb, actions)
            # forward_loss, selected_codes, pred_next_state_emb, zqx = self.compute_vq_loss2(state_emb, next_state_emb, actions)
            forward_loss, selected_codes, pred_next_state_emb, quantize = self.compute_vq_loss3(state_emb, next_state_emb, actions)
            
            forward_loss *= self.coef_vq
        else:
            if self.uncert:
                pred_next_state_emb, pred_next_state_std = self.forward_uncert_pred_net(state_emb, actions)
                # mse_loss = F.mse_loss(pred_next_state_emb, next_state_emb, reduction="none")
                mse_loss = th.norm(pred_next_state_emb - next_state_emb, dim=-1, p=2).unsqueeze(-1)  # size(1, 2048)
                mse_loss = mse_loss.repeat(1, 1, pred_next_state_emb.size(-1))
                forward_loss = th.mean(th.exp(-pred_next_state_std) * mse_loss + pred_next_state_std * self.uncertainty_budget)
            else:
                # only forward
                pred_next_state_emb = self.forward_pred_net(state_emb, actions)
                forward_loss = th.norm(pred_next_state_emb - next_state_emb, dim=2, p=2).mean()

        if self.backward:
            pred_next_state_emb = self.backward_pred_net(next_state_emb, actions) # size([1, 2048, 128])
            backward_loss = th.norm(pred_next_state_emb - state_emb, dim=2, p=2).mean()
            forward_loss += backward_loss * self.coef_back  # 1.21

        inverse_loss = F.nll_loss(
            F.log_softmax(th.flatten(pred_actions, 0, 1), dim=-1),
            target=th.flatten(actions, 0, 1),
        )
        perm = th.randperm(state_emb.shape[1])
        state_emb_perm = state_emb[:, perm, :]
        next_state_emb_perm = next_state_emb[:, perm, :]  


        smoothl1 = th.nn.SmoothL1Loss(reduction="none")
        mse = th.nn.MSELoss(reduction="none")  
        cosine = th.nn.CosineSimilarity(dim=-1)

        # 1. compute transition distribution similarity
        if self.vq_vae:
            pred_next_state_emb = quantize # zqx
        if self.tran_sim == 'smooth_l1':
            transition_dist = smoothl1(pred_next_state_emb, pred_next_state_emb[:, perm, :])
        elif self.tran_sim == 'mse':
            transition_dist = mse(pred_next_state_emb, pred_next_state_emb[:, perm, :]).mean(-1).mul(-1).exp()
        elif self.tran_sim == 'cosine':
            transition_dist = cosine(pred_next_state_emb, pred_next_state_emb[:, perm, :]) # size(1, 2048)

        elif self.tran_sim == 'mahalanobis':
            transition_dist = self.mahalanobis_dist(pred_next_state_emb[0], pred_next_state_emb[0, perm, :]).unsqueeze(0)
        # if self.vq_vae:
        #     transition_dist += cosine(selected_codes, selected_codes[:, perm, :])


        # 2. compute state distribution similarity
        if self.bisim_delta:
            if self.lat_sim == 'smooth_l1':
                sim = smoothl1(state_emb[0], state_emb_perm[0])
            elif self.lat_sim == 'mse':
                sim = mse(state_emb[0], state_emb_perm[0]).mul(-1).exp() # size(2048,128)    
                sim = sim.mean(-1).unsqueeze(0)
            elif self.lat_sim == 'mse_delta':
                delta = th.abs(next_state_emb[0] - state_emb[0])
                delta_perm = delta[perm]
                delta_transit = th.abs(delta - delta_perm) # size(2048,128)    
                sim = mse(state_emb[0], state_emb_perm[0]).mul(-1).exp() # size(2048,128)    
                sim = (sim * delta_transit).mean(-1).unsqueeze(0)

            elif self.lat_sim == 'cosine':
                sim = cosine(state_emb[0], state_emb_perm[0]).unsqueeze(0)
            elif self.lat_sim == 'mahalanobis':
                sim = self.mahalanobis_dist

        if self.sr:
            current_sr = self.sr(obs, firsts)[range(obs.size(1)), actions.reshape(-1)]  # size(2048. 7, 128) -> (2048, 128)
            with th.no_grad():
                next_action, _, _ = self.step(obs, firsts)  
                target_sr = self.sr(next_obs, next_firsts)  # size(2048. 7, 128)
                target_sr = state_emb[0] + ~firsts.T * self.discount * target_sr[range(obs.size(1)), th.tensor(next_action).reshape(-1)]  # (2048, 128)
            sr_loss = F.mse_loss(current_sr, target_sr)
            sr_perm = self.sr(obs[:, perm], firsts[:, perm])[range(obs[:, perm].size(1)), actions[:, perm].reshape(-1)]
            if self.sr_sim == 'smooth_l1':
                sr_loss_perm = smoothl1(current_sr, sr_perm)
            elif self.sr_sim == 'mse':
                sr_loss_perm = mse(current_sr, sr_perm).mean(-1).unsqueeze(0)
            elif self.sr_sim == 'cosine':
                sr_loss_perm = cosine(current_sr, sr_perm).unsqueeze(0)
            elif self.sr_sim == 'mahalanobis':
                sr_loss_perm = self.mahalanobis_dist(current_sr, sr_perm)

            # forward_loss += sr_loss * self.coef_sr
            # forward_loss += F.mse_loss(sr_loss_perm, transition_dist) * 0.1 + sr_loss * 100

        # if self.bisim:
        #     bisim_loss = th.abs(z_dist - transition_dist).mean() * self.coef_bisim
        #     bisim_loss_next = th.abs(z_dist_next - transition_dist).mean() * self.coef_bisim
        #     bisim_loss = (bisim_loss + bisim_loss_next) / 2
        #     forward_loss += bisim_loss

        # if self.bisim_delta:
            # delta = th.abs(state_emb[0] - next_state_emb[0]) # size(2048, 128)
            # delta_perm = delta[perm]
            # delta_transit = F.mse_loss(delta, delta_perm, reduction='none')
            # sim = F.mse_loss(state_emb[0], state_emb_perm[0], reduction='none').mul(-1).exp() # size(2048, 128)

            # loss = sim * delta_transit
            # forward_loss += loss.mean() * self.coef_bisim_delta                
            # forward_loss += th.abs(loss.mean(-1) - transition_dist.squeeze(0)).mean() * self.coef_bisim_delta

            # without delta
            # forward_loss += th.abs(sim.mean(-1) - transition_dist.squeeze(0)).mean() * self.coef_bisim_delta

            # action_unique = th.unique(actions)
            # index_list = []
            # for a in action_unique:
            #     a_indices = (actions == a).nonzero(as_tuple=True)[1]
            #     index_list.append(a_indices)
            # for i in index_list:
            #     delta_bucket = delta[i]
            #     state_bucket = state_emb[0, i]  # size(269, 128)
            #     transition_dist_bucket = transition_dist[0, i]

            #     # shuffle_index = th.randperm(state_bucket.size(0))
            #     # state_bucket_perm = state_bucket[shuffle_index]
            #     # delta_bucket_perm = delta_bucket[shuffle_index]

            #     ##########
            #     state_bucket_perm = state_emb_perm[0, i]
            #     delta_bucket_perm = delta_perm[i]
            #     ##########

            #     delta_transit = F.mse_loss(delta_bucket, delta_bucket_perm, reduction='none')  # ||delta_i-delta_j||^2
            #     sim = F.mse_loss(state_bucket, state_bucket_perm, reduction='none').mul(-1).exp()  # e^(-||z_i-z_j||^2)
            #     loss = (sim * delta_transit).mean(-1) 
            #     # forward_loss += th.abs(loss - transition_dist_bucket).mean() * self.coef_bisim_delta
            #     forward_loss += F.mse_loss(loss, transition_dist_bucket).mean() * self.coef_bisim_delta


        if self.uniform:
            forward_loss += self.lunif(state_emb[0]) * self.coef_uniform

        if self.hinge:
            #  E_(s_t, s_k)max(0,epsilon-||f(s_t)-f(s_k)||^2
            epsilon = 0.1
            distance = th.norm(state_emb[0] - state_emb_perm[0], p=2, dim=1) ** 2
            # Apply the hinge loss function
            hinge_loss = F.relu(epsilon - distance)            
            hinge_loss = th.mean(hinge_loss)
            forward_loss += hinge_loss * self.coef_hinge  # 0.0029

        if self.lap:
            # orthonormality loss
            cov = th.matmul(state_emb[0], state_emb[0].T)
            I = th.eye(*cov.size()).to(self.device)
            off_diag = ~I.bool()
            orth_loss_diag = - 2 * cov.diag().mean()
            orth_loss_off_diag = cov[off_diag].pow(2).mean()
            orth_loss = orth_loss_diag * 0.01 + orth_loss_off_diag
            forward_loss += orth_loss * self.coef_lap  # 0.2295

        if self.contrastive_batch_wise:
            p = th.einsum("sd, td -> st", state_emb[0], next_state_emb[0])
            i = th.eye(*p.size(), device=self.device)
            off_diag = ~i.bool()
            batch_wise_loss = p[off_diag].pow(2).mean() - 2 * p.diag().mean() * 0.1
            forward_loss += batch_wise_loss * self.coef_contrastive # 0.0892         



        if self.sr and self.bisim_delta:
            forward_loss += (F.mse_loss(sim, sr_loss_perm + transition_dist) + sr_loss) * self.coef_bisim_delta # 0.122

        
        return pg_loss, vf_loss, entropy, forward_loss, inverse_loss, extra_out


def train(config):
    # Setup logger
    env_name = config["worker_kwargs"]["env_kwargs"]["env_kwargs"]["id"]
    task_name = "-".join(env_name.split("-")[1:-1])
    run_dir = os.path.join(
        config["run_cfg"]["log_dir"],
        task_name,
        f"run_{config['run_cfg']['run_id']}",
        f"back_{config['worker_kwargs']['policy_kwargs']['backward']}_{config['worker_kwargs']['policy_kwargs']['coef_back']}",
        f"vq_vae_{config['worker_kwargs']['policy_kwargs']['vq_vae']}_{config['worker_kwargs']['policy_kwargs']['coef_vq']}",
        f"bisim_delta_{config['worker_kwargs']['policy_kwargs']['bisim_delta']}_{config['worker_kwargs']['policy_kwargs']['coef_bisim_delta']}_{config['worker_kwargs']['policy_kwargs']['lat_sim']}",
        f"sr_{config['worker_kwargs']['policy_kwargs']['sr']}_{config['worker_kwargs']['policy_kwargs']['sr_sim']}",
        f"tran_sim_{config['worker_kwargs']['policy_kwargs']['tran_sim']}",
        f"hinge_{config['worker_kwargs']['policy_kwargs']['hinge']}_{config['worker_kwargs']['policy_kwargs']['coef_hinge']}",
        f"uni_{config['worker_kwargs']['policy_kwargs']['uniform']}_{config['worker_kwargs']['policy_kwargs']['coef_uniform']}",
        f"lap_{config['worker_kwargs']['policy_kwargs']['lap']}_{config['worker_kwargs']['policy_kwargs']['coef_lap']}",
        f"constr_batch_{config['worker_kwargs']['policy_kwargs']['contrastive_batch_wise']}_{config['worker_kwargs']['policy_kwargs']['coef_contrastive']}",
        f"coef_intr{config['intrinsic_reward_coef']}",
        f"coef_invloss{config['worker_kwargs']['inverse_loss_coef']}_for{config['worker_kwargs']['forward_loss_coef']}",
    )
    logger.configure(dir=run_dir, format_strs=["csv", "stdout", "wandb"])
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

    # Modify worker to add bcd intrinsic rewards
    class BCDPPOWorker(PPOWorker):
        def __init__(self, *args, forward_loss_coef, inverse_loss_coef, **kwargs):
            super().__init__(*args, **kwargs)
            self.ep_cnts = [dict() for _ in range(self.env.num)]
            self.forward_loss_coef = forward_loss_coef
            self.inverse_loss_coef = inverse_loss_coef
            self.uncert = kwargs['policy_kwargs']['uncert']
            self.permute = kwargs['policy_kwargs']['permute']
            self.vq_vae = kwargs['policy_kwargs']['vq_vae']
            self.coef_vq = kwargs['policy_kwargs']['coef_vq']

        def collect_batch(self) -> Tuple[dict, np.ndarray, np.ndarray]:
            """
            Additionally, collect next obs
            """
            # Update episodic unique states set before collecting experience
            reward, obs, first = self.env.observe()
            state_keys = [tuple(x) for x in obs.reshape(obs.shape[0], -1).tolist()]
            for env_idx, (key, ep_cnt) in enumerate(zip(state_keys, self.ep_cnts)):
                if first[env_idx]:
                    ep_cnt.clear()
                ep_cnt[key] = 1 + ep_cnt.get(key, 0)

            # Rollout
            batch = defaultdict(list)
            for _ in range(self.n_steps):
                reward, obs, first = self.env.observe()
                action, value, logpacs = self.policy.step(
                    obs[None, ...], first[None, ...]
                )
                batch["obs"].append(obs)
                batch["first"].append(first)
                batch["action"].append(action.squeeze(0))
                batch["value"].append(value.squeeze(0))
                batch["logpac"].append(logpacs.squeeze(0))
                self.env.act(action.squeeze(0))
                reward, next_obs, next_first = self.env.observe()
                batch["next_first"].append(next_first)
                # Calculate BCD intrinsic reward
                
                with th.no_grad():
                    state_emb, _ = self.policy.state_emb_net.extract_features(
                        th.as_tensor(obs[None, ...]).float().to(self.device),
                        th.as_tensor(first[None, ...]).float().to(self.device),
                    )
                    next_state_emb, _ = self.policy.state_emb_net.extract_features(
                        th.as_tensor(next_obs[None, ...]).float().to(self.device),
                        th.as_tensor(next_first[None, ...]).float().to(self.device),
                    )
                    if self.vq_vae:
                        _, _, pred_next_state_emb, _ = self.policy.compute_vq_loss3(state_emb, next_state_emb, th.as_tensor(action).to(self.device))
                    else:
                        if self.uncert:
                            pred_next_state_emb, pred_next_state_std = self.policy.forward_uncert_pred_net(state_emb, th.as_tensor(action).to(self.device))
                        else:
                            pred_next_state_emb = self.policy.forward_pred_net(
                                state_emb, th.as_tensor(action).to(self.device)
                            )
                    
                bcd_rew = th.norm(next_state_emb - pred_next_state_emb, dim=2, p=2)
                bcd_rew = bcd_rew.cpu().numpy().squeeze(0)
                # Record episodic visitation count and calculate curiosity
                ep_curiosity = np.zeros(shape=(self.env.num,), dtype=np.float32)
                state_keys = [
                    tuple(x) for x in next_obs.reshape(next_obs.shape[0], -1).tolist()
                ]
                for env_idx, (key, ep_cnt) in enumerate(zip(state_keys, self.ep_cnts)):
                    if next_first[env_idx]:
                        ep_cnt.clear()
                        ep_cnt[key] = 1
                    else:
                        ep_cnt[key] = 1 + ep_cnt.get(key, 0)
                        if config["ep_curiosity"] == "visit":
                            ep_curiosity[env_idx] = ep_cnt[key] == 1
                        elif config["ep_curiosity"] == "count":
                            ep_curiosity[env_idx] = 1 / np.sqrt(ep_cnt[key])
                # Add into batch
                batch["bcd_rew"].append(bcd_rew)
                batch["ep_curiosity"].append(ep_curiosity)
                batch["reward"].append(reward)
                batch["next_obs"].append(next_obs)
            # Concatenate
            batch["reward"] = np.asarray(batch["reward"], dtype=np.float32)
            batch["obs"] = np.asarray(batch["obs"], dtype=obs.dtype)
            batch["next_obs"] = np.asarray(batch["next_obs"], dtype=next_obs.dtype)
            batch["first"] = np.asarray(batch["first"], dtype=np.bool)
            batch["next_first"] = np.asarray(batch["next_first"], dtype=np.bool)
            batch["ep_curiosity"] = np.asarray(batch["ep_curiosity"], dtype=np.float32)
            batch["bcd_rew"] = np.asarray(batch["bcd_rew"], dtype=np.float32)
            batch["action"] = np.asarray(batch["action"])
            batch["value"] = np.asarray(batch["value"], dtype=np.float32)
            batch["logpac"] = np.asarray(batch["logpac"], dtype=np.float32)
            return batch, next_obs, next_first

        def process_batch(
            self, batch: dict, last_obs: np.ndarray, last_first: np.ndarray, scheduler_step: int
        ):
            """
            Add BCD intrinsic reward
            """
            intrinsic_rewards = batch["bcd_rew"]
            if self.permute:
                to_shuffle = intrinsic_rewards[intrinsic_rewards != 0]
                np.random.shuffle(to_shuffle)
                intrinsic_rewards[intrinsic_rewards != 0] = to_shuffle
            if config["ep_curiosity"] in ("visit", "count"):
                intrinsic_rewards *= batch["ep_curiosity"]
            if config["intrinsic_only"]:
                batch["reward"] = intrinsic_rewards * config["intrinsic_reward_coef"]
            elif config["dropout_int_rew"]: # ep_curiosity => none
                # adaptive dropout
                rate = [0.0075 * scheduler_step + 0.16] * intrinsic_rewards.shape[0]
                rand_mat = np.random.rand(*intrinsic_rewards.shape)
                mask = rand_mat < rate
                batch["ep_curiosity"] = mask
                batch["reward"] = intrinsic_rewards * mask
            else:
                batch["reward"] += intrinsic_rewards * config["intrinsic_reward_coef"]
            super().process_batch(batch=batch, last_obs=last_obs, last_first=last_first, scheduler_step=scheduler_step)
            return batch

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
                    
                    (
                        pg_loss,
                        vf_loss,
                        entropy,
                        forward_loss,
                        inverse_loss,
                        extra_out,
                    ) = self.policy.loss(
                        obs=batch["obs"][sub_indices].swapaxes(0, 1),
                        next_obs=batch["next_obs"][sub_indices].swapaxes(0, 1),
                        advs=batch["adv"][sub_indices].swapaxes(0, 1),
                        firsts=batch["first"][sub_indices].swapaxes(0, 1),
                        next_firsts=batch["next_first"][sub_indices].swapaxes(0, 1),
                        actions=batch["action"][sub_indices].swapaxes(0, 1),
                        old_values=batch["value"][sub_indices].swapaxes(0, 1),
                        old_logpacs=batch["logpac"][sub_indices].swapaxes(0, 1),
                        rnn_states=rnn_states,
                        clip_range=cur_clip_range,
                        vf_clip_range=cur_vf_clip_range,
                        normalize_adv=self.normalize_adv,
                    )
                    total_loss = (
                        pg_loss
                        + self.vf_loss_coef * vf_loss
                        - self.entropy_coef * entropy
                        + self.forward_loss_coef * forward_loss
                        + self.inverse_loss_coef * inverse_loss
                    )
                    total_loss.backward()
                    self.pre_optim_step_hook()
                    self.optimizer.step()
                    # Saving statistics
                    stats_dict["policy_loss"].append(pg_loss.item())
                    stats_dict["value_loss"].append(vf_loss.item())
                    stats_dict["forward_loss"].append(forward_loss.item())
                    stats_dict["inverse_loss"].append(inverse_loss.item())
                    stats_dict["entropy"].append(entropy.item())
                    stats_dict["total_loss"].append(total_loss.item())
                    stats_dict["ep_percentage"].append(batch["ep_curiosity"][sub_indices].sum()/len(batch["ep_curiosity"][sub_indices]))
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

    # Create worker
    worker = BCDPPOWorker(**config["worker_kwargs"])

    # Create buffer
    buffer_size = worker.env.num * worker.n_steps
    buffer = Buffer(max_size=buffer_size, sequence_length=worker.n_steps)

    # Training
    n_iters = int(config["run_cfg"]["n_timesteps"] / worker.env.num / worker.n_steps)
    for i in range(n_iters):
        t_start = time.perf_counter()
        # Collect data
        worker.collect(scheduler_step=i, buffer=buffer)
        # Learn on data
        stats_dict = worker.learn(scheduler_step=i, buffer=buffer)
        # Logging
        ret = worker.env.callmethod("get_ep_stat_mean", "r")
        finish = worker.env.callmethod("get_ep_stat_mean", "finish")
        logger.logkv("time", time.perf_counter() - t_start)
        logger.logkv("iter", i + 1)
        logger.logkv("return", ret)
        logger.logkv("success", finish)
        for key, value in stats_dict.items():
            logger.logkv(key, value)
        logger.dumpkvs()

    # Save model
    th.save(worker.policy.state_dict(), os.path.join(run_dir, "policy.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--n_timesteps", type=int, default=int(4e7))
    parser.add_argument("--env_id", type=str, default="MiniGrid-KeyCorridorS4R3-v0")
    parser.add_argument("--n_envs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4) 
    parser.add_argument("--n_steps", type=float, default=128)
    parser.add_argument("--n_epochs", type=float, default=4)
    parser.add_argument("--n_minibatches", type=int, default=8)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--intrinsic_only", action="store_true")
    parser.add_argument("--intrinsic_reward_coef", type=float, default=1e-3)
    parser.add_argument(
        "--ep_curiosity", type=str, choices=("visit", "count", "none"), default="none"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bisim", action='store_true', default=False)
    parser.add_argument("--coef_bisim", type=float, default=1e2)
    parser.add_argument("--forward_loss_coef", type=float, default=10)
    parser.add_argument("--inverse_loss_coef", type=float, default=0.1)
    parser.add_argument("--dropout_int_rew", type=bool, default=False)

    parser.add_argument("--lap", action='store_true', default=False)
    parser.add_argument("--coef_lap", type=float, default=1e2)
    parser.add_argument("--contrastive_batch_wise", action='store_true', default=False)
    parser.add_argument("--coef_contrastive", type=float, default=1)
    parser.add_argument("--bisim_delta", action='store_true', default=False)
    parser.add_argument("--coef_bisim_delta", type=float, default=1)
    parser.add_argument("--uncert", action='store_true', default=False)
    parser.add_argument("--permute", action='store_true', default=False)
    parser.add_argument("--uniform", action='store_true', default=False)
    parser.add_argument("--coef_uniform", type=float, default=1)
    parser.add_argument("--sr", action='store_true', default=False)
    parser.add_argument("--hinge", action='store_true', default=False)
    parser.add_argument("--coef_hinge", type=float, default=1)
    parser.add_argument("--vq_vae", action='store_true', default=False)
    parser.add_argument("--coef_vq", type=float, default=1)
    
    parser.add_argument("--backward", action='store_true', default=False)
    parser.add_argument("--coef_back", type=float, default=1)

    parser.add_argument("--lat_sim", type=str, choices=("smoothl1", "mse", "cosine", "mse_delta"), default="none")
    parser.add_argument("--tran_sim", type=str, choices=("smoothl1", "mse", "cosine"), default="none")
    parser.add_argument("--sr_sim", type=str, choices=("smoothl1", "mse", "cosine"), default="none")
    args = parser.parse_args()
    config = {
        # Run
        "run_cfg": {
            "run_id": args.run_id,
            "log_dir": f"./exps/bcd_{args.ep_curiosity}/",
            "n_timesteps": args.n_timesteps,
        },
        # Agent
        "worker_kwargs": {
            "env_fn": make_gym3_env,
            "env_kwargs": {
                "env_fn": make_gym_env,
                "num": args.n_envs,
                "env_kwargs": {"id": args.env_id},
                "use_subproc": False,
            },
            "policy_fn": "__main__.BCDPPODiscretePolicy",
            "policy_kwargs": {
                "extractor_fn": "cnn",
                "extractor_kwargs": {
                    "input_shape": (3, 7, 7),
                    "conv_kwargs": (
                        {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                        {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                        {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                    ),
                    "activation": nn.ELU,
                    "hiddens": (512,),
                    "final_activation": nn.ReLU,
                },
                "n_actions": 7,
                "bisim": args.bisim,
                "coef_bisim": args.coef_bisim,
                "lap": args.lap,
                "coef_lap": args.coef_lap,
                "contrastive_batch_wise": args.contrastive_batch_wise,
                "coef_contrastive": args.coef_contrastive,
                "bisim_delta": args.bisim_delta,
                "coef_bisim_delta": args.coef_bisim_delta,
                "uncert": args.uncert,
                "permute": args.permute,
                "uniform": args.uniform,
                "coef_uniform": args.coef_uniform,
                "sr": args.sr,
                "hinge": args.hinge,
                "coef_hinge": args.coef_hinge,
                "lat_sim": args.lat_sim,
                "tran_sim": args.tran_sim,
                "sr_sim": args.sr_sim,
                "vq_vae": args.vq_vae,
                "coef_vq": args.coef_vq,
                "backward": args.backward,
                "coef_back": args.coef_back,
            },
            "forward_loss_coef": args.forward_loss_coef,
            "inverse_loss_coef": args.inverse_loss_coef,
            "optimizer_fn": "torch.optim.Adam",
            "optimizer_kwargs": {"lr": args.lr},
            "n_steps": args.n_steps,
            "n_epochs": args.n_epochs,
            "n_minibatches": args.n_minibatches,
            "discount_gamma": args.discount_gamma,
            "gae_lambda": args.gae_lambda,
            "normalize_adv": True,
            "clip_range": args.clip_range,
            "entropy_coef": args.entropy_coef,
            "device": args.device,
        },
        "ep_curiosity": args.ep_curiosity,
        "intrinsic_only": args.intrinsic_only,
        "intrinsic_reward_coef": args.intrinsic_reward_coef,
        "dropout_int_rew": args.dropout_int_rew,
    }

    train(config)
