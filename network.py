from typing import Any
import torch as th
from torch import nn
from torch.nn import functional as F

from torch.autograd import Function

from minirl.common.policy import Extractor


class StateEmbeddingNet(Extractor):
    def __init__(self, input_shape, embedding_size) -> None:
        super().__init__(
            extractor_fn="cnn",
            extractor_kwargs={
                "input_shape": input_shape,
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
                        "out_channels": embedding_size,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ),
                "hiddens": (),
                "activation": nn.ELU,
            },
        )


class ForwardDynamicNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.forward_dynamics = nn.Sequential(
            nn.Linear(128 + self.num_actions, 256),
            nn.ReLU(),
        )
        self.fd_out = nn.Linear(256, 128)

    def forward(self, state_embedding, action):
        # Embedding shape: T x B x C
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = th.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb

class ForwardDynamicUncertaintyNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.forward_dynamics_mean = nn.Sequential(
            nn.Linear(128 + self.num_actions, 256),
            nn.ReLU(),
        )
        self.fd_out_mean = nn.Linear(256, 128)
        self.forward_dynamics_std = nn.Sequential(
            nn.Linear(128 + self.num_actions, 256),
            nn.ReLU(),
        )
        self.fd_out_std = nn.Linear(256, 128)

    @staticmethod
    def normalize(x):
        return x / th.sqrt(th.pow(x, 2).sum(dim=-1, keepdim=True))

    def forward(self, state_embedding, action):
        # Embedding shape: T x B x C
        # state_embedding = self.normalize(state_embedding)

        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = th.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb_mean = self.fd_out_mean(self.forward_dynamics_mean(inputs))
        next_state_emb_std = self.fd_out_std(self.forward_dynamics_std(inputs))

        return next_state_emb_mean, next_state_emb_std


class InverseDynamicNet(nn.Module):
    def __init__(self, num_actions, embedding_size):
        super().__init__()
        self.num_actions = num_actions
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(2 * embedding_size, 256),
            nn.ReLU(),
        )
        self.id_out = nn.Linear(256, num_actions)

    @staticmethod
    def normalize(x):
        return x / th.sqrt(th.pow(x, 2).sum(dim=-1, keepdim=True))

    def forward(self, state_embedding, next_state_embedding):
        # Embedding shape: T x B x C
        # state_embedding = self.normalize(state_embedding)
        # next_state_embedding = self.normalize(next_state_embedding)

        inputs = th.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits

class SR_rep(Extractor):
    def __init__(self, input_shape, num_actions, embedding_dim) -> None:
        super().__init__(
            extractor_fn="cnn",
            extractor_kwargs={
                "input_shape": input_shape,
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
                        "out_channels": embedding_dim * num_actions,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ),
                "hiddens": (),
                "activation": nn.ELU,
            },
        )
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

    def forward(self, obs, firsts):
        emb, _ = self.extract_features(obs, firsts)
        return emb.reshape(-1, self.num_actions, self.embedding_dim)


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx: Any, inputs, codebook):
        with th.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = th.sum(codebook ** 2, dim=1)
            inputs_sqr = th.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Euclidean distance
            distances = th.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = th.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            
            ctx.mark_non_differentiable(indices)

            return indices
        
    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx: Any, inputs, codebook):
        indices = vector_quantization(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = th.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)


vector_quantization = VectorQuantization.apply
vector_quantization_st = VectorQuantizationStraightThrough.apply
__all__ = [vector_quantization, vector_quantization_st]


class VQEncoder(nn.Module):
    def __init__(self, num_actions, embedding_size, code_dim):
        super().__init__()
        self.num_actions = num_actions
        self.forward_dynamics = nn.Sequential(
            nn.Linear(2 * embedding_size + self.num_actions, 64),
            nn.ReLU(),
        )
        self.fd_out = nn.Linear(64, code_dim)

    def forward(self, state_embedding, action, next_state_embedding):
        # Embedding shape: T x B x C
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = th.cat((state_embedding, action_one_hot, next_state_embedding), dim=2)
        latent = self.fd_out(self.forward_dynamics(inputs))
        return latent

class VQDecoder(nn.Module):
    def __init__(self, embedding_size, code_dim):
        super().__init__()
        self.forward_dynamics = nn.Sequential(
            nn.Linear(code_dim, 64),
            nn.ReLU(),
        )
        self.fd_out = nn.Linear(64, embedding_size)

    def forward(self, zqx):
        # Embedding shape: T x B x C
        x_decoder = self.fd_out(self.forward_dynamics(zqx))
        return x_decoder


class BackDynamicNet(nn.Module):
    def __init__(self, num_actions, embedding_dim):
        super().__init__()
        self.num_actions = num_actions
        self.backward_dynamics = nn.Sequential(
            nn.Linear(embedding_dim + self.num_actions, 256),
            nn.ReLU(),
        )
        self.fd_out = nn.Linear(256, embedding_dim)

    def forward(self, next_state_embedding, action):
        # Embedding shape: T x B x C
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = th.cat((next_state_embedding, action_one_hot), dim=2)
        pred_state_emb = self.fd_out(self.backward_dynamics(inputs))
        return pred_state_emb