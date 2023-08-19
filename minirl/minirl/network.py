from more_itertools import pairwise
from typing import Optional, Sequence, Tuple, Type

import torch as th
from torch import nn

from minirl.torch_utils import infer_fc_input_dim

mapping = {}


def register(name: str):
    def _thunk(network_class: Type[nn.Module]):
        mapping[name] = network_class
        return network_class

    return _thunk


def get_network_class(name: str):
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError("Unknown network class: {}".format(name))


@register("mlp")
class MLP(nn.Module):
    """
    Simple fully connected network
    """

    def __init__(
        self,
        input_dim: int,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        final_activation: Type[nn.Module] = nn.Identity,
    ):
        super().__init__()
        hiddens = (input_dim, *hiddens)
        n_layers = len(hiddens) - 1
        layers = []
        if n_layers > 0:
            for i, (n_in, n_out) in enumerate(pairwise(hiddens), start=1):
                layers.append(nn.Linear(in_features=n_in, out_features=n_out))
                layers.append(activation() if i < n_layers else final_activation())

        self.fc = nn.Sequential(*layers)
        self.output_dim = hiddens[-1]
        self.is_recurrent = False

    def forward(self, x):
        x = self.fc(x)
        return x


@register("cnn")
class CNN(nn.Module):
    """
    Simple convolutional network
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conv_kwargs: Tuple[dict, ...],
        activation: Type[nn.Module] = nn.ReLU,
        hiddens: Optional[Tuple[int, ...]] = (256,),
        final_activation: Type[nn.Module] = nn.Identity,
    ):
        super().__init__()
        c, h, w = input_shape

        # Build conv layers
        n_channels = (c, *(kwargs["out_channels"] for kwargs in conv_kwargs))
        convs = []
        for i, n_in in enumerate(n_channels[:-1]):
            convs.append(nn.Conv2d(in_channels=n_in, **conv_kwargs[i]))
            convs.append(activation())
        convs.append(nn.Flatten())
        self.convs = nn.Sequential(*convs)
        output_dim = infer_fc_input_dim(self.convs, input_shape)

        # Build fully connected layers
        self.hiddens = hiddens
        if hiddens is not None:
            self.mlp = MLP(
                input_dim=output_dim,
                hiddens=hiddens,
                activation=activation,
                final_activation=final_activation,
            )
            self.output_dim = self.mlp.output_dim
        self.is_recurrent = False

    def forward(self, x):
        x = self.convs(x)
        if self.hiddens is not None:
            x = self.mlp(x)
        return x


@register("rnn")
class RNN(nn.Module):
    """
    Simple recurrent network
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: Type[nn.RNNBase] = nn.LSTM,
        rnn_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.recurrent_layers = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **(rnn_kwargs or {}),
        )
        self.output_dim = hidden_size
        self.is_recurrent = True

    def forward(self, x, states=None):
        x, states = self.recurrent_layers(x, states)
        return x, states


@register("mlp_rnn")
class MLPRNN(nn.Module):
    """
    Simple recurrent network
    """

    def __init__(
        self,
        input_dim: int,
        rnn_hidden_size: int,
        num_rnn_layers: int,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        rnn_type: Type[nn.RNNBase] = nn.LSTM,
        rnn_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.mlp = MLP(
            input_dim=input_dim,
            activation=activation,
            hiddens=hiddens,
            final_activation=activation,
        )
        self.rnn = RNN(
            input_size=self.mlp.output_dim,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            **(rnn_kwargs or {}),
        )
        self.output_dim = rnn_hidden_size
        self.is_recurrent = True

    def forward(self, x, states=None):
        T, B, *shape = x.shape
        x = self.mlp(x.view(T * B, *shape)).view(T, B, -1)
        x, states = self.rnn(x, states)
        return x, states


@register("cnn_rnn")
class CNNRNN(nn.Module):
    """
    Simple recurrent network
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conv_kwargs: Tuple[dict, ...],
        rnn_hidden_size: int,
        num_rnn_layers: int,
        activation: Type[nn.Module] = nn.ReLU,
        hiddens: Optional[Tuple[int, ...]] = (256,),
        rnn_type: Type[nn.RNNBase] = nn.LSTM,
        rnn_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.cnn = CNN(
            input_shape=input_shape,
            conv_kwargs=conv_kwargs,
            activation=activation,
            hiddens=hiddens,
            final_activation=activation,
        )
        self.rnn = rnn_type(
            input_size=self.cnn.output_dim,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            **(rnn_kwargs or {}),
        )
        self.output_dim = rnn_hidden_size
        self.is_recurrent = True

    def forward(self, x, states=None):
        T, B, *shape = x.shape
        x = self.cnn(x.view(T * B, *shape)).view(T, B, -1)
        x, states = self.rnn(x, states)
        return x, states


@register("nature_cnn")
class NatureCNN(CNN):
    """
    CNN network used in the paper
    "Human Level Control Through Deep Reinforcement Learning"
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        hiddens: Optional[Tuple[int, ...]] = (512,),
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            input_shape=input_shape,
            conv_kwargs=(
                {"out_channels": 32, "kernel_size": 8, "stride": 4},
                {"out_channels": 64, "kernel_size": 4, "stride": 2},
                {"out_channels": 64, "kernel_size": 3, "stride": 1},
            ),
            hiddens=hiddens,
            activation=activation,
            final_activation=activation,
        )


class ImpalaBlock(nn.Module):
    """
    Conv sequence in Impala CNN
    """

    def __init__(self, in_channels, depth):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=depth, kernel_size=3, padding=1
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, padding=1),
        )
        self.res2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        return x


@register("impala_cnn")
class ImpalaCNN(nn.Module):
    """
    CNN network used in the paper
    "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        depths: Tuple[int, ...] = (16, 32, 32),
        hiddens: Optional[Tuple[int, ...]] = (256,),
    ):
        super().__init__()
        c, h, w = input_shape

        depths = (c, *depths)
        self.convs = nn.Sequential(
            *(ImpalaBlock(n_in, depth) for n_in, depth in pairwise(depths)),
            nn.Flatten(),
            nn.ReLU(),
        )
        output_dim = infer_fc_input_dim(self.convs, input_shape)
        if hiddens is not None:
            layers = []
            for n_channel in hiddens:
                layers.append(nn.Linear(in_features=output_dim, out_features=n_channel))
                layers.append(nn.ReLU())
                output_dim = n_channel
            self.fc = nn.Sequential(*layers)
        self.hiddens = hiddens
        self.output_dim = output_dim

    def forward(self, x):
        x = self.convs(x)
        if self.hiddens is not None:
            x = self.fc(x)
        return x
