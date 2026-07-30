
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blocks import src_utils

class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class _ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_fn,
        norm: str,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int
    ):
        """PyTorch module implementing a residual block module used in `TCN_net`.
        Parameters
        ----------
        num_filters
            The number of filters in a convolutional layer of the TCN.
        kernel_size
            The size of every kernel in a convolutional layer.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout_fn
            The dropout function to be applied to every convolutional layer.
        norm
            Boolean value indicating whether to use weight normalization.
        nr_blocks_below
            The number of residual blocks before the current one.
        num_layers
            The number of convolutional layers.
        input_size
            The dimensionality of the input time series of the whole network.
        target_size
            The dimensionality of the output time series of the whole network.
        Inputs
        ------
        x of shape `(batch_size, in_dimension, input_chunk_length)`
            Tensor containing the features of the input sequence.
            in_dimension is equal to `input_size` if this is the first residual block,
            in all other cases it is equal to `num_filters`.
        Outputs
        -------
        y of shape `(batch_size, out_dimension, input_chunk_length)`
            Tensor containing the output sequence of the residual block.
            out_dimension is equal to `output_size` if this is the last residual block,
            in all other cases it is equal to `num_filters`.
        """
        super(_ResidualBlock, self).__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        if norm == "weightnorm":
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1
            ), nn.utils.weight_norm(self.conv2)
            self.layer_norm = False

        elif norm == "layernorm":
            self.layer_norm = True
            self.layernorm1, self.layernorm2 = src_utils.LayerNorm(num_filters), src_utils.LayerNorm(output_dim) # [num_filters, 150]
        else:
            self.layer_norm = False

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base**self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0))
        x = self.conv1(x)
        if self.layer_norm:
            x = self.layernorm1(x)
        x = self.dropout_fn(F.relu(x))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.layer_norm:
            x = self.layernorm2(x)
        
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual

        return x

class TCN_net(nn.Module):
    def __init__(
        self,
        max_input_length: int,
        input_size: int,
        kernel_size: int,
        num_filters: int,

        num_layers: Optional[int] = None,
        dilation_base: int = 2,
        norm: str = "weightnorm",
        nr_params: int = 1,
        dropout: float = 0.2,
    ):

        """PyTorch module implementing a dilated TCN module used in `TCNModel`.
        Parameters
        ----------
        input_size
            The dimensionality of the input time series.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        num_layers
            The number of convolutional layers.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout
            The dropout rate for every convolutional layer.
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.
        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_size)`
            Tensor containing the features of the input sequence.
        Outputs
        -------
        y of shape `(batch_size, input_chunk_length, target_size, nr_params)`
            Tensor containing the predictions of the next 'output_chunk_length' points in the last
            'output_chunk_length' entries of the tensor. The entries before contain the data points
            leading up to the first prediction, all in chronological order.
        """

        super(TCN_net, self).__init__()

        # Defining parameters
        self.input_chunk_length = max_input_length # max processable length 
        self.input_size = input_size
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.nr_params = 1
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)
        # If num_layers is not passed, compute number of layers needed for full history coverage
        if num_layers is None and dilation_base > 1:
            num_layers = math.ceil(
                math.log(
                    (self.input_chunk_length - 1)
                    * (dilation_base - 1)
                    / (kernel_size - 1)
                    / 2
                    + 1,
                    dilation_base,
                )
            )
            print("Number of layers chosen: " + str(num_layers))
        elif num_layers is None:
            num_layers = math.ceil(
                (self.input_chunk_length - 1) / (kernel_size - 1) / 2
            )
            print("Number of layers chosen: " + str(num_layers))
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(
                num_filters,
                kernel_size,
                dilation_base,
                self.dropout,
                norm,
                i,
                num_layers,
                self.input_size,
                num_filters * nr_params
            )
            self.res_blocks_list.append(res_block)

        self.res_blocks = nn.ModuleList(self.res_blocks_list)


    def forward(self, x_in, transpose_ = True):
        x = x_in
        length_ = x.shape[1]
        assert length_ <= self.input_chunk_length 
        
        # data is of size (batch_size, input_chunk_length, input_size)
        batch_size = x.size(0)
        x = x.permute(0,2, 1)

        for res_block in self.res_blocks_list:
            x = res_block(x)

        x = x.permute(0, 2, 1)
        x = x.view(batch_size, length_, self.n_filters)
        return x