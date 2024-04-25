# Copyright 2023 The vLLM team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple, Union

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.worker.lora_engine import LoRaWeight
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)

from .random import get_cuda_rng_tracker
from .utils import (
    divide,
    VocabUtility,
)

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}

def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False,
                                  *, params_dtype=None):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    if params_dtype is None:
        params_dtype = torch.get_default_dtype()

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        init_method: method to initialize weights.
        params_dtype
        use_cpu_initialization
        perform_initialization
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                 init_method=init.xavier_normal_,
                 params_dtype: torch.dtype=None,
                 use_cpu_initialization: bool=False,
                 perform_initialization: bool=True):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight, self.num_embeddings, self.embedding_dim,
                    self.num_embeddings_per_partition, 0, init_method,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 params_dtype=None,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, self.world_size)
        self.skip_bias_add = skip_bias_add

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=params_dtype))
            if perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.output_size_per_partition, 0, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=0, stride=stride)

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    dtype=params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)


    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        input_parallel = input_
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class LoRaColumnParallelLinear(ColumnParallelLinear):
    # Lora implemented in a dense layer
    def __init__(
        self,
        # adapter_name: str,
        in_features: int,
        out_features: int,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        params_dtype = None,
        bias=True, gather_output=True,
        init_method=init.xavier_normal_, stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        use_cpu_initialization=False,
        perform_initialization=True,
        **kwargs,
    ):
        ColumnParallelLinear.__init__(self, in_features, out_features, bias=bias, gather_output=gather_output,
                                      init_method=init_method, stride=stride, 
                                      keep_master_weight_for_test=keep_master_weight_for_test, skip_bias_add=skip_bias_add,
                                      params_dtype=params_dtype, use_cpu_initialization=use_cpu_initialization,
                                      perform_initialization=perform_initialization)
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.adapter_mapping = {}
        self.num_adapters = 0
        self.merged_adapter = None
        self.lora_As = None
        self.lora_Bs = None
        self.active_adaptes = []
        self.active_lora_As = None
        self.active_lora_Bs = None
        self.max_r = 1
        # Freezing the pre-trained weight matrix
        self.kwargs = kwargs

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.adapter_mapping[adapter_name] = self.num_adapters
        self.r[self.num_adapters] = r
        self.lora_alpha[self.num_adapters] = lora_alpha
        self.num_adapters += 1

        # Actual trainable parameters
        if r > self.max_r and self.lora_As is not None:
            padding_As = torch.zeros(self.lora_As.shape[0], self.input_size, r - self.max_r).cuda()
            padding_Bs = torch.zeros(self.lora_Bs.shape[0], r - self.max_r, self.output_size_per_partition).cuda()
            self.lora_As = torch.cat((self.lora_As, padding_As), dim=-1)
            self.lora_Bs = torch.cat((self.lora_Bs, padding_Bs), dim=1)
            self.max_r = r
            lora_A = torch.randn(1, self.input_size, r).cuda()
            lora_B = torch.randn(1, r, self.output_size_per_partition).cuda()
        else:
            self.max_r = max(r, self.max_r)
            lora_A = torch.randn(1, self.input_size, self.max_r).cuda()
            lora_B = torch.randn(1, self.max_r, self.output_size_per_partition).cuda()
            lora_A[:, :, r:] = 0
            lora_B[:, r:, :] = 0
        if r > 0:
            self.scaling[adapter_name] = lora_alpha / r
        if self.lora_As == None:
            self.lora_As = lora_A
            self.lora_Bs = lora_B
        else:
            self.lora_As = torch.cat((self.lora_As, lora_A), dim=0)
            self.lora_Bs = torch.cat((self.lora_Bs, lora_B), dim=0)
            
        self.to(self.weight.device)

    def all_lora(self, x: torch.Tensor, adapter_mapping: torch.Tensor, lora_weight: LoRaWeight):
        return torch.einsum('bk, bi, ikr, ird->bd',
                x,
                adapter_mapping,
                lora_weight.active_lora_As, # [:len(lora_weight.active_idx)],
                lora_weight.active_lora_Bs # [:len(lora_weight.active_idx)]
            )

    def forward(self, input_: torch.Tensor, adapter_mapping: torch.Tensor, lora_weight: Optional[LoRaWeight]):
        bias = self.bias if not self.skip_bias_add else None

        input_parallel = input_
        # Matrix multiply.
        if lora_weight != None and lora_weight.merged_adapter == None:
            assert input_.shape[0] == adapter_mapping.shape[0]
            assert adapter_mapping.shape[1] == len(lora_weight.active_idx)
            output_parallel = F.linear(input_parallel, self.weight, bias) + self.all_lora(input_parallel, adapter_mapping, lora_weight)
        else:
            output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    

class QKV_LoRaColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        params_dtype = None,
        bias=True, gather_output=True,
        init_method=init.xavier_normal_, stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        use_cpu_initialization=False,
        perform_initialization=True,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        ColumnParallelLinear.__init__(self, in_features, out_features, bias=bias, gather_output=gather_output,
                                      init_method=init_method, stride=stride, 
                                      keep_master_weight_for_test=keep_master_weight_for_test, skip_bias_add=skip_bias_add,
                                      params_dtype=params_dtype, use_cpu_initialization=use_cpu_initialization,
                                      perform_initialization=perform_initialization)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.adapter_mapping = {}
        self.num_adapters = 0
        self.merged_adapter = None
        self.lora_As = None
        self.lora_Bs = None
        self.active_adaptes = []
        self.active_lora_As = None
        self.active_lora_Bs = None
        self.in_features = in_features
        self.out_features = out_features
        self.max_r = 1 # max(qkv_r)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.kwargs = kwargs

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def update_layer(self, adapter_name, qkv_r, qkv_lora_alpha, qkv_lora_dropout, init_lora_weights):
        assert len(qkv_r) == len(qkv_lora_alpha) == len(qkv_lora_dropout) == 3
        self.adapter_mapping[adapter_name] = self.num_adapters
        self.r[self.num_adapters] = qkv_r
        self.lora_alpha[self.num_adapters] = qkv_lora_alpha
        self.num_adapters += 1
        # TODO: move scaling into lora_A or lora_B
        self.scaling[adapter_name] = [qkv_lora_alpha[i] / qkv_r[i] if qkv_r[i] != 0 else 0 for i in range(3)]

        r = max(qkv_r)
        if r > self.max_r and self.lora_As != None:
            padding_As = torch.zeros(self.lora_As.shape[0], self.in_features, 3 * (r - self.max_r)).cuda()
            padding_Bs = torch.zeros(self.lora_Bs.shape[0], 3 * (r - self.max_r), self.output_size_per_partition).cuda()
            self.lora_As = torch.cat((self.lora_As, padding_As), dim=-1)
            self.lora_Bs = torch.cat((self.lora_Bs, padding_Bs), dim=1)
            self.max_r = r
            lora_A = torch.randn(1, self.in_features, 3 * r).cuda()
            lora_B = torch.randn(1, 3 * r, self.output_size_per_partition).cuda()
        else:
            self.max_r = max(self.max_r, r)
            lora_A = torch.randn(1, self.in_features, 3 * self.max_r).cuda()
            lora_B = torch.randn(1, 3 * self.max_r, self.output_size_per_partition).cuda()
        
        for i in range(3):
            lora_A[:, :, qkv_r[i]:(i+1)*self.max_r] = 0
            lora_B[:, qkv_r[i]:(i+1)*self.max_r, :] = 0
        if self.lora_As == None:
            self.lora_As = lora_A
            self.lora_Bs = lora_B
        else:
            self.lora_As = torch.cat((self.lora_As, lora_A), dim=0)
            self.lora_Bs = torch.cat((self.lora_Bs, lora_B), dim=0)
            
        self.to(self.weight.device)

    def all_lora(self, x: torch.Tensor, adapter_mapping: torch.Tensor, lora_weight: LoRaWeight):
        return torch.einsum('bk, bi, ikr, ird->bd',
                x,
                adapter_mapping,
                lora_weight.active_lora_As, # [:len(lora_weight.active_idx)],
                lora_weight.active_lora_Bs # [:len(lora_weight.active_idx)]
            )

    def forward(self, input_: torch.Tensor, adapter_mapping: torch.Tensor, lora_weight: Optional[LoRaWeight]):
        bias = self.bias if not self.skip_bias_add else None

        input_parallel = input_
        # Matrix multiply.
        if lora_weight != None and lora_weight.merged_adapter == None:
            assert input_.shape[0] == adapter_mapping.shape[0]
            assert adapter_mapping.shape[1] == len(lora_weight.active_idx)
            output_parallel = F.linear(input_parallel, self.weight, bias) + self.all_lora(input_parallel, adapter_mapping, lora_weight)
        else:
            output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
        reduce_results:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 params_dtype=None,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 reduce_results=True,
                 ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Divide the weight matrix along the last dimension.
        self.world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.world_size)
        self.skip_bias_add = skip_bias_add

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=params_dtype))
            if perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.input_size_per_partition, 1, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=1, stride=stride)
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size,
                    dtype=params_dtype))

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.weight_t = self.weight.t()

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        if self.reduce_results and self.world_size > 1:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        else:
            output_ = output_parallel

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


class LoRaRowParallelLinear(RowParallelLinear):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        params_dtype = None,
        bias=True, input_is_parallel=False,
        init_method=init.xavier_normal_, stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        use_cpu_initialization=False,
        perform_initialization=True,
        **kwargs,
    ):
        RowParallelLinear.__init__(self, in_features, out_features, bias=bias, input_is_parallel=input_is_parallel,
                                      init_method=init_method, stride=stride, 
                                      keep_master_weight_for_test=keep_master_weight_for_test, skip_bias_add=skip_bias_add,
                                      params_dtype=params_dtype, use_cpu_initialization=use_cpu_initialization,
                                      perform_initialization=perform_initialization)
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.adapter_mapping = {}
        self.num_adapters = 0
        self.merged_adapter = None
        self.lora_As = None
        self.lora_Bs = None
        self.active_adaptes = []
        self.active_lora_As = None
        self.active_lora_Bs = None
        self.max_r = 1
        # Freezing the pre-trained weight matrix
        # self.all_lora_opt = torch.compile(self.all_lora)
        self.kwargs = kwargs

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.adapter_mapping[adapter_name] = self.num_adapters
        self.r[self.num_adapters] = r
        self.lora_alpha[self.num_adapters] = lora_alpha
        self.num_adapters += 1

        # Actual trainable parameters
        if r > self.max_r and self.lora_As != None:
            padding_As = torch.zeros(self.lora_As.shape[0], self.input_size_per_partition, r - self.max_r).cuda()
            padding_Bs = torch.zeros(self.lora_Bs.shape[0], r - self.max_r, self.output_size).cuda()
            self.lora_As = torch.cat((self.lora_As, padding_As), dim=-1)
            self.lora_Bs = torch.cat((self.lora_Bs, padding_Bs), dim=1)
            self.max_r = r
            lora_A = torch.randn(1, self.input_size_per_partition, r).cuda()
            lora_B = torch.randn(1, r, self.output_size).cuda()
        else:
            self.max_r = max(self.max_r, r)
            lora_A = torch.randn(1, self.input_size_per_partition, self.max_r).cuda()
            lora_B = torch.randn(1, self.max_r, self.output_size).cuda()
            lora_A[:, :, r:] = 0
            lora_B[:, r:, :] = 0
        if r > 0:
            self.scaling[adapter_name] = lora_alpha / r
        if self.lora_As == None:
            self.lora_As = lora_A
            self.lora_Bs = lora_B
        else:
            self.lora_As = torch.cat((self.lora_As, lora_A), dim=0)
            self.lora_Bs = torch.cat((self.lora_Bs, lora_B), dim=0)
            
        self.to(self.weight.device)

    def all_lora(self, x: torch.Tensor, adapter_mapping: torch.Tensor, lora_weight: LoRaWeight):
        return torch.einsum('bk, bi, ikr, ird->bd',
                x,
                adapter_mapping,
                lora_weight.active_lora_As, # [:len(lora_weight.active_idx)],
                lora_weight.active_lora_Bs # [:len(lora_weight.active_idx)]
            )

    def forward(self, input_: torch.Tensor, adapter_mapping: torch.Tensor, lora_weight: Optional[LoRaWeight]):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if lora_weight != None and lora_weight.merged_adapter == None:
            assert input_.shape[0] == adapter_mapping.shape[0]
            assert adapter_mapping.shape[1] == len(lora_weight.active_idx)
            output_parallel = F.linear(input_parallel, self.weight) + self.all_lora(input_parallel, adapter_mapping, lora_weight)
        else:
            output_parallel = F.linear(input_parallel, self.weight)
        if self.reduce_results and self.world_size > 1:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        else:
            output_ = output_parallel

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias