"""Utilities for selecting and loading models."""
from typing import Type, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig
from vllm.model_executor.models import *  # pylint: disable=wildcard-import
from vllm.model_executor.weight_utils import initialize_dummy_weights

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "AquilaModel": AquilaForCausalLM,
    "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    "BloomForCausalLM": BloomForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "LlamaForCausalLM_peft": LlamaForCausalLM_peft,
    "LLaMAForCausalLM_peft": LlamaForCausalLM_peft,
    "MPTForCausalLM": MPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "RWForCausalLM": FalconForCausalLM,
}


def _get_model_architecture(config: PretrainedConfig, is_peft: bool = False) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if is_peft:
            arch = arch + "_peft"
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_model(model_config: ModelConfig, 
              parallel_config: Optional[ParallelConfig] = None,
              to_gpu: bool = True,
              is_peft: bool = False) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config, is_peft)
    torch.set_default_dtype(model_config.dtype)

    # Create a model instance.
    # The weights will be initialized as empty tensors.
    if parallel_config == None:
        model = model_class(model_config.hf_config)
    else:
        model = model_class(model_config.hf_config, parallel_config)
    if model_config.use_dummy_weights:
        if to_gpu:
            # print("Using dummy weights on GPU")
            model = model.cuda()
        # else:
        #     print("Using dummy weights on CPU")
        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        initialize_dummy_weights(model)
    else:
        # Load the weights from the cached or downloaded files.
        model.load_weights(model_config.model, model_config.download_dir,
                           model_config.use_np_weights)
        if to_gpu:
            model = model.cuda()
    return model.eval()
