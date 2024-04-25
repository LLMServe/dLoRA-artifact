# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only OPT model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import OPTConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage
)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear,
    LoRaColumnParallelLinear, QKV_LoRaColumnParallelLinear, LoRaRowParallelLinear)
from vllm.model_executor.parallel_utils.pipeline_parallel import (send_forward, recv_forward)
from vllm.sequence import SequenceOutputs
from vllm.config import ParallelConfig
from vllm.worker.lora_engine import OPTLoRaEngine, LoRaWeight

KVCache = Tuple[torch.Tensor, torch.Tensor]


class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKV_LoRaColumnParallelLinear(
            # "default",
            embed_dim,
            3 * embed_dim,
            # [0, 0, 0], [0, 0, 0], [0.05, 0.05, 0.05],
            bias=bias,
            gather_output=False,
            perform_initialization=False
        )

        self.out_proj = LoRaRowParallelLinear(
            # "default",
            embed_dim,
            embed_dim,
            # 16, 32, 0.05,
            bias=bias,
            input_is_parallel=True,
            perform_initialization=False
        )

        # self.qkv_proj = ColumnParallelLinear(embed_dim,
        #                                      3 * embed_dim,
        #                                      bias=bias,
        #                                      gather_output=False,
        #                                      perform_initialization=False)
        # self.out_proj = RowParallelLinear(embed_dim,
        #                                   embed_dim,
        #                                   bias=bias,
        #                                   input_is_parallel=True,
        #                                   perform_initialization=False)
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   scale=self.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
        lora_weights: Tuple[Optional[LoRaWeight], Optional[LoRaWeight]],
    ) -> torch.Tensor:
        qkv_lora, out_lora = lora_weights
        qkv, _ = self.qkv_proj(hidden_states, input_metadata.adapter_mapping, qkv_lora)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        key_cache, value_cache = kv_cache
        attn_output = self.attn(q, k, v, key_cache, value_cache,
                                input_metadata, cache_event)
        output, _ = self.out_proj(attn_output, input_metadata.adapter_mapping, out_lora)
        return output

    def merge(self, adapter: Union[int, str]):
        self.qkv_proj.merge(adapter)
        self.out_proj.merge(adapter)

    def unmerge(self):
        self.qkv_proj.unmerge()
        self.out_proj.unmerge()


class OPTDecoderLayer(nn.Module):

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.activation_fn = get_act_fn(config.activation_function)

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = LoRaColumnParallelLinear(self.embed_dim,
                                        config.ffn_dim,
                                        bias=config.enable_bias,
                                        gather_output=False,
                                        perform_initialization=False)
        self.fc2 = LoRaRowParallelLinear(config.ffn_dim,
                                     self.embed_dim,
                                     bias=config.enable_bias,
                                     input_is_parallel=True,
                                     perform_initialization=False)
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
        lora_weights: Optional[OPTLoRaEngine.lora_type],
        lora_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        if lora_weights is None:
            lora_weights = (None, None, None, None)
        if lora_event is not None:
            lora_event.wait()
        qkv_lora, out_lora, fc1_lora, fc2_lora = lora_weights
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       input_metadata=input_metadata,
                                       cache_event=cache_event,
                                       lora_weights=(qkv_lora, out_lora))
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states, input_metadata.adapter_mapping, fc1_lora)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states, input_metadata.adapter_mapping, fc2_lora)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states

    def merge(self, adapter: Union[int, str]):
        self.self_attn.merge(adapter)
        self.fc1.merge(adapter)
        self.fc2.merge(adapter)

    def unmerge(self):
        self.self_attn.unmerge()
        self.fc1.unmerge()
        self.fc2.unmerge()


class OPTDecoder(nn.Module):

    def __init__(self, config: OPTConfig, parallel_config: ParallelConfig):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        if is_pipeline_first_stage():
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.word_embed_proj_dim,
                perform_initialization=False)
            # Positional embeddings are replicated (not sharded).
            self.embed_positions = OPTLearnedPositionalEmbedding(
                config.max_position_embeddings, config.hidden_size)

            if config.word_embed_proj_dim != config.hidden_size:
                self.project_in = nn.Linear(config.word_embed_proj_dim,
                                            config.hidden_size,
                                            bias=False)
            else:
                self.project_in = None

        if is_pipeline_last_stage():
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.word_embed_proj_dim,
                perform_initialization=False)
            # Project out & in will be replicated if they exist.
            if config.word_embed_proj_dim != config.hidden_size:
                self.project_out = nn.Linear(config.hidden_size,
                                            config.word_embed_proj_dim,
                                            bias=False)
            else:
                self.project_out = None

            # Note that the only purpose of `config._remove_final_layer_norm` is to
            # keep backward compatibility with checkpoints that have been fine-tuned
            # before transformers v4.20.1
            # see https://github.com/facebookresearch/metaseq/pull/164
            if config.do_layer_norm_before and not config._remove_final_layer_norm:
                self.final_layer_norm = nn.LayerNorm(
                    config.hidden_size,
                    elementwise_affine=config.layer_norm_elementwise_affine)
            else:
                self.final_layer_norm = None

        num_layers = config.num_hidden_layers // parallel_config.pipeline_parallel_size

        # if is_pipeline_last_stage():
        #     num_layers -= 3
        # else:
        #     num_layers += 1

        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config) for _ in range(num_layers)])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
        lora_weights: Optional[OPTLoRaEngine] = None,
        lora_events: Optional[List[torch.cuda.Event]] = None,
    ) -> torch.Tensor:
        if is_pipeline_first_stage():
            inputs_embeds = self.embed_tokens(input_ids)
            pos_embeds = self.embed_positions(positions)
            if self.project_in is not None:
                inputs_embeds = self.project_in(inputs_embeds)
            hidden_states = inputs_embeds + pos_embeds
        else:
            shape = [input_ids.shape[0], self.config.word_embed_proj_dim]
            hidden_states = recv_forward(shape, self.parallel_config)

        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            if lora_weights is None:
                lora_weight = None
            else:
                lora_weight = lora_weights.gpu_lora_weights[i]
            if lora_events is None:
                lora_event = None
            else:
                lora_event = lora_events[i]
            layer = self.layers[i]
            hidden_states = layer(hidden_states, kv_caches[i], input_metadata,
                                  cache_event, lora_weight, lora_event)

        if is_pipeline_last_stage():
            if self.final_layer_norm is not None:
                hidden_states = self.final_layer_norm(hidden_states)
            if self.project_out is not None:
                hidden_states = self.project_out(hidden_states)
        else:
            send_forward(hidden_states, self.parallel_config)
            
        return hidden_states

    def merge(self, adapter: Union[int, str]):
        for layer in self.layers:
            layer.merge(adapter)

    def unmerge(self):
        for layer in self.layers:
            layer.unmerge()


class OPTModel(nn.Module):

    def __init__(self, config: OPTConfig, parallel_config: ParallelConfig):
        super().__init__()
        self.decoder = OPTDecoder(config, parallel_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
        lora_weights: Optional[OPTLoRaEngine] = None,
        lora_events: Optional[List[torch.cuda.Event]] = None,
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions, kv_caches, input_metadata,
                            cache_events, lora_weights, lora_events)

    def merge(self, adapter: Union[int, str]):
        self.decoder.merge(adapter)

    def unmerge(self):
        self.decoder.unmerge()


class OPTForCausalLM(nn.Module):

    def __init__(self, config, parallel_config: ParallelConfig):
        super().__init__()
        self.config = config
        self.model = OPTModel(config, parallel_config)
        # TODO(zhuohan): create a new weight after implementing pipeline
        #                parallelism
        if is_pipeline_last_stage():
            self.lm_head_weight = self.model.decoder.embed_tokens.weight
            self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
        lora_weights: Optional[OPTLoRaEngine] = None,
        lora_events: Optional[List[torch.cuda.Event]] = None,
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events, lora_weights, lora_events)
        if is_pipeline_last_stage():
            next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                    input_metadata)
        else:
            next_tokens = {}
        return next_tokens

    def merge(self, lora_engine: OPTLoRaEngine, adapter: int):
        assert adapter in lora_engine.gpu_lora_models
        if adapter == lora_engine.merged_adapter:
            return
        idx = lora_engine.gpu_lora_models.index(adapter)
        lora_engine.merged_adapter = idx

        for layer, layer_lora_weight in zip(self.model.decoder.layers, lora_engine.gpu_lora_weights):
            qkv_lora, out_lora, fc1_lora, fc2_lora = layer_lora_weight
            qkv_proj = layer.self_attn.qkv_proj
            qkv_lora.merge(qkv_proj.weight.data, idx)
            out_proj = layer.self_attn.out_proj
            out_lora.merge(out_proj.weight.data, idx)
            fc1 = layer.fc1
            fc1_lora.merge(fc1.weight.data, idx)
            fc2 = layer.fc2
            fc2_lora.merge(fc2.weight.data, idx)

    def unmerge(self, lora_engine: OPTLoRaEngine):
        if lora_engine.merged_adapter == None:
            return
        lora_engine.merged_adapter = None
        for layer, layer_lora_weight in zip(self.model.decoder.layers, lora_engine.gpu_lora_weights):
            qkv_lora, out_lora, fc1_lora, fc2_lora = layer_lora_weight
            qkv_proj = layer.self_attn.qkv_proj
            qkv_lora.unmerge(qkv_proj.weight.data)
            out_proj = layer.self_attn.out_proj
            out_lora.unmerge(out_proj.weight.data)
            fc1 = layer.fc1
            fc1_lora.unmerge(fc1.weight.data)
            fc2 = layer.fc2
            fc2_lora.unmerge(fc2.weight.data)


    _column_parallel_weights = [
        "embed_tokens.weight", "fc1.weight", "fc1.bias"
    ]
    _row_parallel_weights = ["out_proj.weight", "fc2.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "lm_head.weight" in name:
                continue

            if name.startswith("decoder."):
                name = "model." + name

            is_attention_weight = False
            for stride_id, att_weight_name in enumerate(
                ["q_proj", "k_proj", "v_proj"]):
                if att_weight_name not in name:
                    continue
                param = state_dict[name.replace(att_weight_name, "qkv_proj")]
                shard_size = param.shape[0] // 3
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            param = state_dict[name]
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)

