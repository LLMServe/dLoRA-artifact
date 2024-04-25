"""A GPU worker class."""
import os
from typing import Dict, List, Tuple, Optional, Union
import time

import torch
import torch.distributed as dist

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, LoRaConfig, ExecType)
from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata, SequenceOutputs, SequenceGroup, SequenceStatus
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import get_gpu_memory
from vllm.core.block_manager import BlockSpaceManager
from vllm.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_group, get_tensor_model_parallel_group, get_data_parallel_group)
from vllm.worker import lora_engine

from torch.profiler import profile, record_function, ProfilerActivity

from peft import create_lora_model, LoraConfig, TaskType, inject_adapter_in_model
from transformers import PretrainedConfig

from vllm.logger import init_logger
logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]

class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        lora_config: LoRaConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
        exec_type: ExecType = ExecType.LORA,
        num_model_per_group: int = 1,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.exec_type = exec_type
        # TODO: hard-coded multiple models
        self.models = {}
        self.num_models = num_model_per_group
        # dummy init for profiling available gpu memory
        self.active = [0]
        self.num_active_models = 1
        self.active_lora_model_mapping = {0: 0}
        self.adjust_cnt = 0
        self.adjust_time = 0.0
        self.execute_time = 0.0

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

    def init_model(self, engine_id: int):
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("TORCH_NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)

        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)
        
        self.init_cpu_group()

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        if self.exec_type == ExecType.REPLICATED:
            #TODO: hard-coded multiple models
            start_model_id = 0 # engine_id * self.num_models
            self.num_model_on_gpu = _get_num_model_in_gpu(self.model_config.hf_config)
            for i in range(self.num_models):
                to_gpu = i < self.num_model_on_gpu + start_model_id
                self.models[i+start_model_id] = get_model(self.model_config, parallel_config=self.parallel_config, to_gpu=to_gpu)
            self.model = self.models[start_model_id]
            self.gpu_model_ids = [i+start_model_id for i in range(self.num_model_on_gpu)]
        elif self.exec_type == ExecType.PEFT:
            #TODO: hard-coded multiple models
            start_model_id = 0 # engine_id * self.num_models
            self.model = get_model(self.model_config, parallel_config=self.parallel_config, is_peft=True)
            if self.model_config.hf_config.model_type =='opt':
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'], inference_mode=True, r=16, lora_alpha=1, lora_dropout=0)
            elif self.model_config.hf_config.model_type =='llama':
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], inference_mode=True, r=16, lora_alpha=1, lora_dropout=0)
            else:
                assert False, f"not supported model type {self.model_config.hf_config.model_type}"

            self.peft_config = peft_config
            self.active_adapter = -1
            self.peft_exec_model = None
        else:
            self.model = get_model(self.model_config, parallel_config=self.parallel_config)
            self.lora_engine = lora_engine._MODEL_LORA_MAPPING[self.model_config.hf_config.model_type](
                self.model_config.hf_config, self.model_config, self.lora_config, self.parallel_config, self.device)

        self.cpu_key_cache: List[torch.tensor] = []
        self.cpu_value_cache: List[torch.tensor] = []

    def init_cpu_group(self):
        assert dist.is_initialized()
        self.cpu_group = dist.new_group(backend="gloo")

    def merge(self, adapter: int):
        self.model.merge(self.lora_engine, adapter)

    def unmerge(self):
        self.model.unmerge(self.lora_engine)

    def adjust_lora_adapter(self, gpu_model_list: List[int], active_model_list: List[int]) -> bool:
        if self.active == active_model_list and self.adjust_cnt > 0:
            return False
        print("adjust_lora_adapter", active_model_list)
        start = time.time()
        self.active = active_model_list.copy()
        swapped = self.lora_engine.adjust_lora_adapter(gpu_model_list, active_model_list)
        if len(active_model_list) > 1:
            self.unmerge()
        else:
            self.merge(active_model_list[0])
        self.active_lora_model_mapping = {active_model_list[i]: i for i in range(len(active_model_list))}
        self.num_active_models = len(active_model_list)
        torch.cuda.synchronize()
        self.adjust_cnt += 1
        adjust_time = time.time() - start
        self.adjust_time += adjust_time
        logger.info(f"{self.rank} {self.adjust_cnt}, adjust time: {adjust_time}, for {self.active}, total time: {self.adjust_time}")
        return swapped

    @torch.inference_mode()
    def profile_inference_time(
        self,
        seq_len: Optional[int] = None,
    ):
        # Modify from profile_num_available_blocks.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
        for group_id in range(max_num_seqs):
            if seq_len == None:
                seq_len = (max_num_batched_tokens // max_num_seqs +
                        (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                model_id=0,
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seqs)
        # Execute the model.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        start_time = time.time()
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        torch.cuda.synchronize()
        interval = time.time() - start_time
        print(f"profile time {interval: .5f}, len {seq_len}, average {interval / seq_len: .5f}")

        return interval / seq_len
    

    @torch.inference_mode()
    def profile_available_gpu_memory(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> Tuple[int, int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                model_id=self.active[0],
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seqs)

        # Execute the model.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None,
            lora_weights=None,
            lora_events=None,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        available_gpu_memory = total_gpu_memory * gpu_memory_utilization - peak_memory
        torch.cuda.empty_cache()

        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        
        if self.exec_type == ExecType.LORA:
            lora_size = self.lora_engine.get_size()
        else:
            lora_size = 100925440 #TODO: hard-coded

        return available_gpu_memory, lora_size, cache_block_size

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        allocated_gpu_memory: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(allocated_gpu_memory // cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        return num_gpu_blocks, num_cpu_blocks
    
    def set_output(self, output):
        import logging
        fh = logging.FileHandler(output)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        self.adjust_time = 0.0

    def init_cache_engine_and_lora(self, cache_config: CacheConfig, init_active_lora_types: List[int]) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.cpu_cache = self.cache_engine.cpu_cache

        if self.exec_type == ExecType.LORA:
            self.active = init_active_lora_types.copy()
            for idx, model_id in enumerate(init_active_lora_types):
                self.active_lora_model_mapping[model_id] = idx
            self.num_active_models = len(init_active_lora_types)
            self.lora_config.gpu_capacity = len(init_active_lora_types)
            
            self.lora_engine.allocate_gpu_lora_weight(init_active_lora_types)
            self.lora_events = self.lora_engine.events

        elif self.exec_type == ExecType.PEFT:
            print("injecting adapter in model")
            self.active = init_active_lora_types.copy()
            peft_configs = {str(i): self.peft_config for i in self.active}
            self.model = create_lora_model(peft_configs, self.model, str(self.active[0]))

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        model_id: Optional[int] = None, # if this is none, it means that we use lora to handle multiple models
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        model_mapping: List[int] = []

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            if model_id != None and seq_group_metadata.model_id != model_id:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            if self.exec_type == ExecType.LORA:
                assert seq_group_metadata.model_id in self.active_lora_model_mapping
                mapped_model_id = self.active_lora_model_mapping[seq_group_metadata.model_id]
                model_mapping.extend([mapped_model_id] * prompt_len)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(prompt_tokens)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([0] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            if model_id != None and seq_group_metadata.model_id != model_id:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)
                if self.exec_type == ExecType.LORA:
                    assert seq_group_metadata.model_id in self.active_lora_model_mapping
                    mapped_model_id = self.active_lora_model_mapping[seq_group_metadata.model_id]
                    model_mapping.append(mapped_model_id)

                context_len = seq_data.get_len()
                position = context_len - 1
                input_positions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)
        model_mapping = _pad_to_alignment(model_mapping, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.tensor(input_tokens, dtype=torch.long, device='cuda')
        positions_tensor = torch.tensor(input_positions, dtype=torch.long, device='cuda')
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int, device='cuda')
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int, device='cuda')
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables
        ]
        block_tables_tensor = torch.tensor(padded_block_tables, dtype=torch.int, device='cuda')

        if model_id == None:
            # Converse int to one-hot
            model_mapping = torch.LongTensor(model_mapping).view(-1, 1)
            adapter_tensor = torch.IntTensor(len(input_tokens), self.num_active_models)
            adapter_tensor.zero_()
            adapter_tensor.scatter_(1, model_mapping, 1)
            adapter_tensor = adapter_tensor.cuda()
        else:
            adapter_tensor = None

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            if model_id != None and seq_group_metadata.model_id != model_id:
                continue
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
            adapter_mapping=adapter_tensor,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        gpu_model_list: List[int],
        active_model_list: List[int],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> Dict[int, SequenceOutputs]:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True,) as prof:
        # If we do not use lora, we need to execute models seperatelly
        if self.exec_type != ExecType.LORA:
            output: Dict[int, SequenceOutputs] = {}
            # Execute the model.
            for i in range(self.num_models):
                # Prepare input tensors.
                input_tokens, input_positions, input_metadata = self._prepare_inputs(
                    seq_group_metadata_list, model_id=i)
                
                if len(input_tokens) == 0:
                    continue

                if self.exec_type == ExecType.REPLICATED:

                    if i not in self.gpu_model_ids:
                        print("model", i, "to gpu")
                        self.models[self.gpu_model_ids[0]] = self.models[self.gpu_model_ids[0]].to("cpu")
                        self.gpu_model_ids.pop(0)
                        self.models[i] = self.models[i].to("cuda")
                        self.gpu_model_ids.append(i)
                    else: # lru
                        self.gpu_model_ids.remove(i)
                        self.gpu_model_ids.append(i)
                    exec_model = self.models[i]
                else:
                    self.model.set_adapter(str(i))
                    exec_model = self.model.model
                model_output = exec_model(
                    input_ids=input_tokens,
                    positions=input_positions,
                    kv_caches=self.gpu_cache,
                    input_metadata=input_metadata,
                    cache_events=cache_events,
                    lora_weights=None,
                    lora_events=None,
                )
                output.update(model_output)

        else:
            adjusted = self.adjust_lora_adapter(gpu_model_list, active_model_list)
            input_tokens, input_positions, input_metadata = self._prepare_inputs(
                seq_group_metadata_list)

            if adjusted:
                lora_events = self.lora_events
            else:
                lora_events = None
            output = self.model(
                input_ids=input_tokens,
                positions=input_positions,
                kv_caches=self.gpu_cache,
                input_metadata=input_metadata,
                cache_events=cache_events,
                lora_weights=self.lora_engine,
                lora_events=lora_events,
            )
        return output


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))

# only works for 80GB GPU
def _get_num_model_in_gpu(config: PretrainedConfig):
    name = getattr(config, "_name_or_path", [])
    if "7b" in name:
        return 4
    else:
        return 2