import enum
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union, Set

from vllm.config import CacheConfig, SchedulerConfig, LoRaConfig, ExecType
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceOutputs,
                           SequenceStatus, RequestMetadata)

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_ids: List[str],
        num_prompt_tokens: int,
        num_generation_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_ids = prompt_ids
        self.num_prompt_tokens = num_prompt_tokens
        self.num_generation_tokens = num_generation_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: LoRaConfig,
        num_models: int,
        models_in_gpu: List[int],
        exec_type: bool,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.num_models = num_models
        self.models_in_gpu = sorted(models_in_gpu.copy())
        self.active = self.models_in_gpu.copy()
        self.exec_type = exec_type
        self.req_model_cnt = {i: 0 for i in range(num_models)}

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy_name = scheduler_config.policy
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")

        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
        )

        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []
        # Sequence groups in the READY state.
        self.ready: List[SequenceGroup] = []

        # for credit schedule
        self.init_credit = 10000.0 * (num_models - 1)
        self.credit_right_threshold = 10000.0 * num_models
        self.credit_left_threshold = self.credit_right_threshold - 5000.0
        self.credit_base = 0.0
        self.lora_credit = {i: self.init_credit for i in range(num_models)}
        # profling results from llama-7b
        self.merge_time = 0.015
        self.unmerge_time = 0.027
        self.switch_cost = 0.02
        self.speed_rate = self.merge_time / self.unmerge_time
        self.merged = False
        self.aimd_mul = 1.1
        self.aimd_add = 0.05
        self.merge_batch_cnt = 0
        self.unmerge_batch_cnt = 0
        self.credit_cost = 1.0 / (self.scheduler_config.max_num_seqs)
        self.batch_credit_cost = 1.0 / (self.speed_rate * self.scheduler_config.max_num_seqs)
        self.previous_credit_models = []
        self.merge_right_threshold = 1.0
        self.merge_left_threshold = self.speed_rate
        self.num_iters = 0
        self.max_model = -1
        self.virtual_switch_cnt = 0

        self.preempt_cnt = 0
        self.swap_cnt = 0

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.req_model_cnt[seq_group.model_id] += 1
        self.waiting.append(seq_group)

    def insert_seq_group(self, seq_group: SequenceGroup) -> None:
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.WAITING
        self.waiting.append(seq_group)
        self.req_model_cnt[seq_group.model_id] += 1

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped, self.ready]:
            for seq_group in state_queue:
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.seqs:
                        if seq.is_finished():
                            continue
                        self.free_seq(seq, SequenceStatus.FINISHED_ABORTED)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped or self.ready

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped) + len(self.ready)
    
    def get_reqs_metadata(self, engine_id: int) -> List[RequestMetadata]:
        req_metadata = []
        for state_queue in [self.waiting, self.running, self.swapped, self.ready]:
            for seq_group in state_queue:
                num_blocks = sum(len(seq.logical_token_blocks) for seq in seq_group.seqs)
                in_gpu = seq_group.seqs[0].status == SequenceStatus.RUNNING or seq_group.seqs[0].status == SequenceStatus.READY
                metadata = RequestMetadata(seq_group.request_id, seq_group.model_id, engine_id, num_blocks, in_gpu)
                req_metadata.append(metadata)
        return req_metadata

    def fetch_seq_groups(self, request_ids: List[str]) -> List[SequenceGroup]:
        seq_groups: Set[SequenceGroup] = set()
        updated_state_queues = []
        for state_queue in [self.waiting, self.ready, self.running, self.swapped]:
            for seq_group in state_queue:
                if seq_group.request_id in request_ids:
                    seq_groups.add(seq_group)
                    self.req_model_cnt[seq_group.model_id] -= 1
            state_queue = [seq_group for seq_group in state_queue if seq_group not in seq_groups]
            updated_state_queues.append(state_queue)
        self.waiting, self.ready, self.running, self.swapped = updated_state_queues 
        assert len(seq_groups) == len(request_ids)
        for seq_group in seq_groups:
            for seq in seq_group.seqs:
                self.block_manager.free(seq)
        return list(seq_groups)

    def get_swapped_block_num(self) -> int:
        num_tokens = 0
        for seq_group in self.swapped:
            for seq in seq_group.get_seqs():
                num_tokens += seq.get_len()
        return (num_tokens + self.cache_config.block_size - 1) // self.cache_config.block_size
    
    def _strict_fcfs_schedule(self, max_model_cnt) -> SchedulerOutputs:
        queued: List[SequenceGroup] = []
        queued.extend(self.waiting)
        queued.extend(self.ready)
        queued.extend(self.swapped)
        now = time.time()
        queued = self.policy.sort_by_priority(now, queued)
        scheduled = []
        model_set = set()
        model_cnt = 0

        for seq_group in queued:
            if seq_group.model_id not in model_set:
                if model_cnt >= max_model_cnt:
                    break
                model_set.add(seq_group.model_id)
                model_cnt += 1
            scheduled.append(seq_group)
        return self._prepare_scheduleroutputs(scheduled)
    
    def _always_merge_schedule(self) -> SchedulerOutputs:
        queued: List[SequenceGroup] = []
        queued.extend(self.waiting)
        queued.extend(self.ready)
        queued.extend(self.swapped)
        now = time.time()
        queued = self.policy.sort_by_priority(now, queued)
        scheduled = []

        if queued:
            model_req_mapping = {i: [] for i in range(self.num_models)}
            for seq_group in queued:
                model_req_mapping[seq_group.model_id].append(seq_group)
            model_id = queued[0].model_id
            scheduled = model_req_mapping[model_id]
            self.active = [model_id]

        return self._prepare_scheduleroutputs(scheduled)
    
    def _adjust_credit(self, scheduled_by_fcfs: List[SequenceGroup], scheduled: List[SequenceGroup], credit_cost: float):

        # sort scheduled by whether the seq_group is in scheduled_by_fcfs
        scheduled = sorted(scheduled, key=lambda x: x in scheduled_by_fcfs, reverse=True)

        if len(scheduled_by_fcfs) == len(scheduled):
            for seq_group, fcfs_seq_group in zip(scheduled, scheduled_by_fcfs):
                if seq_group not in scheduled_by_fcfs:
                    self.lora_credit[seq_group.model_id] -= credit_cost
                    self.lora_credit[fcfs_seq_group.model_id] += credit_cost

        else:
            assert credit_cost == self.batch_credit_cost
            for seq_group in scheduled_by_fcfs:
                if seq_group not in scheduled:
                    self.lora_credit[scheduled[0].model_id] -= credit_cost / 2
                    self.lora_credit[seq_group.model_id] += credit_cost / 2

    def _adjust_merge_threshold(self):
        if self.num_iters == 0:
            return
        t_merge = self.merge_time * self.num_iters
        t_unmerge = self.unmerge_time * self.num_iters
        if self.merged:
            t_merge += self.switch_cost
        else:
            t_unmerge += self.switch_cost
            t_merge += self.switch_cost * self.virtual_switch_cnt
        
        tput_merge = self.merge_batch_cnt / t_merge
        tput_unmerge = self.unmerge_batch_cnt / t_unmerge

        if self.merged:
            if tput_merge > tput_unmerge and self.merge_right_threshold - self.aimd_add > self.merge_left_threshold:
                self.merge_right_threshold -= self.aimd_add
                print("tput_merge", tput_merge, "tput_unmerge", tput_unmerge, "merge_right_threshold", self.merge_right_threshold)
            elif tput_merge < tput_unmerge:
                self.merge_right_threshold *= self.aimd_mul
                print("tput_merge", tput_merge, "tput_unmerge", tput_unmerge, "merge_right_threshold", self.merge_right_threshold)
        else:
            if tput_unmerge > tput_merge and self.merge_left_threshold + self.aimd_add < self.merge_right_threshold:
                self.merge_left_threshold += self.aimd_add
                print("tput_merge", tput_merge, "tput_unmerge", tput_unmerge, "merge_left_threshold", self.merge_left_threshold)
            elif tput_unmerge < tput_merge:
                self.merge_left_threshold /= self.aimd_mul
                print("tput_merge", tput_merge, "tput_unmerge", tput_unmerge, "merge_left_threshold", self.merge_left_threshold)
        
        self.merge_batch_cnt = 0
        self.unmerge_batch_cnt = 0
        self.virtual_switch_cnt = 0
        self.max_model = -1

    
    def _credit_schedule(self) -> SchedulerOutputs:

        # here we assume that the number of seqs in one seq_group is 1 and batch size is self.scheduler_config.max_num_seqs(ignoring the limitation of gpu memory)
        scheduled_by_fcfs: List[SequenceGroup] = []
        scheduled_by_credit: List[SequenceGroup] = []
        scheduled_by_batch: List[SequenceGroup] = []
        to_merge = False

        queued: List[SequenceGroup] = []
        queued.extend(self.waiting)
        queued.extend(self.ready)
        queued.extend(self.swapped)
        now = time.time()
        queued = self.policy.sort_by_priority(now, queued)
        if not queued:
            return self._prepare_scheduleroutputs([])

        scheduled_by_fcfs = queued[:self.scheduler_config.max_num_seqs]

        model_req_mapping = {i: [] for i in range(self.num_models)}
        for seq_group in queued:
            model_req_mapping[seq_group.model_id].append(seq_group)
        
        scheduled_by_credit = []
        num_to_schedule = self.scheduler_config.max_num_seqs
        # check if any model's credit is over threshold
        # if so, schedule it

        self.lora_credit = dict(sorted(self.lora_credit.items(), key=lambda x: x[1], reverse=True))

        credit_models: List[int] = []

        for model_id in self.lora_credit.keys():
            if self.lora_credit[model_id] < self.credit_left_threshold:
                break
            if self.lora_credit[model_id] >= self.credit_right_threshold or (model_id in self.previous_credit_models and self.lora_credit[model_id] >= self.credit_left_threshold):
                num = min(num_to_schedule, len(model_req_mapping[model_id]))
                scheduled_by_credit.extend(model_req_mapping[model_id][:num])
                credit_models.append(model_id)
                num_to_schedule -= num
                if num_to_schedule == 0:
                    break
        self.previous_credit_models = credit_models

        if num_to_schedule == self.scheduler_config.max_num_seqs: # try to schedule by batch
            model_req_mapping = {model_id: model_req_mapping[model_id] for model_id in model_req_mapping if self.lora_credit[model_id] >= self.credit_base}
            model_req_mapping = dict(sorted(model_req_mapping.items(), key=lambda x: len(x[1]), reverse=True))
            max_model = list(model_req_mapping.keys())[0]
            max_num_model_reqs = len(model_req_mapping[max_model])
            if len(self.active) == 1 and self.active[0] in model_req_mapping and len(model_req_mapping[self.active[0]]) >= self.merge_left_threshold * len(scheduled_by_fcfs): # use previous batch
                assert self.merged == True or self.num_models == 1
                to_merge = True
                self.num_iters += 1
                scheduled_by_batch = model_req_mapping[self.active[0]]
                scheduled = scheduled_by_batch
                self._adjust_credit(scheduled_by_fcfs, scheduled, self.batch_credit_cost)
            elif max_num_model_reqs >= self.merge_right_threshold * len(scheduled_by_fcfs): # use batch
                self._adjust_merge_threshold()
                to_merge = True
                self.merged = True
                self.num_iters = 1
                scheduled_by_batch = model_req_mapping[max_model][:self.scheduler_config.max_num_seqs]
                scheduled = scheduled_by_batch
                self.active = [max_model]
                if max_model not in self.models_in_gpu:
                    self.models_in_gpu[0] = max_model
                self._adjust_credit(scheduled_by_fcfs, scheduled, self.batch_credit_cost)

            else: # use fcfs
                if self.max_model != max_model:
                    self.virtual_switch_cnt += 1
                    self.max_model = max_model
                scheduled_by_batch = model_req_mapping[max_model][:self.scheduler_config.max_num_seqs]
                scheduled = scheduled_by_fcfs
            
            self.merge_batch_cnt += min(len(scheduled_by_batch), self.scheduler_config.max_num_seqs)
            self.unmerge_batch_cnt += min(len(scheduled_by_fcfs), self.scheduler_config.max_num_seqs)


        elif num_to_schedule == 0: # use credit
            scheduled = scheduled_by_credit
            self._adjust_credit(scheduled_by_fcfs, scheduled, self.credit_cost)
        else: # use credit + fcfs
            scheduled = scheduled_by_credit
            for seq_group in scheduled_by_fcfs:
                if seq_group not in scheduled:
                    scheduled.append(seq_group)
                    num_to_schedule -= 1
                    if num_to_schedule == 0:
                        break
            self._adjust_credit(scheduled_by_fcfs, scheduled, self.credit_cost)

        if not to_merge:
            if self.merged:
                self._adjust_merge_threshold()
                self.merged = False
                self.num_iters = 1
            else:
                self.num_iters += 1

            model_set = set()
            model_cnt = 0
            scheduled_capacity = []
            for id, seq_group in enumerate(scheduled):
                if seq_group.model_id not in model_set:
                    if model_cnt >= self.lora_config.gpu_capacity:
                        continue
                    model_set.add(seq_group.model_id)
                    model_cnt += 1
                scheduled_capacity.append(seq_group)

            scheduled = scheduled_capacity

            assert len(model_set) <= len(self.models_in_gpu)

            self.models_in_gpu = sorted(self.models_in_gpu, key=lambda x: self.req_model_cnt[x])
            idx = 0
            for model_id in model_set:
                if model_id not in self.models_in_gpu:
                    while self.models_in_gpu[idx] in model_set:
                        idx += 1
                    self.models_in_gpu[idx] = model_id
                    idx += 1

            self.models_in_gpu = sorted(self.models_in_gpu)
            self.active = self.models_in_gpu.copy()
        

        return self._prepare_scheduleroutputs(scheduled)


    def _prepare_scheduleroutputs(self, seq_groups: List[SequenceGroup]) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Reserve new token slots for the running sequence groups.
        preempted: List[SequenceGroup] = []
        ignored_seq_groups: List[SequenceGroup] = []
        prompt_ids = []
        num_input_tokens = 0
        num_generation_tokens = 0
        scheduled: List[SequenceGroup] = []
        model_set = set()

        for seq_group in seq_groups:
            if seq_group in self.ready:
                self.ready.remove(seq_group)
                while not self.block_manager.can_append_slot(seq_group):
                    if self.ready:
                        # Preempt the lowest-priority sequence groups.
                        victim_seq_group = self.ready.pop(-1)
                        self._preempt(victim_seq_group, blocks_to_swap_out)
                        preempted.append(victim_seq_group)
                    else:
                        # No other sequence groups can be preempted.
                        # Preempt the current sequence group.
                        self._preempt(seq_group, blocks_to_swap_out)
                        preempted.append(seq_group)
                        break
                else:
                    # The total number of sequences in the RUNNING state should not
                    # exceed the maximum number of sequences.
                    num_new_seqs = seq_group.num_seqs(
                        status=SequenceStatus.READY)
                    num_curr_seqs = sum(
                        seq_group.num_seqs(status=SequenceStatus.READY)
                        for seq_group in scheduled)
                    if (num_curr_seqs + num_new_seqs >
                            self.scheduler_config.max_num_seqs):
                        self.ready.append(seq_group)
                        break
                    # Append new slots to the sequence group.
                    self._append_slot(seq_group, blocks_to_copy)
                    scheduled.append(seq_group)
                    num_generation_tokens += seq_group.num_seqs(status=SequenceStatus.READY)
            
            elif seq_group in self.swapped:
                if blocks_to_swap_out:
                    continue
                if not self.block_manager.can_swap_in(seq_group):
                    continue
                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
                num_curr_seqs = sum(
                    seq_group.num_seqs(status=SequenceStatus.READY)
                    for seq_group in scheduled)
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                self.swapped.remove(seq_group)
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slot(seq_group, blocks_to_copy)
                scheduled.append(seq_group)
                num_generation_tokens += seq_group.num_seqs(status=SequenceStatus.READY)

            elif seq_group in self.waiting:
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.remove(seq_group)
                    continue

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    continue

                # If the number of batched tokens exceeds the limit, stop.
                if (num_input_tokens + num_generation_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    # print("too large tokens")
                    continue

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.num_seqs(
                    status=SequenceStatus.WAITING)
                num_curr_seqs = sum(
                    seq_group.num_seqs(status=SequenceStatus.READY)
                    for seq_group in scheduled)
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                self.waiting.remove(seq_group)
                self._allocate(seq_group)
                num_input_tokens += num_prompt_tokens
                scheduled.append(seq_group)
                prompt_ids.append(seq_group.request_id)
        

        # Update the status of the sequence groups to RUNNING.
        for seq_group in scheduled:
            model_set.add(seq_group.model_id)
            for seq in seq_group.get_seqs():
                if not seq.is_finished():
                    seq.status = SequenceStatus.RUNNING
        self.running.extend(scheduled)
        
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=scheduled,
            prompt_ids=prompt_ids,
            num_prompt_tokens=num_input_tokens,
            num_generation_tokens=num_generation_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
        )

        return scheduler_outputs


    def _batch_schedule_v2(self, active: List[int]) -> SchedulerOutputs:

        queued: List[SequenceGroup] = []

        queued.extend(self.waiting)
        queued.extend(self.ready)

        queued = [grp for grp in queued if grp.model_id in active]
        num_seqs = sum([grp.num_seqs() for grp in queued])
        if num_seqs < self.scheduler_config.max_num_seqs:
            return self._batch_schedule()
        else:
            return self._batch_schedule(active)


    # Batch requests to the same model with higher priority
    def _batch_schedule(self, active: Optional[List[int]] = None) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        queued: List[SequenceGroup] = []
        model_req_mapping: Dict[int, List[SequenceGroup]] = {}
        scheduled: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        scheduled_mapping: Dict[int, List[SequenceGroup]] = {}
        ignored_seq_groups: List[SequenceGroup] = []
        prompt_ids = []
        num_input_tokens = 0
        num_generation_tokens = 0

        now = time.time()

        queued.extend(self.waiting)
        queued.extend(self.ready)
        queued.extend(self.swapped)
        queued = self.policy.sort_by_priority(now, queued)
        if not queued:
            return SchedulerOutputs(
                scheduled_seq_groups=[],
                prompt_ids=[],
                num_prompt_tokens=0,
                num_generation_tokens=0,
                blocks_to_swap_in={},
                blocks_to_swap_out={},
                blocks_to_copy={},
                ignored_seq_groups=[],
            )

        # sort reqs based on the number of same model reqs
        for seq_group in queued:
            if seq_group.model_id in model_req_mapping:
                model_req_mapping[seq_group.model_id].append(seq_group)
            else:
                model_req_mapping[seq_group.model_id] = [seq_group]

        model_req_mapping = dict(sorted(model_req_mapping.items(), key=lambda x: len(x[1]), reverse=True))            
        max_model = list(model_req_mapping.keys())[0]
        max_num_model_reqs = len(model_req_mapping[max_model])
        
        # merge first
        if max_num_model_reqs >= self.scheduler_config.max_num_seqs:
            active_model_req_mapping = {max_model: model_req_mapping[max_model]}
            deactive_model_req_mapping = {k: v for k, v in model_req_mapping.items() if k != max_model}
        # no change second
        elif active is not None:
            active_model_req_mapping = {k: v for k, v in model_req_mapping.items() if k in active}
            deactive_model_req_mapping = {k: v for k, v in model_req_mapping.items() if k not in active}
            active_max_model = list(active_model_req_mapping.keys())[0]
            active_max_num_model_reqs = len(active_model_req_mapping[active_max_model])
        # regard all as active last
        else:
            active_model_req_mapping = model_req_mapping.copy()
            deactive_model_req_mapping = {}
            active_max_model = -1
            active_max_num_model_reqs = 0
        
        num_seq_groups = 0

        sorted_req_list = []
        for model_id, reqs in active_model_req_mapping.items():
            sorted_req_list.extend(reqs)
        for model_id, reqs in deactive_model_req_mapping.items():
            sorted_req_list.extend(reqs)
        self.ready = [req for req in sorted_req_list if req in self.ready]

        for model_id, reqs in active_model_req_mapping.items():
            if num_seq_groups + len(reqs) >= self.scheduler_config.max_num_seqs:
                num_seq_groups = self.scheduler_config.max_num_seqs
                scheduled_mapping[model_id] = reqs[:self.scheduler_config.max_num_seqs]
                break
            else:
                num_seq_groups += len(reqs)
                scheduled_mapping[model_id] = reqs

        # threshold for batch schedule
        if len(scheduled_mapping) > 1:
            self.active = self.models_in_gpu.copy()
            return self._schedule()
        
        model_cnt = 0
        scheduled_models: List[int] = []
        
        for model_id, seq_groups in scheduled_mapping.items():
            num_curr_seqs = sum(
                            seq_group.num_seqs(status=SequenceStatus.READY)
                            for seq_group in scheduled)
            if num_curr_seqs >= self.scheduler_config.max_num_seqs:
                break

            for seq_group in seq_groups:
                if seq_group in self.ready:
                    self.ready.remove(seq_group)
                    while not self.block_manager.can_append_slot(seq_group):
                        if self.ready:
                            # Preempt the lowest-priority sequence groups.
                            victim_seq_group = self.ready.pop(-1)
                            self._preempt(victim_seq_group, blocks_to_swap_out)
                            preempted.append(victim_seq_group)
                        else:
                            # No other sequence groups can be preempted.
                            # Preempt the current sequence group.
                            self._preempt(seq_group, blocks_to_swap_out)
                            preempted.append(seq_group)
                            break
                    else:
                        # The total number of sequences in the RUNNING state should not
                        # exceed the maximum number of sequences.
                        num_new_seqs = seq_group.num_seqs(
                            status=SequenceStatus.READY)
                        num_curr_seqs = sum(
                            seq_group.num_seqs(status=SequenceStatus.READY)
                            for seq_group in scheduled)
                        if (num_curr_seqs + num_new_seqs >
                                self.scheduler_config.max_num_seqs):
                            self.ready.append(seq_group)
                            break
                        # Append new slots to the sequence group.
                        self._append_slot(seq_group, blocks_to_copy)
                        scheduled.append(seq_group)
                        num_generation_tokens += seq_group.num_seqs(status=SequenceStatus.READY)
                
                elif seq_group in self.swapped:
                    if blocks_to_swap_out:
                        continue
                    if not self.block_manager.can_swap_in(seq_group):
                        continue
                    # The total number of sequences in the RUNNING state should not
                    # exceed the maximum number of sequences.
                    num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
                    num_curr_seqs = sum(
                        seq_group.num_seqs(status=SequenceStatus.READY)
                        for seq_group in scheduled)
                    if (num_curr_seqs + num_new_seqs >
                            self.scheduler_config.max_num_seqs):
                        break

                    self.swapped.remove(seq_group)
                    self._swap_in(seq_group, blocks_to_swap_in)
                    self._append_slot(seq_group, blocks_to_copy)
                    scheduled.append(seq_group)
                    num_generation_tokens += seq_group.num_seqs(status=SequenceStatus.READY)

                elif seq_group in self.waiting:
                    num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                    if num_prompt_tokens > self.prompt_limit:
                        logger.warning(
                            f"Input prompt ({num_prompt_tokens} tokens) is too long"
                            f" and exceeds limit of {self.prompt_limit}")
                        for seq in seq_group.get_seqs():
                            seq.status = SequenceStatus.FINISHED_IGNORED
                        ignored_seq_groups.append(seq_group)
                        self.waiting.remove(seq_group)
                        continue

                    # If the sequence group cannot be allocated, stop.
                    if not self.block_manager.can_allocate(seq_group):
                        continue

                    # If the number of batched tokens exceeds the limit, stop.
                    if (num_input_tokens + num_generation_tokens + num_prompt_tokens >
                            self.scheduler_config.max_num_batched_tokens):
                        continue

                    # The total number of sequences in the RUNNING state should not
                    # exceed the maximum number of sequences.
                    num_new_seqs = seq_group.num_seqs(
                        status=SequenceStatus.WAITING)
                    num_curr_seqs = sum(
                        seq_group.num_seqs(status=SequenceStatus.READY)
                        for seq_group in scheduled)
                    if (num_curr_seqs + num_new_seqs >
                            self.scheduler_config.max_num_seqs):
                        break

                    self.waiting.remove(seq_group)
                    self._allocate(seq_group)
                    num_input_tokens += num_prompt_tokens
                    scheduled.append(seq_group)
                    prompt_ids.append(seq_group.request_id)

            model_cnt += 1
            scheduled_models.append(model_id)
            if model_cnt >= self.lora_config.gpu_capacity:
                break

        self.active = sorted(scheduled_models)

        # sort models in gpu according to the number of reqs
        self.models_in_gpu = sorted(self.models_in_gpu, key=lambda x: self.req_model_cnt[x])
        idx = 0
        for model_id in scheduled_models:
            if model_id not in self.models_in_gpu:
                while self.models_in_gpu[idx] in scheduled_models:
                    idx += 1
                self.models_in_gpu[idx] = model_id
                idx += 1

        self.models_in_gpu = sorted(self.models_in_gpu)

        # Update the status of the sequence groups to RUNNING.
        for seq_group in scheduled:
            for seq in seq_group.get_seqs():
                if not seq.is_finished():
                    seq.status = SequenceStatus.RUNNING
        self.running.extend(scheduled)
        
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=scheduled,
            prompt_ids=prompt_ids,
            num_prompt_tokens=num_input_tokens,
            num_generation_tokens=num_generation_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
        )
        if self.has_unfinished_seqs() and not scheduled:
            return self._schedule()
        return scheduler_outputs

    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.time()

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the READY state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.ready = self.policy.sort_by_priority(now, self.ready)

        # Reserve new token slots for the running sequence groups.
        scheduled: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        ruleout: List[SequenceGroup] = []
        model_set = set()
        model_cnt = 0
        while self.ready:
            seq_group = self.ready.pop(0)
            if seq_group.model_id not in model_set and model_cnt >= self.lora_config.gpu_capacity:
                ruleout.append(seq_group)
                continue
            while not self.block_manager.can_append_slot(seq_group):
                if self.ready:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.ready.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                if seq_group.model_id not in model_set:
                    model_set.add(seq_group.model_id)
                    model_cnt += 1
                self._append_slot(seq_group, blocks_to_copy)
                scheduled.append(seq_group)

        self.ready.extend(ruleout)
        ruleout = []

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        while self.swapped and not blocks_to_swap_out:
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            if seq_group in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = sum(
                seq_group.num_seqs(status=SequenceStatus.READY)
                for seq_group in scheduled)
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            seq_group = self.swapped.pop(0)
            if seq_group.model_id not in model_set:
                if model_cnt >= self.lora_config.gpu_capacity:
                    ruleout.append(seq_group)
                    continue
                else:
                    model_set.add(seq_group.model_id)
                    model_cnt += 1
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            scheduled.append(seq_group)

        self.swapped.extend(ruleout)
        ruleout = []

        num_generation_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.READY)
            for seq_group in scheduled)

        # Join waiting sequences if possible.
        ignored_seq_groups: List[SequenceGroup] = []
        prompt_ids = []
        num_input_tokens = 0
        if not self.swapped:
            # scheduled: List[SequenceGroup] = []
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                if (num_input_tokens + num_generation_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.num_seqs(
                    status=SequenceStatus.WAITING)
                num_curr_seqs = sum(
                    seq_group.num_seqs(status=SequenceStatus.READY)
                    for seq_group in scheduled)
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.waiting.pop(0)
                if seq_group.model_id not in model_set:
                    if model_cnt >= self.lora_config.gpu_capacity:
                        ruleout.append(seq_group)
                        continue
                    else:
                        model_set.add(seq_group.model_id)
                        model_cnt += 1
                self._allocate(seq_group)
                num_input_tokens += num_prompt_tokens
                scheduled.append(seq_group)
                prompt_ids.append(seq_group.request_id)

        ruleout.extend(self.waiting)
        self.waiting = ruleout

        # sort models in gpu according to the number of reqs
        if self.exec_type == ExecType.LORA:
            self.models_in_gpu = sorted(self.models_in_gpu, key=lambda x: self.req_model_cnt[x])
            idx = 0
            for model_id in model_set:
                if model_id not in self.models_in_gpu:
                    while self.models_in_gpu[idx] in model_set:
                        idx += 1
                    self.models_in_gpu[idx] = model_id
                    idx += 1

            self.models_in_gpu = sorted(self.models_in_gpu)
            self.active = self.models_in_gpu.copy()

        # Update the status of the sequence groups to RUNNING.
        for seq_group in scheduled:
            for seq in seq_group.get_seqs():
                if not seq.is_finished():
                    seq.status = SequenceStatus.RUNNING

        self.running.extend(scheduled)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=scheduled,
            prompt_ids=prompt_ids,
            num_prompt_tokens=num_input_tokens,
            num_generation_tokens=num_generation_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
        )
        return scheduler_outputs


    def _old_schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.time()
        scheduled: List[SequenceGroup] = []
        ignored_seq_groups: List[SequenceGroup] = []
        prompt_ids = []

        # Join waiting sequences if possible.
        if not self.swapped:
            num_batched_tokens = 0
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                prompt_limit = min(
                    self.scheduler_config.max_model_len,
                    self.scheduler_config.max_num_batched_tokens)
                if num_prompt_tokens > prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    break

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                if (num_batched_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.num_seqs(
                    status=SequenceStatus.WAITING)
                num_curr_seqs = sum(
                    seq_group.num_seqs(status=SequenceStatus.READY)
                    for seq_group in scheduled)
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                num_batched_tokens += num_prompt_tokens
                scheduled.append(seq_group)
                prompt_ids.append(seq_group.request_id)

            if scheduled:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_ids=prompt_ids,
                    num_prompt_tokens=num_batched_tokens,
                    num_generation_tokens=0,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                for seq_group in scheduled:
                    for seq in seq_group.get_seqs():
                        if not seq.is_finished():
                            seq.status = SequenceStatus.RUNNING
                self.running.extend(scheduled)
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.ready = self.policy.sort_by_priority(now, self.ready)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.ready:
            seq_group = self.ready.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if self.ready:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.ready.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                scheduled.append(seq_group)

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        while self.swapped and not blocks_to_swap_out:
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            if seq_group in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = sum(
                seq_group.num_seqs(status=SequenceStatus.READY)
                for seq_group in scheduled)
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            scheduled.append(seq_group)

        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.READY)
            for seq_group in scheduled)

        for seq_group in scheduled:
            for seq in seq_group.get_seqs():
                if not seq.is_finished():
                    seq.status = SequenceStatus.RUNNING
        self.running.extend(scheduled)
        
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=scheduled,
            prompt_ids=[],
            num_prompt_tokens=0,
            num_generation_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
            
        if self.exec_type == ExecType.LORA:
            if self.policy_name == "fcfs":
                scheduler_outputs = self._strict_fcfs_schedule(self.lora_config.gpu_capacity)
            elif self.policy_name == "credit":
                scheduler_outputs = self._credit_schedule()
            else:
                scheduler_outputs = self._always_merge_schedule()
        else:
            scheduler_outputs = self._strict_fcfs_schedule(1)

        # Create input data structures.
        now = time.time()
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_group.start_time = now
            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            is_prompt = seq_group.request_id in scheduler_outputs.prompt_ids
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                model_id=seq_group.model_id,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def update(
        self,
        seq_outputs: Dict[int, SequenceOutputs],
    ) -> List[SequenceGroup]:

        # Remove iteration-finished groups from RUNNING.
        now = time.time()
        scheduled: List[SequenceGroup] = []
        i = 0
        while i < len(self.running):
            seq_group = self.running[i]
            found = False
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                if seq.seq_id in seq_outputs:
                    seq_group.exec_time += now - seq_group.start_time
                    scheduled.append(seq_group)
                    found = True
                    break
            if found:
                self.running.pop(i)
                continue
            i += 1

        # Update the scheduled sequences and free blocks.
        for seq_group in scheduled:
            # Process beam search results before processing the new tokens.
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                output = seq_outputs[seq.seq_id]
                if seq.seq_id != output.parent_seq_id:
                    # The sequence is a fork of the parent sequence (beam
                    # search). Free the current sequence.
                    self.block_manager.free(seq)
                    # Fork the parent sequence.
                    parent_seq = seq_group.find(output.parent_seq_id)
                    parent_seq.fork(seq)
                    self.block_manager.fork(parent_seq, seq)

            # Process the new tokens.
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Append a new token to the sequence.
                output = seq_outputs[seq.seq_id]
                seq.append_token_id(output.output_token, output.logprobs)
                seq.status = SequenceStatus.READY
        self.ready.extend(scheduled)
        return scheduled

    def free_seq(self, seq: Sequence, finish_status: SequenceStatus) -> None:
        seq.status = finish_status
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.ready = [
            seq_group for seq_group in self.ready
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.READY

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.READY):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not supported. In such a case,
        # we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            seqs = seq_group.get_seqs(status=SequenceStatus.READY)
            if len(seqs) == 1:
                preemption_mode = PreemptionMode.SWAP
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self.preempt_cnt += 1
            print("preempt_cnt:", self.preempt_cnt)
            # print("preempt by recompute")
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self.swap_cnt += 1
            print("swap_cnt:", self.swap_cnt)
            # print("preempt by swap")
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, "Invalid preemption mode."

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.READY)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.READY

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.READY):
            seq.status = SequenceStatus.SWAPPED
