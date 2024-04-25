from typing import List, Optional, Dict, Tuple
import asyncio
import time
import copy
import math
import random
from enum import Enum

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, ExecType)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sequence import (Sequence, SequenceGroup, SequenceGroupMetadata,
                           SequenceStatus, RequestMetadata)
from vllm.engine.migration_ilp import MigrationILP

from vllm.engine.ray_utils import ray
from ray.air.util.torch_dist import init_torch_dist_process_group

_GB = 1 << 30
PCIE_BANDWIDTH = 32 * _GB

class MigrationType(Enum):
    DISPATCH_ONLY = 1
    DISPATCH_MIG = 2
    PERIOD_MIG = 3

class EngineManager:
    """EngineManager is responsible for managing the engines."""

    def __init__(
        self,
        exec_type: ExecType,
        migration_type: MigrationType,
        num_groups: int,
        num_models: int, # generally num_models >= num_groups
        engine_args: AsyncEngineArgs
        ):
        """Initialize the EngineManager."""
        self.exec_type = exec_type
        self.migration_type = migration_type
        self.num_groups = num_groups
        self.num_models = num_models
        self.num_model_per_group = num_models // num_groups
        self.model_engine_mapping: Dict[int, List[int]] = {} # model_id -> [engine_id]
        self.engine_model_mapping = {} # engine_id -> [model_id]
        self.engine_num_requests = [0] * num_groups
        self.engine_num_free_blocks = [0] * num_groups
        self.bs = engine_args.max_num_seqs
        self.cpu_swap_space = engine_args.swap_space * _GB
        self.engine_req_model_cnt: Dict[int, Dict[int, int]] = {}
        self.reqs_metadata: Dict[int, List[RequestMetadata]] = {}
        self.model_exec_info = {i: [0, 0.0] for i in range(num_models)}
        self.model_avg_exec_time = []
        # parameters for migration
        self.default_exec_time = 5.0
        self.migration_interval = 10
        self.migration_req_thres = 16
        self.engine_exec_cost: Dict[int, float] = {} # approximated execution cost of each engine
        self.merge_speed_ratio = 0.6 # speed ratio of merge engine
        self.select_lock = asyncio.Lock()
        self.background_loop = None
        self.engines: List[AsyncLLMEngine] = []
        self.engine_id = 0
        self.all_workers = []

        for i in range(num_groups):
            self.engines.append(AsyncLLMEngine.from_engine_args(engine_args, i))
            self.all_workers.extend(self.engines[-1].workers)
        init_torch_dist_process_group(self.all_workers, backend="nccl")
        outputs = []
        for engine in self.engines:
            outputs.append(engine.engine._init_workers_ray_cont.remote())
        results = ray.get(outputs)
        self.available_gpu_memorys = [result[0] for result in results]
        # here we assert all lora types have the same weight size if parallel config is the same
        self.lora_weight_sizes = [result[1] for result in results]
        self.cache_block_sizes = [result[2] for result in results]
        self.num_gpu_blocks = [0] * self.num_groups
        self.num_cpu_blocks = [int(self.cpu_swap_space // self.cache_block_sizes[i]) for i in range(self.num_groups)]
        self.lora_load_cost = self.lora_weight_sizes[0] / PCIE_BANDWIDTH

        if self.exec_type != ExecType.REPLICATED:
            dist = [1] * self.num_models
            all = sum(dist)
            self.expected_lora_distribution = [(num+1e-7) / all for num in dist]

            self.engine_model_mapping = {i: [] for i in range(self.num_groups)}
            self.model_engine_mapping = {i: [] for i in range(self.num_models)}
            self.find_best_lora_weight_schedule(True, self.expected_lora_distribution)
            self.engine_lora_capacity = [0] * self.num_groups
            for engine_id, model_ids in self.engine_model_mapping.items():
                self.engine_lora_capacity[engine_id] = len(model_ids)

            for engine in self.engines:
                self.num_gpu_blocks[engine.engine_id] = int(self.available_gpu_memorys[engine.engine_id] // self.cache_block_sizes[engine.engine_id])
                ray.get(engine.engine.init_cont.remote(self.available_gpu_memorys[engine.engine_id], self.engine_model_mapping[engine.engine_id]))
                engine.models_in_gpu = ray.get(engine.engine.get_models_in_gpu.remote())
                engine.num_free_blocks = ray.get(engine.engine.get_num_free_blocks.remote())

        else:
            if self.exec_type == ExecType.REPLICATED:
                for i in range(num_models):
                    self.model_engine_mapping[i] = [i // self.num_model_per_group]
                for i in range(num_groups):
                    self.engine_model_mapping[i] = [i * self.num_model_per_group + j for j in range(self.num_model_per_group)]

            for engine in self.engines:
                ray.get(engine.engine.init_cont.remote(self.available_gpu_memorys[engine.engine_id], self.engine_model_mapping[engine.engine_id]))
            

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and not self.background_loop.done())

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        print("start background loop")
        self.background_loop = asyncio.get_event_loop().create_task(
            self.run_loop())


    def get_migration_info(self):
        need_migration = False
        self.reqs_metadata = {}
        self.model_exec_info = {i: [0, 0.0] for i in range(self.num_models)}
        self.model_avg_exec_time = [0.0] * self.num_models
        for engine in self.engines:
            reqs_metadata, model_exec_time = engine.get_migration_info()
            if len(reqs_metadata) > self.bs:
                need_migration = True
            self.reqs_metadata[engine.engine_id] = reqs_metadata
            for model_id, exec_info in model_exec_time.items():
                self.model_exec_info[model_id][0] += exec_info[0]
                self.model_exec_info[model_id][1] += exec_info[1]
        for model_id, exec_info in self.model_exec_info.items():
            self.model_avg_exec_time[model_id] = exec_info[1] / exec_info[0] if exec_info[0] > 0 else self.default_exec_time
        return need_migration

    async def run_loop(self):
        while True:
            await asyncio.sleep(self.migration_interval)
            await self.migration_schedule()

    async def get_engine_req_model_cnt(self):
        for engine in self.engines:
            if self.migration_type == MigrationType.DISPATCH_MIG:
                req_model_cnt = ray.get(engine.engine.get_req_model_cnt.remote())
            else:
                req_model_cnt = await engine.get_req_model_cnt()
            cost = 0.0
            res_cnt = 0
            for _, cnt in req_model_cnt.items():
                res_cnt += cnt % self.bs
                cost += cnt // self.bs * self.merge_speed_ratio
            cost += res_cnt // self.bs
            self.engine_exec_cost[engine.engine_id] = cost
            self.engine_req_model_cnt[engine.engine_id] = req_model_cnt
            self.engine_num_requests[engine.engine_id] = sum(req_model_cnt.values())
            

    async def select_engine(self, request_id, prompt, sampling_params, model_id: int) -> AsyncLLMEngine:
        if self.exec_type == ExecType.LORA and self.migration_type == MigrationType.PERIOD_MIG and not self.is_running:
            self.start_background_loop()
        """Select the engine for the model."""

        if self.exec_type != ExecType.REPLICATED:
            self.expected_lora_distribution[model_id] += 1
            await self.get_engine_req_model_cnt()
            candidate_engines = {}
            sub_engine_model_ids = {}

            if len(self.model_engine_mapping[model_id]) == 0:
                engine_id = self.engine_num_requests.index(min(self.engine_num_requests))
                self.model_engine_mapping[model_id] = [engine_id]
                self.engine_model_mapping[engine_id].append(model_id)
                print(f"add model {model_id} to engine {engine_id}")

            for engine_id in self.model_engine_mapping[model_id]:
                candidate_engines[engine_id] = self.engine_exec_cost[engine_id]
            if self.migration_type == MigrationType.DISPATCH_MIG:
                for engine_id, req_model_cnt in self.engine_req_model_cnt.items():
                    assert model_id in req_model_cnt
                    if engine_id in candidate_engines:
                        continue
                    for sub_model_id in self.engine_model_mapping[engine_id]:
                        if req_model_cnt[sub_model_id] == 0 and len(self.model_engine_mapping[sub_model_id]) > 1:
                            candidate_engines[engine_id] = self.engine_exec_cost[engine_id] + self.lora_load_cost
                            sub_engine_model_ids[engine_id] = sub_model_id
                            break

            # find the engine with the minimum cost
            min_cost = math.inf
            min_engine_id = None
            for engine_id, cost in candidate_engines.items():
                if cost < min_cost:
                    min_cost = cost
                    min_engine_id = engine_id

            if min_engine_id in sub_engine_model_ids:
                print("substitute model", sub_engine_model_ids[min_engine_id], "with model", model_id, "in engine", min_engine_id, "with", min_cost, "cost")
                sub_model_id = sub_engine_model_ids[min_engine_id]
                self.engine_model_mapping[min_engine_id].remove(sub_model_id)
                self.model_engine_mapping[sub_model_id].remove(min_engine_id)
                self.engine_model_mapping[min_engine_id].append(model_id)
                self.model_engine_mapping[model_id].append(min_engine_id)

            eng_id = min_engine_id

        elif self.exec_type == ExecType.REPLICATED:
            eng_id = self.model_engine_mapping[model_id][0]

        engine: AsyncLLMEngine = self.engines[eng_id]
        if self.exec_type == ExecType.REPLICATED:
            model_id = model_id % self.num_model_per_group
        results_generator = engine.generate(prompt, sampling_params, request_id, model_id)
        return engine, results_generator


    def greedy_placement(self, arrival_rates: List[int]):
        placement = {}
        cnts = [0] * self.num_groups
        arrival_rates_mapping = {i: arrival_rates[i] for i in range(self.num_models)}
        sorted_arrival_rates = dict(sorted(arrival_rates_mapping.items(), key=lambda x: x[1], reverse=True))
        for model_id, rate in sorted_arrival_rates.items():
            engine_id = cnts.index(min(cnts))
            cnts[engine_id] += rate
            placement[model_id] = engine_id

        self.engine_model_mapping = {i: [] for i in range(self.num_groups)}
        for model_id, engine_id in placement.items():
            self.model_engine_mapping[model_id] = [engine_id]
            self.engine_model_mapping[engine_id].append(model_id)
        for engine_id, model_ids in self.engine_model_mapping.items():
            self.engines[engine_id].adjust_lora_adapter(model_ids)

    def calc_min_bt(self, expected_lora_distribution: List[float]):
        min_bt = math.inf
    
        for lora_type in range(self.num_models):
            total_throughput = 0
            for engine_id in range(self.num_groups):
                if lora_type in self.engine_model_mapping[engine_id]:
                    max_throughput_on_this_replica = self.available_gpu_memorys[engine_id] # since partitioned kvcache size is the same, calc max_tput is  to calc available gpu memory
                    total_throughput += max_throughput_on_this_replica
            min_bt = min(min_bt, total_throughput / expected_lora_distribution[lora_type])
        
        return min_bt
        
    def find_best_lora_weight_schedule(self, is_init: bool, expected_lora_distribution: List[float], current_lora_distribution: List[int] = None, engine_lora_capacity: List[int] = None):
        """Find the best lora weight schedule."""
        if current_lora_distribution is None:
            current_lora_distribution = [0 for _ in range(self.num_models)]
        num_lora_replicas = 0
        best_bt = 0
        update_flag = True

        engine_ids = [i for i in range(self.num_groups)]
        models_not_allocated = [i for i in range(self.num_models) if len(self.model_engine_mapping[i]) == 0]

        while update_flag:
            update_flag = False
            
            next_lora_type = 0
            for i in range(self.num_models):
                if current_lora_distribution[i] / (num_lora_replicas + 1e-7) - expected_lora_distribution[i] < current_lora_distribution[next_lora_type] / (num_lora_replicas + 1e-7) - expected_lora_distribution[next_lora_type]:
                    next_lora_type = i

                if next_lora_type not in models_not_allocated and len(models_not_allocated) > 0:
                    next_lora_type = models_not_allocated[0]

            # sort engine ids by number of lora weights on it
            if engine_lora_capacity is None:
                engine_ids = sorted(engine_ids, key=lambda engine_id: len(self.engine_model_mapping[engine_id]))
            else:
                engine_ids = sorted(engine_ids, key=lambda engine_id: engine_lora_capacity[engine_id] - len(self.engine_model_mapping[engine_id]), reverse=True)
                engine_ids = [id for id in engine_ids if engine_lora_capacity[id] > len(self.engine_model_mapping[id])]
            for engine_id in engine_ids:
                if next_lora_type in self.engine_model_mapping[engine_id]:
                    continue
                self.engine_model_mapping[engine_id].append(next_lora_type)
                self.available_gpu_memorys[engine_id] -= self.lora_weight_sizes[engine_id]
                new_bt = self.calc_min_bt(expected_lora_distribution)
                if new_bt >= best_bt:
                    current_lora_distribution[next_lora_type] += 1
                    num_lora_replicas += 1
                    best_bt = new_bt
                    update_flag = True
                    if next_lora_type in models_not_allocated:
                        models_not_allocated.remove(next_lora_type)
                    break
                else:
                    self.engine_model_mapping[engine_id].remove(next_lora_type)
                    self.available_gpu_memorys[engine_id] += self.lora_weight_sizes[engine_id]
                    if models_not_allocated:
                        update_flag = True

        if is_init:
                min_replicas = self.num_groups + self.num_models - 1
                if self.migration_type == MigrationType.PERIOD_MIG:
                    next_engine_id = 0
                    next_lora_type = random.randint(0, self.num_models-1) # random int
                    while num_lora_replicas < min_replicas:
                        while len(self.engine_model_mapping[next_engine_id]) >= self.num_models or next_lora_type in self.engine_model_mapping[next_engine_id]:
                            next_engine_id = (next_engine_id + 1) % self.num_groups
                        self.engine_model_mapping[next_engine_id].append(next_lora_type)
                        self.available_gpu_memorys[next_engine_id] -= self.lora_weight_sizes[next_engine_id]
                        next_engine_id = (next_engine_id + 1) % self.num_groups
                        current_lora_distribution[next_lora_type] += 1
                        num_lora_replicas += 1
                else:
                    next_engine_id = 0
                    next_lora_type = 0
                    while num_lora_replicas < min_replicas:
                        while len(self.engine_model_mapping[next_engine_id]) >= self.num_models:
                            next_engine_id = (next_engine_id + 1) % self.num_groups
                        while next_lora_type in self.engine_model_mapping[next_engine_id]:
                            next_lora_type = (next_lora_type + 1) % self.num_models
                        self.engine_model_mapping[next_engine_id].append(next_lora_type)
                        self.available_gpu_memorys[next_engine_id] -= self.lora_weight_sizes[next_engine_id]
                        next_engine_id = (next_engine_id + 1) % self.num_groups
                        next_lora_type = (next_lora_type + 1) % self.num_models
                        current_lora_distribution[next_lora_type] += 1
                        num_lora_replicas += 1


        self.model_engine_mapping = {i: [] for i in range(self.num_models)}
        for engine_id, model_ids in self.engine_model_mapping.items():
            for model_id in model_ids:
                self.model_engine_mapping[model_id].append(engine_id)

    async def migration_schedule(self) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
        """Find the migration schedule."""
        async with self.select_lock:
            need_migration = self.get_migration_info()

            # TODO: decide when to not use migration
            engine_block_cnt = {i: sum([reqs.num_blocks for reqs in self.reqs_metadata[i]]) for i in range(self.num_groups)}
            engine_block_cnt = sorted(engine_block_cnt.items(), key=lambda x: x[1])
            most_block_engine_id = engine_block_cnt[-1][0]
            most_block_engine_cnt = engine_block_cnt[-1][1]
            least_block_engine_id = engine_block_cnt[0][0]
            least_block_engine_cnt = engine_block_cnt[0][1]
            if most_block_engine_cnt >= self.num_gpu_blocks[most_block_engine_id] * 0.9 and least_block_engine_cnt < self.num_gpu_blocks[least_block_engine_id] * 0.9:
                ilp_engines = sorted([most_block_engine_id, least_block_engine_id])

                print(most_block_engine_id, "kvcache is full, balance with", least_block_engine_id)

            else:
                engine_req_cnt = {i: len(reqs) for i, reqs in self.reqs_metadata.items()}
                num_reqs = sum(engine_req_cnt.values())
                engine_req_cnt = sorted(engine_req_cnt.items(), key=lambda x: x[1])
                matched_engines = []
                while engine_req_cnt:
                    avg_num_reqs = num_reqs / len(engine_req_cnt)
                    most_req_engine_id, most_req_engine_cnt = engine_req_cnt.pop()
                    num_reqs -= most_req_engine_cnt
                    delta = most_req_engine_cnt - avg_num_reqs
                    if delta < self.migration_req_thres:
                        return
                    assert delta >= 0
                    for engine_id, cnt in engine_req_cnt:
                        if delta <= 0 or len(matched_engines) > 0:
                            break
                        to_fill = avg_num_reqs - cnt
                        if to_fill <= 0:
                            break
                        to_fill = min(to_fill, delta)
                        matched_engines.append(engine_id)
                        delta -= to_fill
                    
                    if matched_engines:
                        break

                if not matched_engines:
                    return
                
                ilp_engines = sorted([most_req_engine_id] + matched_engines)
                
                print(most_req_engine_id, "is too much, balance with", matched_engines)

            ilp_engine_mapping = {ilp_engines[i]: i for i in range(len(ilp_engines))}

            ilp_reqs_metadata = []
            ilp_models = set()
            for engine_id in ilp_engines:
                for model_id in self.engine_model_mapping[engine_id]:
                    ilp_models.add(model_id)
            ilp_models = sorted(list(ilp_models))
            ilp_model_mapping = {ilp_models[i]: i for i in range(len(ilp_models))}
            for engine_id in ilp_engines:
                for req_metadata in self.reqs_metadata[engine_id]:
                    req_metadata.engine_id = ilp_engine_mapping[engine_id]
                    req_metadata.model_id = ilp_model_mapping[req_metadata.model_id]
                ilp_reqs_metadata.extend(self.reqs_metadata[engine_id])

            ilp_num_groups = len(ilp_engines)
            ilp_num_models = len(ilp_models)
            ilp_num_gpu_blocks = [self.num_gpu_blocks[engine_id] for engine_id in ilp_engines]
            ilp_num_cpu_blocks = [self.num_cpu_blocks[engine_id] for engine_id in ilp_engines]
            ilp_lora_capacity = [len(self.engine_model_mapping[engine_id]) for engine_id in ilp_engines]
            ilp_model_avg_exec_time = [self.model_avg_exec_time[ilp_model_mapping[model_id]] for model_id in ilp_models]

            ilp_model_engine_mapping = {i: [] for i in range(ilp_num_models)}
            for engine_id, model_ids in self.engine_model_mapping.items():
                if engine_id not in ilp_engines:
                    continue
                for model_id in model_ids:
                    if model_id in ilp_engine_mapping:
                        ilp_model_engine_mapping[ilp_model_mapping[model_id]].append(ilp_engine_mapping[engine_id])

            migration_reqs = {i: {j: [] for j in range(self.num_groups)} for i in range(self.num_groups)}
            migration_ilp = MigrationILP(ilp_reqs_metadata, ilp_num_groups, ilp_num_models, ilp_num_gpu_blocks, ilp_num_cpu_blocks, ilp_lora_capacity, ilp_model_avg_exec_time, 0.1, PCIE_BANDWIDTH / self.cache_block_sizes[0], ilp_model_engine_mapping)
            req_migration_decision, lora_migration_decision, lora_weight_cnt = migration_ilp.solve()
            if req_migration_decision is None:
                print("migration ILP failed")
                return

            for src_engine_id, decision in req_migration_decision.items():
                src_engine_id = ilp_engines[src_engine_id]
                for dst_engine_id, req_ids in decision.items():
                    dst_engine_id = ilp_engines[dst_engine_id]
                    if not req_ids:
                        continue
                    print("move", len(req_ids), "requests from", src_engine_id, "to", dst_engine_id)
                    seq_groups = self.engines[src_engine_id].fetch_seq_groups(req_ids)
                    migration_reqs[src_engine_id][dst_engine_id] = seq_groups

            for engine_id, model_ids in lora_migration_decision.items():
                self.engine_model_mapping[ilp_engines[engine_id]] = [ilp_models[model_id] for model_id in model_ids]
            lora_weight_cnt = [0] * self.num_models
            print("engine_model_mapping", self.engine_model_mapping)
            self.model_engine_mapping = {i: [] for i in range(self.num_models)}
            for engine_id, model_ids in self.engine_model_mapping.items():
                for model_id in model_ids:
                    self.model_engine_mapping[model_id].append(engine_id)
                    lora_weight_cnt[model_id] += 1
            self.find_best_lora_weight_schedule(False, self.expected_lora_distribution, lora_weight_cnt, self.engine_lora_capacity)

            for src_engine_id, decision in migration_reqs.items():
                for dst_engine_id, seq_groups in decision.items():
                    if not seq_groups:
                        continue
                    req_ids = [seq_group.request_id for seq_group in seq_groups]
                    dst_seq_groups = copy.deepcopy(seq_groups)
                    request_streams = self.engines[src_engine_id].remove_requests(req_ids)
                    assert len(request_streams) == len(req_ids)
                    self.engines[dst_engine_id].add_requests(request_streams)
                    dst_seq_groups = self.engines[dst_engine_id].insert_seq_groups(dst_seq_groups)


    def set_output(self, output) -> None:
        """Set the output of the engine."""
        for engine in self.engines:
            ray.get(engine.engine.set_output.remote(output))