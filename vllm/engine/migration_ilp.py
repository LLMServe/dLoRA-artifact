import pulp
from typing import Dict, List, Tuple, Set
import time
import multiprocessing

from vllm.sequence import RequestMetadata

class MigrationILP(object):
    def __init__(self, reqs_metadata: List[RequestMetadata], num_groups: int, num_models: int, engine_gpu_blocks: List[int], engine_cpu_blocks: List[int], engine_lora_capacity: List[int], lora_exec_time: List[int], alpha: float, bw: int, model_engine_mapping: Dict[int, List[int]]):
        print("num_reqs", len(reqs_metadata), "num_groups", num_groups, "num_models", num_models)
        self.reqs_metadata_mapping = {i : reqs_metadata[i] for i in range(len(reqs_metadata))}
        self.num_reqs = len(reqs_metadata) # i
        self.num_groups = num_groups # j
        self.num_models = num_models # k
        self.engine_gpu_blocks = engine_gpu_blocks
        self.engine_cpu_blocks = engine_cpu_blocks
        self.engine_lora_capacity = engine_lora_capacity
        self.lora_exec_time = lora_exec_time
        self.alpha = alpha
        self.B = bw
        self.model_engine_mapping = model_engine_mapping
        self.prob = pulp.LpProblem('MigrationILP', pulp.LpMinimize)
        self.x = pulp.LpVariable.dicts('x', ((i, j) for i in range(self.num_reqs) for j in range(self.num_groups)), lowBound=0)
        self.y = pulp.LpVariable.dicts('y', ((i, j) for i in range(self.num_models) for j in range(self.num_groups)), cat='Binary')
        self.max_time = pulp.LpVariable('max_time', lowBound=0, cat='Continuous')
        self.max_mem = pulp.LpVariable.dicts('z', ((j) for j in range(self.num_groups)), lowBound=0)
        self.__add_constraints()
        self.__set_init_value()
        self.__set_objective()

    def __add_constraints(self):
        # constraint 1
        for i in range(self.num_reqs):
            self.prob += pulp.lpSum([self.x[(i, j)] for j in range(self.num_groups)]) >= 1

        # constraint 2
        for k in range(self.num_models):
            num_reqs = len([req_ for req_ in self.reqs_metadata_mapping.values() if req_.model_id == k])
            for j in range(self.num_groups):
                self.prob += num_reqs * self.y[(k, j)] >= pulp.lpSum([self.x[(i, j)] for i in range(self.num_reqs) if self.reqs_metadata_mapping[i].model_id == k])
            
        # constraint 4
        for j in range(self.num_groups):
            self.prob += pulp.lpSum([self.y[(k, j)] for k in range(self.num_models)]) <= self.engine_lora_capacity[j]

    def __set_init_value(self):
        for i in range(self.num_reqs):
            for j in range(self.num_groups):
                self.x[(i, j)].setInitialValue(self.reqs_metadata_mapping[i].engine_id == j)
        for k in range(self.num_models):
            for j in range(self.num_groups):
                self.y[(k, j)].setInitialValue(j in self.model_engine_mapping[k])

    def __set_objective(self):
        for j in range(self.num_groups):
            self.prob += self.max_mem[j] >= (pulp.lpSum([self.reqs_metadata_mapping[i].num_blocks * self.x[(i, j)] for i in range(self.num_reqs)]) - self.engine_gpu_blocks[j]) / self.B
            self.prob += self.max_time >= pulp.lpSum([self.lora_exec_time[self.reqs_metadata_mapping[i].model_id] * self.x[(i, j)] for i in range(self.num_reqs)]) + pulp.lpSum([0.05 * self.x[(i, j)] for i in range(self.num_reqs) if self.reqs_metadata_mapping[i].engine_id != j]) + self.max_mem[j]

        self.prob += self.max_time

    def solve(self) -> Tuple[int, Dict[int, List[str]], List[int]]:
        verbose = False
        req_migration_mapping = {i: {j: [] for j in range(self.num_groups)} for i in range(self.num_groups)}
        lora_weight_mapping = {i: [] for i in range(self.num_groups)}
        lora_weight_cnt = [0 for i in range(self.num_models)]
        start = time.time()
        time_limit = 600
        solver = pulp.PULP_CBC_CMD(mip=True,
                                msg=verbose,
                                timeLimit=time_limit,
                                threads=multiprocessing.cpu_count())
        self.prob.solve(solver)
        print(f"Solve time: {time.time() - start}, num variable {len(self.prob.variables())}, num constraints {len(self.prob.constraints)}", flush=True)
        for i in range(self.num_reqs):
            src_group = self.reqs_metadata_mapping[i].engine_id
            req_id = self.reqs_metadata_mapping[i].request_id
            model_id = self.reqs_metadata_mapping[i].model_id
            max_j = 0
            max_val = 0.0
            for j in range(self.num_groups):
                if self.x[(i, j)].value() > max_val:
                    max_val = self.x[(i, j)].value()
                    max_j = j
            assert self.y[(model_id, max_j)].value() == 1, f"Model {model_id} is not assigned to group {max_j}, value {max_val}, but {self.y[(model_id, max_j)].value()}"
            if src_group != max_j:
                if verbose:
                    print(f"Request {i}({req_id}) with type {model_id} is assigned to group {max_j} on GPU with value {max_val}")
                req_migration_mapping[src_group][max_j].append(req_id)
        for k in range(self.num_models):
            for j in range(self.num_groups):
                if self.y[(k, j)].value() == 1:
                    if verbose:
                        print(f"Model {k} is assigned to group {j}")
                    lora_weight_mapping[j].append(k)
                    lora_weight_cnt[k] += 1
        return req_migration_mapping, lora_weight_mapping, lora_weight_cnt
    
