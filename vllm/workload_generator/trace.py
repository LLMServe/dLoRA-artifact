import argparse
import random
import pandas as pd
import numpy as np
import heapq

from typing import Iterable, Optional, Dict, List, Tuple, Iterator
from collections import OrderedDict

from vllm.workload_generator.workload import ArrivalProcess, GammaProcess, PoissonProcess

def load_trace(trace_name, trace_dir, duration_list, end_d, end_h, end_m, start_d=0, start_h=0, start_m=0, need_sort=False):
    if trace_name == "azure_v1":
        usecols = ['HashOwner', 'HashApp', 'HashFunction'] + duration_list
        trace_md = pd.read_csv(trace_dir, usecols=usecols)
        trace_md['InvocationSum'] = 0
        for duration in duration_list:
            trace_md['InvocationSum'] += trace_md[duration]
        trace_md = trace_md[trace_md['InvocationSum'] > 0]
        function_name = trace_md.groupby(['HashOwner', 'HashApp', 'HashFunction']).size().reset_index(name='FunctionName')
        function_name['FunctionName'] = np.arange(0, len(function_name))
        assert len(trace_md) == len(function_name)
        trace_md = trace_md.merge(function_name, on=['HashOwner', 'HashApp', 'HashFunction'], how='inner')   
        if need_sort:
            sorted_md = trace_md.sort_values(by=['InvocationSum'], ascending=False)
            names = sorted_md['FunctionName'].to_numpy()
        else:
            names = function_name['FunctionName'].to_numpy() 
        return trace_md, names # 
    elif trace_name == "azure_v2":
        usecols = ['app', 'func', 'end_timestamp']
        trace_md = pd.read_csv(trace_dir, usecols=usecols)
        function_name = trace_md.groupby(['app', 'func']).size().reset_index(name='name')
        function_name['name'] = np.arange(0, len(function_name))
        trace_md = trace_md.merge(function_name, on=['app', 'func'], how='inner')
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
        trace_md = trace_md[(trace_md['end_timestamp'] >= start_timestamp_seconds) & (trace_md['end_timestamp'] < end_timestamp_seconds)]
        if need_sort:
            names, counts = np.unique(trace_md['name'], return_counts=True)
            sorted_indices = np.argsort(-counts)
            names = names[sorted_indices]
        else:
            names = function_name['name'].to_numpy() 

        return trace_md, names

    
def generate_from_iterators(iterators: Dict[int, Iterator], num_reqs: int):
    heap: List[Tuple[float, int]] = []

    for model_id, iter in iterators.items():
        heapq.heappush(heap, (next(iter), model_id))
    
    result: List[Tuple[float, int]] = []
    num_generated = 0

    while num_generated < num_reqs:
        val, model_id = heapq.heappop(heap)
        result.append((val, model_id))
        num_generated += 1

        next_val = next(iterators[model_id])
        heapq.heappush(heap, (next_val, model_id))

    return result

def timestr_to_dhm(time_str):
    dhm = time_str.split(sep=".")
    if len(dhm) != 3:
        raise RuntimeError("Wrong format for `start_time`.")
    day = int(dhm[0])
    hour = int(dhm[1])
    min = int(dhm[2])
    return day, hour, min

class Trace:
    def __init__(
        self, 
        trace_name: str, 
        trace_dir: str,
        start_time: str, 
        end_time: str,
        need_sort: bool = False,
    ):
        self.trace_name = trace_name
        self.trace_dir = trace_dir
        self.start_d, self.start_h, self.start_m = timestr_to_dhm(start_time)
        self.end_d, self.end_h, self.end_m = timestr_to_dhm(end_time)
        self.start_mnt = self.start_d * 24 * 60 + self.start_h * 60 + self.start_m
        self.end_mnt = self.start_d * 24 * 60 + self.end_h * 60 + self.end_m
        self.duration = self.end_mnt - self.start_mnt
        if trace_name == "azure_v1":
            # now only support for one day in maf1
            # and must assert trace_dir corresponds to the day
            assert self.end_d == self.start_d
            if self.start_d < 9:
                trace_dir += f"invocations_per_function_md.anon.d0{self.start_d+1}.csv"
            else:
                trace_dir += f"invocations_per_function_md.anon.d{self.start_d+1}.csv"
            self.duration_list = [str(i) for i in range(self.start_mnt+1, self.end_mnt+1)]
            self.function_histogram, self.function_names = load_trace(trace_name, trace_dir, self.duration_list, self.end_d, self.end_h, self.end_m, self.start_d, self.start_h, self.start_m, need_sort=need_sort)
        elif trace_name == "azure_v2":
            trace_dir += 'AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt'
            self.function_arrivals, self.function_names = load_trace(trace_name, trace_dir, [], self.end_d, self.end_h, self.end_m, self.start_d, self.start_h, self.start_m, need_sort=need_sort)
        else:
            raise NotImplementedError(f"trace_name {trace_name} not supported")


    def map_model(self, num_models: int, function_names: Iterable[int], map_stride: int = 1):
        mapping: Dict[int, List[int]] = {}
        num_functions = len(function_names)
        assert num_functions >= num_models, f"#function {num_functions} < #models {num_models}"
        rest_stride = map_stride
        model_id = 0
        for idx, func in enumerate(function_names):
            if model_id not in mapping:
                mapping[model_id] = [func]
            else:
                mapping[model_id].append(func)
            rest_stride -= 1
            if rest_stride == 0:
                rest_stride = map_stride
                model_id = (model_id + 1) % num_models
        return mapping
    
    def replay_to_workload(self, num_models: int, num_reqs: int, arrival_distribution: str="gamma", 
                           interval_minutes: int=5, tot_rate: float = 4.0, cv: float = 1.0, map_stride: int = 1) -> List[Tuple[int, float]]:
        
        model_mapping = self.map_model(num_models, self.function_names, map_stride)
        model_histogram: Dict[int, np.array] = {}
        model_arrivals: Dict[int, np.array] = {}
        if self.trace_name == "azure_v1":
            for model, functions in model_mapping.items():
                model_cnts = np.zeros((self.duration,))
                for func in functions:
                    func_cnts = self.function_histogram.loc[self.function_histogram['FunctionName']==func].iloc[0][self.duration_list].to_numpy()
                    model_cnts = np.add(model_cnts, func_cnts)
                model_histogram[model] = model_cnts
        elif self.trace_name == "azure_v2":
            func_mapping = {}
            for model, functions in model_mapping.items():
                for func in functions:
                    func_mapping[func] = model
            for func, model in func_mapping.items():
                func_arrivals = self.function_arrivals[self.function_arrivals['name'] == func]['end_timestamp'].to_numpy()
                if model not in model_arrivals:
                    model_arrivals[model] = func_arrivals
                else:
                    model_arrivals[model] = np.concatenate((model_arrivals[model], func_arrivals))

            for model, arrivals in model_arrivals.items():
                model_arrivals[model] = np.sort(arrivals)            

        dataset: Dict[int, np.array] = {}
        num_intervals = (self.duration + interval_minutes - 1) // interval_minutes
        if self.trace_name == "azure_v1":
            for model, model_cnts in model_histogram.items():
                assert len(model_cnts) == self.duration
                accumulated = np.zeros((num_intervals,))
                for i in range(accumulated.size):
                    start = i * interval_minutes
                    end = (i + 1) * interval_minutes if (i + 1) * interval_minutes <= self.duration else self.duration
                    accumulated[i] = np.sum(model_cnts[start:end])
                dataset[model] = accumulated
        else:
            intervals = np.arange(self.start_mnt, self.end_mnt, interval_minutes)
            if intervals[-1] != self.end_mnt:
                intervals = np.append(intervals, self.end_mnt)
            for m in model_arrivals:
                arrivals = model_arrivals[m]
                interval_dataset = []
                for i in range(intervals.size - 1):
                    tmp = arrivals[arrivals >= intervals[i] * 60]
                    tmp = tmp[tmp < intervals[i+1] * 60]
                    interval_dataset.append(len(tmp))
                dataset[m] = np.array(interval_dataset)

        distributions = self.estimate_parameters_with_histogram(dataset, arrival_distribution, tot_rate, cv)

        num_reqs_per_interval = num_reqs // num_intervals

        replay_trace: List[Tuple[float, int]] = []
        start = 0
        for i in range(num_intervals):
            iterator_list: Dict[int, Iterator] = {}
            for model, arrival_process in distributions.items():
                if arrival_process[i] is None:
                    continue
                iterator_list[model] = arrival_process[i].get_iterator(start, interval_minutes)
            num_reqs_i = num_reqs_per_interval
            if i < num_reqs % num_intervals:
                num_reqs_i += 1
            interval_trace = generate_from_iterators(iterator_list, num_reqs_i)
            replay_trace.extend(interval_trace)
            start, _ = replay_trace[-1]

        workload: List[Tuple[int, float]] = []
        models_cnt = [0] * num_models
        pre_arrival = 0
        for arrival, model in replay_trace:
            models_cnt[model] += 1
            assert pre_arrival <= arrival
            workload.append((model, arrival - pre_arrival))
            pre_arrival = arrival

        print(models_cnt)
        # print(workload)
        print("last arrival:", arrival)

        return workload

    def estimate_parameters_with_histogram(self,
                                           dataset,
                                           arrival_distribution="exponential",
                                           tot_rate=4.0,
                                           cv=1.0) -> Dict[int, List[ArrivalProcess]]:
        if arrival_distribution not in ["exponential", "gamma"]:
            raise NotImplementedError(f"We can only use histogram data for exponential or gamma distribution, "
                                      f"got {arrival_distribution}")
        distributions: Dict[int, List[ArrivalProcess]] = {}
        sum_hist = None
        for _, histogram in dataset.items():
            if sum_hist is None:
                sum_hist = list(histogram)
            else:
                sum_hist += histogram

        for model, histogram in dataset.items():
            distributions[model] = []
            for id, h in enumerate(histogram):
                if h == 0:
                    distributions[model].append(None)
                else:
                    rate_ratio = h / sum_hist[id]
                    arrival_rate = rate_ratio * tot_rate
                    if arrival_distribution == "exponential":
                        distributions[model].append(PoissonProcess(arrival_rate))
                    else:
                        distributions[model].append(GammaProcess(arrival_rate, cv))
        return distributions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_path', type=str, default='~/maf2/', help='')
    args = parser.parse_args()

    trace = Trace("azure_v2", args.trace_path, '0.0.0', '0.6.0', need_sort=True)
    trace.replay_to_workload(8, 400, interval_minutes=60, cv=1, tot_rate=4, map_stride=1)



