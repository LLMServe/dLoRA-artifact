import argparse
import random
import pandas as pd
import numpy as np

# invocations_per_function_md.anon.d[01-14].csv
#     Field	Description
#     HashOwner	unique id of the application owner 1
#     HashApp	unique id for application name 1
#     HashFunction	unique id for the function name within the app 1
#     Trigger	trigger for the function2
#     1 .. 1440	1440 fields, with the number of invocations of the function per each minute of the 24h period in the file3

# function_durations_percentiles.anon.d[01-14].csv
#     Field	Description
#     HashOwner	unique id of the application owner
#     HashApp	unique id for application name
#     HashFunction	unique id for the function name within the app
#     Average	Average execution time (ms) across all invocations of the 24-period 4
#     Count	Number of executions used in computing the average5
#     Minimum	Minimum execution time for the 24-hour period6
#     Maximum	Maximum execution time for the 24-hour period6
#     percentile_Average_0	Weighted 0th-percentile of the execution time average7
#     percentile_Average_1	Weighted 1st-percentile of the execution time average7
#     percentile_Average_25	Weighted 25th-percentile of the execution time average7
#     percentile_Average_50	Weighted 50th-percentile of the execution time average7
#     percentile_Average_75	Weighted 75th-percentile of the execution time average7
#     percentile_Average_99	Weighted 99th-percentile of the execution time average7
#     percentile_Average_100	Weighted 100th-percentile of the execution time average7

# app_memory_percentiles.anon.d[01..12].csv
#     Field	Description
#     HashOwner	unique id of the application owner
#     HashApp	unique id for application name
#     SampleCount	Number of samples used for computing the average
#     AverageAllocatedMb	Average allocated memory across all SampleCount measurements throughout this 24h period8
#     AverageAllocatedMb_pct1	1st percentile of the average allocated memory 9
#     AverageAllocatedMb_pct5	5th percentile of the average allocated memory 9
#     AverageAllocatedMb_pct25	25th percentile of the average allocated memory 9
#     AverageAllocatedMb_pct50	50th percentile of the average allocated memory 9
#     AverageAllocatedMb_pct75	75th percentile of the average allocated memory 9
#     AverageAllocatedMb_pct95	95th percentile of the average allocated memory 9
#     AverageAllocatedMb_pct99	99th percentile of the average allocated memory 9
#     AverageAllocatedMb_pct100	100th percentile of the average allocated memory 9

duration_list = []

def applfy_fn(row):
    submit_time_list = []
    global duration_list
    for duration in duration_list:
        num_invocation = row[duration]
        for i in range(num_invocation):
            submit_time = (int(duration) + i / num_invocation) * 60 * 1000
            submit_time_list.append(submit_time)

    image_name = row['FunctionName'] # row['HashOwner'] + '-' + row['HashApp'] + '-' + row['HashFunction']
    cold_start_time = row['Maximum'] - row['Average']
    return pd.Series({
        'submit_time': submit_time_list,
        'image_name': image_name,
        'execution_time': row['Average'],
        'hot_start_time': 0,
        'cold_start_time': cold_start_time,
        'vcpu': 0,
        'memory': row['AverageAllocatedMb'],
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--invocations_per_function_md_path', type=str, required=True, help='')
    parser.add_argument('--function_durations_percentiles_path', type=str, required=True, help='')
    parser.add_argument('--app_memory_percentiles_path', type=str, required=True, help='')
    parser.add_argument('--duration', type=int, required=True, help='')
    parser.add_argument('--num_function', type=int, required=True, help='')
    parser.add_argument('--load_frac', type=float, default=1.0, help='')
    parser.add_argument('--workload_config_path', type=str, required=True, help='')
    parser.add_argument('--trace_type', type=int, default=0, help='')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    duration_list = [str(i) for i in range(1, args.duration + 1)]
    usecols = ['HashOwner', 'HashApp', 'HashFunction'] + duration_list
    invocations_per_function_md = pd.read_csv(args.invocations_per_function_md_path, usecols=usecols)

    function_durations_percentiles = pd.read_csv(args.function_durations_percentiles_path, usecols=['HashOwner', 'HashApp', 'HashFunction', 'Average', 'Maximum'])
    invocations_per_function_md = invocations_per_function_md.merge(function_durations_percentiles, on=['HashOwner', 'HashApp', 'HashFunction'], how='inner')
    invocations_per_function_md = invocations_per_function_md[invocations_per_function_md['Average'] > 0]

    invocations_per_function_md['InvocationSum'] = 0
    for duration in duration_list:
        invocations_per_function_md['InvocationSum'] += invocations_per_function_md[duration]
    invocations_per_function_md = invocations_per_function_md[invocations_per_function_md['InvocationSum'] > 0]
    if args.trace_type == 0: #sample
        invocations_per_function_md.sort_values(by=['InvocationSum'], ascending=False, inplace=True)
        invocations_per_function_md = invocations_per_function_md.sample(n=args.num_function, random_state=0, replace=False)
    elif args.trace_type == 1: #random
        invocations_per_function_md = invocations_per_function_md.sample(n=args.num_function, random_state=0, replace=False)
    elif args.trace_type == 2: #rare
        invocations_per_function_md.sort_values(by=['InvocationSum'], ascending=False, inplace=True)
        invocations_per_function_md = invocations_per_function_md[:len(invocations_per_function_md)//8] # 8 is magic number
        invocations_per_function_md = invocations_per_function_md.sample(n=args.num_function, random_state=0, replace=False)
    else:
        assert False

    function_count_per_app = invocations_per_function_md.groupby(['HashOwner', 'HashApp']).size().reset_index(name='FunctionCount')
    invocations_per_function_md = invocations_per_function_md.merge(function_count_per_app, on=['HashOwner', 'HashApp'], how='inner')

    function_name = invocations_per_function_md.groupby(['HashOwner', 'HashApp', 'HashFunction']).size().reset_index(name='FunctionName')
    function_name['FunctionName'] = np.arange(1, len(function_name) + 1)
    print(function_name)
    invocations_per_function_md = invocations_per_function_md.merge(function_name, on=['HashOwner', 'HashApp', 'HashFunction'], how='inner')    

    app_memory_percentiles = pd.read_csv(args.app_memory_percentiles_path, usecols=['HashOwner', 'HashApp', 'AverageAllocatedMb'])
    invocations_per_function_md = invocations_per_function_md.merge(app_memory_percentiles, on=['HashOwner', 'HashApp'], how='inner')

    invocations_per_function_md['AverageAllocatedMb'] = (invocations_per_function_md['AverageAllocatedMb'] / invocations_per_function_md['FunctionCount']).astype(int)

    trace = invocations_per_function_md.apply(applfy_fn, axis=1).explode('submit_time')
    trace = trace.sample(frac=args.load_frac, random_state=0, replace=False)
    trace.sort_values(by=['submit_time'], ascending=True, inplace=True)
    trace['task_id'] = np.arange(1, len(trace) + 1)
    print(trace)

    trace.to_csv(args.workload_config_path, index=False)