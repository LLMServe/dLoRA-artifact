import subprocess
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_gpu", type=int, default=8)
args = parser.parse_args()
num_gpu = args.num_gpu
batch_size = 1
running = False
num_iter = 0

sm_list = [0] * num_gpu
mem_list = [0] * num_gpu


# Use pipe to get output of nvidia-smi pmon
try:
    process = subprocess.Popen(["nvidia-smi", "pmon"], stdout=subprocess.PIPE, universal_newlines=True)
except subprocess.CalledProcessError as e:
    print("Failed to run nvidia-smi:", e.output)
    exit()

with open("pmon.log", "w") as json_file:
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue

        fields = line.split()

        # Skip header
        if len(fields) != 8:
            continue
        
        # Skip unused GPU
        gpu = int(fields[0])
        if gpu >= num_gpu:
            continue

        # No running process
        if fields[3] == '-':
            continue

        sm = float(fields[3])
        mem = float(fields[4])
        if gpu == 0:
            running = False
        if sm > 0 and mem > 0:
            running = True
        if gpu == num_gpu - 1 and not running:
            continue

        sm_list[gpu] += sm
        mem_list[gpu] += mem

        if gpu == num_gpu - 1:
            num_iter += 1
            if num_iter % batch_size == 0:
                for gpu_id in range(num_gpu):
                    sm_list[gpu_id] /= batch_size
                    mem_list[gpu_id] /= batch_size

                    data = {
                        "gpu": gpu_id,
                        "sm": sm_list[gpu_id],
                        "mem": mem_list[gpu_id],
                        "time": time.time()
                    }
                    json.dump(data, json_file, indent=None)
                    json_file.write("\n")
                    json_file.flush()

                    sm_list[gpu_id] = 0
                    mem_list[gpu_id] = 0
