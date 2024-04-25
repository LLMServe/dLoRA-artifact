import json

num_gpu = 8
json_list = []

for gpu_id in range(num_gpu):
    json_list.append({"gpu": gpu_id, "sm": [], "mem": []})

with open("pmon.log", "r") as json_file:
    for line in json_file:
        data = json.loads(line)
        gpu_id = data["gpu"]
        if data["sm"] > 0:
            json_list[gpu_id]["sm"].append(data["sm"])
            json_list[gpu_id]["mem"].append(data["mem"])

for gpu_id in range(num_gpu):
    json_list[gpu_id]["sm"] = sum(json_list[gpu_id]["sm"]) / len(json_list[gpu_id]["sm"])
    json_list[gpu_id]["mem"] = sum(json_list[gpu_id]["mem"]) / len(json_list[gpu_id]["mem"])

with open("pmon.json", "w") as json_file:
    json.dump(json_list, json_file, indent=4)
