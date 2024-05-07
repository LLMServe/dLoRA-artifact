# d-LoRA Artifact

This is the artifact for the paper "dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving". We are going to reproduce the main results in the paper. 
<!-- This project is based on vLLM. [[Link]](https://github.com/vllm-project/vllm) -->

## Setup the environment

### Installation

Similar to the installation of vLLM. [[Link]](https://docs.vllm.ai/en/latest/getting_started/installation.html#build-from-source)

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash ./Anaconda3-2024.02-1-Linux-x86_64.sh
conda create -n dlora python=3.9 -y
conda activate dlora
conda install nvidia/label/cuda-12.2.0::cuda-toolkit
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

git clone https://github.com/LLMServe/dLoRA-artifact.git

cd dLoRA-artifact
pip install -e .
```

To run PEFT equipped with selective batching and PagedAttention, install customized-version PEFT with:
```bash
git submodule update --init --recursive
cd PEFT-Dist
pip install -e .
```

For visualization, install following packages:
```bash
conda install -n dlora ipykernel
pip install matplotlib pandas
```

### Dataset

Dataset:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Save the dataset in ~/ShareGPT_V3_unfiltered_cleaned_split.json.

Supported public trace:
- Microsoft azure_v1 trace. [[Intrduction]](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) [[Download]](https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz)
- Microsoft azure_v2 trace. [[Introduction]](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md) [[Download]](https://github.com/Azure/AzurePublicDataset/raw/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar)


Download and unzip them.
Save azure_v1 in directory ~/maf1/ and azure_v2 in directory ~/maf2/.

### Allocate GPUs in the cluster

```bash
cd ./ae_scripts
bash ./alloc.sh ${num_svr}
```

For end-to-end experiments, num_svr=4;
For others, num_svr=1.

The information of allocated machines are needed in the following sections.

After allocating, you can get the name and ip of machines by
```bash
squeue -u wubingyang
```
For example, if you get one machine with name HOST-10-140-60-7 and ip 10.140.60.7, then $machines and $master_ip in the following sections are set to them, respectively.
If you want to allocate and run dLoRA on multiple machines, $machines in the following sections should be like HOST-10-140-60-[7,10,13,96] and $master_ip is one of them(e.g., 10.140.60.10).

Also notice that once the terminal which runs command *salloc* is closed or terminated, the allocated machines would be released, so you should run *alloc.sh* and *start_server.sh* in the same terminal.

## Motivation

### Figure 2 (b) (Section 2)

First start the server by
```bash
bash ./start_server.sh $machines 1 8 32 $master_ip 7 1 3
```

Wait for server ready(i.e., wait until seeing output "INFO: Uvicorn running on http://$master_ip:8000 (Press CTRL+C to quit)"), then run:
```bash
bash ./fig2b.sh $master_ip
```

Run `figure2.ipynb` to generate the figure `figure2.pdf`.

### Figure 5 (Section 4)

```bash
./figure5.sh
```

The figure `figure5.pdf` will be generated in the current directory to show results.

### Figure 7 (Section 5)

Run `figure7.ipynb` to generate the figure `figure7.pdf`.

## End-to-end performance (Section 7.2, Figure 9)

Notice: experiments in this section would take several hours.

For each execution type(vLLM, Peft, dLoRA) and each model(7b, 13b, 70b), start server:
```bash
bash ./start_server.sh $machines $tp $num_groups $num_models $master_ip $model $exec_type 3
```
where $model is in {7, 13, 70} and $exec_type is in {1, 2, 3}.
If model is 70b, then tp=4, num_groups=8, num_models=32, else tp=1, num_groups=32, num_models=128.

Wait for server ready, then run:
```bash
bash ./fig9_${trace}_${model}.sh $master_ip $exec_type
```
where $trace is in {maf1, maf2} and $model is in {7b, 13b, 70b}.

Use `figure9.ipynb` to generate the figure `figure9.pdf`.


## Ablation studies (Figure 10-13)

### Figure 10 (Section 7.3)

For each policy(unmerged-only, merged-only, dLoRA), start server:
```bash
bash ./start_server.sh $machines 1 1 8 $master_ip 7 3 3 $policy
```
where $policy is in {fcfs, merge, credit}.

Wait for server ready, then run:
```bash
bash ./fig10.sh $master_ip $policy
```

To plot figure `figure10a.pdf` and `figure10b.pdf`, run `figure10.ipynb`.

### Figure 11 (Section 7.4)

#### Figure 11 (a)

For each migration_type(RR, Proactive, dLoRA), start server:
```bash
bash ./start_server.sh $machines 1 8 32 $master_ip 7 3 $migration_type
```
where $migration_type is in {1, 2, 3}.

Wait for server ready, then run:
```bash
bash ./fig11a.sh $master_ip $migration_type
```

#### Figure 11 (b)

For each migration_type(RR, Proactive, dLoRA), start server:
```bash
bash ./start_server.sh $machines 1 8 32 $master_ip 7 3 $migration_type
```
where $migration_type is in {1, 2, 3}.

Wait for server ready, then run:
```bash
bash ./fig11b.sh $master_ip $migration_type
```

#### Visualization

To plot figure `figure11a.pdf` and `figure11b.pdf`, run `figure11.ipynb`.


### Figure 12 (Section 7.5)

Notice: experiments in this section would take several hours.

For each execution type(vLLM, Peft, dLoRA), start server:
```bash
bash ./start_server.sh $machines 1 8 $num_models $master_ip 7 $exec_type 3
```
where $num_models is in {16, 32, 64, 128} and $exec_type is in {1, 2, 3}.

Wait for server ready, then run:
```bash
bash ./fig12.sh $master_ip $exec_type $num_models
```

To plot figure `figure12.pdf`, run `figure12.ipynb`.

### Figure 13 (Section 7.5)

Data is generated from experiment for Figure 11 (a). Please run that experiment first. To plot figure `figure13.pdf`, run `figure13.ipynb`.