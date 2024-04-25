#! /bin/bash
srun --unbuffered -p llm_s --gres=gpu:1 -n1 -N1 --quotatype=auto -c 16 python figure5.py