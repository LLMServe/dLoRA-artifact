#!/bin/bash

num_svr=$1

salloc -p llm_s --gres=gpu:8 -c 128 -N${num_svr} -n${num_svr}