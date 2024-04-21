#!/usr/bin/env bash

export TORCH_LOGS="all"
RANK=0 python main.py &
RANK=1 python main.py &
wait
