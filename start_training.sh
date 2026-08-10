#!/bin/bash
cd /Users/eulogikdeveloper/Documents/PolyWhisper
sleep 3600
nohup /opt/homebrew/bin/python3 train_local.py > polywhisper_output/nohup.log 2>&1 &
echo $! > polywhisper_output/train.pid
echo "Training started at $(date)"
echo "PID: $!"