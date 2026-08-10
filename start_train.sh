#!/bin/bash
cd /Users/eulogikdeveloper/Documents/PolyWhisper
sleep 3600
nohup .venv/bin/python3 train_local.py > polywhisper_output/nohup.log 2>&1 &
echo 0 > polywhisper_output/train.pid
echo "Training started at Wed Jul 22 15:47:03 IST 2026"
echo "PID: 0"
