cd /Users/eulogikdeveloper/Documents/PolyWhisper
sleep 5040
nohup .venv/bin/python train_local.py > polywhisper_output/nohup.log 2>&1 &
echo $! > polywhisper_output/train.pid
echo "Training started at $(date)"
echo "PID: $!"
