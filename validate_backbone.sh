#!/bin/bash
# Controlled backbone validation: whisper-base+LoRA vs whisper-small+LoRA
# Same data (3000 IndicVoices hi), same recipe (3 epochs, rank16, encoder-lora),
# eval on FLEURS hi test. Decides the product backbone.
set -x
export PYTHONPATH="$PWD"
TS=$(date +%s)

# 1) base+LoRA
.venv/bin/python3 train_v3.py --langs hi --model-size base --max-samples 3000 \
    --epochs 3 --batch-size 4 --encoder-lora --tag _val_base > polywhisper_output/val_base_train.log 2>&1
.venv/bin/python3 eval_lang_pure.py --lang hi --model-size base \
    --adapter hi_best_val_base.pt \
    --test-json polywhisper_output/data/fleurs_hi_in_test.json \
    --out polywhisper_output/eval_hi_val_base.json \
    --max-new-tokens 256 --encoder-lora >> polywhisper_output/val_base_train.log 2>&1

# 2) small+LoRA
.venv/bin/python3 train_v3.py --langs hi --model-size small --max-samples 3000 \
    --epochs 3 --batch-size 4 --encoder-lora --tag _val_small > polywhisper_output/val_small_train.log 2>&1
.venv/bin/python3 eval_lang_pure.py --lang hi --model-size small \
    --adapter hi_best_val_small.pt \
    --test-json polywhisper_output/data/fleurs_hi_in_test.json \
    --out polywhisper_output/eval_hi_val_small.json \
    --max-new-tokens 256 --encoder-lora >> polywhisper_output/val_small_train.log 2>&1

echo "VALIDATION DONE $(date)" > polywhisper_output/validation_done.flag