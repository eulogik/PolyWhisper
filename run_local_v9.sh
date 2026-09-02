#!/usr/bin/env bash
# PolyWhisper v9 — Local launcher (MPS, auto-resume, auto-eval)
# Usage:  ./run_local_v9.sh              # foreground
#         nohup ./run_local_v9.sh &      # background
#         tmux new -s pw9 ./run_local_v9.sh  # tmux (recommended)
#
# What it does:
#   1. Trains 5 langs: hi/ta/te without augmentations, bn/mr with augmentations
#   2. Auto-resumes from checkpoint if interrupted
#   3. Auto-runs FLEURS eval + normalize_ortho after training
#   4. Logs everything to logs/

set -euo pipefail

# ─── Config ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
OUTPUT_DIR="${SCRIPT_DIR}/polywhisper_output_v9"
LOG_DIR="${SCRIPT_DIR}/logs"
STATE_FILE="${OUTPUT_DIR}/training_state.json"
EPOCHS=5
LR=1e-4
BATCH_SIZE=4
MAX_RUNTIME_HOURS=12
WER_EVAL_EVERY=500
TAG="_v9"
LANGS="hi,ta,te,bn,mr"
# hi/ta/te: no augmentations (they have enough data; augmentations cause degeneration)
# bn/mr: full SpecAugment + speed perturb (they benefit dramatically)
AUGMENT_LANGS="bn,mr"

# whisper-small snapshot path (local cache)
WHISPER_MODEL="${HOME}/.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d"

# ─── Environment ──────────────────────────────────────────────────────
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Load HF_TOKEN from .env
if [ -f "${SCRIPT_DIR}/.env" ]; then
    HF_TOKEN=$(grep "^HF_TOKEN=" "${SCRIPT_DIR}/.env" | cut -d= -f2-)
    export HF_TOKEN
    echo "[launcher] HF_TOKEN loaded from .env"
else
    echo "[launcher] ERROR: .env not found (need HF_TOKEN for IndicVoices)"
    exit 1
fi

# ─── Pre-flight checks ───────────────────────────────────────────────
echo "========================================"
echo "  PolyWhisper v9 — Local Training"
echo "  $(date)"
echo "========================================"

# Check MPS
"$PYTHON" -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'" || {
    echo "ERROR: MPS not available"; exit 1
}
echo "[check] MPS: OK"

# Check whisper model
[ -d "$WHISPER_MODEL" ] || {
    echo "ERROR: whisper-small not found at $WHISPER_MODEL"; exit 1
}
echo "[check] whisper-small: OK"

# Check disk space (need ~5GB free for checkpoints)
AVAIL_KB=$(df -k / | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1048576))
echo "[check] Disk: ${AVAIL_GB}GB free"
[ "$AVAIL_GB" -ge 3 ] || {
    echo "ERROR: Need at least 3GB free, only ${AVAIL_GB}GB available"; exit 1
}

# Check HF_TOKEN
[ -n "${HF_TOKEN:-}" ] || {
    echo "ERROR: HF_TOKEN not set"; exit 1
}
echo "[check] HF_TOKEN: OK"

echo ""

# ─── Directories ──────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/adapters_v3" "$OUTPUT_DIR/data" "$LOG_DIR"
ADAPTER_DIR="${OUTPUT_DIR}/adapters_v3"
DATA_DIR="${OUTPUT_DIR}/data"

# ─── Resume state ─────────────────────────────────────────────────────
if [ -f "$STATE_FILE" ]; then
    echo "[resume] Found state file: $STATE_FILE"
    EPOCH_DONE=$( "$PYTHON" -c "import json; s=json.load(open('$STATE_FILE')); print(len(s.get('lang_done',{})))" 2>/dev/null || echo "0")
    echo "[resume] Completed lang-epoch pairs: $EPOCH_DONE"
else
    echo "[resume] No state file — starting fresh"
fi

# ─── Training ─────────────────────────────────────────────────────────
echo ""
echo "=== TRAINING ==="
echo "  Langs: $LANGS"
echo "  Augment only: $AUGMENT_LANGS"
echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH_SIZE"
echo "  LR: $LR"
echo "  Output: $OUTPUT_DIR"
echo ""

TRAIN_LOG="${LOG_DIR}/train_v9_$(date +%Y%m%d_%H%M%S).log"

"$PYTHON" "${SCRIPT_DIR}/train_v3.py" \
    --base-model "$WHISPER_MODEL" \
    --langs "$LANGS" \
    --model-size small \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --encoder-lora \
    --save-dir "$OUTPUT_DIR" \
    --tag "$TAG" \
    --max-runtime-hours $MAX_RUNTIME_HOURS \
    --wer-eval-every $WER_EVAL_EVERY \
    --augment-langs "$AUGMENT_LANGS" \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[launcher] Training exited with code $TRAIN_EXIT"
    echo "[launcher] Check log: $TRAIN_LOG"
    echo "[launcher] Can be resumed by re-running this script"
fi

echo ""
echo "=== TRAINING COMPLETE ==="
echo ""

# ─── Eval ─────────────────────────────────────────────────────────────
echo "=== FLEURS EVAL ==="

EVAL_LOG="${LOG_DIR}/eval_v9_$(date +%Y%m%d_%H%M%S).log"

for LANG in hi ta te bn mr; do
    ADAPTER="${ADAPTER_DIR}/${LANG}_best_prod.pt"
    if [ ! -f "$ADAPTER" ]; then
        # Try alternate naming from state
        ADAPTER_ALT=$(ls "${ADAPTER_DIR}/${LANG}_"*"_best"*".pt" 2>/dev/null | head -1)
        if [ -n "$ADAPTER_ALT" ]; then
            ADAPTER="$ADAPTER_ALT"
        else
            echo "  [${LANG}] No adapter found — skipping"
            continue
        fi
    fi

    TEST_JSON="${DATA_DIR}/fleurs_${LANG}_in_test.json"
    if [ ! -f "$TEST_JSON" ]; then
        # Try alternate naming
        TEST_JSON_ALT=$(ls "${DATA_DIR}/fleurs_${LANG}"*"_test.json" 2>/dev/null | head -1)
        if [ -n "$TEST_JSON_ALT" ]; then
            TEST_JSON="$TEST_JSON_ALT"
        else
            echo "  [${LANG}] No test json — skipping"
            continue
        fi
    fi

    OUT="${OUTPUT_DIR}/eval_${LANG}_pure_fleurs.json"

    echo "  [${LANG}] Eval: adapter=$(basename $ADAPTER) test=$(basename $TEST_JSON)"
    "$PYTHON" "${SCRIPT_DIR}/eval_lang_pure.py" \
        --lang "$LANG" \
        --adapter "$ADAPTER" \
        --adapter-dir "$ADAPTER_DIR" \
        --test-json "$TEST_JSON" \
        --out "$OUT" \
        --model-size small \
        --encoder-lora \
        --num-beams 5 \
        2>&1 | tee -a "$EVAL_LOG"

    # normalize_ortho.py compares vanilla vs pure expert (needs both files)
    # For v9 we just report raw WER; normalization is informational only
    # Skip gracefully if normalize_ortho can't find required files
done

echo ""
echo "=== EVAL COMPLETE ==="
echo ""

# ─── Summary ──────────────────────────────────────────────────────────
echo "=== RESULTS SUMMARY ==="
echo ""
printf "%-6s  %8s  %s\n" "LANG" "WER" "ADAPTER"
printf "%-6s  %8s  %s\n" "------" "--------" "-------"
for LANG in hi ta te bn mr; do
    RAW="${OUTPUT_DIR}/eval_${LANG}_pure_fleurs.json"
    ADAPTER="${ADAPTER_DIR}/${LANG}_best_prod.pt"
    RAW_WER=$( "$PYTHON" -c "import json; print(f'{json.load(open(\"$RAW\"))[\"wer\"]:.1f}%')" 2>/dev/null || echo "N/A")
    ADAPTER_EXISTS="no"
    [ -f "$ADAPTER" ] && ADAPTER_EXISTS="yes"
    printf "%-6s  %8s  %s\n" "$LANG" "$RAW_WER" "$ADAPTER_EXISTS"
done
echo ""

# ─── Upload to HF ─────────────────────────────────────────────────────
echo "=== UPLOADING TO HF ==="
"$PYTHON" - <<UPLOAD_EOF
import os, json
from huggingface_hub import HfApi

api = HfApi()
output_dir = "${OUTPUT_DIR}"
tag = "${TAG}"

for lang in ["hi", "ta", "te", "bn", "mr"]:
    # Upload adapter
    adapter = os.path.join(output_dir, "adapters_v3", f"{lang}_best_prod.pt")
    if os.path.exists(adapter):
        api.upload_file(
            path_or_fileobj=adapter,
            path_in_repo=f"polywhisper_output_local/adapters_v3/{lang}_best_prod.pt",
            repo_id="eulogik/polywhisper",
            repo_type="model",
        )
        print(f"  uploaded adapter {lang}")
    
    # Upload eval results
    for suffix in ["", "_norm"]:
        eval_file = os.path.join(output_dir, f"eval_{lang}_pure_fleurs{suffix}.json")
        if os.path.exists(eval_file):
            api.upload_file(
                path_or_fileobj=eval_file,
                path_in_repo=f"polywhisper_output_local/eval_{lang}_pure_fleurs{suffix}_v9.json",
                repo_id="eulogik/polywhisper",
                repo_type="model",
            )
            print(f"  uploaded eval {lang}{suffix}")

print("Upload complete")
UPLOAD_EOF

echo ""
echo "========================================"
echo "  ALL DONE — $(date)"
echo "========================================"
echo "  Logs: ${LOG_DIR}/"
echo "  Results: ${OUTPUT_DIR}/"
echo "========================================"
