#!/bin/bash
PY=".venv/bin/python3"
S=polywhisper_output
echo "[$(date)] START 256-token re-evals → standard filenames"

for lang_pair in "ta ta_best_ta.pt" "te te_best_te_v2.pt" "bn bn_best_bn_v2.pt" "mr mr_best_mr.pt"; do
  lang=$(echo $lang_pair | cut -d' ' -f1)
  adapter=$(echo $lang_pair | cut -d' ' -f2)
  echo "[$(date)] === $lang vanilla (256 tok) ==="
  $PY eval_vanilla.py --language $lang --test-json $S/data/fleurs_${lang}_in_test.json --out $S/eval_vanilla_fleurs_${lang}.json --max-new-tokens 256
  echo "[$(date)] === $lang pure (256 tok) ==="
  $PY eval_lang_pure.py --lang $lang --adapter $adapter --test-json $S/data/fleurs_${lang}_in_test.json --out $S/eval_${lang}_pure_fleurs.json --max-new-tokens 256
  echo "[$(date)] === $lang DONE ==="
done

echo "[$(date)] ALL 8 EVALS DONE"
echo "[$(date)] Running normalize_ortho.py..."
$PY normalize_ortho.py
echo "[$(date)] COMPLETE"
