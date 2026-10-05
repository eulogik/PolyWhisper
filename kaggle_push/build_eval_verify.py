#!/usr/bin/env python3
"""Build the secrets-free eval-verification Kaggle notebook.

Everything comes from public resources (HF repo, HF datasets, GitHub);
results are saved to /kaggle/working and pulled via the Kaggle API.
No HF_TOKEN, no GH_TOKEN, nothing to attach.

Usage: .venv/bin/python3 kaggle_push/build_eval_verify.py
Push:  kaggle kernels push -p kaggle_push/eval_verify
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "eval_verify"
OUT.mkdir(parents=True, exist_ok=True)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


cells = [
    md(["# PolyWhisper eval verification (no secrets needed)\n",
        "\n",
        "All inputs are public: scripts + adapters from `eulogik/polywhisper` on HF, ",
        "FLEURS audio from `google/fleurs`. Results land in `/kaggle/working` ",
        "and are pulled via the Kaggle API. Nothing to attach, just run.\n",
        "\n",
        "Runs: mr prod beam-5 rep1.0; bn/mr clean beam-1 rep1.0 (full); ",
        "te prod beam-1 rep1.0 on the live test split."]),
    code(["!pip install -q transformers datasets accelerate peft torch torchaudio soundfile librosa resampy scipy tqdm huggingface_hub"]),
    code(["import shutil\n",
          "from huggingface_hub import hf_hub_download\n",
          "REPO = 'eulogik/polywhisper'\n",
          "for s in ['eval_lang_pure.py', 'train_v3.py', 'normalize_ortho.py']:\n",
          "    p = hf_hub_download(REPO, s, repo_type='model')\n",
          "    shutil.copy(p, s)\n",
          "    print(' ', s)"]),
    code(["import shutil\n",
          "from huggingface_hub import hf_hub_download\n",
          "REPO = 'eulogik/polywhisper'\n",
          "import os\n",
          "os.makedirs('adapters', exist_ok=True)\n",
          "ADS = ['polywhisper_output_gpu0/adapters_v3/te_best_prod.pt',\n",
          "       'polywhisper_output_gpu0/adapters_v3/bn_best_prod.pt',\n",
          "       'polywhisper_output_gpu1/adapters_v3/mr_best_prod.pt',\n",
          "       'polywhisper_output_gpu0_clean/adapters_v3/bn_best_clean.pt',\n",
          "       'polywhisper_output_gpu1_clean/adapters_v3/mr_best_clean.pt']\n",
          "for a in ADS:\n",
          "    p = hf_hub_download(REPO, a, repo_type='model')\n",
          "    shutil.copy(p, 'adapters/' + p.split('/')[-1])\n",
          "    print(' ', a)"]),
    code(["import json\n",
          "from pathlib import Path\n",
          "import numpy as np\n",
          "import soundfile as sf\n",
          "from scipy import signal\n",
          "from datasets import load_dataset\n",
          "DATA_DIR = Path('polywhisper_output/data')\n",
          "DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
          "def resample(audio, sr):\n",
          "    a = np.asarray(audio, dtype=np.float32)\n",
          "    return a if sr == 16000 else signal.resample(a, int(len(a) * 16000 / sr)).astype(np.float32)\n",
          "for lang in ['te_in', 'bn_in', 'mr_in']:\n",
          "    adir = DATA_DIR / f'audio_fleurs_{lang}'\n",
          "    adir.mkdir(parents=True, exist_ok=True)\n",
          "    recs = []\n",
          "    ds = load_dataset('google/fleurs', lang, split='test', streaming=True)\n",
          "    for item in ds:\n",
          "        a = item['audio']\n",
          "        au = resample(a['array'], a['sampling_rate'])\n",
          "        tx = item['transcription'].strip()\n",
          "        if len(au) < 1600 or len(au) > 30 * 16000 or len(tx) < 3:\n",
          "            continue\n",
          "        wp = adir / f'{len(recs):06d}.wav'\n",
          "        sf.write(str(wp), au, 16000)\n",
          "        recs.append({'wav': str(wp), 'text': tx})\n",
          "    json.dump(recs, open(DATA_DIR / f'fleurs_{lang}_test.json', 'w'))\n",
          "    print(lang, len(recs), 'records')"]),
    code(["!python eval_lang_pure.py --lang mr --model-size small --adapter mr_best_prod.pt --adapter-dir adapters --test-json polywhisper_output/data/fleurs_mr_in_test.json --out /kaggle/working/eval_mr_prod_beam5_rep10.json --max-new-tokens 256 --encoder-lora --num-beams 5 "]),
    code(["!python eval_lang_pure.py --lang bn --model-size small --adapter bn_best_clean.pt --adapter-dir adapters --test-json polywhisper_output/data/fleurs_bn_in_test.json --out /kaggle/working/eval_bn_clean_beam1_rep10.json --max-new-tokens 256 --encoder-lora --num-beams 1 "]),
    code(["!python eval_lang_pure.py --lang mr --model-size small --adapter mr_best_clean.pt --adapter-dir adapters --test-json polywhisper_output/data/fleurs_mr_in_test.json --out /kaggle/working/eval_mr_clean_beam1_rep10.json --max-new-tokens 256 --encoder-lora --num-beams 1 "]),
    code(["!python eval_lang_pure.py --lang te --model-size small --adapter te_best_prod.pt --adapter-dir adapters --test-json polywhisper_output/data/fleurs_te_in_test.json --out /kaggle/working/eval_te_prod_beam1_rep10.json --max-new-tokens 256 --encoder-lora --num-beams 1 "]),
    code(["import json, glob\n",
          "for f in sorted(glob.glob('/kaggle/working/eval_*.json')):\n",
          "    d = json.load(open(f))\n",
          "    print(f.split('/')[-1], 'raw WER=%.1f' % d['wer'], 'n=%d' % len(d['samples']))\n",
          "print('ALL EVALS DONE')"]),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10.0"}},
      "nbformat": 4, "nbformat_minor": 4}

nb_path = OUT / "polywhisper-eval-verify.ipynb"
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

meta = {
    "id": "eulogikdevelopers/polywhisper-eval-verify",
    "title": "Polywhisper Eval Verify",
    "code_file": "polywhisper-eval-verify.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
with open(OUT / "kernel-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

nb2 = json.load(open(nb_path))
assert nb2["nbformat"] == 4 and len(nb2["cells"]) == 10
for i, c in enumerate(nb2["cells"]):
    assert set(c) >= {"cell_type", "metadata", "source"}, i
print(f"built+validated: {len(nb2['cells'])} cells -> {nb_path}")
