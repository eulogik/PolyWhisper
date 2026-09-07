#!/usr/bin/env python3
"""Build the v9 Kaggle notebook + kernel-metadata (durable, in-repo).

Regenerates kaggle_push/v9e/polywhisper-v9-per-lang-aug.ipynb deterministically
via json (no hand-written notebook JSON -> no control-char bugs).
v4 changes vs v3: fail-fast dep assert (resampy!), HF heartbeat phases.
Usage: .venv/bin/python3 kaggle_push/build_v9e.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "v9e"
OUT.mkdir(parents=True, exist_ok=True)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


BEAT = [
    "import os, datetime\n",
    "def beat(phase):\n",
    "    try:\n",
    "        tok = os.environ.get('HF_TOKEN', '')\n",
    "        if not tok:\n",
    "            print(f'[beat] {phase} (no token, local only)', flush=True)\n",
    "            return\n",
    "        from huggingface_hub import HfApi\n",
    "        msg = f\"{phase} @ {datetime.datetime.now(datetime.timezone.utc).isoformat()}\"\n",
    "        open('/tmp/phase.txt', 'w').write(msg + '\\n')\n",
    "        HfApi(token=tok).upload_file('/tmp/phase.txt', 'polywhisper_v9_kernel_progress.txt',\n",
    "            repo_id='eulogik/polywhisper', repo_type='model',\n",
    "            commit_message=f'v9 progress: {phase}')\n",
    "        print('[beat]', msg, flush=True)\n",
    "    except Exception as e:\n",
    "        print('[beat] failed:', str(e)[:150], flush=True)\n",
]

cells = [
    md(["# PolyWhisper v9 — Per-Language Augmentation (eulogikdevelopers)\n",
        "\n",
        "hi/ta/te: NO augmentations (enough data; SpecAugment caused token-loop degeneration)\n",
        "bn/mr: full SpecAugment + speed perturb (bn -14%, mr -51% in v8)\n",
        "\n",
        "Account-independent: backbone `openai/whisper-small` downloads from HF at runtime "
        "(pinned revision). No /kaggle/input dataset dependency. 2-GPU orchestrator when "
        "2 GPUs present, sequential fallback on 1 GPU."]),
    code(["!pip install -q transformers datasets accelerate peft torch torchaudio evaluate "
          "jiwer sentencepiece huggingface_hub soundfile librosa resampy"]),
    code(["import resampy, librosa, soundfile, torch, transformers, datasets, peft, accelerate\n",
          "print('deps OK: resampy', resampy.__version__,\n",
          "      '| torch', torch.__version__,\n",
          "      '| transformers', transformers.__version__)\n",
          ] + BEAT + ["beat('v4-pip-done')\n"]),
    code(["import os, shutil\n",
          "for d in ['scripts', 'polywhisper_output_gpu0', 'polywhisper_output_gpu1']:\n",
          "    if os.path.exists(d): shutil.rmtree(d)\n",
          "os.makedirs('scripts', exist_ok=True)\n",
          "print('cleaned')"]),
    code(["import shutil\n",
          "from huggingface_hub import hf_hub_download\n",
          "for s in ['train_v3.py', 'eval_lang_pure.py', 'kaggle_train_resumable.py', "
          "'normalize_ortho.py']:\n",
          "    p = hf_hub_download('eulogik/polywhisper', s, repo_type='model')\n",
          "    shutil.copy(p, f'scripts/{s}')\n",
          "    print(' ', s)\n",
          "t = open('scripts/train_v3.py').read()\n",
          "assert 'AUGMENT_LANGS' in t, 'STALE train_v3.py on HF — aborting'\n",
          "o = open('scripts/kaggle_train_resumable.py').read()\n",
          "assert '\"--augment-langs\"' in o and '\"bn,mr\"' in o, "
          "'STALE orchestrator on HF — aborting'\n",
          "print('version check OK: per-language augmentation present')\n",
          "beat('v4-scripts-ok')\n"]),
    code(["import os\n",
          "try:\n",
          "    from kaggle_secrets import UserSecretsClient\n",
          "    tok = UserSecretsClient().get_secret('HF_TOKEN')\n",
          "    if tok:\n",
          "        os.environ['HF_TOKEN'] = tok\n",
          "        print('HF_TOKEN secret attached: yes (checkpoints will persist to HF)')\n",
          "    else:\n",
          "        print('HF_TOKEN secret empty: HF upload skipped, "
          "pull outputs via kernels output API')\n",
          "except Exception as e:\n",
          "    print('HF_TOKEN secret not attached — HF upload skipped, "
          "pull outputs via kernels output API:', str(e)[:120])\n",
          "import torch\n",
          "n = torch.cuda.device_count()\n",
          "print('CUDA GPUs:', n, [torch.cuda.get_device_name(i) for i in range(n)] if n else [])\n"]),
    code(["import os, shutil, subprocess, sys\n",
          "from pathlib import Path\n",
          "os.chdir('scripts')\n",
          "import torch\n",
          "n = torch.cuda.device_count()\n",
          "print('GPUs:', n, flush=True)\n",
          "beat(f'v4-training-start gpus={n}')\n",
          "if n >= 2:\n",
          "    print('2+ GPUs: launching 2-GPU orchestrator (train + eval + HF upload)', flush=True)\n",
          "    r = subprocess.run([sys.executable, 'kaggle_train_resumable.py'])\n",
          "    print('orchestrator rc=', r.returncode, flush=True)\n",
          "else:\n",
          "    print('single GPU: sequential fallback (all 5 langs, per-lang augment inside train_v3)',\n",
          "          flush=True)\n",
          "    sys.path.insert(0, '.')\n",
          "    import kaggle_train_resumable as K\n",
          "    SD = 'polywhisper_output_gpu0'\n",
          "    if os.environ.get('HF_TOKEN'):\n",
          "        K.api.token = os.environ['HF_TOKEN']\n",
          "        K.fetch_remote(SD)\n",
          "    r = subprocess.run([sys.executable, 'train_v3.py', '--langs', 'hi,ta,te,bn,mr',\n",
          "        '--model-size', 'small', '--epochs', '5', '--batch-size', '4',\n",
          "        '--encoder-lora', '--tag', '_prod', '--save-dir', SD,\n",
          "        '--max-runtime-hours', '11', '--wer-eval-every', '500',\n",
          "        '--augment-langs', 'bn,mr'])\n",
          "    print('train rc=', r.returncode, flush=True)\n",
          "    for p in Path(SD, 'data').glob('audio_indicvoices_*'):\n",
          "        shutil.rmtree(p, ignore_errors=True)\n",
          "    print('freed re-downloadable indicvoices FLAC (kept fleurs test audio for eval)',\n",
          "          flush=True)\n",
          "    for lang in ['hi', 'ta', 'te', 'bn', 'mr']:\n",
          "        try:\n",
          "            K.eval_lang(lang, SD)\n",
          "        except Exception as e:\n",
          "            print(f'eval {lang} FAILED: {e}', flush=True)\n",
          "    try:\n",
          "        K.upload_tree(SD, K.remote_snapshot(), force=True)\n",
          "    except Exception as e:\n",
          "        print('final upload failed:', str(e)[:200], flush=True)\n",
          "print('MAIN DONE', flush=True)\n",
          "beat('v4-main-done')\n"]),
    code(["from pathlib import Path\n",
          "print('cwd check + output manifest:')\n",
          "for d in ['polywhisper_output_gpu0', 'polywhisper_output_gpu1']:\n",
          "    p = Path(d)\n",
          "    if not p.exists():\n",
          "        print(f'  {d}: MISSING')\n",
          "        continue\n",
          "    ad = p / 'adapters_v3'\n",
          "    pts = sorted(x.name for x in ad.glob('*.pt')) if ad.exists() else []\n",
          "    evs = sorted(x.name for x in p.glob('eval_*.json'))\n",
          "    print(f'  {d}: adapters={pts}')\n",
          "    print(f'  {d}: evals={evs}')\n"]),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10.0"}},
      "nbformat": 4, "nbformat_minor": 4}

nb_path = OUT / "polywhisper-v9-per-lang-aug.ipynb"
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

meta = {
    "id": "eulogikdevelopers/polywhisper-v9-per-lang-aug",
    "title": "Polywhisper V9 Per Lang Aug",
    "code_file": "polywhisper-v9-per-lang-aug.ipynb",
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

# validate
nb2 = json.load(open(nb_path))
assert nb2["nbformat"] == 4 and len(nb2["cells"]) == 8
for i, c in enumerate(nb2["cells"]):
    assert set(c) >= {"cell_type", "metadata", "source"}, i
    if c["cell_type"] == "code":
        body = "\n".join(l for l in c["source"]
                         if not l.lstrip().startswith(("!", "%")))
        compile(body, f"<cell{i}>", "exec")
assert "resampy" in "".join(nb2["cells"][1]["source"])
assert any("beat(" in "".join(c["source"]) for c in nb2["cells"]
           if c["cell_type"] == "code")
print(f"v4 built+validated: {len(nb2['cells'])} cells -> {nb_path}")
