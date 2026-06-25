import torch
import numpy as np
import soundfile as sf
import json
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor
from huggingface_hub import hf_hub_download, HfApi


FLEURS_LANGUAGES = {"en": "en_us", "hi": "hi_in"}


def download_fleurs_split(language: str, split: str, cache_dir: str = "data") -> list[dict]:
    config = FLEURS_LANGUAGES[language]
    cache_path = Path(cache_dir) / f"fleurs_{language}_{split}.jsonl"

    if cache_path.exists():
        print(f"Loading cached FLEURS {language}/{split} metadata")
        with open(cache_path) as f:
            return [json.loads(line) for line in f]

    print(f"Downloading FLEURS {language}/{split} manifest...")
    manifest_path = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{config}/audio/{split}/audiofolder/manifest.jsonl",
        repo_type="dataset",
    )

    with open(manifest_path) as f:
        records = [json.loads(line) for line in f]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for rec in records:
            f.write(json.dumps({"path": rec["path"], "transcription": rec["transcription"]}) + "\n")

    return [{"path": r["path"], "transcription": r["transcription"]} for r in records]


class FleursDataset(Dataset):
    def __init__(
        self,
        language: str,
        split: str = "train",
        max_audio_length: float = 10.0,
        max_label_length: int = 256,
        max_samples: int = None,
        cache_dir: str = "data",
    ):
        self.language = language
        self.max_audio_length = max_audio_length
        self.max_label_length = max_label_length
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.cache_dir = cache_dir

        config = FLEURS_LANGUAGES[language]
        records = download_fleurs_split(language, split, cache_dir)

        if max_samples:
            records = records[:max_samples]

        self.records = records
        self.audio_dir = Path(cache_dir) / "fleurs_audio" / language / split
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self._pre_download_audio()

    def _pre_download_audio(self):
        api = HfApi()
        config = FLEURS_LANGUAGES[language]

        for i, rec in enumerate(self.records):
            filename = rec["path"]
            local_path = self.audio_dir / filename

            if not local_path.exists():
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    hf_hub_download(
                        repo_id="google/fleurs",
                        filename=f"data/{config}/audio/{split}/audiofolder/{filename}",
                        repo_type="dataset",
                        local_dir=self.audio_dir,
                    )
                except Exception as e:
                    print(f"Warning: failed to download {filename}: {e}")
                    local_path = None

            self.records[i]["_local_path"] = str(local_path) if local_path and local_path.exists() else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["transcription"]

        if self.language == "en":
            text = text.lower().strip()

        local_path = rec.get("_local_path")
        if local_path and Path(local_path).exists():
            audio, sr = sf.read(local_path)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            audio = audio.astype(np.float32)
        else:
            audio = np.zeros(1600, dtype=np.float32)

        max_samples = int(self.max_audio_length * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return {
            "audio": audio,
            "text": text,
        }


class PolyWhisperCollator:
    def __init__(self, processor: WhisperProcessor, max_label_length: int = 256):
        self.processor = processor
        self.max_label_length = max_label_length

    def __call__(self, batch):
        audios = [item["audio"] for item in batch]
        texts = [item["text"] for item in batch]

        audio_inputs = self.processor.feature_extractor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(30.0 * 16000),
        )

        labels = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_label_length,
            truncation=True,
        )

        labels["input_ids"] = labels["input_ids"].masked_fill(
            labels["attention_mask"] == 0, -100
        )

        return {
            "input_features": audio_inputs["input_features"],
            "labels": labels["input_ids"],
            "label_attention_mask": labels["attention_mask"],
        }


def get_dataloader(
    language: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 0,
    max_audio_length: float = 30.0,
    max_label_length: int = 256,
    max_samples: int = None,
):
    dataset = FleursDataset(
        language=language,
        split=split,
        max_audio_length=max_audio_length,
        max_label_length=max_label_length,
        max_samples=max_samples,
    )
    collator = PolyWhisperCollator(
        processor=WhisperProcessor.from_pretrained("openai/whisper-base"),
        max_label_length=max_label_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
