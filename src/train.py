import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import time
from pathlib import Path

from model import PolyWhisper, AdapterConfig
from data import get_dataloader


def train_adapter(
    model: PolyWhisper,
    language: str,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    save_dir: str = "models/adapters",
    device: str = "mps",
    log_interval: int = 50,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.train()

    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [{language}]")

        for batch in progress:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            decoder_input_ids = labels[:, :-1]
            decoder_target_ids = labels[:, 1:]

            logits = model(
                audio_features=input_features,
                decoder_input_ids=decoder_input_ids,
                language=language,
            )

            loss = criterion(logits.view(-1, logits.size(-1)), decoder_target_ids.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if global_step >= warmup_steps:
                scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                current_lr = optimizer.param_groups[0]["lr"]
                progress.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})

        avg_train_loss = epoch_loss / num_batches
        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = save_dir / f"{language}_best.pt"
            model.save_adapters(str(save_path))
            print(f"  Saved best model to {save_path}")

    history_path = save_dir / f"{language}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


@torch.no_grad()
def evaluate(model: PolyWhisper, dataloader, criterion, device: str):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        decoder_input_ids = labels[:, :-1]
        decoder_target_ids = labels[:, 1:]

        logits = model(
            audio_features=input_features,
            decoder_input_ids=decoder_input_ids,
            language=model.languages[0],
        )

        loss = criterion(logits.view(-1, logits.size(-1)), decoder_target_ids.reshape(-1))
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def compute_wer(predictions: list[str], references: list[str]) -> float:
    import re
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    total_errors = 0
    total_words = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)

        dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
        for i in range(len(pred_tokens) + 1):
            dp[i][0] = i
        for j in range(len(ref_tokens) + 1):
            dp[0][j] = j

        for i in range(1, len(pred_tokens) + 1):
            for j in range(1, len(ref_tokens) + 1):
                if pred_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        total_errors += dp[len(pred_tokens)][len(ref_tokens)]
        total_words += len(ref_tokens)

    return total_errors / max(total_words, 1) * 100
