import torch
import argparse
from model import PolyWhisper, AdapterConfig
from train import train_adapter, compute_wer
from data import get_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train PolyWhisper adapter")
    parser.add_argument("--language", type=str, required=True, help="Language code (en, hi)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--save-dir", type=str, default="models/adapters")
    parser.add_argument("--max-audio-length", type=float, default=15.0)
    args = parser.parse_args()

    print(f"Training PolyWhisper adapter for: {args.language}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    model = PolyWhisper(
        model_name="openai/whisper-base",
        languages=[args.language],
        freeze_encoder=True,
    )

    trainable = model.get_trainable_params()
    print(f"\nTrainable parameters:")
    for k, v in trainable.items():
        print(f"  {k}: {v/1e6:.1f}M")
    print(f"  Total trainable: {sum(trainable.values())/1e6:.1f}M")

    print(f"\nLoading data...")
    train_loader = get_dataloader(
        language=args.language,
        split="train",
        batch_size=args.batch_size,
        max_audio_length=args.max_audio_length,
    )
    val_loader = get_dataloader(
        language=args.language,
        split="validation",
        batch_size=args.batch_size,
        max_audio_length=args.max_audio_length,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    history = train_adapter(
        model=model,
        language=args.language,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )

    print(f"\nTraining complete! Best val loss: {min(history['val_loss']):.4f}")
    print(f"Adapter saved to: {args.save_dir}/{args.language}_best.pt")


if __name__ == "__main__":
    main()
