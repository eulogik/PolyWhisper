import torch
from model import PolyWhisper, AdapterConfig


def test_model():
    print("=" * 60)
    print("PolyWhisper Model Smoke Test")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\n1. Loading model...")
    model = PolyWhisper(
        model_name="openai/whisper-base",
        languages=["en", "hi"],
        freeze_encoder=True,
    )
    model = model.to(device)
    model.eval()
    print("   Model loaded successfully!")

    trainable = model.get_trainable_params()
    print(f"\n2. Parameter counts:")
    for k, v in trainable.items():
        print(f"   {k}: {v/1e6:.1f}M params")
    total = sum(trainable.values())
    print(f"   Total trainable: {total/1e6:.1f}M")

    print("\n3. Testing forward pass (English)...")
    batch_size = 2
    decoder_len = 20

    dummy_audio = torch.randn(batch_size, 80, 3000).to(device)
    dummy_decoder = torch.randint(0, 1000, (batch_size, decoder_len)).to(device)

    with torch.no_grad():
        logits = model(
            audio_features=dummy_audio,
            decoder_input_ids=dummy_decoder,
            language="en",
        )
    print(f"   Output shape: {logits.shape}")
    assert logits.shape == (batch_size, decoder_len, 1024), f"Unexpected shape: {logits.shape}"
    print("   Forward pass OK!")

    print("\n4. Testing forward pass (Hindi)...")
    with torch.no_grad():
        logits_hi = model(
            audio_features=dummy_audio,
            decoder_input_ids=dummy_decoder,
            language="hi",
        )
    print(f"   Output shape: {logits_hi.shape}")
    assert logits_hi.shape == (batch_size, decoder_len, 1024)
    print("   Forward pass OK!")

    print("\n5. Testing language identification...")
    with torch.no_grad():
        lang_out = model.detect_language(dummy_audio[0:1])
    print(f"   Detected: {lang_out['language']} (confidence: {lang_out['confidence']:.3f})")
    print(f"   All probs: {lang_out['all_probs']}")
    print("   Language ID OK!")

    print("\n6. Testing adapter save/load...")
    save_path = "/tmp/test_adapter.pt"
    model.save_adapters(save_path)

    model2 = PolyWhisper(
        model_name="openai/whisper-base",
        languages=["en", "hi"],
        freeze_encoder=True,
    )
    model2.load_adapters(save_path, device=device)
    model2 = model2.to(device)
    print("   Save/Load OK!")

    print("\n7. Verifying adapter weights are independent...")
    en_params_1 = dict(model.adapters["en"].named_parameters())
    en_params_2 = dict(model2.adapters["en"].named_parameters())
    for name in en_params_1:
        assert torch.equal(en_params_1[name].cpu(), en_params_2[name].cpu()), f"Mismatch in {name}"
    print("   Weights match after load!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\nModel is ready for training on {device}")
    print(f"Trainable params: {total/1e6:.1f}M (adapters + language ID)")


if __name__ == "__main__":
    test_model()
