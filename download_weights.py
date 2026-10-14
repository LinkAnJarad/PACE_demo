"""Weights verification & downloader script for the Predictive Processing demo."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

REQUIRED_CHECKPOINTS = {
    "gru_fold_2_best.pt": (1_000_000, "Dialogue-Context GRU Prior (CV5 Fold 2)"),
    "wav2vec2_fold_1_best.pt": (500_000_000, "Acoustic Likelihood Wav2Vec2 (CV5 Fold 1)"),
    "emonet_8.pth": (50_000_000, "EmoNet Pretrained Feature Extractor"),
    "emonet_visual_likelihood_fold_1_best.pt": (50_000_000, "Visual Likelihood Head (CV5 Fold 1)"),
}


def verify_weights() -> bool:
    print("\n=========================================================================")
    print("  Checking Predictive Processing Model Checkpoints")
    print("=========================================================================")
    all_ok = True
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, (min_size, desc) in REQUIRED_CHECKPOINTS.items():
        file_path = CHECKPOINTS_DIR / filename
        if not file_path.is_file():
            print(f"  [MISSING] {filename} - {desc}")
            all_ok = False
        elif file_path.stat().st_size < min_size:
            print(f"  [CORRUPT] {filename} is smaller than expected ({file_path.stat().st_size} bytes)")
            all_ok = False
        else:
            mb_size = file_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {filename} ({mb_size:.1f} MB) - {desc}")

    if all_ok:
        print("  All required checkpoints are present and verified.\n")
    else:
        print("\n  Warning: Some model checkpoints are missing or incomplete.")
        print("  Please ensure checkpoints are placed in the 'checkpoints/' directory.\n")
    return all_ok


if __name__ == "__main__":
    verify_weights()
