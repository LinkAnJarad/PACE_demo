"""Cache text embeddings and Acoustic Likelihood outputs for IEMOCAP utterances.

This pre-processing step avoids recomputing expensive features every epoch.
Produces two files:
  - text_embeddings.npz: {utterance_id: embedding_vector} for all transcriptions
  - sensory_outputs.npz (legacy filename): {utterance_id: [V_mean, A_mean, V_sigma, A_sigma]}

Run once before training the Expectation Model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_processing.config import ExperimentConfig
from predictive_processing.data.iemocap_conversation_dataloader import (
    IEMOCAPConversationLoader,
)
from predictive_processing.models.sensory_model import SensoryModel


# ── Text embedding ─────────────────────────────────────────────────────────


def cache_text_embeddings(
    loader: IEMOCAPConversationLoader,
    config: ExperimentConfig,
    device: str = "cpu",
    batch_size: int = 64,
) -> None:
    """Compute and cache sentence embeddings for all utterance transcriptions."""
    output_path = config.cached_embeddings_dir / "text_embeddings.npz"
    if output_path.exists():
        print(f"Text embeddings already cached at {output_path}")
        return

    print(f"Loading text encoder: {config.text_encoder_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
    model = AutoModel.from_pretrained(config.text_encoder_model).to(device)
    model.eval()

    # Collect all transcripts
    utt_ids = []
    texts = []
    for utt in loader.utterances:
        if utt.transcript:
            utt_ids.append(utt.utterance_id)
            texts.append(utt.transcript)
        else:
            # Use empty string for missing transcripts (will get a zero-ish embedding)
            utt_ids.append(utt.utterance_id)
            texts.append("")

    print(f"Encoding {len(texts)} transcriptions...")
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        batch_texts = texts[i : i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # Mean pooling over token embeddings (masked)
        attention_mask = encoded["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        all_embeddings.append(embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # Save as {utterance_id: embedding}
    save_dict = {}
    for uid, emb in zip(utt_ids, all_embeddings):
        save_dict[uid] = emb

    np.savez_compressed(str(output_path), **save_dict)
    print(f"Saved text embeddings ({all_embeddings.shape}) to {output_path}")


# ── Sensory model outputs ─────────────────────────────────────────────────


def _load_audio(wav_path: Path, sampling_rate: int = 16_000) -> np.ndarray:
    """Load and resample a wav file to target sampling rate."""
    waveform, source_sr = sf.read(str(wav_path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if source_sr != sampling_rate:
        waveform = resample_poly(waveform, sampling_rate, source_sr).astype(np.float32)
    return np.asarray(waveform, dtype=np.float32)


def cache_sensory_outputs(
    loader: IEMOCAPConversationLoader,
    config: ExperimentConfig,
    device: str = "cpu",
) -> None:
    """Run the frozen Acoustic Likelihood on all IEMOCAP utterances and cache outputs."""
    output_path = config.cached_embeddings_dir / "sensory_outputs.npz"
    if output_path.exists():
        print(f"Acoustic Likelihood outputs already cached at {output_path}")
        return

    print("Loading Acoustic Likelihood (Wav2Vec2)...")
    sensory = SensoryModel.from_checkpoint(
        base_model_name=config.sensory_base_model,
        checkpoint_path=config.sensory_checkpoint,
        device=device,
    )

    utt_ids = []
    all_outputs = []  # Each: [V_mean, A_mean, V_sigma, A_sigma]

    batch_size = config.sensory_batch_size
    utterances = [u for u in loader.utterances if u.wav_path.exists()]

    print(f"Processing {len(utterances)} utterances through Acoustic Likelihood...")

    for i in tqdm(range(0, len(utterances), batch_size), desc="Sensory inference"):
        batch_utts = utterances[i : i + batch_size]

        # Load and process audio
        audio_arrays = []
        for utt in batch_utts:
            audio = _load_audio(utt.wav_path, config.sampling_rate)
            if config.max_audio_seconds is not None:
                max_samples = int(config.max_audio_seconds * config.sampling_rate)
                audio = audio[:max_samples]
            audio_arrays.append(torch.tensor(audio, dtype=torch.float32))

        # Pad to same length
        input_values = pad_sequence(audio_arrays, batch_first=True, padding_value=0.0)
        attention_mask = torch.zeros(input_values.shape, dtype=torch.long)
        for j, seq in enumerate(audio_arrays):
            attention_mask[j, : len(seq)] = 1

        # Run model
        outputs = sensory.predict(input_values, attention_mask)

        for j, utt in enumerate(batch_utts):
            utt_ids.append(utt.utterance_id)
            all_outputs.append(
                np.array(
                    [
                        float(outputs["valence_mean"][j]),
                        float(outputs["arousal_mean"][j]),
                        float(outputs["valence_sigma"][j]),
                        float(outputs["arousal_sigma"][j]),
                    ],
                    dtype=np.float32,
                )
            )

    save_dict = {}
    for uid, out in zip(utt_ids, all_outputs):
        save_dict[uid] = out

    np.savez_compressed(str(output_path), **save_dict)
    print(f"Saved sensory outputs for {len(utt_ids)} utterances to {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Cache embeddings for the PP experiment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-text", action="store_true", help="Skip text embedding caching")
    parser.add_argument("--skip-sensory", action="store_true", help="Skip sensory model caching")
    args = parser.parse_args()

    config = ExperimentConfig()

    print("Loading IEMOCAP conversational data...")
    loader = IEMOCAPConversationLoader(
        iemocap_root=config.iemocap_root,
        sessions=None,  # Load ALL sessions for caching
        only_improvised=config.only_improvised,
    )
    info = loader.summary()
    print(f"  Loaded {info['num_utterances']} utterances across {info['num_dialogs']} dialogs")

    if not args.skip_text:
        cache_text_embeddings(loader, config, device=args.device)

    if not args.skip_sensory:
        cache_sensory_outputs(loader, config, device=args.device)

    print("\nDone! Cached embeddings are in:", config.cached_embeddings_dir)


if __name__ == "__main__":
    main()
