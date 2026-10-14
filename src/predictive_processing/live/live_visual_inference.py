"""Live Visual Likelihood Inference using EmoNet.

Decodes video frames from browser MediaRecorder WebM bytes, extracts and normalizes
facial crops, and runs the EmoNetVisualLikelihood PyTorch model to return
calibrated univariate Gaussian distributions for valence and arousal.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "emonet") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "emonet"))

try:
    from emonet.models import EmoNet
except ImportError:
    EmoNet = None

DEFAULT_EMONET_PRETRAINED = PROJECT_ROOT / "checkpoints" / "emonet_8.pth"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "emonet_visual_likelihood_fold_1_best.pt"

_MODEL_CACHE: dict[str, Any] = {}


class EmoNetVisualLikelihood(nn.Module):
    """Visual Likelihood model wrapping EmoNet with temporal pooling and dual uncertainty heads."""

    def __init__(
        self,
        emonet_pretrained_path: str | Path = DEFAULT_EMONET_PRETRAINED,
        hidden_dim: int = 128,
        freeze_fan: bool = True,
        freeze_convs: bool = False,
    ):
        super().__init__()
        if EmoNet is None:
            raise ImportError("emonet.models.EmoNet could not be imported.")
        self.emonet = EmoNet(n_expression=8)
        if Path(emonet_pretrained_path).exists():
            state_dict = torch.load(str(emonet_pretrained_path), map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.emonet.load_state_dict(state_dict, strict=False)

        feature_dim = 256
        self.mean_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # [Valence, Arousal]
        )

        self.scale_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # [sigma_V, sigma_A]
            nn.Softplus(),
        )
        self.tanh = nn.Tanh()

    def extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim feature vectors from a batch of frames [N, 3, 256, 256]."""
        x = F.relu(self.emonet.bn1(self.emonet.conv1(x)), True)
        x = F.max_pool2d(self.emonet.conv2(x), 2, stride=2)
        x = self.emonet.conv3(x)
        x = self.emonet.conv4(x)

        previous = x
        hg_features = []
        for i in range(self.emonet.num_modules):
            hg = self.emonet._modules["m" + str(i)](previous)
            ll = hg
            ll = self.emonet._modules["top_m_" + str(i)](ll)
            ll = F.relu(self.emonet._modules["bn_end" + str(i)](self.emonet._modules["conv_last" + str(i)](ll)), True)
            tmp_out = self.emonet._modules["l" + str(i)](ll)
            if i < self.emonet.num_modules - 1:
                ll = self.emonet._modules["bl" + str(i)](ll)
                tmp_out_ = self.emonet._modules["al" + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
            hg_features.append(ll)

        hg_features_cat = torch.cat(tuple(hg_features), dim=1)
        if getattr(self.emonet, "attention", False):
            mask = torch.sum(tmp_out, dim=1, keepdim=True)
            hg_features_cat *= mask
            emo_feat = torch.cat((x, hg_features_cat), dim=1)
        else:
            emo_feat = torch.cat([x, hg_features_cat, tmp_out], dim=1)

        emo_feat_conv1D = self.emonet.conv1x1_input_emo_2(emo_feat)
        final_features = self.emonet.emo_net_2(emo_feat_conv1D)
        final_features = self.emonet.avg_pool_2(final_features)
        batch_size = final_features.shape[0]
        return final_features.view(batch_size, 256)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expects frames shape [B, K, 3, 256, 256]."""
        B, K, C, H, W = frames.shape
        flat_frames = frames.view(B * K, C, H, W)
        frame_feats = self.extract_frame_features(flat_frames)  # [B * K, 256]
        turn_feats = frame_feats.view(B, K, 256).mean(dim=1)    # [B, 256]

        means = self.mean_head(turn_feats)
        scales = self.scale_head(turn_feats) + 1e-4

        return {
            "valence_mean": self.tanh(means[:, 0]),
            "arousal_mean": self.tanh(means[:, 1]),
            "valence_sigma": scales[:, 0],
            "arousal_sigma": scales[:, 1],
        }


def get_or_load_visual_model(
    checkpoint_path: str | Path | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> EmoNetVisualLikelihood | None:
    """Load and cache the fine-tuned EmoNetVisualLikelihood model."""
    ckpt = str(checkpoint_path or DEFAULT_CHECKPOINT)
    if ckpt in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt]

    if not Path(ckpt).exists() and not DEFAULT_EMONET_PRETRAINED.exists():
        print(f"Warning: Neither {ckpt} nor {DEFAULT_EMONET_PRETRAINED} was found.")
        return None

    try:
        model = EmoNetVisualLikelihood(emonet_pretrained_path=DEFAULT_EMONET_PRETRAINED)
        if Path(ckpt).exists():
            state = torch.load(ckpt, map_location="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        _MODEL_CACHE[ckpt] = model
        return model
    except Exception as exc:
        print(f"Failed to load EmoNetVisualLikelihood: {exc}")
        return None


def extract_frames_from_video_bytes(
    video_bytes: bytes,
    num_samples: int = 8,
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Extract and sample N face-centered frames from raw video bytes."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames: list[np.ndarray] = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 30  # Fallback estimate

        sample_indices = set(np.linspace(0, max(0, total_frames - 1), num_samples, dtype=int))
        frame_idx = 0

        while cap.isOpened() and len(frames) < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in sample_indices or total_frames < num_samples:
                # Convert BGR to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Center square crop
                h, w, _ = rgb.shape
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                cropped = rgb[top : top + min_dim, left : left + min_dim]
                resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
                # Normalize to [0, 1]
                norm_frame = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)  # [3, 256, 256]
                frames.append(norm_frame)
            frame_idx += 1
        cap.release()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not frames:
        # Fallback dummy neutral frame
        dummy = np.zeros((3, target_size[0], target_size[1]), dtype=np.float32)
        frames = [dummy] * num_samples

    while len(frames) < num_samples:
        frames.append(frames[-1])

    return np.stack(frames[:num_samples], axis=0)  # [K, 3, 256, 256]


def predict_visual_likelihood(
    video_bytes: bytes,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fallback_mu: float = 0.0,
    fallback_sigma: float = 0.40,
) -> dict[str, float]:
    """Run EmoNetVisualLikelihood inference on video bytes and return Gaussian params.

    Returns:
        Dict with keys:
            valence_mu, valence_sigma, arousal_mu, arousal_sigma
    """
    model = get_or_load_visual_model(checkpoint_path, device=device)
    if model is None:
        return {
            "valence_mu": fallback_mu,
            "valence_sigma": fallback_sigma,
            "arousal_mu": fallback_mu,
            "arousal_sigma": fallback_sigma,
        }

    try:
        frames_np = extract_frames_from_video_bytes(video_bytes, num_samples=8)
        # [1, K, 3, 256, 256]
        frames_tensor = torch.tensor(frames_np, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frames_tensor)
            val_mu = float(outputs["valence_mean"][0].cpu())
            val_sigma = float(outputs["valence_sigma"][0].cpu())
            aro_mu = float(outputs["arousal_mean"][0].cpu())
            aro_sigma = float(outputs["arousal_sigma"][0].cpu())

        return {
            "valence_mu": float(np.clip(val_mu, -1.0, 1.0)),
            "valence_sigma": float(max(val_sigma, 0.05)),
            "arousal_mu": float(np.clip(aro_mu, -1.0, 1.0)),
            "arousal_sigma": float(max(aro_sigma, 0.05)),
        }
    except Exception as exc:
        print(f"Visual Likelihood inference error: {exc}")
        return {
            "valence_mu": fallback_mu,
            "valence_sigma": fallback_sigma,
            "arousal_mu": fallback_mu,
            "arousal_sigma": fallback_sigma,
        }
