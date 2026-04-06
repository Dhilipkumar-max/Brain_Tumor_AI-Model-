"""Model loading utility using the Singleton pattern.

Uses absolute path resolution so it works on Render cloud and locally.
"""

import os
import logging
from typing import Optional
import torch
from monai.networks.nets import UNet
from brain_tumor_ai.config import DEVICE

logger = logging.getLogger(__name__)

# Absolute path — works on Render and local machines
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_model.pth")

# Global singleton cache
_cached_model: Optional[torch.nn.Module] = None


def load_model() -> torch.nn.Module:
    """Loads and returns the 3D UNet model (singleton).

    Tries to load pretrained weights from `models/brain_model.pth`.
    Falls back to random initialisation if weights are absent (demo mode).

    Returns:
        torch.nn.Module: Initialised, eval-mode model on CPU.
    """
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    try:
        device = torch.device("cpu")

        # Architecture must match training config exactly
        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        )

        # Load pretrained weights if available
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=device)
            )
            logger.info(f"Pretrained weights loaded from: {MODEL_PATH}")
        else:
            logger.warning(
                f"No weights at {MODEL_PATH} — running in demo mode "
                "(random initialisation). Predictions will not be clinically accurate.)"
            )

        model.to(device)
        model.eval()
        logger.info(f"Model ready on {device}.")

        _cached_model = model
        return _cached_model

    except Exception as e:
        logger.error(f"Model initialisation failed: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
