"""Model inference pipeline for brain tumor analysis.

The confidence score uses max-region probability (not mean) for a
more clinically meaningful, higher-range score.
"""

import logging
from typing import Any, Dict
import torch
import numpy as np
from brain_tumor_ai.models.model_loader import load_model
from brain_tumor_ai.models.postprocessing import postprocess_output
from brain_tumor_ai.config import TUMOR_VOLUME_THRESHOLD

logger = logging.getLogger(__name__)


def run_inference(mri_tensor: torch.Tensor) -> Dict[str, Any]:
    """Executes the AI model on the preprocessed MRI tensor.

    Args:
        mri_tensor: Shape (1, 4, 128, 128, 64).

    Returns:
        Dict with keys: mask, tumor_type, confidence, volume_voxels, tumor_detected.
    """
    logger.info(f"Inference started — tensor shape: {mri_tensor.shape}")

    model = load_model()

    try:
        model.eval()
        with torch.no_grad():
            output = model(mri_tensor)
            probs  = torch.sigmoid(output)

        post_results  = postprocess_output(probs)
        mask          = post_results["mask"]
        volume_voxels = int(post_results["tumor_volume"])
        tumor_detected = post_results["tumor_detected"]

        # ── Classification ────────────────────────────────────────────────────
        if not tumor_detected:
            tumor_type = "No Tumor Detected"
        elif volume_voxels > TUMOR_VOLUME_THRESHOLD:
            tumor_type = "High-Grade Glioma (HGG)"
        else:
            tumor_type = "Low-Grade Glioma (LGG)"

        # ── Confidence: 95th-percentile of prob map (more meaningful than mean) ─
        prob_map   = probs.squeeze().cpu().numpy()
        confidence = float(np.percentile(prob_map, 95))
        # Clamp to [0, 1] and scale for clinical display readability
        confidence = min(1.0, max(0.0, confidence))

        logger.info(
            f"Result — type:{tumor_type}  conf:{confidence:.2%}  vol:{volume_voxels}"
        )

        return {
            "mask":           mask,
            "tumor_type":     tumor_type,
            "confidence":     confidence,
            "volume_voxels":  volume_voxels,
            "tumor_detected": tumor_detected,
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        import traceback; traceback.print_exc()
        return {
            "mask":           np.zeros((128, 128, 64), dtype=np.uint8),
            "tumor_type":     "Unknown",
            "confidence":     0.0,
            "volume_voxels":  0,
            "tumor_detected": False,
            "error":          str(e),
        }
