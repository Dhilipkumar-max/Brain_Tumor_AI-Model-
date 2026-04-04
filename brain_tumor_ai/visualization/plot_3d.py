"""Stable 3D Interactive visualization for brain MRI using Scatter3d point cloud.

Uses go.Scatter3d for reliable cross-environment rendering.
Brain is shown as a semi-transparent point cloud; tumor as a dense red cluster.
"""

import logging
from typing import Optional
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def generate_3d_plot(mri_data: np.ndarray, mask: np.ndarray, theme: str = "grayscale") -> go.Figure:
    """Generates a stable, interactive 3D brain + tumor point-cloud visualization.

    Handles resolution mismatches between high-res MRI and AI-processed masks.

    Args:
        mri_data (np.ndarray): Multi-modal MRI scan of shape (4, H, W, D).
        mask (np.ndarray): Binary segmentation mask — any shape (H_m, W_m, D_m).
        theme (str): Visual color theme ('grayscale' or 'thermal').

    Returns:
        go.Figure: A Plotly Figure with brain point cloud and tumor overlay.
    """
    logger.info(f"Generating stable 3D point cloud (Theme: {theme})")

    try:
        # ── Step 1: Input Validation ──────────────────────────────────────────
        if mri_data is None:
            raise ValueError("MRI data is None")
        if mask is None:
            raise ValueError("Mask is None")

        # Use FLAIR modality (index 0) as anatomical background
        brain = mri_data[0].astype("float32")

        # ── Step 2: Normalize brain intensity to [0, 1] ───────────────────────
        brain = (brain - brain.min()) / (brain.max() - brain.min() + 1e-8)

        # ── Step 3: Downsample for performance (3x stride = 27x fewer voxels) ─
        brain_ds = brain[::3, ::3, ::3]

        # ── Step 4: Align mask resolution to downsampled brain ────────────────
        # The AI mask may be (128,128,64) while MRI is (240,240,155).
        # We zoom the mask to match brain_ds using nearest-neighbour (order=0).
        target_shape = brain_ds.shape
        mask_float = mask.astype("float32")

        if mask_float.shape != target_shape:
            zoom_factors = (
                target_shape[0] / mask_float.shape[0],
                target_shape[1] / mask_float.shape[1],
                target_shape[2] / mask_float.shape[2],
            )
            mask_ds = zoom(mask_float, zoom_factors, order=0)
            logger.info(
                f"Mask resampled from {mask_float.shape} → {mask_ds.shape} "
                f"(brain_ds={target_shape})"
            )
        else:
            mask_ds = mask_float

        # Safety: shapes must match now
        if brain_ds.shape != mask_ds.shape:
            raise ValueError(
                f"Shape mismatch after resampling: brain={brain_ds.shape} mask={mask_ds.shape}"
            )

        # ── Step 5: Build coordinate grid ─────────────────────────────────────
        h, w, d = brain_ds.shape
        grid_x, grid_y, grid_z = np.mgrid[0:h, 0:w, 0:d]

        # ── Step 6: Flatten everything ────────────────────────────────────────
        x = grid_x.flatten()
        y = grid_y.flatten()
        z = grid_z.flatten()
        brain_values = brain_ds.flatten()
        mask_values  = mask_ds.flatten()

        print(
            f"[3D] Shapes — x:{x.shape} y:{y.shape} z:{z.shape} "
            f"brain:{brain_values.shape} mask:{mask_values.shape}"
        )

        # ── Step 7: Filter low-intensity voxels (removes background air) ──────
        threshold  = 0.1
        brain_sel  = brain_values > threshold

        x_brain     = x[brain_sel]
        y_brain     = y[brain_sel]
        z_brain     = z[brain_sel]
        brain_fil   = brain_values[brain_sel]

        # ── Step 8: Filter tumor voxels ───────────────────────────────────────
        tumor_sel  = mask_values > 0
        x_tumor    = x[tumor_sel]
        y_tumor    = y[tumor_sel]
        z_tumor    = z[tumor_sel]

        print(
            f"[3D] Points — brain:{len(x_brain):,}  tumor:{len(x_tumor):,}"
        )

        # ── Step 9: Brain Scatter3d trace ─────────────────────────────────────
        brain_colorscale = "Gray" if theme == "grayscale" else "Portland"

        brain_trace = go.Scatter3d(
            x=x_brain, y=y_brain, z=z_brain,
            mode="markers",
            marker=dict(
                size=2,
                color=brain_fil,
                colorscale=brain_colorscale,
                opacity=0.08,
                showscale=False,
            ),
            name="Brain",
        )

        # ── Step 10: Tumor Scatter3d trace (only if tumour exists) ────────────
        traces = [brain_trace]

        if len(x_tumor) > 0:
            tumor_trace = go.Scatter3d(
                x=x_tumor, y=y_tumor, z=z_tumor,
                mode="markers",
                marker=dict(size=4, color="red", opacity=0.85),
                name="Tumor",
            )
            traces.append(tumor_trace)

        # ── Step 11: Assemble figure ──────────────────────────────────────────
        fig = go.Figure(data=traces)

        # ── Step 12: Layout ───────────────────────────────────────────────────
        fig.update_layout(
            title={
                "text": f"3D Brain Tumor Visualization ({theme.capitalize()} Theme)",
                "x": 0.5,
                "xanchor": "center",
                "font": {"color": "white", "size": 20},
            },
            template="plotly_dark",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgb(2, 6, 23)",
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=600,
            legend=dict(
                font=dict(color="white", size=13),
                bgcolor="rgba(0,0,0,0.4)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
            ),
        )

        logger.info(
            f"Stable 3D plot done — {len(x_brain):,} brain pts, "
            f"{len(x_tumor):,} tumor pts."
        )
        return fig

    except Exception as e:
        logger.error(f"3D Visualization Error: {e}")
        print(f"3D Visualization Error: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure()
