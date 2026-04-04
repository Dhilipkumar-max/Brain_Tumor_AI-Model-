"""Stable 3D Interactive visualization for brain MRI using Scatter3d point cloud.

Uses go.Scatter3d for reliable cross-environment rendering.
Brain is shown as a semi-transparent point cloud; tumor as a dense red cluster.
"""

import logging
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def generate_3d_plot(mri_data: np.ndarray, mask: np.ndarray, theme: str = "grayscale") -> go.Figure:
    """Generates a stable, interactive 3D brain + tumor point-cloud visualization.

    Args:
        mri_data (np.ndarray): Multi-modal MRI scan of shape (4, H, W, D).
        mask (np.ndarray): Binary segmentation mask — any shape (H_m, W_m, D_m).
        theme (str): Visual color theme ('grayscale', 'thermal', 'rainbow', 'plasma').

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

        brain = mri_data[0].astype("float32")

        # ── Step 2: Normalize [0, 1] ──────────────────────────────────────────
        brain = (brain - brain.min()) / (brain.max() - brain.min() + 1e-8)

        # ── Step 3: Downsample (3x stride = 27x fewer voxels for 60fps) ───────
        brain_ds = brain[::3, ::3, ::3]

        # ── Step 4: Align mask to brain_ds via zoom (handles resolution diff) ─
        target_shape = brain_ds.shape
        mask_float = mask.astype("float32")

        if mask_float.shape != target_shape:
            zf = (
                target_shape[0] / mask_float.shape[0],
                target_shape[1] / mask_float.shape[1],
                target_shape[2] / mask_float.shape[2],
            )
            mask_ds = zoom(mask_float, zf, order=0)
            logger.info(f"Mask resampled {mask_float.shape} → {mask_ds.shape}")
        else:
            mask_ds = mask_float

        # ── Step 5: Build coordinate grid ─────────────────────────────────────
        h, w, d = brain_ds.shape
        grid_x, grid_y, grid_z = np.mgrid[0:h, 0:w, 0:d]

        # ── Step 6: Flatten ───────────────────────────────────────────────────
        x            = grid_x.flatten()
        y            = grid_y.flatten()
        z            = grid_z.flatten()
        brain_values = brain_ds.flatten()
        mask_values  = mask_ds.flatten()

        print(f"[3D] Shapes — x:{x.shape} brain:{brain_values.shape} mask:{mask_values.shape}")

        # ── Step 7: Filter background air voxels (keep brain tissue only) ─────
        threshold = 0.15
        brain_sel = brain_values > threshold
        x_brain   = x[brain_sel]
        y_brain   = y[brain_sel]
        z_brain   = z[brain_sel]
        brain_fil = brain_values[brain_sel]

        # ── Step 8: Filter tumor voxels ───────────────────────────────────────
        tumor_sel = mask_values > 0
        x_tumor   = x[tumor_sel]
        y_tumor   = y[tumor_sel]
        z_tumor   = z[tumor_sel]

        print(f"[3D] Points — brain:{len(x_brain):,}  tumor:{len(x_tumor):,}")

        # ── Step 9: Theme → colorscale mapping ───────────────────────────────
        colorscale_map = {
            "grayscale": "gray",
            "thermal":   "hot",
            "rainbow":   "rainbow",
            "plasma":    "plasma",
        }
        brain_colorscale = colorscale_map.get(theme, "blues")

        # ── Step 10: Traces ───────────────────────────────────────────────────
        brain_trace = go.Scatter3d(
            x=x_brain, y=y_brain, z=z_brain,
            mode="markers",
            marker=dict(
                size=2,
                color=brain_fil,
                colorscale=brain_colorscale,
                cmin=0.15,
                cmax=1.0,
                opacity=0.35,       # increased from 0.08 → visible on white bg
                showscale=True,
                colorbar=dict(
                    title=dict(text="Intensity", font=dict(color="black")),
                    thickness=12,
                    tickfont=dict(color="black"),
                    x=1.02,
                ),
            ),
            name="Brain",
        )

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

        # ── Step 12: Layout (white background, matches 2D slide style) ────────
        fig.update_layout(
            title={
                "text": f"3D Brain Tumor Visualization ({theme.capitalize()} Theme)",
                "x": 0.5,
                "xanchor": "center",
                "font": {"color": "black", "size": 18, "family": "Arial"},
            },
            paper_bgcolor="white",          # outer canvas → white
            plot_bgcolor="white",
            scene=dict(
                xaxis=dict(visible=False, backgroundcolor="white"),
                yaxis=dict(visible=False, backgroundcolor="white"),
                zaxis=dict(visible=False, backgroundcolor="white"),
                bgcolor="white",            # 3D scene → white
            ),
            margin=dict(l=0, r=60, t=60, b=0),
            height=600,
            legend=dict(
                font=dict(color="black", size=13),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        )

        logger.info(f"3D done: {len(x_brain):,} brain, {len(x_tumor):,} tumor pts.")
        return fig

    except Exception as e:
        logger.error(f"3D Visualization Error: {e}")
        print(f"3D Visualization Error: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure()
