"""Professional 3D brain + tumor surface rendering using Marching Cubes + Mesh3d.

Fixes applied (matching the 10-step spec):
  Step 1  — brain = mri_data[0]  →  use MRI channel, NOT the mask.
  Step 2  — Normalise brain: (x-min)/(max-min+1e-8).
  Step 3  — marching_cubes(brain, level=0.2) for the brain surface.
  Step 4  — Brain Mesh3d uses intensity=verts[:,2] + colorscale=theme.
  Step 5  — mask = (mask > 0.5).astype(int)  strict binarisation.
  Step 6  — Skip tumor mesh when mask.sum() < 100.
  Step 7  — marching_cubes on the CLEANED mask only (never raw noisy map).
  Step 8  — Tumor Mesh3d: color="red", opacity=0.8.
  Step 9  — data=[brain_mesh]; append tumor only if valid.
  Step 10 — xaxis_visible=False etc. removes square/grid artifact.
"""

import logging
from typing import Optional
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import zoom, gaussian_filter

logger = logging.getLogger(__name__)

try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    logger.warning("scikit-image not found — 3D mesh unavailable, using scatter fallback.")

# Step 4: valid Plotly colorscale names (not HTML colour strings)
_THEME_COLORSCALE = {
    "grayscale": "Greys",
    "thermal":   "Hot",
    "rainbow":   "Rainbow",
    "plasma":    "Plasma",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_3d_plot(
    mri_data: np.ndarray,
    mask: np.ndarray,
    theme: str = "grayscale",
) -> go.Figure:
    if _HAS_SKIMAGE:
        return _mesh_pipeline(mri_data, mask, theme)
    return _scatter_fallback(mri_data, mask, theme)


# ─────────────────────────────────────────────────────────────────────────────
# Marching-Cubes pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _mesh_pipeline(mri_data: np.ndarray, mask: np.ndarray, theme: str) -> go.Figure:
    logger.info(f"Marching-Cubes 3D render — theme={theme}")
    try:
        if mri_data is None:
            raise ValueError("mri_data is None")
        if mask is None:
            raise ValueError("mask is None")

        # ── Step 1: brain from channel 0 (FLAIR) ──────────────────────────────
        brain = mri_data[0].astype("float32")

        # ── Step 2: normalise ─────────────────────────────────────────────────
        brain = (brain - brain.min()) / (brain.max() - brain.min() + 1e-8)

        # Downsample 2× for performance
        brain_ds = brain[::2, ::2, ::2]

        # Align mask to brain_ds resolution
        target = brain_ds.shape
        mask_f = mask.astype("float32")
        if mask_f.shape != target:
            zf = tuple(t / s for t, s in zip(target, mask_f.shape))
            mask_ds = zoom(mask_f, zf, order=1)
        else:
            mask_ds = mask_f

        # Smooth brain for nicer isosurface
        brain_smooth = gaussian_filter(brain_ds, sigma=1.0)

        # ── Step 3: brain surface ─────────────────────────────────────────────
        brain_verts, brain_faces, _, _ = marching_cubes(
            brain_smooth, level=0.2, allow_degenerate=False
        )
        logger.info(f"Brain mesh — verts:{len(brain_verts):,}  faces:{len(brain_faces):,}")

        # ── Step 4: brain mesh with theme colorscale ──────────────────────────
        colorscale = _THEME_COLORSCALE.get(theme, "Greys")
        brain_mesh = go.Mesh3d(
            x=brain_verts[:, 0],
            y=brain_verts[:, 1],
            z=brain_verts[:, 2],
            i=brain_faces[:, 0],
            j=brain_faces[:, 1],
            k=brain_faces[:, 2],
            intensity=brain_verts[:, 2],
            colorscale=colorscale,
            showscale=False,
            opacity=0.25,
            flatshading=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.4, roughness=0.4, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=150),
            name="Brain",
            showlegend=True,
        )

        # ── Step 9: start with brain ──────────────────────────────────────────
        data = [brain_mesh]

        # ── Steps 5-8: tumor ──────────────────────────────────────────────────

        # Step 5: strict binary mask
        mask_bin = (mask_ds > 0.5).astype(int)

        # Step 6: skip if too sparse
        tumor_mesh = None
        if mask_bin.sum() < 100:
            logger.info("Tumor mask < 100 voxels — skipping tumor mesh.")
        else:
            # Keep largest connected component to remove noise blobs
            try:
                from scipy.ndimage import label as scipy_label
                labeled, n = scipy_label(mask_bin)
                if n > 1:
                    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
                    best = int(np.argmax(sizes)) + 1
                    mask_bin = (labeled == best).astype(int)
                    logger.info(f"Kept largest of {n} components ({sizes[best-1]} vx)")
            except Exception as ce:
                logger.warning(f"CCA skipped: {ce}")

            # Light smooth AFTER cleaning — never run MC on raw noisy map
            mask_smooth = gaussian_filter(mask_bin.astype("float32"), sigma=0.8)

            # Step 7: extract tumor surface from cleaned mask
            try:
                tumor_verts, tumor_faces, _, _ = marching_cubes(
                    mask_smooth, level=0.5, allow_degenerate=False
                )
                logger.info(f"Tumor mesh — verts:{len(tumor_verts):,}  faces:{len(tumor_faces):,}")

                # Step 8: tumor mesh
                tumor_mesh = go.Mesh3d(
                    x=tumor_verts[:, 0],
                    y=tumor_verts[:, 1],
                    z=tumor_verts[:, 2],
                    i=tumor_faces[:, 0],
                    j=tumor_faces[:, 1],
                    k=tumor_faces[:, 2],
                    color="red",
                    opacity=0.8,
                    flatshading=False,
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.6, roughness=0.3),
                    lightposition=dict(x=100, y=200, z=150),
                    name="Tumor",
                    showlegend=True,
                )
            except Exception as te:
                logger.warning(f"Tumor surface failed: {te}")

        # Step 9: append tumor only if valid
        if tumor_mesh is not None:
            data.append(tumor_mesh)

        fig = go.Figure(data=data)
        _apply_layout(fig, theme)
        return fig

    except Exception as e:
        logger.error(f"Mesh pipeline error: {e}")
        import traceback; traceback.print_exc()
        return go.Figure()


# ─────────────────────────────────────────────────────────────────────────────
# Scatter3d fallback
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_fallback(mri_data: np.ndarray, mask: np.ndarray, theme: str) -> go.Figure:
    logger.info("Using Scatter3d fallback (scikit-image not installed)")
    try:
        brain = mri_data[0].astype("float32")
        brain = (brain - brain.min()) / (brain.max() - brain.min() + 1e-8)
        brain_ds = brain[::3, ::3, ::3]

        target = brain_ds.shape
        mask_f = mask.astype("float32")
        if mask_f.shape != target:
            zf = tuple(t / s for t, s in zip(target, mask_f.shape))
            mask_ds = zoom(mask_f, zf, order=0)
        else:
            mask_ds = mask_f

        mask_bin = (mask_ds > 0.5).astype(int)

        h, w, d = brain_ds.shape
        gx, gy, gz = np.mgrid[0:h, 0:w, 0:d]
        x, y, z = gx.flatten(), gy.flatten(), gz.flatten()
        bv, mv = brain_ds.flatten(), mask_bin.flatten()

        sel = bv > 0.15
        cs_map = {"grayscale": "Greys", "thermal": "Hot", "rainbow": "Rainbow", "plasma": "Plasma"}
        data = [go.Scatter3d(
            x=x[sel], y=y[sel], z=z[sel], mode="markers",
            marker=dict(size=2, color=bv[sel], colorscale=cs_map.get(theme, "Greys"), opacity=0.12),
            name="Brain",
        )]
        if mv.sum() >= 100:
            tsel = mv > 0
            data.append(go.Scatter3d(
                x=x[tsel], y=y[tsel], z=z[tsel], mode="markers",
                marker=dict(size=4, color="red", opacity=0.85),
                name="Tumor",
            ))
        fig = go.Figure(data=data)
        _apply_layout(fig, theme)
        return fig
    except Exception as e:
        logger.error(f"Scatter fallback error: {e}")
        return go.Figure()


# ─────────────────────────────────────────────────────────────────────────────
# Shared layout — Step 10: all axes hidden → no square/grid artifacts
# ─────────────────────────────────────────────────────────────────────────────

def _apply_layout(fig: go.Figure, theme: str) -> None:
    fig.update_layout(
        title={
            "text": f"3D Brain Tumor Visualization ({theme.capitalize()} Theme)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "white", "size": 18, "family": "Arial, sans-serif"},
        },
        paper_bgcolor="#020617",
        scene=dict(
            # Step 10: hide all scene axes to eliminate the square artifact
            xaxis=dict(visible=False, showgrid=False, showbackground=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, showbackground=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, showbackground=False, zeroline=False),
            bgcolor="#020617",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2), up=dict(x=0, y=0, z=1)),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=620,
        legend=dict(
            font=dict(color="white", size=13),
            bgcolor="rgba(255,255,255,0.08)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
    )
