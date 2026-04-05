"""Professional 3D brain + tumor surface rendering using Marching Cubes + Mesh3d.

Replaces point-cloud approach with smooth ISO-surface extraction for
medical-grade, publication-quality 3D visualization.
"""

import logging
from typing import Optional
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import zoom, gaussian_filter

logger = logging.getLogger(__name__)

# ── lazy import so the module still loads if skimage is missing ──────────────
try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    logger.warning("scikit-image not found — 3D mesh unavailable, using scatter fallback.")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_3d_plot(
    mri_data: np.ndarray,
    mask: np.ndarray,
    theme: str = "grayscale",
) -> go.Figure:
    """Generate a smooth-surface 3D brain + tumor visualization.

    Args:
        mri_data: Multi-modal MRI of shape (4, H, W, D).
        mask:     Binary segmentation mask, any resolution.
        theme:    Color theme: 'grayscale' | 'thermal' | 'rainbow' | 'plasma'.

    Returns:
        Interactive Plotly Figure with Mesh3d surfaces.
    """
    if _HAS_SKIMAGE:
        return _mesh_pipeline(mri_data, mask, theme)
    else:
        return _scatter_fallback(mri_data, mask, theme)


# ─────────────────────────────────────────────────────────────────────────────
# Marching-Cubes Mesh Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _mesh_pipeline(mri_data, mask, theme):
    logger.info(f"Marching-Cubes 3D render — theme={theme}")
    try:
        # ── 1. Validate ───────────────────────────────────────────────────────
        if mri_data is None:
            raise ValueError("MRI data is None")
        if mask is None:
            raise ValueError("Mask is None")

        brain = mri_data[0].astype("float32")

        # ── 2. Normalize ──────────────────────────────────────────────────────
        brain = (brain - brain.min()) / (brain.max() - brain.min() + 1e-8)

        # ── 3. Downsample 2× for performance ──────────────────────────────────
        brain_ds = brain[::2, ::2, ::2]

        # Align mask resolution → brain_ds shape
        target = brain_ds.shape
        mask_f = mask.astype("float32")
        if mask_f.shape != target:
            zf = tuple(t / s for t, s in zip(target, mask_f.shape))
            mask_ds = zoom(mask_f, zf, order=0)
            logger.info(f"Mask resampled {mask_f.shape} → {mask_ds.shape}")
        else:
            mask_ds = mask_f

        # ── 4. Clean mask: strict binary + keep ONLY largest connected region ──
        # Critical: noisy/scattered mask → cube artifact in marching cubes
        from scipy.ndimage import label as scipy_label
        mask_bin = (mask_ds > 0.5).astype(np.uint8)

        # Keep only the largest connected component (eliminates noise blobs)
        labeled, num_components = scipy_label(mask_bin)
        if num_components > 1:
            sizes = [(labeled == i).sum() for i in range(1, num_components + 1)]
            largest_label = int(np.argmax(sizes)) + 1
            mask_bin = (labeled == largest_label).astype(np.uint8)
            logger.info(f"Kept largest component ({sizes[largest_label-1]} vx) of {num_components} found.")

        mask_ds = mask_bin.astype("float32")

        # ── 4. Smooth brain slightly for nicer surface ─────────────────────────
        brain_smooth = gaussian_filter(brain_ds, sigma=1.0)

        # ── 5. Extract brain surface (Marching Cubes) ─────────────────────────
        brain_level = 0.25   # Isovalue — tune 0.15-0.35 to expose more/less tissue
        verts, faces, normals, _ = marching_cubes(brain_smooth, level=brain_level,
                                                   allow_degenerate=False)
        print(f"[3D Mesh] Brain — verts:{len(verts):,}  faces:{len(faces):,}")

        # ── 6. Brain mesh colour by theme (high-contrast for dark background) ─
        theme_color = {
            "grayscale": "#C0C0C0",   # silver — stands out on black
            "thermal":   "#FF8C69",   # light salmon / coral
            "rainbow":   "#00BFFF",   # deep sky blue
            "plasma":    "#DA70D6",   # orchid / violet
        }.get(theme, "#C0C0C0")

        brain_mesh = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=theme_color,
            opacity=0.35,
            flatshading=False,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.4,
                roughness=0.4,
                fresnel=0.2,
            ),
            lightposition=dict(x=100, y=200, z=150),
            name="Brain",
            showlegend=True,
        )

        # ── 7. Tumor surface — safe extraction with noise guard ────────────────
        traces = [brain_mesh]

        tumor_voxel_count = int(mask_ds.sum())
        MIN_TUMOR_VOXELS = 50

        if tumor_voxel_count >= MIN_TUMOR_VOXELS:
            try:
                # Light smooth after cleaning — preserves shape, rounds edges
                mask_smooth = gaussian_filter(mask_ds, sigma=1.0)
                t_verts, t_faces, _, _ = marching_cubes(
                    mask_smooth, level=0.5, allow_degenerate=False
                )
                print(f"[3D Mesh] Tumor — voxels:{tumor_voxel_count}  "
                      f"verts:{len(t_verts):,}  faces:{len(t_faces):,}")

                tumor_mesh = go.Mesh3d(
                    x=t_verts[:, 0], y=t_verts[:, 1], z=t_verts[:, 2],
                    i=t_faces[:, 0], j=t_faces[:, 1], k=t_faces[:, 2],
                    color="red",
                    opacity=0.85,
                    flatshading=False,
                    lighting=dict(
                        ambient=0.4,
                        diffuse=0.8,
                        specular=0.6,
                        roughness=0.3,
                    ),
                    lightposition=dict(x=100, y=200, z=150),
                    name="Tumor",
                    showlegend=True,
                )
                traces.append(tumor_mesh)
            except Exception as te:
                logger.warning(f"Tumor surface extraction failed: {te}")

        # ── 8. Assemble and layout ─────────────────────────────────────────────
        fig = go.Figure(data=traces)
        _apply_layout(fig, theme)
        logger.info("Mesh3d figure built successfully.")
        return fig

    except Exception as e:
        logger.error(f"Mesh pipeline error: {e}")
        print(f"[3D Mesh Error] {e}")
        import traceback; traceback.print_exc()
        return go.Figure()


# ─────────────────────────────────────────────────────────────────────────────
# Scatter3d fallback (no scikit-image)
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_fallback(mri_data, mask, theme):
    """Lightweight point-cloud fallback when skimage is unavailable."""
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

        h, w, d = brain_ds.shape
        gx, gy, gz = np.mgrid[0:h, 0:w, 0:d]
        x, y, z = gx.flatten(), gy.flatten(), gz.flatten()
        bv = brain_ds.flatten()
        mv = mask_ds.flatten()

        sel = bv > 0.15
        cs_map = {"grayscale":"gray","thermal":"hot","rainbow":"rainbow","plasma":"plasma"}
        brain_trace = go.Scatter3d(
            x=x[sel], y=y[sel], z=z[sel], mode="markers",
            marker=dict(size=2, color=bv[sel],
                        colorscale=cs_map.get(theme,"gray"), opacity=0.12),
            name="Brain",
        )
        traces = [brain_trace]
        tsel = mv > 0
        if tsel.any():
            traces.append(go.Scatter3d(
                x=x[tsel], y=y[tsel], z=z[tsel], mode="markers",
                marker=dict(size=4, color="red", opacity=0.85),
                name="Tumor",
            ))
        fig = go.Figure(data=traces)
        _apply_layout(fig, theme)
        return fig
    except Exception as e:
        print(f"[Scatter Fallback Error] {e}")
        return go.Figure()


# ─────────────────────────────────────────────────────────────────────────────
# Shared layout
# ─────────────────────────────────────────────────────────────────────────────

def _apply_layout(fig: go.Figure, theme: str) -> None:
    fig.update_layout(
        title={
            "text": f"3D Brain Tumor Visualization ({theme.capitalize()} Theme)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "white", "size": 18, "family": "Arial, sans-serif"},
        },
        paper_bgcolor="#020617",        # GFG dark background
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="#020617"),
            yaxis=dict(visible=False, backgroundcolor="#020617"),
            zaxis=dict(visible=False, backgroundcolor="#020617"),
            bgcolor="#020617",           # 3D scene — dark
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
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
