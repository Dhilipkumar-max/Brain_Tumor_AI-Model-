"""Professional 2D MRI Visualization module with tumor overlay support.

This module generates high-quality 2D slice views (Axial, Sagittal, Coronal) 
of MRI scans with optional tumor segmentation overlays.
"""

import os
import logging
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

def generate_2d_views(mri_data: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Generates a professional 3-view 2D MRI plot with correct slicing.

    Args:
        mri_data (np.ndarray): Multi-modal MRI of shape (4, H, W, D).
        mask (Optional[np.ndarray]): Binary segmentation mask (H_m, W_m, D_m).

    Returns:
        str: Absolute path to the saved visualization image (mri_overlay.png).
    """
    logger.info("Generating professional 2D slice views...")

    try:
        # 1. Component Selection
        # Use FLAIR (index 0) for the grayscale background
        brain = mri_data[0]
        
        # 2. Extract Dimensions and Implement Safe Indexing
        h, w, d = brain.shape
        mid_h = min(h // 2, h - 1)
        mid_w = min(w // 2, w - 1)
        mid_d = min(d // 2, d - 1)

        # 3. Extract Correct Middle Slices (Orthogonal Planes)
        slice_axial = brain[:, :, mid_d]       # Axial: (H, W) -> top view
        slice_sagittal = brain[:, mid_w, :]    # Sagittal: (H, D) -> side view
        slice_coronal = brain[mid_h, :, :]     # Coronal: (W, D) -> front view

        # MANDATORY Debug Logging
        print(f"Brain shape: {brain.shape}")
        print(f"Axial shape: {slice_axial.shape}")
        print(f"Sagittal shape: {slice_sagittal.shape}")
        print(f"Coronal shape: {slice_coronal.shape}")

        # 4. Plot Layout Setup
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='white')
        
        # Views mapping
        views = [
            (slice_axial, "Axial View"),
            (slice_sagittal, "Sagittal View"),
            (slice_coronal, "Coronal View")
        ]

        for i, (img_slice, title) in enumerate(views):
            # Show MRI background in grayscale
            axes[i].imshow(img_slice.T if i != 0 else img_slice, cmap='gray', origin='lower')
            axes[i].set_title(title, color='black', fontsize=14, fontweight='bold', pad=10)
            axes[i].axis('off')

            # 5. Handle Overlay (Mask Alignment)
            if mask is not None:
                h_m, w_m, d_m = mask.shape
                mid_h_m = min(h_m // 2, h_m - 1)
                mid_w_m = min(w_m // 2, w_m - 1)
                mid_d_m = min(d_m // 2, d_m - 1)
                
                # Extract corresponding mask slice using ITS OWN shape info and safe indexing
                if i == 0:  # Axial
                    m_slice = mask[:, :, mid_d_m]
                elif i == 1:  # Sagittal
                    m_slice = mask[:, mid_w_m, :]
                else:  # Coronal
                    m_slice = mask[mid_h_m, :, :]

                # Only overlay if there's predictive data
                if np.max(m_slice) > 0:
                    # Robust Alignment: Zoom mask subset to match MRI resolution
                    zoom_factor = (img_slice.shape[0] / m_slice.shape[0], 
                                   img_slice.shape[1] / m_slice.shape[1])
                    
                    aligned_mask = zoom(m_slice, zoom_factor, order=0) # order 0 for labels
                    
                    # Apply Color Overlay (jet/red) with transparency
                    overlay = aligned_mask.T if i != 0 else aligned_mask
                    axes[i].imshow(overlay, cmap='jet', alpha=0.4, origin='lower')

        # 6. Save and Finish
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        # Using specific output name requested in previous turn or keeping consistent
        output_path = os.path.join(output_dir, "mri_overlay.png")

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor='white', dpi=150)
        plt.close(fig)
        
        logger.info(f"Corrected 2D Views saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Visualization pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ""
