"""Quick diagnostic: run generate_3d_plot with realistic data and print every step."""
import numpy as np
import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

print("=== 3D VISUALIZATION DIAGNOSTIC ===\n")

try:
    # Simulate what app.py provides — data shape from loader is (4, H, W, D)
    # After inference, mask is (H, W, D) but at processed resolution (128,128,64)
    # The raw MRI data could be different — let's test both scenarios
    
    print("Scenario A: MRI (4,128,128,64) + Mask (128,128,64)")
    mri_data = np.random.rand(4, 128, 128, 64).astype("float32")
    mri_data[0] = mri_data[0] * 0.9 + 0.15  # ensure above threshold
    mask = np.zeros((128, 128, 64), dtype=np.uint8)
    mask[40:80, 40:80, 20:40] = 1

    from brain_tumor_ai.visualization.plot_3d import generate_3d_plot
    fig = generate_3d_plot(mri_data, mask, "grayscale")
    
    print(f"  Figure type: {type(fig)}")
    print(f"  Number of traces: {len(fig.data)}")
    if len(fig.data) > 0:
        t0 = fig.data[0]
        print(f"  Trace[0] type: {type(t0).__name__}")
        print(f"  Trace[0] x len: {len(t0.x) if t0.x is not None else 'None'}")
    
    print("\nScenario B: MRI (4,240,240,155) + Mask (128,128,64)  ← SHAPE MISMATCH CASE")
    mri_big = np.random.rand(4, 240, 240, 155).astype("float32")
    mri_big[0] = mri_big[0] * 0.9 + 0.15
    mask_small = np.zeros((128, 128, 64), dtype=np.uint8)
    mask_small[40:80, 40:80, 20:40] = 1
    
    try:
        fig2 = generate_3d_plot(mri_big, mask_small, "grayscale")
        print(f"  Figure traces: {len(fig2.data)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  → This is the real issue: MRI and mask have different shapes!")
        print(f"  → mri_data[0] shape after ::3: {mri_big[0][::3,::3,::3].shape}")
        print(f"  → mask shape after ::3:        {mask_small[::3,::3,::3].shape}")
    
except Exception as e:
    print(f"FATAL: {e}")
    traceback.print_exc()
