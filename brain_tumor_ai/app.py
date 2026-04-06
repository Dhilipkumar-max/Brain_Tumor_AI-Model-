"""Brain Tumor AI — Gradio web interface.

Premium dark-mode medical UI with full analysis pipeline:
  Upload 4 MRI modality .npy files → AI inference → 2D/3D visualisation + report.
"""

import os
import logging
from typing import Any, List, Tuple
import gradio as gr
import numpy as np
import torch

from brain_tumor_ai.preprocessing.loader import load_mri_data
from brain_tumor_ai.preprocessing.transforms import preprocess_mri
from brain_tumor_ai.models.inference import run_inference
from brain_tumor_ai.visualization.plot_2d import generate_2d_views
from brain_tumor_ai.visualization.plot_3d import generate_3d_plot
from brain_tumor_ai.reports.generator import generate_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brain_tumor_ai")

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium dark medical theme
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Root tokens ── */
:root {
    --primary:      #0F9D58;
    --primary-glow: #22c55e;
    --bg-deep:      #020617;
    --bg-card:      #0f172a;
    --bg-input:     #1e293b;
    --border:       rgba(34,197,94,0.25);
    --text-main:    #e2e8f0;
    --text-muted:   #94a3b8;
    --radius:       12px;
    --shadow:       0 4px 24px rgba(15,157,88,0.12);
}

/* ── Page background ── */
body, .gradio-container {
    background: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Header band ── */
.header-band {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 0 20px;
    text-align: center;
    margin-bottom: 24px;
}
.header-band h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #0F9D58, #22c55e, #4ade80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.header-band p {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin: 0;
}

/* ── Cards ── */
.gr-block, .gr-box, .gr-form, .gr-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
}

/* ── Input labels ── */
label, .gr-label, .svelte-1gfkn6j {
    color: var(--text-main) !important;
    font-weight: 500 !important;
}

/* ── Upload area ── */
.gr-file-upload, .upload-container {
    background: var(--bg-input) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.3s ease;
}
.gr-file-upload:hover {
    border-color: var(--primary-glow) !important;
}

/* ── Primary button ── */
.gr-button-primary, button[variant="primary"] {
    background: linear-gradient(135deg, #0F9D58, #22c55e) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 18px rgba(34,197,94,0.35) !important;
    letter-spacing: 0.3px;
}
.gr-button-primary:hover, button[variant="primary"]:hover {
    background: linear-gradient(135deg, #22c55e, #4ade80) !important;
    box-shadow: 0 0 28px rgba(34,197,94,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Dropdown ── */
.gr-dropdown select, select {
    background: var(--bg-input) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Tabs ── */
.gr-tab-nav button {
    color: var(--text-muted) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}
.gr-tab-nav button.selected {
    color: var(--primary-glow) !important;
    border-bottom-color: var(--primary-glow) !important;
}

/* ── Stat cards row ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin: 16px 0 8px;
}
.stat-card {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    text-align: center;
    transition: box-shadow 0.3s ease;
}
.stat-card:hover { box-shadow: 0 0 16px rgba(34,197,94,0.2); }
.stat-card .value {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--primary-glow);
    line-height: 1;
}
.stat-card .label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── Report markdown ── */
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: var(--primary-glow) !important;
}
.gr-markdown { color: var(--text-main) !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(mri_files: List[Any], theme: str) -> Tuple[Any, Any, str]:
    """End-to-end analysis pipeline triggered by the UI button."""
    logger.info("Analysis triggered via UI...")

    if not mri_files or len(mri_files) == 0:
        err = (
            "## ⚠️ No Files Uploaded\n\n"
            "Please upload **4 MRI modality files** (.npy):\n"
            "- `flair.npy`\n- `t1.npy`\n- `t1ce.npy`\n- `t2.npy`"
        )
        return None, None, err

    try:
        data    = load_mri_data(mri_files)
        tensor  = preprocess_mri(data)
        results = run_inference(tensor)
        mask    = results["mask"]

        image_path     = generate_2d_views(data, mask)
        plot_3d_figure = generate_3d_plot(data, mask, theme)
        report_text    = generate_report(results)

        logger.info("Analysis completed successfully.")
        return image_path, plot_3d_figure, report_text

    except ValueError as ve:
        logger.warning(f"Validation: {ve}")
        return None, None, f"## ❌ Validation Error\n\n{ve}"
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback; logger.error(traceback.format_exc())
        return None, None, f"## 🔧 System Error\n\n{e}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="AI Brain Tumor Analysis",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
) as app:

    # ── Hero header ──────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-band">
        <h1>🧠 Brain Tumor AI Analysis</h1>
        <p>Medical-grade 3D MRI segmentation & classification  ·  Powered by MONAI 3D UNet</p>
    </div>
    """)

    # ── Main layout ──────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left panel — controls ────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📂 Upload MRI Scans")
            gr.Markdown(
                "_Upload the 4 required modalities as `.npy` files. "
                "Filename must contain: `flair`, `t1`, `t1ce`, `t2`._",
            )

            mri_upload = gr.File(
                label="MRI Modalities (.npy)",
                file_count="multiple",
                file_types=[".npy"],
            )

            gr.Markdown("### 🎨 Visualization")
            theme_choice = gr.Dropdown(
                label="Color Theme",
                choices=["grayscale", "thermal", "rainbow", "plasma"],
                value="grayscale",
                interactive=True,
            )

            analyze_btn = gr.Button(
                "🔍  Run Analysis",
                variant="primary",
                size="lg",
            )

            gr.HTML("""
            <div style="margin-top:20px;padding:14px;background:#0f172a;
                        border:1px solid rgba(34,197,94,0.2);border-radius:10px;">
                <p style="color:#94a3b8;font-size:0.82rem;margin:0;line-height:1.6;">
                    ⚠️ <strong style="color:#e2e8f0;">Disclaimer:</strong>
                    This tool is for <em>research and educational purposes only</em>.
                    Not a substitute for professional medical diagnosis.
                </p>
            </div>
            """)

        # ── Right panel — results ────────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():

                # Tab 1 — 2D slices
                with gr.TabItem("🖼️  2D Slices"):
                    slice_output = gr.Image(
                        label="Multi-Plane MRI Views (Axial · Sagittal · Coronal)",
                        type="filepath",
                        show_download_button=True,
                    )

                # Tab 2 — 3D interactive
                with gr.TabItem("📊  3D Interactive"):
                    plot_output = gr.Plot(
                        label="Interactive 3D Brain + Tumor Rendering",
                    )
                    gr.HTML("""
                    <p style="color:#64748b;font-size:0.8rem;text-align:center;margin:6px 0 0;">
                        🖱️ Drag to rotate  ·  Scroll to zoom  ·  Right-click to pan
                    </p>
                    """)

                # Tab 3 — Clinical report
                with gr.TabItem("📄  Clinical Report"):
                    report_output = gr.Markdown(
                        label="AI Clinical Analysis Report",
                        value="> _Run the analysis to generate a detailed report._",
                    )

    # ── Wire up ───────────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=run_analysis,
        inputs=[mri_upload, theme_choice],
        outputs=[slice_output, plot_output, report_output],
        show_progress="full",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Local dev entry
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Gradio dev server...")
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
    )
