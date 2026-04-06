"""FastAPI entry point for Render deployment.

Exposes the Brain Tumor AI as a REST API alongside the Gradio web UI.
Run locally:  uvicorn main:app --reload
Run on Render: start command is  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import io
import sys
import logging

# ── Import Path Management ──────────────────────────────────────────────────
# Ensure project root is in sys.path for robust imports in cloud environments
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brain_tumor_api")

# ── Gradio app (imported lazily to avoid circular issues) ─────────────────────
from brain_tumor_ai.app import app as gradio_app

# ── FastAPI core ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor AI API",
    description="Medical-grade 3D MRI Brain Tumor Analysis — powered by MONAI 3D UNet",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "🧠 Brain Tumor AI is running",
        "status": "healthy",
        "ui": "/gradio",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/")
async def predict_npy(file: UploadFile = File(...)):
    """Accept a single .npy file (shape 4,H,W,D) and return classification + confidence."""
    try:
        from brain_tumor_ai.preprocessing.transforms import preprocess_mri
        from brain_tumor_ai.models.inference import run_inference

        content = await file.read()
        data = np.load(io.BytesIO(content))  # expect (4, H, W, D)

        if data.ndim != 4 or data.shape[0] != 4:
            raise HTTPException(
                status_code=422,
                detail=f"Expected array shape (4, H, W, D), got {data.shape}",
            )

        import torch
        tensor = preprocess_mri(data)
        results = run_inference(tensor)

        return {
            "tumor_detected": bool(results["tumor_detected"]),
            "tumor_type": results["tumor_type"],
            "confidence": round(float(results["confidence"]), 4),
            "volume_voxels": int(results["volume_voxels"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Mount Gradio as a sub-app at /gradio ─────────────────────────────────────
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
