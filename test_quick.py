#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test suite for BiasConsist repository.
Validates dependencies, model initialization, face detection, metric calculations, and Web App UI.
Run:
    python test_quick.py
"""

import sys
import os
from pathlib import Path

# Ensure root directory on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_imports():
    print("[1/5] Testing core dependencies...")
    import torch
    import torchvision
    import transformers
    import gradio
    import cv2
    import pandas
    import numpy
    import sklearn
    print(f"      PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    print(f"      Gradio: {gradio.__version__}")
    print(f"      Transformers: {transformers.__version__}")
    print("      -> Core dependencies imported successfully.")


def test_metrics():
    print("[2/5] Testing Video AUROC & EER metric calculation...")
    from src.metrics import calculate_video_auc, calculate_eer
    import numpy as np

    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.15, 0.85, 0.95])
    auc, fpr, tpr, ths = calculate_video_auc(labels, scores)
    eer = calculate_eer(labels, scores)

    assert auc == 1.0, f"Expected AUC=1.0, got {auc}"
    assert eer == 0.0, f"Expected EER=0.0, got {eer}"
    print(f"      -> Metrics test passed: AUROC={auc:.4f}, EER={eer:.4f}")


def test_retinaface():
    print("[3/5] Testing RetinaFace face detection on sample video...")
    import cv2
    from src.retinaface import prepare_model

    video_path = ROOT / "sample_videos" / "real_face_sample.mp4"
    assert video_path.exists(), f"Sample video not found at: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read frame from sample video"

    detector = prepare_model(det_thres=0.5)
    faces, landmarks = detector.detect(frame)
    assert faces is not None and len(faces) > 0, "No face detected in sample video"
    print(f"      -> RetinaFace passed: detected {len(faces)} face(s) in {video_path.name}")


def test_bias_model_architecture():
    print("[4/5] Testing BiasConsistency model architecture instantiation...")
    from src.model.BiasConsistency import BiasConsistencyDetector
    import torch

    detector = BiasConsistencyDetector()
    detector.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = detector(dummy_input)
    assert logits.shape == (1, 2), f"Expected logits shape (1, 2), got {logits.shape}"
    print("      -> BiasConsistency forward pass passed: output shape (1, 2)")


def test_app_build():
    print("[5/5] Testing Gradio Web App compilation...")
    from app.run import build_ui

    demo = build_ui()
    assert demo is not None, "Failed to build Gradio UI"
    print("      -> Gradio interface built successfully.")


def main():
    print("=" * 60)
    print("  BiasConsist: Quick Smoke Test & Verification Suite")
    print("=" * 60)

    test_imports()
    test_metrics()
    test_retinaface()
    test_bias_model_architecture()
    test_app_build()

    print("=" * 60)
    print("  [SUCCESS] All 5 smoke tests passed successfully!")
    print("  The repository is fully verified and ready for evaluation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
