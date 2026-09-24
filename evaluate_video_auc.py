#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script tính chỉ số Video AUROC (Video Area Under the ROC Curve) cho deepfake detection.
Hỗ trợ tính từ file CSV kết quả hoặc từ mảng dự đoán trực tiếp.

Cách dùng:
1. Từ dòng lệnh với file CSV:
   python evaluate_video_auc.py --csv outputs/tmp/gradio_app/inference_log.csv

2. Trong mã nguồn Python:
   from src.metrics import calculate_video_auc, compute_video_auc_from_frames
   auc, fpr, tpr, ths = calculate_video_auc(video_labels, video_scores)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.metrics import (
    calculate_eer,
    calculate_video_auc,
    compute_video_auc_from_frames,
    infer_label_from_path,
)


def evaluate_from_csv(
    csv_path: str,
    input_col: Optional[str] = None,
    score_col: Optional[str] = None,
    label_col: Optional[str] = None,
    reduce: str = "mean",
) -> dict:
    """
    Tính Video AUC từ file CSV ghi nhận kết quả inference.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"File CSV {csv_path} rỗng!")

    print(f"Đã nạp {len(df)} dòng từ: {csv_path}")
    print(f"Các cột có sẵn: {list(df.columns)}")

    # 1. Xác định cột chứa tên/đường dẫn file video
    if input_col is None:
        for candidate in ["input", "files", "video", "file", "path", "filename"]:
            if candidate in df.columns:
                input_col = candidate
                break
    if input_col is None:
        input_col = df.columns[0]

    # 2. Xác định cột điểm dự đoán (score/probability of fake)
    if score_col is None:
        candidates = ["avg_p_fake", "median_p_fake", "prob_class_1", "p_fake", "score", "fake_prob", "prob"]
        for candidate in candidates:
            if candidate in df.columns:
                score_col = candidate
                break
    if score_col is None:
        raise ValueError(f"Không tự động tìm thấy cột điểm dự đoán trong {list(df.columns)}. Vui lòng chỉ định --score_col.")

    # 3. Xác định cột nhãn thực tế (ground truth label)
    if label_col is None:
        for candidate in ["label", "labels", "y_true", "target", "ground_truth"]:
            if candidate in df.columns:
                label_col = candidate
                break

    # Nếu chưa có cột nhãn, suy đoán nhãn từ đường dẫn file
    if label_col is not None and label_col in df.columns:
        labels = df[label_col].to_numpy()
    else:
        print("Không có sẵn cột nhãn. Tự động suy đoán nhãn (0 = Real, 1 = Fake) từ đường dẫn file...")
        inferred = [infer_label_from_path(p) for p in df[input_col]]
        df["inferred_label"] = inferred
        valid_df = df.dropna(subset=["inferred_label"]).copy()
        valid_df["inferred_label"] = valid_df["inferred_label"].astype(int)
        if len(valid_df) < len(df):
            print(f"Bỏ qua {len(df) - len(valid_df)} mẫu không xác định được nhãn.")
        df = valid_df
        label_col = "inferred_label"
        labels = df[label_col].to_numpy()

    scores = df[score_col].to_numpy(dtype=float)
    video_ids = df[input_col].to_numpy()

    # Kiểm tra xem đây là dữ liệu cấp video hay cấp frame
    unique_videos = np.unique(video_ids)
    if len(unique_videos) < len(video_ids):
        print(f"Phát hiện dữ liệu cấp frame ({len(video_ids)} frames thuộc {len(unique_videos)} videos). Đang tổng hợp...")
        res = compute_video_auc_from_frames(video_ids, scores, labels, reduce=reduce)
        auc = res["video_auc"]
        eer = res["video_eer"]
        num_vids = res["num_videos"]
        num_real = res["num_real"]
        num_fake = res["num_fake"]
    else:
        auc, fpr, tpr, ths = calculate_video_auc(labels, scores)
        y_score_2d = np.column_stack([1.0 - scores, scores])
        eer = calculate_eer(labels, y_score_2d) if len(np.unique(labels)) == 2 else float("nan")
        num_vids = len(scores)
        num_real = int((labels == 0).sum())
        num_fake = int((labels == 1).sum())

    print("\n" + "=" * 50)
    print("           KẾT QUẢ ĐÁNH GIÁ VIDEO AUC")
    print("=" * 50)
    print(f"Tổng số video đánh giá : {num_vids}")
    print(f" - Số video Real (0)   : {num_real}")
    print(f" - Số video Fake (1)   : {num_fake}")
    print(f"Cột điểm sử dụng       : {score_col}")
    if np.isnan(auc):
        print("⚠️ CẢNH BÁO: Không thể tính AUC vì tập dữ liệu chỉ chứa 1 lớp nhãn!")
    else:
        print(f"🎯 Video AUROC         : {auc:.4f} ({auc * 100:.2f}%)")
        if not np.isnan(eer):
            print(f"🎯 Video EER           : {eer:.4f} ({eer * 100:.2f}%)")
    print("=" * 50 + "\n")

    return {
        "video_auc": auc,
        "video_eer": eer,
        "num_videos": num_vids,
        "num_real": num_real,
        "num_fake": num_fake,
    }


def main():
    parser = argparse.ArgumentParser(description="Tính chỉ số Video AUROC cho Deepfake Detection")
    parser.add_argument("--csv", type=str, default="outputs/tmp/gradio_app/inference_log.csv", help="Đường dẫn file CSV chứa kết quả")
    parser.add_argument("--input_col", type=str, default=None, help="Tên cột chứa đường dẫn file video")
    parser.add_argument("--score_col", type=str, default=None, help="Tên cột chứa xác suất dự đoán Fake")
    parser.add_argument("--label_col", type=str, default=None, help="Tên cột chứa nhãn thực tế (0/1)")
    parser.add_argument("--reduce", type=str, default="mean", choices=["mean", "median", "max"], help="Phương thức gộp frame ('mean', 'median', 'max')")
    args = parser.parse_args()

    evaluate_from_csv(
        csv_path=args.csv,
        input_col=args.input_col,
        score_col=args.score_col,
        label_col=args.label_col,
        reduce=args.reduce,
    )


if __name__ == "__main__":
    main()
