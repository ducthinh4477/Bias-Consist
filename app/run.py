import os

# set GRADIO_TEMP_DIR
os.environ["GRADIO_TEMP_DIR"] = "./tmp/gradio"


import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import gradio as gr
import imageio
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Ensure project root on sys.path
try:
    import autorootcwd
except Exception:
    THIS = Path(__file__).resolve()
    ROOT = THIS.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from detector import align_face
from src.config import Config
from src.hf.modeling_gend import GenD as GenD_HF
from src.metrics import calculate_eer, calculate_video_auc, infer_label_from_path
from src.model.BiasConsistency import BiasConsistencyDetector
from src.model.Effort import Effort
from src.model.ForAda import ForAda
from src.model.GenD import GenD as GenD_Train
from src.retinaface import RetinaFace, prepare_model

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CKPT = "checkpoints/best_model.ckpt"
DEFAULT_BIAS_CKPT = "weights/BiasConsist/bias_consistency.pth"
DEFAULT_EFFORT_CKPT = "weights/Effort/effort_clip_L14_trainOn_FaceForensic.pth"
DEFAULT_FORADA_CKPT = "weights/ForAda/forada_checkpoint.pth"
DEFAULT_GEND_ID = "yermandy/GenD_CLIP_L_14"
GEND_MODELS = ["GenD (CLIP-ViT-L/14)"]
OUTPUT_DIR = Path("outputs/tmp/gradio_app")
TABLE_HEADERS = [
    "input",
    "num_frames",
    "num_faces",
    "avg_p_fake",
    "median_p_fake",
    "prediction",
    "ground_truth",
    "result",
]

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "English": {
        "app_title": "BiasConsist",
        "app_badge": "Benchmark",
        "app_subtitle": "Multi-Model Deepfake Detection & Comparative Evaluation Platform",
        "language_label": "Language",
        "sidebar_config_title": "Configuration",
        "model_source_label": "Model Architecture",
        "bias_ckpt_label": "BiasConsist Checkpoint Path",
        "gend_model_label": "GenD Model Architecture",
        "effort_ckpt_label": "Effort Checkpoint Path",
        "forada_ckpt_label": "ForAda Checkpoint Path",
        "local_ckpt_label": "Local Checkpoint Path",
        "upload_label": "Upload Media (MP4, AVI, MOV, MKV, JPG, PNG, WEBP)",
        "advanced_settings": "Advanced Detection Settings",
        "face_thresh_label": "Face Detection Threshold",
        "scale_label": "Face Alignment Crop Scale",
        "target_size_label": "Target Face Size (px, -1 for original)",
        "stride_label": "Video Frame Stride",
        "max_frames_label": "Max Frames per Video (-1 for all)",
        "max_faces_label": "Max Faces per Frame",
        "run_btn": "Run Detection",
        "status_ready": "**System Status: Ready** — Select a model, upload media files, and click **Run Detection**.",
        "status_loading_model": "**System Status: Loading Model** — Initializing model weights and feature extractor...",
        "status_loading_detector": "**System Status: Initializing Detector** — Loading RetinaFace detector...",
        "status_collecting_inputs": "**System Status: Parsing Inputs** — Scanning media files...",
        "status_no_inputs": "**System Status: Notice** — No valid input media files found.",
        "status_calculating_progress": "**System Status: Preparing Frames** — Calculating total video frames...",
        "status_starting_inference": "**System Status: Processing** — Running deepfake inference...",
        "status_complete": "**System Status: Completed** — Evaluation and video AUROC metrics ready.",
        "progress_desc": "Processing frames ({current}/{total})",
        "input_preview_label": "Input Media Preview",
        "output_preview_label": "Annotated Output Preview",
        "results_title": "Detection Results & Video AUROC Evaluation",
        "copy_btn": "Copy Table",
        "export_btn": "Export CSV",
        "summary_overview": "**Overview:** {count} files processed | **Avg Fake:** `{avg:.4f}` | **Median Fake:** `{med:.4f}`",
        "auc_title": "### Video AUROC & EER Metrics:",
        "auc_mean": "- **Video AUC (Mean Aggregation):** `{auc:.4f}` ({pct:.2f}%)",
        "auc_median": "- **Video AUC (Median Aggregation):** `{auc:.4f}` ({pct:.2f}%)",
        "eer_text": "- **Video EER (Equal Error Rate):** `{eer:.4f}` ({pct:.2f}%)",
        "auc_details": "- **Details:** Evaluated on {count} videos with inferred ground-truth ({real} Real, {fake} Fake)",
        "auc_info_title": "### Video AUROC Information:",
        "auc_info_p1": "- You are processing **{count} video(s)**, all having ground-truth label **{cls}**.",
        "auc_info_p2": "- By statistical definition, AUROC measures separation between two distributions (Real vs Fake). Therefore, calculating Video AUC requires **at least 1 Real video and 1 Fake video**.",
        "auc_info_p3": "- **To view Video AUC:** Upload or provide a folder containing both Real and Fake videos (or run `python evaluate_video_auc.py`).",
        "table_summary_row": "[VIDEO AUC SUMMARY]",
        "table_info_row": "[VIDEO AUC INFO]",
        "correct": "CORRECT",
        "incorrect": "INCORRECT",
        "unknown": "UNKNOWN",
        "need_two_classes": "Need >= 2 classes",
        "real_and_fake": "Real & Fake",
        "current_has": "Current: {cls}",
        "need_rf": "Need Real+Fake",
    },
    "Tiếng Việt": {
        "app_title": "BiasConsist",
        "app_badge": "Nền tảng Đánh giá",
        "app_subtitle": "Hệ thống Phát hiện & So sánh Đối chuẩn Mô hình Deepfake",
        "language_label": "Ngôn ngữ",
        "sidebar_config_title": "Cấu hình phân tích",
        "model_source_label": "Kiến trúc mô hình",
        "bias_ckpt_label": "Đường dẫn Checkpoint BiasConsist",
        "gend_model_label": "Kiến trúc mô hình GenD",
        "effort_ckpt_label": "Đường dẫn Checkpoint Effort",
        "forada_ckpt_label": "Đường dẫn Checkpoint ForAda",
        "local_ckpt_label": "Đường dẫn Checkpoint cục bộ",
        "upload_label": "Tải tệp đa phương tiện (MP4, AVI, MOV, MKV, JPG, PNG, WEBP)",
        "advanced_settings": "Cài đặt phát hiện nâng cao",
        "face_thresh_label": "Ngưỡng phát hiện khuôn mặt",
        "scale_label": "Tỉ lệ cắt viền khuôn mặt",
        "target_size_label": "Kích thước khuôn mặt (px, -1 để giữ nguyên)",
        "stride_label": "Bước nhảy khung hình video (stride)",
        "max_frames_label": "Số khung hình tối đa mỗi video (-1 là tất cả)",
        "max_faces_label": "Số khuôn mặt tối đa mỗi khung hình",
        "run_btn": "Bắt đầu phát hiện",
        "status_ready": "**Trạng thái hệ thống: Sẵn sàng** — Chọn mô hình, tải tệp cần kiểm tra và nhấn **Bắt đầu phát hiện**.",
        "status_loading_model": "**Trạng thái hệ thống: Đang tải mô hình** — Khởi tạo trọng số và bộ trích xuất đặc trưng...",
        "status_loading_detector": "**Trạng thái hệ thống: Khởi tạo detector** — Đang nạp mô hình RetinaFace...",
        "status_collecting_inputs": "**Trạng thái hệ thống: Thu thập tệp đầu vào** — Đang quét danh sách tệp...",
        "status_no_inputs": "**Trạng thái hệ thống: Chú ý** — Không tìm thấy tệp đầu vào hợp lệ.",
        "status_calculating_progress": "**Trạng thái hệ thống: Chuẩn bị khung hình** — Đang tính toán tổng số frame...",
        "status_starting_inference": "**Trạng thái hệ thống: Đang xử lý** — Bắt đầu phân tích phát hiện deepfake...",
        "status_complete": "**Trạng thái hệ thống: Hoàn thành** — Đã tính toán xong kết quả và chỉ số Video AUROC.",
        "progress_desc": "Đang xử lý khung hình ({current}/{total})",
        "input_preview_label": "Xem trước tệp đầu vào",
        "output_preview_label": "Xem trước kết quả gắn nhãn",
        "results_title": "Kết quả phát hiện & Đánh giá Video AUROC",
        "copy_btn": "Sao chép bảng",
        "export_btn": "Xuất CSV",
        "summary_overview": "**Tổng quan:** {count} tệp đã xử lý | **Avg Fake:** `{avg:.4f}` | **Median Fake:** `{med:.4f}`",
        "auc_title": "### Chỉ số Video AUROC & EER:",
        "auc_mean": "- **Video AUC (Mean Aggregation):** `{auc:.4f}` ({pct:.2f}%)",
        "auc_median": "- **Video AUC (Median Aggregation):** `{auc:.4f}` ({pct:.2f}%)",
        "eer_text": "- **Video EER (Equal Error Rate):** `{eer:.4f}` ({pct:.2f}%)",
        "auc_details": "- **Chi tiết:** Tính trên {count} video nhận diện được nhãn ({real} Real, {fake} Fake)",
        "auc_info_title": "### Thông tin về Video AUROC:",
        "auc_info_p1": "- Bạn đang xử lý **{count} video**, tất cả đều thuộc nhãn **{cls}**.",
        "auc_info_p2": "- Theo định nghĩa thống kê học máy, AUROC đo lường độ phân tách giữa 2 phân phối (Real vs Fake). Do đó bắt buộc cần **ít nhất 1 video Real và 1 video Fake** để tính được chỉ số AUC.",
        "auc_info_p3": "- **Cách xem Video AUC:** Hãy tải lên hoặc chọn thư mục chứa cả video Real & Fake (hoặc chạy lệnh `python evaluate_video_auc.py`).",
        "table_summary_row": "[TỔNG KẾT VIDEO AUC]",
        "table_info_row": "[THÔNG TIN VIDEO AUC]",
        "correct": "ĐÚNG",
        "incorrect": "SAI",
        "unknown": "CHƯA RÕ",
        "need_two_classes": "Cần >= 2 lớp",
        "real_and_fake": "Real & Fake",
        "current_has": "Đang có: {cls}",
        "need_rf": "Cần Real+Fake",
    },
}

TRANSLATIONS["EN"] = TRANSLATIONS["English"]
TRANSLATIONS["VI"] = TRANSLATIONS["Tiếng Việt"]

torch.set_float32_matmul_precision("high")


class DeepfakeDetector:
    """Handles model loading, caching, and inference for deepfake detection."""

    def __init__(self):
        self.model_cache: Dict[str, Dict] = {}
        self.detector_cache: Dict[float, RetinaFace] = {}

    def _get_dtype(self, precision: str) -> torch.dtype:
        """Determine torch dtype from precision string."""
        precision = (precision or "").lower()
        if DEVICE == "cpu":
            return torch.float32
        if "bf16" in precision:
            return torch.bfloat16
        if "16" in precision:
            return torch.float16
        return torch.float32

    def load_model(
        self, model_source: str, model_id: str
    ) -> Tuple[Union[GenD_Train, GenD_HF, BiasConsistencyDetector, Effort, ForAda], Callable, torch.dtype]:
        """Load and cache the deepfake detection models."""
        cache_key = f"{model_source}::{model_id}::{DEVICE}"
        if cache_key in self.model_cache:
            return (
                self.model_cache[cache_key]["model"],
                self.model_cache[cache_key]["preproc"],
                self.model_cache[cache_key]["dtype"],
            )

        # Clear cache to free memory from previous models
        self.model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if model_source in ("GenD", "Hugging Face"):
            target_id = DEFAULT_GEND_ID if model_id in GEND_MODELS or not model_id else model_id
            model = GenD_HF.from_pretrained(target_id)
            model.eval()
            model.to(DEVICE)
            preproc = model.feature_extractor.preprocess
            dtype = torch.float32
        elif model_source == "BiasConsist":
            ckpt_path = model_id or DEFAULT_BIAS_CKPT
            model = BiasConsistencyDetector.load_from_checkpoint(ckpt_path, device=DEVICE)
            preproc = model.preprocess
            dtype = torch.float32
        elif model_source == "Effort":
            ckpt_path = model_id or DEFAULT_EFFORT_CKPT
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            config = Config()
            model = Effort(config)
            model.load_checkpoint(ckpt_path)
            model.eval()
            model.to(DEVICE)
            preproc = model.get_preprocessing()
            dtype = torch.float32
        elif model_source == "ForAda":
            ckpt_path = model_id or DEFAULT_FORADA_CKPT
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            config = Config()
            model = ForAda(config)
            model.load_checkpoint(ckpt_path)
            model.eval()
            model.to(DEVICE)
            preproc = model.get_preprocessing()
            dtype = torch.float32
        else:
            ckpt_path = model_id
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu")
            hparams = ckpt.get("hyper_parameters", {})
            precision = hparams.get("precision", "32-true")
            dtype = self._get_dtype(precision)

            config = Config(**hparams)
            model = GenD_Train(config)
            model.eval()
            model.load_state_dict(ckpt["state_dict"], strict=True)
            model.to(DEVICE)

            preproc = model.get_preprocessing()

        self.model_cache[cache_key] = {"model": model, "preproc": preproc, "dtype": dtype}
        return model, preproc, dtype

    def load_detector(self, face_thresh: float = 0.5) -> RetinaFace:
        """Load and cache the face detector."""
        face_thresh = float(face_thresh)
        if face_thresh in self.detector_cache:
            return self.detector_cache[face_thresh]
        model = prepare_model(face_thresh)
        self.detector_cache[face_thresh] = model
        return model

    def infer_faces(
        self,
        frame_bgr: np.ndarray,
        detector: RetinaFace,
        model: Union[GenD_Train, GenD_HF],
        preproc: Callable,
        dtype: torch.dtype,
        scale: float = 1.3,
        target_size: Optional[int] = None,
        max_faces: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, float]]:
        """Detect faces and run inference on them."""
        try:
            xyxy, landmarks = detector.detect(frame_bgr)
        except Exception:
            return []

        if xyxy is None or len(xyxy) == 0:
            return []

        # Select faces sorted by area (largest first) when limiting
        indices = list(range(len(xyxy)))
        indices.sort(key=lambda idx: (xyxy[idx][2] - xyxy[idx][0]) * (xyxy[idx][3] - xyxy[idx][1]), reverse=True)
        if max_faces is not None:
            indices = indices[: max(1, max_faces)]

        results = []
        for i in indices:
            lms = landmarks[i]
            try:
                aligned_face, _ = align_face(
                    frame_bgr,
                    lms,
                    target_size=(target_size, target_size) if target_size else None,
                    scale=scale,
                )
            except Exception:
                continue

            # Convert to PIL Image
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(aligned_face)

            with torch.no_grad():
                batch = preproc(pil_img).unsqueeze(0).to(DEVICE)
                if DEVICE == "cuda" and dtype in (torch.float16, torch.bfloat16):
                    batch = batch.to(dtype)

                out = model(batch)

                if hasattr(out, "logits_labels"):
                    probs = out.logits_labels.softmax(dim=-1).detach().cpu().numpy()[0]
                elif isinstance(out, torch.Tensor):
                    probs = out.softmax(dim=-1).detach().cpu().numpy()[0]
                elif isinstance(model, GenD_Train):
                    probs = out.logits_labels.softmax(dim=1).detach().cpu().numpy()[0]
                else:
                    probs = out.softmax(dim=-1).detach().cpu().numpy()[0]

                p_fake = float(probs[1])

            results.append((xyxy[i], p_fake))

        return results

    def annotate_frame(
        self, frame_bgr: np.ndarray, faces: List[Tuple[np.ndarray, float]], avg_fake: Optional[float] = None
    ) -> np.ndarray:
        """Annotate frame with bounding boxes and probabilities."""
        vis = frame_bgr.copy()
        for bbox, p_fake in faces:
            x1, y1, x2, y2 = map(int, bbox[:4])
            # Interpolate color from green (p_fake=0) to red (p_fake=1)
            blue = 0
            green = int(255 * (1 - p_fake))
            red = int(255 * p_fake)
            color = (blue, green, red)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"fake: {p_fake:.3f}"
            org = (x1 + 6, max(20, y1 + 20))
            cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if avg_fake is not None:
            msg = f"Avg fake: {avg_fake:.3f}"
            org = (8, 28)
            cv2.putText(vis, msg, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, msg, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return vis


class MediaProcessor:
    """Handles processing of images and videos."""

    def __init__(self, detector: DeepfakeDetector):
        self.detector = detector

    def process_image(
        self,
        img_path: str,
        detector: RetinaFace,
        model: Union[GenD_Train, GenD_HF],
        preproc: Callable,
        dtype: torch.dtype,
        scale: float,
        target_size: Optional[int],
        out_dir: Path,
        max_faces: Optional[int] = None,
        progress_updater: Optional[Callable[[int], None]] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Process a single image."""
        try:
            img_rgb = iio.imread(img_path)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {img_path} ({e})")

        faces = self.detector.infer_faces(img, detector, model, preproc, dtype, scale, target_size, max_faces)
        p_fake_vals = [pf for _, pf in faces]
        avg_fake = float(np.mean(p_fake_vals)) if p_fake_vals else 0.0
        med_fake = float(np.median(p_fake_vals)) if p_fake_vals else 0.0

        annotated = self.detector.annotate_frame(img, faces, avg_fake)
        out_path = out_dir / (Path(img_path).stem + "_annot.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)

        if progress_updater is not None:
            progress_updater(1)

        metrics = {
            "num_frames": 1,
            "num_faces": float(len(faces)),
            "avg_p_fake": avg_fake,
            "median_p_fake": med_fake,
        }
        return str(out_path), metrics

    def process_video(
        self,
        vid_path: str,
        detector: RetinaFace,
        model: Union[GenD_Train, GenD_HF],
        preproc: Callable,
        dtype: torch.dtype,
        scale: float,
        target_size: Optional[int],
        out_dir: Path,
        stride: int = 1,
        max_frames: int = -1,
        max_faces: Optional[int] = None,
        progress_updater: Optional[Callable[[int], None]] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Process a video."""
        try:
            meta = iio.immeta(vid_path, plugin="pyav")
            orig_fps = float(meta.get("fps", 25.0))
        except Exception:
            orig_fps = 25.0

        out_path = out_dir / (Path(vid_path).stem + "_annot.mp4")
        out_fps = max(1.0, orig_fps / max(1, stride))

        processed = 0
        frame_idx = 0
        p_fake_values: List[float] = []
        total_faces = 0
        writer = None

        try:
            for frame_rgb in iio.imiter(vid_path, plugin="pyav"):
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(
                        str(out_path),
                        fps=out_fps,
                        codec="libx264",
                        quality=None,
                        pixelformat="yuv420p",
                        output_params=["-preset", "fast", "-crf", "23"],
                    )

                faces = self.detector.infer_faces(frame, detector, model, preproc, dtype, scale, target_size, max_faces)
                total_faces += len(faces)
                if faces:
                    p_fake_values.extend([pf for _, pf in faces])
                    running_avg = float(np.mean(p_fake_values))
                    vis = self.detector.annotate_frame(frame, faces, running_avg)
                else:
                    running_avg = float(np.mean(p_fake_values)) if p_fake_values else 0.0
                    vis = self.detector.annotate_frame(frame, [], running_avg)

                # Convert BGR to RGB for imageio
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                writer.append_data(vis_rgb)

                processed += 1
                frame_idx += 1
                if progress_updater is not None:
                    progress_updater(1)
                if max_frames != -1 and processed >= max_frames:
                    break
        finally:
            if writer is not None:
                writer.close()

        avg_fake = float(np.mean(p_fake_values)) if p_fake_values else 0.0
        med_fake = float(np.median(p_fake_values)) if p_fake_values else 0.0

        metrics = {
            "num_frames": float(processed),
            "num_faces": float(total_faces),
            "avg_p_fake": avg_fake,
            "median_p_fake": med_fake,
        }
        return str(out_path), metrics


def collect_inputs(files, folder_path: str) -> List[str]:
    """Collect valid media file paths from uploads and folder."""
    paths: List[str] = []
    if files:
        for f in files:
            p = getattr(f, "name", None) or getattr(f, "path", None) or str(f)
            if p and Path(p).suffix.lower() in VIDEO_EXTS.union(IMAGE_EXTS):
                paths.append(p)

    if folder_path:
        root = Path(folder_path)
        if root.is_dir():
            for ext in sorted(VIDEO_EXTS.union(IMAGE_EXTS)):
                paths.extend(str(p) for p in root.rglob(f"*{ext}"))

    # Deduplicate and sort
    seen = set()
    dedup = []
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


DETECTOR = DeepfakeDetector()


def run_inference(
    lang: str,
    model_source: str,
    gend_model: str,
    bias_ckpt: str,
    effort_ckpt: str,
    forada_ckpt: str,
    local_ckpt: str,
    files,
    # folder_path: str,
    face_thresh: float,
    stride: int,
    max_frames: int,
    scale: float,
    target_size: Optional[int],
    max_faces: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """Main inference function for Gradio."""
    t = TRANSLATIONS.get(lang, TRANSLATIONS["English"])

    if target_size == -1:
        target_size = None

    detector_obj = DETECTOR
    processor = MediaProcessor(detector_obj)

    print("Loading model...")
    yield (
        pd.DataFrame(columns=TABLE_HEADERS),
        t["status_loading_model"],
        None,
        None,
    )

    if model_source in ("GenD", "Hugging Face"):
        model_id = DEFAULT_GEND_ID
    elif model_source == "BiasConsist":
        model_id = bias_ckpt
    elif model_source == "Effort":
        model_id = effort_ckpt
    elif model_source == "ForAda":
        model_id = forada_ckpt
    else:
        model_id = local_ckpt

    try:
        model, preproc, dtype = detector_obj.load_model(model_source, model_id)
    except FileNotFoundError as e:
        yield (
            pd.DataFrame(columns=TABLE_HEADERS),
            f"**Notice: Model Checkpoint Not Found** — `{str(e)}`\n\n"
            f"> **Tip**: You can select **GenD** under *Model Architecture* to test immediately (weights download automatically from Hugging Face), or place the checkpoint file in `{model_id}`.",
            None,
            None,
        )
        return

    print("Loading face detector...")
    yield (
        pd.DataFrame(columns=TABLE_HEADERS),
        t["status_loading_detector"],
        None,
        None,
    )
    detector = detector_obj.load_detector(face_thresh)

    print("Collecting inputs...")
    yield (
        pd.DataFrame(columns=TABLE_HEADERS),
        t["status_collecting_inputs"],
        None,
        None,
    )

    inputs = collect_inputs(files, None)
    if not inputs:
        empty_df = pd.DataFrame(columns=TABLE_HEADERS)
        yield (
            empty_df,
            t["status_no_inputs"],
            None,
            None,
        )
        return

    # Calculate total progress units (frames for videos, 1 for images)
    print("Calculating total progress...")
    yield (
        pd.DataFrame(columns=TABLE_HEADERS),
        t["status_calculating_progress"],
        None,
        None,
    )
    total_progress_units = 0
    for p in inputs:
        if is_image(p):
            total_progress_units += 1
        elif is_video(p):
            try:
                props = iio.improps(p, plugin="pyav")
                frame_count = props.shape[0]
                processed_frames = frame_count // max(1, stride)
                if max_frames != -1:
                    processed_frames = min(max_frames, processed_frames)
                total_progress_units += max(1, processed_frames)
            except Exception:
                total_progress_units += 1

    total_progress_units = max(1, total_progress_units)

    current_progress = 0

    def advance_progress(step: int = 1) -> None:
        nonlocal current_progress
        current_progress = min(total_progress_units, current_progress + step)
        fraction = current_progress / total_progress_units if total_progress_units else 1.0
        progress(
            fraction,
            desc=t["progress_desc"].format(current=current_progress, total=total_progress_units),
        )

    progress(0.0, desc=t["progress_desc"].format(current=0, total=total_progress_units))
    print("Starting inference...")
    yield (
        pd.DataFrame(columns=TABLE_HEADERS),
        t["status_starting_inference"],
        None,
        None,
    )

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup directories
    inputs_dir = OUTPUT_DIR / "inputs"
    outputs_dir = OUTPUT_DIR / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    output_files = []
    processed_inputs = []

    for idx, p in enumerate(inputs):
        # Copy input to inputs_dir
        try:
            p_path = Path(p)
            unique_name = f"{p_path.stem}_{uuid.uuid4().hex[:8]}{p_path.suffix}"
            new_input_path = inputs_dir / unique_name
            shutil.copy2(p, new_input_path)
            p = str(new_input_path)
        except Exception as e:
            print(f"Failed to copy input {p}: {e}")

        processed_inputs.append(p)

        try:
            if is_video(p):
                out_p, metrics = processor.process_video(
                    p,
                    detector,
                    model,
                    preproc,
                    dtype,
                    scale,
                    target_size,
                    outputs_dir,
                    stride,
                    max_frames,
                    max_faces if max_faces > 0 else None,
                    advance_progress,
                )
            elif is_image(p):
                out_p, metrics = processor.process_image(
                    p,
                    detector,
                    model,
                    preproc,
                    dtype,
                    scale,
                    target_size,
                    outputs_dir,
                    max_faces if max_faces > 0 else None,
                    advance_progress,
                )
            else:
                continue

            rows.append({"input": p, "output": out_p, **metrics})
            output_files.append(out_p)

        except Exception as e:
            print(f"Error processing {p}: {e}")
            rows.append(
                {
                    "input": p,
                    "output": "",
                    "num_frames": 0,
                    "num_faces": 0,
                    "avg_p_fake": 0.0,
                    "median_p_fake": 0.0,
                    "error": str(e),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty and "input" in df.columns:
        df = df.sort_values("input").reset_index(drop=True)

    # Log to CSV
    log_file = OUTPUT_DIR / "inference_log.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_file.exists()
    df.to_csv(log_file, mode="a", header=write_header, index=False)

    # Prepare display DataFrame
    display_df = df.copy()
    labels = []
    scores_avg = []
    scores_med = []

    if not display_df.empty:
        display_df["input"] = display_df["input"].apply(lambda x: Path(x).name)
        if "output" in display_df.columns:
            display_df = display_df.drop(columns=["output"])
        if "error" in display_df.columns:
            display_df = display_df.drop(columns=["error"])

        # Determine predictions and ground-truth labels
        predictions = []
        ground_truths = []
        results = []

        for idx, row in df.iterrows():
            avg_p = float(row.get("avg_p_fake", 0.0))
            med_p = float(row.get("median_p_fake", 0.0))
            pred_bin = 1 if avg_p >= 0.5 else 0
            pred_str = f"FAKE ({avg_p:.2%})" if pred_bin == 1 else f"REAL ({(1.0 - avg_p):.2%})"
            predictions.append(pred_str)

            raw_path = str(row.get("input", ""))
            lbl = infer_label_from_path(raw_path)
            if lbl is not None:
                labels.append(lbl)
                scores_avg.append(avg_p)
                scores_med.append(med_p)
                gt_str = "FAKE (1)" if lbl == 1 else "REAL (0)"
                res_str = t["correct"] if pred_bin == lbl else t["incorrect"]
            else:
                gt_str = t["unknown"]
                res_str = "-"
            ground_truths.append(gt_str)
            results.append(res_str)

        display_df["prediction"] = predictions
        display_df["ground_truth"] = ground_truths
        display_df["result"] = results

        # Format numeric columns for clean viewing
        if "avg_p_fake" in display_df.columns:
            display_df["avg_p_fake"] = display_df["avg_p_fake"].apply(lambda v: f"{float(v):.4f}" if pd.notnull(v) else "0.0000")
        if "median_p_fake" in display_df.columns:
            display_df["median_p_fake"] = display_df["median_p_fake"].apply(lambda v: f"{float(v):.4f}" if pd.notnull(v) else "0.0000")
        if "num_frames" in display_df.columns:
            display_df["num_frames"] = display_df["num_frames"].apply(lambda v: str(int(v)) if pd.notnull(v) else "0")
        if "num_faces" in display_df.columns:
            display_df["num_faces"] = display_df["num_faces"].apply(lambda v: str(int(v)) if pd.notnull(v) else "0")

    final_status = f"{t['status_complete']}\n\n"
    summary = []
    if not df.empty and "avg_p_fake" in df.columns:
        overall_avg = float(df["avg_p_fake"].mean())
        overall_med = float(df["median_p_fake"].median())
        summary.append(t["summary_overview"].format(count=len(df), avg=overall_avg, med=overall_med))

        # Video AUC & EER Evaluation
        if len(labels) >= 2 and len(set(labels)) == 2:
            auc_avg, _, _, _ = calculate_video_auc(labels, scores_avg)
            auc_med, _, _, _ = calculate_video_auc(labels, scores_med)
            y_score_2d = np.column_stack([1.0 - np.array(scores_avg), np.array(scores_avg)])
            eer_avg = calculate_eer(labels, y_score_2d)
            num_real = sum(1 for l in labels if l == 0)
            num_fake = sum(1 for l in labels if l == 1)

            # Append prominent summary row to Table
            summary_row = {
                "input": t["table_summary_row"],
                "num_frames": f"{len(df)} files",
                "num_faces": "-",
                "avg_p_fake": f"AUC={auc_avg:.4f}",
                "median_p_fake": f"AUC={auc_med:.4f}",
                "prediction": f"EER={eer_avg:.4f}",
                "ground_truth": f"{num_real}R / {num_fake}F",
                "result": f"AUC: {auc_avg:.2%}",
            }
            display_df = pd.concat([display_df, pd.DataFrame([summary_row])], ignore_index=True)

            summary.append(
                f"{t['auc_title']}\n"
                f"{t['auc_mean'].format(auc=auc_avg, pct=auc_avg*100)}\n"
                f"{t['auc_median'].format(auc=auc_med, pct=auc_med*100)}\n"
                f"{t['eer_text'].format(eer=eer_avg, pct=eer_avg*100)}\n"
                f"{t['auc_details'].format(count=len(labels), real=num_real, fake=num_fake)}"
            )
        elif len(labels) > 0 and len(set(labels)) < 2:
            current_cls = "FAKE (1)" if labels[0] == 1 else "REAL (0)"
            info_row = {
                "input": t["table_info_row"],
                "num_frames": f"{len(df)} files",
                "num_faces": "-",
                "avg_p_fake": t["need_two_classes"],
                "median_p_fake": t["real_and_fake"],
                "prediction": "-",
                "ground_truth": t["current_has"].format(cls=current_cls),
                "result": t["need_rf"],
            }
            display_df = pd.concat([display_df, pd.DataFrame([info_row])], ignore_index=True)

            summary.append(
                f"{t['auc_info_title']}\n"
                f"{t['auc_info_p1'].format(count=len(labels), cls=current_cls)}\n"
                f"{t['auc_info_p2']}\n"
                f"{t['auc_info_p3']}"
            )

    # Reorder columns to match TABLE_HEADERS strictly
    display_df = display_df[[col for col in TABLE_HEADERS if col in display_df.columns]]

    if summary:
        final_status += "\n\n".join(summary)

    progress(1.0, desc=t["progress_desc"].format(current=total_progress_units, total=total_progress_units))

    print("Inference complete!")
    yield (
        display_df,
        final_status,
        processed_inputs,
        output_files,
    )


def get_thumbnail(path: str) -> Optional[str]:
    """Get thumbnail image path for preview (image itself or first frame of video)."""
    if is_image(path):
        return path
    if is_video(path):
        return path
    return None


def get_all_inputs(files, folder_path):
    """Get all input paths for preview."""
    return collect_inputs(files, folder_path)


CUSTOM_THEME = gr.themes.Soft(
    font=[
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "Roboto",
        "Lato",
        "Helvetica",
        "Arial",
        "sans-serif",
    ],
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    spacing_size=gr.themes.sizes.spacing_md,
).set(
    body_background_fill="#F8FAFC",
    body_text_color="#0F172A",
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#E2E8F0",
    block_radius="12px",
    block_shadow="0 1px 3px 0 rgba(0, 0, 0, 0.04)",
    button_primary_background_fill="#4F46E5",
    button_primary_background_fill_hover="#4338CA",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#4F46E5",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F8FAFC",
    button_secondary_text_color="#334155",
    button_secondary_border_color="#E2E8F0",
    input_background_fill="#FFFFFF",
    input_border_color="#CBD5E1",
    input_radius="10px",
    table_border_color="#E2E8F0",
    table_even_background_fill="#FFFFFF",
    table_odd_background_fill="#F8FAFC",
)

CUSTOM_CSS = """
/* Typography & System Font */
*, *::before, *::after {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Lato, Helvetica, Arial, sans-serif !important;
}

body, .gradio-container {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    font-size: 0.875rem !important;
    line-height: 1.5 !important;
}

.gradio-container {
    max-width: 1480px !important;
    padding: 16px 28px !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #0F172A !important;
    letter-spacing: -0.015em !important;
    margin-top: 0 !important;
}

/* Top Navbar Strip */
.top-navbar {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    align-items: center !important;
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-between !important;
}

.navbar-brand {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    flex: 1 !important;
}

.navbar-actions {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    min-width: unset !important;
    width: auto !important;
    flex-shrink: 0 !important;
}

.brand-markdown h1 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    margin: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
}

.brand-markdown .badge {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    background: #EEF2FF !important;
    color: #4F46E5 !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
    border: 1px solid #E0E7FF !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
}

.brand-markdown p {
    font-size: 0.82rem !important;
    color: #64748B !important;
    margin: 2px 0 0 0 !important;
}

/* Navbar Language Selector: Sleek Flat Segmented Control [ EN | VI ] */
.nav-lang-toggle {
    background: #F1F5F9 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 2px !important;
    display: inline-flex !important;
    width: auto !important;
    min-width: unset !important;
    margin: 0 !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.03) !important;
}

.nav-lang-toggle > .wrap,
.nav-lang-toggle .wrap {
    display: inline-flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 2px !important;
    padding: 0 !important;
    margin: 0 !important;
    width: auto !important;
}

.nav-lang-toggle input[type="radio"] {
    display: none !important;
}

.nav-lang-toggle label {
    padding: 4px 12px !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    border: none !important;
    background: transparent !important;
    color: #64748B !important;
    transition: all 0.15s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: 38px !important;
    text-align: center !important;
    margin: 0 !important;
    line-height: 1.25 !important;
    user-select: none !important;
    box-shadow: none !important;
}

.nav-lang-toggle label:hover {
    color: #0F172A !important;
    background-color: rgba(255, 255, 255, 0.6) !important;
}

.nav-lang-toggle label.selected {
    background-color: #FFFFFF !important;
    color: #4F46E5 !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

.nav-lang-toggle label span {
    margin: 0 !important;
    padding: 0 !important;
}

/* Main Layout Row */
.main-layout-row {
    gap: 20px !important;
}

/* Left Sidebar Panel: Seamless White Container */
.sidebar-panel {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    padding: 22px 20px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}

/* Remove Card-in-Card nesting inside sidebar */
.sidebar-panel .block,
.sidebar-panel fieldset,
.sidebar-panel .gr-group,
.sidebar-panel .gr-box,
.sidebar-panel .gr-panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.sidebar-section-title {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #94A3B8 !important;
    margin-bottom: 8px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #F1F5F9 !important;
}

.sidebar-section-title h3 {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: #64748B !important;
    margin: 0 !important;
}

/* Inputs, Textboxes, Dropdowns */
input, textarea, select, .gr-input {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #0F172A !important;
    font-size: 0.875rem !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) inset !important;
    transition: all 0.15s ease !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
}

/* Sidebar Dropdown */
.sidebar-dropdown {
    margin-bottom: 14px !important;
}

/* Sidebar File Upload */
.sidebar-file-upload {
    border: 1px dashed #CBD5E1 !important;
    border-radius: 10px !important;
    background-color: #F8FAFC !important;
    padding: 12px !important;
    transition: all 0.2s ease !important;
    margin-bottom: 14px !important;
}
.sidebar-file-upload:hover {
    border-color: #6366F1 !important;
    background-color: #EEF2FF !important;
}

/* Primary Run Button in Sidebar */
.sidebar-run-btn {
    width: 100% !important;
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 12px -2px rgba(79, 70, 229, 0.35) !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin: 4px 0 12px 0 !important;
}
.sidebar-run-btn:hover {
    background: #4338CA !important;
    box-shadow: 0 6px 16px -2px rgba(79, 70, 229, 0.45) !important;
    transform: translateY(-1px) !important;
}
.sidebar-run-btn:active {
    transform: translateY(0) !important;
}

/* Sidebar Accordion */
.sidebar-accordion {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    background: #F8FAFC !important;
    overflow: hidden !important;
    margin-top: 6px !important;
}
.sidebar-accordion > .label-wrap {
    background: #F8FAFC !important;
    padding: 10px 14px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    border-bottom: 1px solid #E2E8F0 !important;
}

/* Main Stage Column */
.main-stage-column {
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}

/* Status Alert Strip */
.status-strip {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 4px solid #4F46E5 !important;
    border-radius: 10px !important;
    padding: 12px 18px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
}
.status-strip p {
    margin: 0 !important;
    color: #334155 !important;
    font-size: 0.875rem !important;
    line-height: 1.45 !important;
}
.status-strip strong {
    color: #0F172A !important;
}

/* Preview Row & Galleries: Fixed height control (390px) to prevent vertical ballooning */
.preview-row {
    gap: 16px !important;
}

.preview-gallery {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    background-color: #FFFFFF !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    height: 390px !important;
    max-height: 390px !important;
    min-height: 390px !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Override Gradio large-screen min-height: 450px rule */
.preview-gallery .fixed-height {
    min-height: unset !important;
    height: 100% !important;
    max-height: 390px !important;
}

.preview-gallery .preview {
    max-height: 390px !important;
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background-color: #0F172A !important;
}

.preview-gallery .media-button {
    height: calc(100% - 50px) !important;
    max-height: 330px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Strict constraint on portrait images and 9:16 vertical videos */
.preview-gallery img,
.preview-gallery video {
    max-height: 320px !important;
    width: auto !important;
    max-width: 100% !important;
    object-fit: contain !important;
    margin: 0 auto !important;
}

.preview-gallery .thumbnails {
    max-height: 55px !important;
}

/* Results Panel Card */
.results-card {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    padding: 18px 22px !important;
    margin-top: 8px !important;
}

.results-header-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 14px !important;
    border-bottom: 1px solid #F1F5F9 !important;
    padding-bottom: 10px !important;
}

.results-title-text h3 {
    margin: 0 !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

.results-btn-group {
    display: flex !important;
    gap: 8px !important;
    justify-content: flex-end !important;
}

.table-action-btn {
    background-color: #FFFFFF !important;
    color: #475569 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 5px 12px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    transition: all 0.15s ease !important;
}
.table-action-btn:hover {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    border-color: #CBD5E1 !important;
}

/* Modern Dataframe / Table: Full horizontal scroll without header truncation */
.modern-table {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    overflow-x: auto !important;
    max-width: 100% !important;
}

/* Enable smooth horizontal scrolling on all Gradio dataframe wrapper levels */
.dataframe-wrap,
.table-wrap,
.table-container,
.modern-table .table-wrap,
.modern-table .dataframe-wrap,
.modern-table .table-container {
    overflow-x: auto !important;
    max-width: 100% !important;
    scrollbar-width: thin !important;
}

.modern-table table,
table {
    border-collapse: collapse !important;
    width: max-content !important;
    min-width: 100% !important;
    font-size: 0.85rem !important;
}

.modern-table thead,
.modern-table tbody,
.modern-table tfoot {
    table-layout: auto !important;
    width: 100% !important;
}

/* Prevent headers and cells from truncating with ellipsis */
th, td,
.modern-table th,
.modern-table td {
    white-space: nowrap !important;
    min-width: 110px !important;
}

/* Extra space for input media path/filename */
th:first-child, td:first-child,
.modern-table th:first-child,
.modern-table td:first-child {
    min-width: 220px !important;
}

.modern-table th {
    background-color: #F8FAFC !important;
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #E2E8F0 !important;
}

.modern-table th span,
.modern-table td span,
.modern-table .cell-wrap {
    white-space: nowrap !important;
    overflow: visible !important;
}

.modern-table td {
    padding: 10px 14px !important;
    border-bottom: 1px solid #F1F5F9 !important;
    color: #1E293B !important;
}

.modern-table tr:hover td {
    background-color: #F8FAFC !important;
}
"""


def build_ui():
    """Build the Gradio interface."""
    t_en = TRANSLATIONS["English"]

    with gr.Blocks(title="BiasConsist - Deepfake Detection & Benchmark Platform", theme=CUSTOM_THEME, css=CUSTOM_CSS) as demo:
        # Top Navigation Bar: Seamless white strip with brand and compact language switch
        with gr.Row(elem_classes=["top-navbar"]):
            with gr.Column(scale=10, elem_classes=["navbar-brand"]):
                app_title_md = gr.Markdown(
                    f"<h1>BiasConsist <span class='badge'>{t_en['app_badge']}</span></h1><p>{t_en['app_subtitle']}</p>",
                    elem_classes=["brand-markdown"],
                )
            with gr.Column(scale=2, min_width=90, elem_classes=["navbar-actions"]):
                lang_radio = gr.Radio(
                    choices=["EN", "VI"],
                    value="EN",
                    label="Language",
                    show_label=False,
                    container=False,
                    elem_classes=["nav-lang-toggle"],
                )

        with gr.Row(elem_classes=["main-layout-row"]):
            # ================= LEFT COLUMN: Unified Sidebar (White, Seamless, No Card-in-Card) =================
            with gr.Column(scale=4, min_width=360, elem_classes=["sidebar-panel"]):
                sidebar_title = gr.Markdown(f"### {t_en['sidebar_config_title']}", elem_classes=["sidebar-section-title"])

                model_source = gr.Dropdown(
                    ["BiasConsist", "GenD", "Effort", "ForAda", "Local Checkpoint"],
                    label=t_en["model_source_label"],
                    value="BiasConsist",
                    interactive=True,
                    elem_classes=["sidebar-dropdown"],
                )

                files = gr.Files(
                    label=t_en["upload_label"],
                    file_count="multiple",
                    file_types=["video", "image", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".bmp", ".webp"],
                    elem_classes=["sidebar-file-upload"],
                )

                # Action Button: Primary Indigo, right beneath file upload
                run_btn = gr.Button(
                    t_en["run_btn"],
                    variant="primary",
                    size="lg",
                    elem_classes=["sidebar-run-btn"],
                )

                # Advanced Settings: Collapsible, holds Checkpoints and Threshold sliders
                with gr.Accordion(t_en["advanced_settings"], open=False, elem_classes=["sidebar-accordion"]) as advanced_accordion:
                    # Model Checkpoint Textboxes (Moved here out of main view!)
                    bias_ckpt = gr.Textbox(label=t_en["bias_ckpt_label"], value=DEFAULT_BIAS_CKPT, visible=True)
                    gend_model = gr.Dropdown(GEND_MODELS, label=t_en["gend_model_label"], value=GEND_MODELS[0], visible=False)
                    effort_ckpt = gr.Textbox(label=t_en["effort_ckpt_label"], value=DEFAULT_EFFORT_CKPT, visible=False)
                    forada_ckpt = gr.Textbox(label=t_en["forada_ckpt_label"], value=DEFAULT_FORADA_CKPT, visible=False)
                    local_ckpt = gr.Textbox(label=t_en["local_ckpt_label"], value=DEFAULT_CKPT, visible=False)

                    face_thresh = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label=t_en["face_thresh_label"])
                    scale = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label=t_en["scale_label"])
                    target_size = gr.Number(value=-1, precision=0, label=t_en["target_size_label"])
                    stride = gr.Slider(1, 10, value=1, step=1, label=t_en["stride_label"])
                    max_frames = gr.Number(value=-1, precision=0, label=t_en["max_frames_label"])
                    max_faces = gr.Slider(1, 10, value=1, step=1, label=t_en["max_faces_label"])

            # ================= RIGHT COLUMN: Main Stage (Preview & Results) =================
            with gr.Column(scale=8, min_width=580, elem_classes=["main-stage-column"]):
                # Compact Status Alert Strip
                with gr.Group(elem_classes=["status-strip"]):
                    status_summary = gr.Markdown(t_en["status_ready"])

                # Preview Galleries with Integrated Native Headers
                with gr.Row(elem_classes=["preview-row"]):
                    with gr.Column(scale=1):
                        input_gallery = gr.Gallery(
                            label=t_en["input_preview_label"],
                            show_label=True,
                            columns=1,
                            object_fit="contain",
                            height=390,
                            preview=True,
                            selected_index=0,
                            elem_classes=["preview-gallery"],
                        )
                    with gr.Column(scale=1):
                        output_gallery = gr.Gallery(
                            label=t_en["output_preview_label"],
                            show_label=True,
                            columns=1,
                            object_fit="contain",
                            height=390,
                            preview=True,
                            selected_index=0,
                            elem_classes=["preview-gallery"],
                        )

                # Results Panel
                with gr.Column(elem_classes=["results-card"]):
                    with gr.Row(elem_classes=["results-header-row"]):
                        results_title = gr.Markdown(f"### {t_en['results_title']}", elem_classes=["results-title-text"])
                        with gr.Row(elem_classes=["results-btn-group"]):
                            copy_btn = gr.Button(t_en["copy_btn"], size="sm", elem_classes=["table-action-btn"])
                            export_btn = gr.Button(t_en["export_btn"], size="sm", elem_classes=["table-action-btn"])

                    table = gr.Dataframe(
                        value=pd.DataFrame(columns=TABLE_HEADERS),
                        headers=TABLE_HEADERS,
                        wrap=False,
                        column_widths=["220px", "110px", "110px", "120px", "130px", "120px", "120px", "110px"],
                        interactive=False,
                        elem_classes=["modern-table"],
                    )

        def change_language(lang):
            t = TRANSLATIONS.get(lang, TRANSLATIONS["English"])
            return (
                f"<h1>BiasConsist <span class='badge'>{t['app_badge']}</span></h1><p>{t['app_subtitle']}</p>",
                f"### {t['sidebar_config_title']}",
                gr.update(label=t["model_source_label"]),
                gr.update(label=t["bias_ckpt_label"]),
                gr.update(label=t["gend_model_label"]),
                gr.update(label=t["effort_ckpt_label"]),
                gr.update(label=t["forada_ckpt_label"]),
                gr.update(label=t["local_ckpt_label"]),
                gr.update(label=t["upload_label"]),
                gr.update(label=t["advanced_settings"]),
                gr.update(label=t["face_thresh_label"]),
                gr.update(label=t["scale_label"]),
                gr.update(label=t["target_size_label"]),
                gr.update(label=t["stride_label"]),
                gr.update(label=t["max_frames_label"]),
                gr.update(label=t["max_faces_label"]),
                gr.update(value=t["run_btn"]),
                t["status_ready"],
                gr.update(label=t["input_preview_label"]),
                gr.update(label=t["output_preview_label"]),
                f"### {t['results_title']}",
                gr.update(value=t["copy_btn"]),
                gr.update(value=t["export_btn"]),
            )

        lang_radio.change(
            fn=change_language,
            inputs=lang_radio,
            outputs=[
                app_title_md,
                sidebar_title,
                model_source,
                bias_ckpt,
                gend_model,
                effort_ckpt,
                forada_ckpt,
                local_ckpt,
                files,
                advanced_accordion,
                face_thresh,
                scale,
                target_size,
                stride,
                max_frames,
                max_faces,
                run_btn,
                status_summary,
                input_gallery,
                output_gallery,
                results_title,
                copy_btn,
                export_btn,
            ],
        )

        copy_btn.click(
            fn=None,
            inputs=[table],
            js="""(table_data) => {
                if (!table_data) return;
                const headers = table_data.headers;
                const data = table_data.data;
                if (!headers || !data) return;
                let text = headers.join(",") + "\\n";
                data.forEach(row => {
                    text += row.join(",") + "\\n";
                });
                navigator.clipboard.writeText(text);
            }""",
        )

        export_btn.click(
            fn=None,
            inputs=[table],
            js="""(table_data) => {
                if (!table_data) return;
                const headers = table_data.headers;
                const data = table_data.data;
                if (!headers || !data) return;
                let text = headers.join(",") + "\\n";
                data.forEach(row => {
                    text += row.join(",") + "\\n";
                });
                const blob = new Blob([text], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'results.csv';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }""",
        )

        def update_model_input(source):
            return (
                gr.update(visible=(source == "BiasConsist")),
                gr.update(visible=(source == "GenD")),
                gr.update(visible=(source == "Effort")),
                gr.update(visible=(source == "ForAda")),
                gr.update(visible=(source == "Local Checkpoint")),
            )

        model_source.change(
            fn=update_model_input,
            inputs=model_source,
            outputs=[bias_ckpt, gend_model, effort_ckpt, forada_ckpt, local_ckpt],
        )

        run_btn.click(
            fn=run_inference,
            inputs=[
                lang_radio,
                model_source,
                gend_model,
                bias_ckpt,
                effort_ckpt,
                forada_ckpt,
                local_ckpt,
                files,
                # folder,
                face_thresh,
                stride,
                max_frames,
                scale,
                target_size,
                max_faces,
            ],
            outputs=[
                table,
                status_summary,
                input_gallery,
                output_gallery,
            ],
        )

        # Update input preview on change
        def update_previews(files_in, folder_in=None):
            return get_all_inputs(files_in, folder_in)

        files.change(
            fn=update_previews,
            inputs=[
                files,
                # folder,
            ],
            outputs=input_gallery,
        )
        # folder.change(fn=update_previews, inputs=[files, folder], outputs=input_gallery)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    returns = ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        #share=True,
    )
    print("Gradio UI launched. Returns:", returns)
