import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Đảm bảo đường dẫn gốc của dự án được thêm vào sys.path và thiết lập làm thư mục làm việc hiện tại
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Đảm bảo terminal console trên Windows hiển thị đúng bảng mã UTF-8 tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thiết lập thư mục tạm thời cho Gradio
os.environ["GRADIO_TEMP_DIR"] = "./tmp/gradio"

import cv2
import gradio as gr
import imageio
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image

from detector import align_face
from src.config import Config
from src.hf.modeling_gend import GenD as GenD_HF
from src.metrics import calculate_eer, calculate_video_auc, infer_label_from_path
from src.model.BiasConsistency import BiasConsistencyDetector
from src.model.Effort import Effort
from src.model.ForAda import ForAda
from src.model.GenD import GenD as GenD_Train
from src.retinaface import RetinaFace, prepare_model

# Các hằng số hệ thống
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

# Tên biến / cột bảng dữ liệu (giữ nguyên tên biến kỹ thuật theo yêu cầu)
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

# Thông tin trọng số mô hình và cơ chế tự động tìm nạp / tải về
WEIGHTS_INFO = {
    "buffalo_l": {
        "path": "weights/models/buffalo_l/det_10g.onnx",
        "name": "RetinaFace (det_10g.onnx)",
        "url": "https://huggingface.co/datasets/theanhntp/Liblib/resolve/ae4357741af379482690fe3e0f2fa6fd32ba33b4/insightface/models/buffalo_l/det_10g.onnx",
        "alt_paths": [
            "C:/GitHub/BiasConsist/weights/models/buffalo_l/det_10g.onnx",
            "../BiasConsist/weights/models/buffalo_l/det_10g.onnx",
        ],
    },
    "BiasConsist": {
        "path": DEFAULT_BIAS_CKPT,
        "name": "BiasConsist (bias_consistency.pth)",
        "url": None,
        "alt_paths": [
            "C:/GitHub/BiasConsist/weights/BiasConsist/bias_consistency.pth",
            "../BiasConsist/weights/BiasConsist/bias_consistency.pth",
        ],
    },
    "Effort": {
        "path": DEFAULT_EFFORT_CKPT,
        "name": "Effort (effort_clip_L14_trainOn_FaceForensic.pth)",
        "url": None,
        "alt_paths": [
            "C:/GitHub/BiasConsist/weights/Effort/effort_clip_L14_trainOn_FaceForensic.pth",
            "../BiasConsist/weights/Effort/effort_clip_L14_trainOn_FaceForensic.pth",
        ],
    },
    "ForAda": {
        "path": DEFAULT_FORADA_CKPT,
        "name": "ForAda (forada_checkpoint.pth)",
        "url": "https://drive.usercontent.google.com/download?id=1UlaAUTtsX87ofIibf38TtfAKIsnA7WVm&export=download&authuser=0",
        "alt_paths": [
            "C:/GitHub/BiasConsist/weights/ForAda/forada_checkpoint.pth",
            "../BiasConsist/weights/ForAda/forada_checkpoint.pth",
        ],
    },
    "forensics_adapter": {
        "path": "weights/forensics_adapter/ViT-L-14.pt",
        "name": "ForAda Backbone (ViT-L-14.pt)",
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "alt_paths": [
            "C:/GitHub/BiasConsist/weights/forensics_adapter/ViT-L-14.pt",
            "../BiasConsist/weights/forensics_adapter/ViT-L-14.pt",
        ],
    },
}


def ensure_weight(key: str, verbose: bool = True) -> bool:
    """Kiểm tra và tự động liên kết / tải trọng số cho mô hình."""
    info = WEIGHTS_INFO.get(key)
    if not info:
        return True

    target_path = Path(info["path"])
    name = info["name"]

    if target_path.is_file() and target_path.stat().st_size > 0:
        if verbose:
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ [{name}] Đã sẵn sàng ({size_mb:.1f} MB)")
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Tìm trong các thư mục cục bộ thay thế
    for alt in info.get("alt_paths", []):
        alt_path = Path(alt)
        if alt_path.is_file() and alt_path.stat().st_size > 0:
            if verbose:
                print(f"  ⚡ [{name}] Tìm thấy tại '{alt}', đang liên kết...")
            try:
                os.link(str(alt_path), str(target_path))
                if verbose:
                    print(f"  ✓ [{name}] Đã tạo hardlink thành công!")
                return True
            except Exception:
                try:
                    shutil.copy2(str(alt_path), str(target_path))
                    if verbose:
                        print(f"  ✓ [{name}] Đã sao chép thành công!")
                    return True
                except Exception as ce:
                    if verbose:
                        print(f"  ⚠ Không thể sao chép từ '{alt}': {ce}")

    # 2. Tải trực tuyến nếu có URL
    url = info.get("url")
    if url:
        if verbose:
            print(f"  ⬇ [{name}] Đang tải tự động từ: {url} ...")
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_f:
                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and verbose:
                        pct = (downloaded / total_size) * 100
                        print(
                            f"\r     Tiến độ: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
            if verbose:
                print(f"\n  ✓ [{name}] Tải xuống hoàn tất thành công!")
            return True
        except Exception as e:
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
            if verbose:
                print(f"\n  ✗ [{name}] Lỗi khi tải trực tuyến: {e}")

    return False


def check_and_download_all_weights():
    """Tự động kiểm tra và tải trước toàn bộ trọng số khi ứng dụng khởi chạy."""
    print("=" * 65)
    print("  BiasConsist: Tự động kiểm tra & Chuẩn bị trọng số mô hình")
    print("=" * 65)
    for key in WEIGHTS_INFO:
        ensure_weight(key, verbose=True)
    print("  ✓ [GenD] Tự động nạp qua HuggingFace Hub ('yermandy/GenD_CLIP_L_14')")
    print("=" * 65)


# Toàn bộ nhãn giao diện và thông báo người dùng bằng Tiếng Việt
UI_TEXT = {
    "app_title": "BiasConsist",
    "app_badge": "Benchmark",
    "app_subtitle": "Nền tảng Phát hiện Deepfake & Đánh giá Đối chuẩn Đa Mô hình",
    "sidebar_config_title": "Cấu hình phân tích",
    "model_source_label": "Kiến trúc mô hình",
    "bias_ckpt_label": "Đường dẫn Checkpoint BiasConsist",
    "gend_model_label": "Kiến trúc mô hình GenD",
    "effort_ckpt_label": "Đường dẫn Checkpoint Effort",
    "forada_ckpt_label": "Đường dẫn Checkpoint ForAda",
    "local_ckpt_label": "Đường dẫn Checkpoint cục bộ",
    "upload_label": "Tải tệp media (MP4, AVI, MOV, MKV, JPG, PNG, WEBP)",
    "advanced_settings": "Cài đặt phát hiện nâng cao",
    "face_thresh_label": "Ngưỡng phát hiện khuôn mặt (Threshold)",
    "scale_label": "Tỉ lệ cắt viền khuôn mặt (Scale)",
    "target_size_label": "Kích thước khuôn mặt mục tiêu (px, -1 để giữ nguyên)",
    "stride_label": "Bước nhảy khung hình video (Stride)",
    "max_frames_label": "Số khung hình tối đa mỗi video (-1 là tất cả)",
    "max_faces_label": "Số khuôn mặt tối đa mỗi khung hình",
    "run_btn": "Bắt đầu phát hiện",
    "status_ready": "**Trạng thái hệ thống: Sẵn sàng** — Chọn mô hình, tải tệp media cần kiểm tra và nhấn **Bắt đầu phát hiện**.",
    "status_loading_model": "**Trạng thái hệ thống: Đang tải mô hình** — Khởi tạo trọng số và bộ trích xuất đặc trưng...",
    "status_loading_detector": "**Trạng thái hệ thống: Khởi tạo detector** — Đang nạp mô hình RetinaFace...",
    "status_collecting_inputs": "**Trạng thái hệ thống: Thu thập tệp đầu vào** — Đang quét danh sách tệp media...",
    "status_no_inputs": "**Trạng thái hệ thống: Chú ý** — Không tìm thấy tệp media đầu vào hợp lệ.",
    "status_calculating_progress": "**Trạng thái hệ thống: Chuẩn bị khung hình** — Đang tính toán tổng số frame...",
    "status_starting_inference": "**Trạng thái hệ thống: Đang xử lý** — Bắt đầu phân tích phát hiện deepfake...",
    "status_complete": "**Trạng thái hệ thống: Hoàn thành**",
    "progress_desc": "Đang xử lý khung hình ({current}/{total})",
    "input_preview_label": "Xem trước media đầu vào",
    "output_preview_label": "Xem trước kết quả gắn nhãn",
    "results_title": "Kết quả phát hiện",
    "copy_btn": "Sao chép bảng",
    "export_btn": "Xuất CSV",
    "correct": "ĐÚNG",
    "incorrect": "SAI",
    "unknown": "CHƯA RÕ",
}

torch.set_float32_matmul_precision("high")


class DeepfakeDetector:
    """Quản lý việc nạp mô hình, lưu đệm và thực hiện suy luận phát hiện deepfake."""

    def __init__(self):
        self.model_cache: Dict[str, Dict] = {}
        self.detector_cache: Dict[float, RetinaFace] = {}

    def _get_dtype(self, precision: str) -> torch.dtype:
        """Xác định kiểu dữ liệu torch dtype từ chuỗi precision."""
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
        """Nạp và lưu đệm các mô hình phát hiện deepfake."""
        cache_key = f"{model_source}::{model_id}::{DEVICE}"
        if cache_key in self.model_cache:
            return (
                self.model_cache[cache_key]["model"],
                self.model_cache[cache_key]["preproc"],
                self.model_cache[cache_key]["dtype"],
            )

        # Xóa cache bộ nhớ GPU khi đổi mô hình
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
            if not os.path.isfile(ckpt_path):
                ensure_weight("BiasConsist", verbose=True)
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Không tìm thấy checkpoint BiasConsist: {ckpt_path}")
            model = BiasConsistencyDetector.load_from_checkpoint(ckpt_path, device=DEVICE)
            preproc = model.preprocess
            dtype = torch.float32
        elif model_source == "Effort":
            ckpt_path = model_id or DEFAULT_EFFORT_CKPT
            if not os.path.isfile(ckpt_path):
                ensure_weight("Effort", verbose=True)
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Không tìm thấy checkpoint Effort: {ckpt_path}")
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
                ensure_weight("ForAda", verbose=True)
            ensure_weight("forensics_adapter", verbose=True)
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Không tìm thấy checkpoint ForAda: {ckpt_path}")
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
                raise FileNotFoundError(f"Không tìm thấy checkpoint cục bộ: {ckpt_path}")

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
        """Nạp và lưu đệm bộ phát hiện khuôn mặt RetinaFace."""
        face_thresh = float(face_thresh)
        if face_thresh in self.detector_cache:
            return self.detector_cache[face_thresh]
        ensure_weight("buffalo_l", verbose=False)
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
        """Phát hiện khuôn mặt và thực hiện suy luận dự đoán deepfake."""
        try:
            xyxy, landmarks = detector.detect(frame_bgr)
        except Exception:
            return []

        if xyxy is None or len(xyxy) == 0:
            return []

        # Sắp xếp khuôn mặt theo diện tích từ lớn đến nhỏ
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
        """Vẽ bounding box và xác suất p_fake lên khung hình."""
        vis = frame_bgr.copy()
        for bbox, p_fake in faces:
            x1, y1, x2, y2 = map(int, bbox[:4])
            # Chuyển đổi màu từ xanh lá (p_fake=0) sang đỏ (p_fake=1)
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
    """Xử lý hình ảnh và video đầu vào."""

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
        """Xử lý một tệp ảnh đơn lẻ."""
        try:
            img_rgb = iio.imread(img_path)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"Không thể đọc tệp ảnh: {img_path} ({e})")

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
        """Xử lý một tệp video."""
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


def collect_inputs(files, folder_path: str = None) -> List[str]:
    """Thu thập danh sách đường dẫn tệp media hợp lệ."""
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

    # Loại bỏ trùng lặp và giữ thứ tự
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
    model_source: str,
    gend_model: str,
    bias_ckpt: str,
    effort_ckpt: str,
    forada_ckpt: str,
    local_ckpt: str,
    files,
    face_thresh: float,
    stride: int,
    max_frames: int,
    scale: float,
    target_size: Optional[int],
    max_faces: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """Hàm suy luận chính thực thi cho giao diện Gradio."""
    t = UI_TEXT

    if target_size == -1:
        target_size = None

    detector_obj = DETECTOR
    processor = MediaProcessor(detector_obj)

    print("[BiasConsist] Đang tải mô hình...")
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
            f"**Thông báo: Không tìm thấy checkpoint mô hình** — `{str(e)}`\n\n"
            f"> **Mẹo**: Bạn có thể chọn **GenD** trong mục *Kiến trúc mô hình* để kiểm thử ngay lập tức (trọng số tự động tải từ Hugging Face), hoặc đặt tệp checkpoint vào `{model_id}`.",
            None,
            None,
        )
        return

    print("[BiasConsist] Đang khởi tạo detector RetinaFace...")
    yield (
        pd.DataFrame(columns=TABLE_HEADERS),
        t["status_loading_detector"],
        None,
        None,
    )
    detector = detector_obj.load_detector(face_thresh)

    print("[BiasConsist] Thu thập danh sách tệp đầu vào...")
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

    print("[BiasConsist] Tính toán tổng số frame cần xử lý...")
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
    print("[BiasConsist] Bắt đầu xử lý phát hiện deepfake...")
    yield (
        pd.DataFrame(columns=TABLE_HEADERS),
        t["status_starting_inference"],
        None,
        None,
    )

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_dir = OUTPUT_DIR / "inputs"
    outputs_dir = OUTPUT_DIR / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    output_files = []
    processed_inputs = []

    for idx, p in enumerate(inputs):
        try:
            p_path = Path(p)
            unique_name = f"{p_path.stem}_{uuid.uuid4().hex[:8]}{p_path.suffix}"
            new_input_path = inputs_dir / unique_name
            shutil.copy2(p, new_input_path)
            p = str(new_input_path)
        except Exception as e:
            print(f"Lỗi khi sao chép tệp đầu vào {p}: {e}")

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
            print(f"Lỗi khi xử lý {p}: {e}")
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

    # Ghi nhật ký vào CSV
    log_file = OUTPUT_DIR / "inference_log.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_file.exists()
    df.to_csv(log_file, mode="a", header=write_header, index=False)

    # Chuẩn bị DataFrame hiển thị
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

        if "avg_p_fake" in display_df.columns:
            display_df["avg_p_fake"] = display_df["avg_p_fake"].apply(
                lambda v: f"{float(v):.4f}" if pd.notnull(v) else "0.0000"
            )
        if "median_p_fake" in display_df.columns:
            display_df["median_p_fake"] = display_df["median_p_fake"].apply(
                lambda v: f"{float(v):.4f}" if pd.notnull(v) else "0.0000"
            )
        if "num_frames" in display_df.columns:
            display_df["num_frames"] = display_df["num_frames"].apply(
                lambda v: str(int(v)) if pd.notnull(v) else "0"
            )
        if "num_faces" in display_df.columns:
            display_df["num_faces"] = display_df["num_faces"].apply(
                lambda v: str(int(v)) if pd.notnull(v) else "0"
            )

    final_status = f"{t['status_complete']}\n\n"
    if not display_df.empty:
        file_reports = []
        for idx, row in display_df.iterrows():
            fname = row.get("input", "")
            avg_f = row.get("avg_p_fake", "0.0000")
            pred = row.get("prediction", "")
            file_reports.append(f"- **{fname}**: `avg_fake = {avg_f}` ({pred})")
        final_status += "\n".join(file_reports)

    display_df = display_df[[col for col in TABLE_HEADERS if col in display_df.columns]]

    progress(1.0, desc=t["progress_desc"].format(current=total_progress_units, total=total_progress_units))

    print("[BiasConsist] Hoàn tất suy luận!")
    yield (
        display_df,
        final_status,
        processed_inputs,
        output_files,
    )


def get_all_inputs(files, folder_path=None):
    """Thu thập toàn bộ đường dẫn đầu vào cho khung xem trước."""
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
/* Typography & Hệ thống Phông chữ */
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

h1, h2, h3, h4 {
    color: #0F172A !important;
    letter-spacing: -0.015em !important;
    margin-top: 0 !important;
}

/* Thanh Navbar trên cùng */
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

.status-pill {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    background: #ECFDF5 !important;
    color: #059669 !important;
    padding: 5px 12px !important;
    border-radius: 9999px !important;
    border: 1px solid #A7F3D0 !important;
    letter-spacing: 0.02em !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
}

.brand-markdown p {
    font-size: 0.82rem !important;
    color: #64748B !important;
    margin: 2px 0 0 0 !important;
}

/* Bố cục chính */
.main-layout-row {
    gap: 20px !important;
}

/* Cột bên trái: Sidebar cấu hình */
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

/* Ô nhập liệu, danh sách chọn */
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

.sidebar-dropdown {
    margin-bottom: 14px !important;
}

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

/* Nút chạy phân tích chính */
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

/* Cột bên phải: Sân khấu chính hiển thị kết quả */
.main-stage-column {
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}

/* Thanh trạng thái hệ thống */
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

/* Hàng xem trước ảnh / video */
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

/* Thẻ kết quả phân tích */
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

/* =====================================================================================
   SỬA LỖI 2 THANH CUỘN NGANG (DUPLICATE HORIZONTAL SCROLLBAR FIX)
   - Khung bao ngoài (.modern-table) ẩn hoàn toàn overflow thừa
   - Chỉ duy nhất lớp .table-wrap được phép cuộn ngang (overflow-x: auto)
   - Triệt tiêu scrollbar lồng nhau bên trong ở thẻ <table> và <tbody>
   ===================================================================================== */
.modern-table {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

/* Chỉ duy nhất phần tử .table-wrap làm thanh cuộn ngang */
.modern-table .table-wrap,
.modern-table .table-container,
.modern-table .dataframe-wrap {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    max-width: 100% !important;
    scrollbar-width: thin !important;
    scrollbar-color: #CBD5E1 #F8FAFC !important;
}

/* Vô hiệu hóa và ẩn hoàn toàn thanh cuộn ở các thẻ table, tbody, thead bên trong */
.modern-table table,
.modern-table tbody,
.modern-table thead,
.modern-table tr,
.gradio-dataframe table,
.gradio-dataframe tbody {
    overflow: visible !important;
    overflow-x: visible !important;
    overflow-y: visible !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.modern-table table::-webkit-scrollbar,
.modern-table tbody::-webkit-scrollbar,
.modern-table thead::-webkit-scrollbar,
.modern-table tr::-webkit-scrollbar,
.gradio-dataframe table::-webkit-scrollbar,
.gradio-dataframe tbody::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
}

/* Tùy chỉnh thanh cuộn ngang mượt mà cho .table-wrap */
.modern-table .table-wrap::-webkit-scrollbar {
    display: block !important;
    height: 6px !important;
}

.modern-table .table-wrap::-webkit-scrollbar-track {
    background: #F8FAFC !important;
    border-radius: 4px !important;
}

.modern-table .table-wrap::-webkit-scrollbar-thumb {
    background-color: #CBD5E1 !important;
    border-radius: 4px !important;
}

.modern-table .table-wrap::-webkit-scrollbar-thumb:hover {
    background-color: #94A3B8 !important;
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

th, td,
.modern-table th,
.modern-table td {
    white-space: nowrap !important;
    min-width: 110px !important;
}

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
    """Khởi tạo toàn bộ giao diện Gradio bằng Tiếng Việt."""
    t = UI_TEXT

    with gr.Blocks(title="BiasConsist - Nền tảng Đánh giá & Phát hiện Deepfake", theme=CUSTOM_THEME, css=CUSTOM_CSS) as demo:
        # Thanh điều hướng phía trên
        with gr.Row(elem_classes=["top-navbar"]):
            with gr.Column(scale=10, elem_classes=["navbar-brand"]):
                gr.Markdown(
                    f"<h1>BiasConsist <span class='badge'>{t['app_badge']}</span></h1><p>{t['app_subtitle']}</p>",
                    elem_classes=["brand-markdown"],
                )
            with gr.Column(scale=2, min_width=140, elem_classes=["navbar-actions"]):
                gr.Markdown("<span class='status-pill'>⚡ Hệ thống sẵn sàng</span>", elem_classes=["brand-markdown"])

        with gr.Row(elem_classes=["main-layout-row"]):
            # ================= CỘT TRÁI: SIDEBAR CẤU HÌNH =================
            with gr.Column(scale=4, min_width=360, elem_classes=["sidebar-panel"]):
                gr.Markdown(f"### {t['sidebar_config_title']}", elem_classes=["sidebar-section-title"])

                model_source = gr.Dropdown(
                    ["BiasConsist", "GenD", "Effort", "ForAda", "Checkpoint cục bộ"],
                    label=t["model_source_label"],
                    value="BiasConsist",
                    interactive=True,
                    elem_classes=["sidebar-dropdown"],
                )

                files = gr.Files(
                    label=t["upload_label"],
                    file_count="multiple",
                    file_types=["video", "image", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".bmp", ".webp"],
                    elem_classes=["sidebar-file-upload"],
                )

                # Nút thực hiện phát hiện
                run_btn = gr.Button(
                    t["run_btn"],
                    variant="primary",
                    size="lg",
                    elem_classes=["sidebar-run-btn"],
                )

                # Cài đặt nâng cao (ẩn các đường dẫn checkpoint và thanh trượt ngưỡng)
                with gr.Accordion(t["advanced_settings"], open=False, elem_classes=["sidebar-accordion"]):
                    bias_ckpt = gr.Textbox(label=t["bias_ckpt_label"], value=DEFAULT_BIAS_CKPT, visible=True)
                    gend_model = gr.Dropdown(GEND_MODELS, label=t["gend_model_label"], value=GEND_MODELS[0], visible=False)
                    effort_ckpt = gr.Textbox(label=t["effort_ckpt_label"], value=DEFAULT_EFFORT_CKPT, visible=False)
                    forada_ckpt = gr.Textbox(label=t["forada_ckpt_label"], value=DEFAULT_FORADA_CKPT, visible=False)
                    local_ckpt = gr.Textbox(label=t["local_ckpt_label"], value=DEFAULT_CKPT, visible=False)

                    face_thresh = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label=t["face_thresh_label"])
                    scale = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label=t["scale_label"])
                    target_size = gr.Number(value=-1, precision=0, label=t["target_size_label"])
                    stride = gr.Slider(1, 10, value=1, step=1, label=t["stride_label"])
                    max_frames = gr.Number(value=-1, precision=0, label=t["max_frames_label"])
                    max_faces = gr.Slider(1, 10, value=1, step=1, label=t["max_faces_label"])

            # ================= CỘT PHẢI: HIỂN THỊ KẾT QUẢ & XEM TRƯỚC =================
            with gr.Column(scale=8, min_width=580, elem_classes=["main-stage-column"]):
                # Dải thông báo trạng thái hệ thống
                with gr.Group(elem_classes=["status-strip"]):
                    status_summary = gr.Markdown(t["status_ready"])

                # Hàng khung xem trước
                with gr.Row(elem_classes=["preview-row"]):
                    with gr.Column(scale=1):
                        input_gallery = gr.Gallery(
                            label=t["input_preview_label"],
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
                            label=t["output_preview_label"],
                            show_label=True,
                            columns=1,
                            object_fit="contain",
                            height=390,
                            preview=True,
                            selected_index=0,
                            elem_classes=["preview-gallery"],
                        )

                # Bảng kết quả phân tích & chỉ số Video AUROC
                with gr.Column(elem_classes=["results-card"]):
                    with gr.Row(elem_classes=["results-header-row"]):
                        gr.Markdown(f"### {t['results_title']}", elem_classes=["results-title-text"])
                        with gr.Row(elem_classes=["results-btn-group"]):
                            copy_btn = gr.Button(t["copy_btn"], size="sm", elem_classes=["table-action-btn"])
                            export_btn = gr.Button(t["export_btn"], size="sm", elem_classes=["table-action-btn"])

                    table = gr.Dataframe(
                        value=pd.DataFrame(columns=TABLE_HEADERS),
                        headers=TABLE_HEADERS,
                        wrap=False,
                        column_widths=["220px", "110px", "110px", "120px", "130px", "120px", "120px", "110px"],
                        interactive=False,
                        elem_classes=["modern-table"],
                    )

        # Sao chép bảng vào Clipboard
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

        # Xuất dữ liệu bảng sang file CSV
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
                a.download = 'ket_qua_danh_gia.csv';
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
                gr.update(visible=(source in ("Checkpoint cục bộ", "Local Checkpoint"))),
            )

        model_source.change(
            fn=update_model_input,
            inputs=model_source,
            outputs=[bias_ckpt, gend_model, effort_ckpt, forada_ckpt, local_ckpt],
        )

        run_btn.click(
            fn=run_inference,
            inputs=[
                model_source,
                gend_model,
                bias_ckpt,
                effort_ckpt,
                forada_ckpt,
                local_ckpt,
                files,
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

        def update_previews(files_in):
            return get_all_inputs(files_in)

        files.change(
            fn=update_previews,
            inputs=[files],
            outputs=input_gallery,
        )

    return demo


if __name__ == "__main__":
    check_and_download_all_weights()
    ui = build_ui()
    returns = ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
    print("[BiasConsist] Giao diện Gradio đã khởi chạy tại:", returns)
