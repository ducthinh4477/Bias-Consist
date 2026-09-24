# BiasConsist: Consistency-Guided Bias Tuning for Deepfake Detection

> **Interactive Web Demo & Multi-Model Deepfake Detection Benchmark Platform**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Gradio UI](https://img.shields.io/badge/Gradio-5.0%2B-yellow.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Giới thiệu

Kho lưu trữ này cung cấp mã nguồn chính thức cho dự án **BiasConsist** (*Consistency-Guided Bias Tuning for Deepfake Detection*), bao gồm:
1. **Nền tảng Thử nghiệm Trực quan (Interactive Web Demo)** chạy bằng Gradio với giao diện SaaS chuẩn Dashboard hai cột, hỗ trợ song ngữ (EN / VI), preview đối chiếu song song và thanh cuộn bảng thông số chi tiết.
2. **Hệ thống Đối chuẩn Đa Mô hình (Comparative Benchmark)** cho phép so sánh trực tiếp BiasConsist với GenD (WACV 2026), Effort, và ForAda.
3. **Công cụ Đánh giá Video AUROC & EER** tự động gom cụm theo từng video từ tập tin kết quả suy luận.

### Về mô hình BiasConsist
- **Tối ưu tham số cực nhỏ (BitFit)**: Chỉ tinh chỉnh các vector bias của mô hình OpenAI CLIP ViT-L/14 (~0.27M tham số có thể huấn luyện, chiếm < 0.1% tổng số tham số backbone).
- **Chuẩn hóa đặc trưng $L_2$**: Chiếu vector đại diện lên siêu cầu $\|\mathbf{f}\|_2 = 1.0$, ngăn chặn overconfidence và bùng nổ logits.
- **Ràng buộc nhất quán yếu-đến-mạnh (Consistency Loss)**: $\mathcal{L}_{cons} = \tau^2 D_{KL}(p^w \parallel p^s)$ triệt tiêu dao động dự đoán giữa các góc nhìn biến dạng.
- **Label Smoothing**: $\epsilon = 0.1$ giúp phân phối xác suất được hiệu chuẩn (calibrated) mượt mà, tối ưu hóa năng lực tổng quát hóa trên dữ liệu chưa từng thấy.

---

## ⚡ Hướng dẫn Bắt đầu Nhanh (Quick Start)

Mã nguồn được thiết kế để bất kỳ ai clone về cũng có thể **chạy kiểm thử ngay lập tức (out-of-the-box)** mà không gặp lỗi thiếu thư viện hay lỗi phụ thuộc đường dẫn tuyệt đối.

### 1. Clone mã nguồn
```bash
git clone https://github.com/ducthinh4477/Bias-Consist.git
cd Bias-Consist
```

### 2. Thiết lập môi trường ảo BiasC (BẮT BUỘC)
Dự án được tối ưu và yêu cầu chạy trong môi trường ảo **`BiasC`** để đảm bảo tương thích các phiên bản PyTorch, CLIP, Transformers và Gradio:

* **Sử dụng Conda (Khuyên dùng):**
  ```bash
  conda create -n BiasC python=3.10 -y
  conda activate BiasC
  ```

* **Hoặc sử dụng `venv`:**
  ```bash
  python -m venv BiasC
  # Trên Windows:
  BiasC\Scripts\activate
  # Trên Linux / macOS:
  source BiasC/bin/activate
  ```

* **Cài đặt danh sách thư viện phụ thuộc:**
  ```bash
  pip install -r requirements.txt
  ```

### 3. Chạy bộ kiểm thử tự động (Smoke Test)
Kiểm tra tính toàn vẹn của PyTorch, RetinaFace phát hiện khuôn mặt, tính toán chỉ số Video AUROC/EER và khởi tạo giao diện Web:
```bash
python test_quick.py
```
> Nếu hiển thị `[SUCCESS] All 5 smoke tests passed successfully!`, hệ thống đã sẵn sàng 100%.

### 4. Đánh giá Video AUROC qua CLI trên dữ liệu mẫu
```bash
python evaluate_video_auc.py --csv sample_videos/sample_eval.csv
```

### 5. Khởi động Web Demo
```bash
python app/run.py
```
Mở trình duyệt truy cập: 👉 **`http://localhost:7860/`** (hoặc `http://127.0.0.1:7860/`).

---

## 🔬 Các Mô hình Tích hợp & Checkpoint Trọng số

| Phương pháp | Backbone | Kỹ thuật Tinh chỉnh | Cơ chế Checkpoint |
| :--- | :---: | :--- | :--- |
| **BiasConsist** *(Proposed)* | CLIP ViT-L/14 | BitFit (Bias-only) + L2 Norm + Consistency Loss | `weights/BiasConsist/bias_consistency.pth` |
| **GenD** *(WACV 2026)* | CLIP ViT-L/14 | Parameter-efficient LayerNorm Adaptation | **Tự động tải từ Hugging Face** (`yermandy/GenD_CLIP_L_14`) |
| **Effort** | CLIP ViT-L/14 | Residual Adapter Tuning | `weights/Effort/effort_clip_L14_trainOn_FaceForensic.pth` |
| **ForAda** | CLIP ViT-L/14 | Frequency-domain Cross-Attention Adaptation | `weights/ForAda/forada_checkpoint.pth` |

> 💡 **Mẹo:** Mô hình đối chuẩn chính **GenD (CLIP ViT-L/14)** đã được tích hợp trọn vẹn cả mã nguồn kiến trúc tại `src/model/gend/` và cơ chế tự động nạp trọng số từ Hugging Face. Người dùng có thể chọn ngay `GenD` trên Web App để kiểm tra suy luận trên video/ảnh thực tế ngay lập tức.

### Cấu trúc thư mục trọng số (khi bổ sung thêm checkpoint thủ công):
```text
weights/
├── BiasConsist/
│   └── bias_consistency.pth
├── Effort/
│   └── effort_clip_L14_trainOn_FaceForensic.pth
└── ForAda/
    └── forada_checkpoint.pth
```

---

## 📊 Bảng Hiệu Năng So Sánh Đối Chuẩn

| Phương pháp | In-domain FF++ (c40) | Cross-dataset DFD | Cross-method DF40 | Trainable Params |
| :--- | :---: | :---: | :--- | :---: |
| **BiasConsist (Ours)** | **99.1%** | **97.9%** | **96.8%** | **~0.27M (<0.1%)** |
| GenD | 98.9% | 97.0% | 95.7% | ~0.08M |
| ForAda | 96.8% | 94.2% | 93.6% | ~12.5M |
| Effort | 93.9% | 96.5% | 95.1% | ~4.2M |

---

## 📁 Cấu trúc Thư mục

```text
Bias-Consist/
├── app/
│   └── run.py                  # Mã nguồn giao diện Web App Gradio (Dashboard SaaS)
├── src/
│   ├── model/
│   │   ├── BiasConsistency.py  # Kiến trúc mô hình BiasConsist
│   │   ├── gend/               # Kiến trúc mô hình GenD CLIP đối chuẩn (WACV 2026)
│   │   │   ├── modeling_gend.py
│   │   │   ├── config.json
│   │   │   └── model_index.json
│   │   ├── effort/             # Mô hình Effort
│   │   ├── forada/             # Mô hình ForAda
│   │   └── fsfm/               # Mô hình FSFM
│   ├── metrics.py              # Thư viện tính Video AUROC & EER
│   └── retinaface.py           # Module phát hiện và căn chỉnh khuôn mặt RetinaFace
├── config/
│   └── datasets/               # Cấu hình danh sách tập dữ liệu kiểm thử
├── sample_videos/              # Video mẫu phục vụ kiểm thử nhanh (Real & Fake)
│   ├── real_face_sample.mp4
│   ├── fake_face_sample.mp4
│   └── sample_eval.csv
├── evaluate_video_auc.py       # Script CLI tính Video AUROC & EER từ CSV
├── detector.py                 # Tiện ích tiền xử lý và cắt khuôn mặt hàng loạt
├── test_quick.py               # Bộ kiểm thử tự động (Smoke Test)
├── BiasConsist.pdf             # Bài báo khoa học BiasConsist
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── .gitignore                  # Cấu hình bỏ qua tệp nhị phân lớn và bộ nhớ đệm
└── README.md                   # Tài liệu hướng dẫn sử dụng
```

---

## 📖 Trích dẫn (Citation)

Nếu bạn sử dụng mã nguồn hoặc phương pháp BiasConsist trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@article{dinh2024biasconsist,
  title={BiasConsist: Consistency-Guided Bias Tuning for Deepfake Detection},
  author={Dinh, Quan Thanh and Nguyen, Vinh-Tiep and others},
  institution={Ho Chi Minh City University of Technology and Education (HCMUTE)},
  year={2024}
}
```

---

## 📄 Bản quyền (License)

Dự án được phân phối dưới giấy phép **MIT License**. Chi tiết xem tại tệp [LICENSE](LICENSE).
