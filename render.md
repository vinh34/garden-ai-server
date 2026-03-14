## Deploy AI Server lên Render

### Render quick setup (copy/paste)

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Set các Environment Variables sau trong Render:

- `ALLOWED_ORIGINS=https://<your-frontend-domain>`
  - Ví dụ: `https://yourname.github.io`
- `MIN_CONF=0.50`
- `MIN_TOP1_TOP2_GAP=0.12` (nếu top-1 và top-2 quá sát nhau thì trả `prediction=null` để tránh nhận nhầm)
- `YOLO_MODEL=models/fruit/best.pt`
  - Nếu chưa có model riêng thì tạm dùng: `YOLO_MODEL=yolov8n.pt`

---

### Gợi ý 1 model đã train nên dùng

Nếu bạn cần **1 lựa chọn duy nhất** để bắt đầu cho bộ class lớn như của bạn, mình khuyên:

- **YOLOv8m custom-trained** trên chính dataset của bạn (10-50+ ảnh/class), lưu tại `models/fruit/best.pt`.

Vì sao chọn model này:

- `yolov8m` thường chính xác hơn `yolov8n` khi số class nhiều (như danh sách seed ID lớn).
- Vẫn deploy được trên Render nếu traffic vừa phải.
- Tương thích trực tiếp với server hiện tại qua biến `YOLO_MODEL`.

Lệnh train gợi ý:

```bash
python scripts/train_fruit_detector.py \
  --dataset-root dataset \
  --base-model yolov8m.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 8
```

### Cách train model trái cây (10-50 ảnh / mỗi loại)

> Bạn train **trên máy local** hoặc cloud notebook trước, sau đó upload file model `best.pt` lên server/repo để Render dùng.

#### 1) Chuẩn bị dữ liệu

Cấu trúc thư mục:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

- Mỗi ảnh phải có file label `.txt` cùng tên trong thư mục `labels/...`.
- Format label YOLO mỗi dòng:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- Nên có khoảng **10-50 ảnh cho mỗi fruit class** (càng nhiều càng tốt).
- Danh sách class hiện tại là **120 seed IDs** và được lấy từ `seed_ids.py` (`ALL_SEED_IDS`).
- Khi gắn nhãn (Label Studio/Roboflow), tên class nên khớp đúng các seed ID này để map chính xác.

#### 2) Cài thư viện

```bash
pip install -r requirements.txt
```

#### 3) Chạy train

```bash
python scripts/train_fruit_detector.py \
  --dataset-root dataset \
  --epochs 120 \
  --imgsz 640 \
  --batch 16
```

Tùy chọn nhẹ hơn nếu máy yếu:

```bash
python scripts/train_fruit_detector.py --dataset-root dataset --epochs 80 --imgsz 512 --batch 8
```

#### 4) Kết quả sau khi train

- Weights tốt nhất được copy về: `models/fruit/best.pt`
- Metadata: `models/fruit/best.json`

#### 5) Deploy model lên Render

- Đảm bảo file `models/fruit/best.pt` tồn tại ở đúng path khi app chạy.
- Trên Render set:
  - `YOLO_MODEL=models/fruit/best.pt`

#### 6) Kiểm tra sau deploy

Mở:

- `https://<render-service>.onrender.com/health`

Nếu thấy `"model":"models/fruit/best.pt"` nghĩa là server đang dùng model đã train.
