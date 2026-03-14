## Deploy AI Server lên Render

### Tạo Web Service

- **Root Directory**: `ai-server`
- **Runtime**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Environment Variables (khuyến nghị)

- **ALLOWED_ORIGINS**: `https://<username>.github.io` (hoặc nhiều origin, phân tách bằng dấu phẩy)
- **MIN_CONF**: `0.35` (tăng lên như `0.45` để “ít đoán”, giảm nhận nhầm)
- **YOLO_MODEL**: `models/fruit/best.pt` (nên trỏ tới model đã train trái cây; nếu không có file sẽ fallback `yolov8n.pt`)

### Huấn luyện model trái cây (10-50 ảnh / loại)

1. Gắn nhãn dữ liệu YOLO cho từng class trái cây (class name đúng `seedId`, ví dụ `tao`, `chuoi`, `xoai`...).
2. Tổ chức dữ liệu:
   - `dataset/images/train`
   - `dataset/images/val`
   - `dataset/labels/train`
   - `dataset/labels/val`
3. Train:

```bash
python scripts/train_fruit_detector.py --dataset-root dataset --epochs 120 --imgsz 640
```

4. Model sau train sẽ nằm ở `models/fruit/best.pt`. Deploy file này cùng server, hoặc set `YOLO_MODEL` tới đường dẫn model tương ứng.

### Test

Mở `https://<render-service>.onrender.com/health`
