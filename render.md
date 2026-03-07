## Deploy AI Server lên Render

### Tạo Web Service

- **Root Directory**: `ai-server`
- **Runtime**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Environment Variables (khuyến nghị)

- **ALLOWED_ORIGINS**: `https://<username>.github.io` (hoặc nhiều origin, phân tách bằng dấu phẩy)
- **MIN_CONF**: `0.35` (tăng lên như `0.45` để “ít đoán”, giảm nhận nhầm)
- **YOLO_MODEL**: `yolov8n.pt` (mặc định)

### Test

Mở `https://<render-service>.onrender.com/health`
