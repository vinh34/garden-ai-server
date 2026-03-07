import io
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

APP_NAME = "Vuon AI Server"

# Render sẽ cấp PORT qua env
PORT = int(os.getenv("PORT", "8000"))

# CORS: đặt ALLOWED_ORIGINS="https://<user>.github.io,https://<user>.github.io/<repo>"
allowed = os.getenv("ALLOWED_ORIGINS", "*").strip()
if allowed == "*" or allowed == "":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [x.strip() for x in allowed.split(",") if x.strip()]

# Ngưỡng để giảm nhận nhầm (tăng lên nếu muốn ít “đoán” hơn)
MIN_CONF = float(os.getenv("MIN_CONF", "0.35"))

# Map COCO class -> seedId trong game
COCO_TO_SEED: Dict[str, Optional[str]] = {
    "apple": "tao",
    "banana": "chuoi",
    "orange": "cam",
    "broccoli": "broccoli",
    "carrot": "ca_rot",
    # các lớp dễ gây nhiễu: không trả seed để tránh sai
    "potted plant": None,
}

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLOv8n pretrained (COCO) – tải weights lần đầu khi server khởi động
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
yolo = YOLO(MODEL_NAME)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "model": MODEL_NAME, "min_conf": MIN_CONF}


def _read_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img


def _to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> Dict[str, Any]:
    file_bytes = await image.read()
    img = _read_image(file_bytes)
    arr = _to_numpy(img)

    # YOLO inference
    results = yolo.predict(arr, verbose=False)
    r0 = results[0]

    detections: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    # boxes: xyxy + conf + cls
    if r0.boxes is not None and len(r0.boxes) > 0:
        names = r0.names  # class_id -> name
        for b in r0.boxes:
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls_name = str(names.get(cls_id, cls_id)).lower()
            xyxy = [float(x) for x in b.xyxy[0].tolist()]

            detections.append({"class": cls_name, "confidence": conf, "bbox_xyxy": xyxy})

            seed_id = COCO_TO_SEED.get(cls_name)
            if seed_id and conf >= MIN_CONF:
                if best is None or conf > float(best["confidence"]):
                    best = {"seedId": seed_id, "confidence": conf, "class": cls_name}

    return {
        "ok": True,
        "source": "yolo",
        "min_conf": MIN_CONF,
        "prediction": best,  # hoặc null
        "detections": sorted(detections, key=lambda x: x["confidence"], reverse=True)[:10],
    }

