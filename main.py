import io
import os
import re
import unicodedata
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

# Ưu tiên model đã train riêng cho trái cây.
DEFAULT_MODEL_CANDIDATES = [
    os.getenv("YOLO_MODEL", "models/fruit/best.pt"),
    "yolov8n.pt",
]

# Các seedId trái cây mà app hỗ trợ khi dùng custom-trained model.
FRUIT_SEED_IDS = {
    "tao",
    "dau_tay",
    "cam",
    "chanh",
    "sung",
    "dua",
    "chuoi",
    "mit",
    "na",
    "luu",
    "nho",
    "dua_hau",
    "du_du",
    "xoai",
    "bo",
    "vai",
    "chom_chom",
    "thanh_long",
    "kiwi",
    "chanh_dau",
    "dau_den",
    "dau_xanh",
    "phuc_bon_tu",
    "le",
    "dao",
    "man",
    "mo",
    "anh_dao",
    "oliu",
    "cha_la",
    "dua_xiem",
    "buoi",
}

# Map COCO class -> seedId trong game
COCO_TO_SEED: Dict[str, Optional[str]] = {
    "apple": "tao",
    "banana": "chuoi",
    "orange": "cam",
    "grape": "nho",
    "pear": "le",
    "peach": "dao",
    "kiwi": "kiwi",
    "mango": "xoai",
    "pineapple": "dua",
    "watermelon": "dua_hau",
    "coconut": "dua_xiem",
    "lemon": "chanh",
    "lime": "chanh",
    "avocado": "bo",
    "pomegranate": "luu",
    "papaya": "du_du",
    "fig": "sung",
    "cherry": "anh_dao",
    "strawberry": "dau_tay",
    "blueberry": "dau_xanh",
    "blackberry": "dau_den",
    "raspberry": "phuc_bon_tu",
    "date": "cha_la",
    # các lớp dễ gây nhiễu: không trả seed để tránh sai
    "potted plant": None,
}

# Alias để ánh xạ nhãn class (EN/VI, có hoặc không dấu) về seedId của game.
FRUIT_CLASS_ALIASES: Dict[str, str] = {
    "tao": "tao",
    "apple": "tao",
    "dau_tay": "dau_tay",
    "strawberry": "dau_tay",
    "cam": "cam",
    "orange": "cam",
    "chanh": "chanh",
    "lemon": "chanh",
    "lime": "chanh",
    "sung": "sung",
    "fig": "sung",
    "dua": "dua",
    "pineapple": "dua",
    "chuoi": "chuoi",
    "banana": "chuoi",
    "mit": "mit",
    "jackfruit": "mit",
    "na": "na",
    "sugar_apple": "na",
    "custard_apple": "na",
    "luu": "luu",
    "pomegranate": "luu",
    "nho": "nho",
    "grape": "nho",
    "dua_hau": "dua_hau",
    "watermelon": "dua_hau",
    "du_du": "du_du",
    "papaya": "du_du",
    "xoai": "xoai",
    "mango": "xoai",
    "bo": "bo",
    "avocado": "bo",
    "vai": "vai",
    "lychee": "vai",
    "chom_chom": "chom_chom",
    "rambutan": "chom_chom",
    "thanh_long": "thanh_long",
    "dragon_fruit": "thanh_long",
    "kiwi": "kiwi",
    "chanh_dau": "chanh_dau",
    "passion_fruit": "chanh_dau",
    "dau_den": "dau_den",
    "blackberry": "dau_den",
    "dau_xanh": "dau_xanh",
    "blueberry": "dau_xanh",
    "phuc_bon_tu": "phuc_bon_tu",
    "raspberry": "phuc_bon_tu",
    "le": "le",
    "pear": "le",
    "dao": "dao",
    "peach": "dao",
    "man": "man",
    "plum": "man",
    "mo": "mo",
    "apricot": "mo",
    "anh_dao": "anh_dao",
    "cherry": "anh_dao",
    "oliu": "oliu",
    "olive": "oliu",
    "cha_la": "cha_la",
    "date": "cha_la",
    "dua_xiem": "dua_xiem",
    "coconut": "dua_xiem",
    "buoi": "buoi",
    "grapefruit": "buoi",
}

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pick_model_name() -> str:
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate == "yolov8n.pt" or os.path.exists(candidate):
            return candidate
    return "yolov8n.pt"


# YOLO pretrained/custom-trained
MODEL_NAME = _pick_model_name()
yolo = YOLO(MODEL_NAME)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "model": MODEL_NAME, "min_conf": MIN_CONF}


def _read_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img


def _to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)


def _normalize_label(value: str) -> str:
    no_diacritics = "".join(
        ch for ch in unicodedata.normalize("NFD", value) if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", "_", no_diacritics.lower()).strip("_")


def _resolve_seed_id(cls_name: str) -> Optional[str]:
    # Ưu tiên map COCO để giữ backward compatibility.
    if cls_name in COCO_TO_SEED:
        return COCO_TO_SEED[cls_name]

    normalized = _normalize_label(cls_name)

    # Hỗ trợ model train riêng có class name đúng bằng seedId trái cây.
    if normalized in FRUIT_SEED_IDS:
        return normalized

    if normalized in FRUIT_CLASS_ALIASES:
        return FRUIT_CLASS_ALIASES[normalized]

    return None


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

            seed_id = _resolve_seed_id(cls_name)
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
