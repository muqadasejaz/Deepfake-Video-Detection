"""
app.py — Streamlit UI for the deepfake detector
----------------------------------------------------------
Run:
    streamlit run app.py

Weights (clip.pth) are downloaded automatically from Google Drive on first
run and cached locally in ./weights/clip.pth for subsequent runs.
"""

import os
import time
import tempfile
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st
import gdown

warnings_silenced = True
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
# Google Drive file id for clip.pth — downloaded automatically on first run
GDRIVE_FILE_ID = "1AJQ-q31xDtDnEAyfI6halTVLwnvjhmCt"
LOCAL_WEIGHTS_PATH = Path("weights") / "clip.pth"

N_FRAMES = 8
THRESHOLD = 0.5

MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
IMG_SIZE = 224
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

st.set_page_config(page_title="Deepfake Detector", layout="centered")


def ensure_weights_downloaded(file_id: str, dest: Path) -> str:
    """Download clip.pth from Google Drive once, then reuse the local copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with st.spinner("Downloading model weights (first run only)..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(dest), quiet=False)
    return str(dest)


# ── Model builder (cached — loads once per session) ───────────────────────
@st.cache_resource(show_spinner="Loading CLIP detector weights...")
def load_model(weights_path: str):
    from transformers import CLIPVisionConfig, CLIPVisionModel

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt.items()}

    # remap backbone.xxx -> backbone.vision_model.xxx
    remapped = {}
    for k, v in state.items():
        if k.startswith("backbone.") and not k.startswith("backbone.vision_model."):
            remapped[k.replace("backbone.", "backbone.vision_model.", 1)] = v
        else:
            remapped[k] = v
    state = remapped

    patch_w = state["backbone.vision_model.embeddings.patch_embedding.weight"]
    hidden_size = patch_w.shape[0]

    if hidden_size == 1024:
        num_heads, num_layers, intermediate_size = 16, 24, 4096
    else:
        num_heads, num_layers, intermediate_size = 12, 12, 3072

    layer_indices = {
        int(k.split("encoder.layers.")[1].split(".")[0])
        for k in state if "encoder.layers." in k
    }
    if layer_indices:
        num_layers = max(layer_indices) + 1

    config = CLIPVisionConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        image_size=224,
        patch_size=16,
        num_channels=3,
        layer_norm_eps=1e-5,
    )

    if "head.weight" in state:
        num_classes = state["head.weight"].shape[0]
    elif "head.0.weight" in state:
        num_classes = state["head.0.weight"].shape[0]
    else:
        num_classes = 2

    class CLIPDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = CLIPVisionModel(config)
            self.head = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.backbone(pixel_values=x)
            return self.head(out.pooler_output)

    model = CLIPDetector()
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    info = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
    return model, info


# ── Preprocessing / video helpers ──────────────────────────────────────────
def preprocess_frame(frame_rgb):
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


def get_frame_count(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def uniform_indices(total, n):
    if total <= 0:
        return []
    n = min(n, total)
    return np.linspace(0, total - 1, n, dtype=int).tolist()


@torch.no_grad()
def score_file(path, model, n_frames, threshold):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in IMG_EXTS:
        raw = cv2.imread(str(path))
        if raw is None:
            return None
        frames, indices = [cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)], [0]
    else:
        total = get_frame_count(path)
        if total == 0:
            return None
        indices = uniform_indices(total, n_frames)
        frames = [f for f in (read_frame(path, i) for i in indices) if f is not None]

    if not frames:
        return None

    batch = torch.stack([preprocess_frame(f) for f in frames])
    logits = model(batch)
    probs = torch.softmax(logits, dim=1).numpy()
    mean_probs = probs.mean(axis=0)

    real_p, fake_p = float(mean_probs[0]), float(mean_probs[1])
    verdict = "FAKE" if fake_p >= threshold else "REAL"

    return {
        "verdict": verdict,
        "fake_prob": fake_p,
        "real_prob": real_p,
        "confidence": max(real_p, fake_p),
        "frames_scored": len(frames),
        "per_frame_fake_prob": probs[:, 1].tolist(),
    }


# ── UI ──────────────────────────────────────────────────────────────────
st.title("Deepfake Detector")

n_frames = N_FRAMES
threshold = THRESHOLD

weights_path = ensure_weights_downloaded(GDRIVE_FILE_ID, LOCAL_WEIGHTS_PATH)
if not Path(weights_path).exists():
    st.error("Model weights could not be downloaded. Check the Google Drive file ID / sharing permissions.")
    st.stop()

model, info = load_model(weights_path)

uploaded = st.file_uploader(
    "Upload a video or image",
    type=[e.strip(".") for e in VIDEO_EXTS | IMG_EXTS],
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Scoring..."):
        t0 = time.time()
        result = score_file(tmp_path, model, n_frames, threshold)
        elapsed = time.time() - t0

    os.unlink(tmp_path)

    if result is None:
        st.error("Could not read the uploaded file.")
    else:
        verdict = result["verdict"]
        color = "🟥" if verdict == "FAKE" else "🟩"
        st.subheader(f"{color} Verdict: {verdict}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Fake probability", f"{result['fake_prob']*100:.1f}%")
        c2.metric("Confidence", f"{result['confidence']*100:.1f}%")
        c3.metric("Frames scored", result["frames_scored"])

        st.caption(f"Inference time: {elapsed:.2f}s")
