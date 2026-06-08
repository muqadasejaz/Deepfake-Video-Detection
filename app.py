"""
DeepFake Detector — Streamlit App
Model: GenConViT-ED (ConvNeXt-tiny + Swin-tiny + Autoencoder)
UI: Tab-based Image & Video analysis with split-column verdict layout
Weights: auto-downloaded from Google Drive on first run, then cached.
Model loaded ONCE via @st.cache_resource.
"""

import streamlit as st

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Standard imports ──────────────────────────────────────────────────────────
import os
import io
import time
import math
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  (must match training notebook CFG exactly)
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    variant          = "ed",
    backbone         = "convnext_tiny.in12k_ft_in1k",
    embedder         = "swin_tiny_patch4_window7_224.ms_in1k",
    img_size         = 224,
    num_classes      = 2,
    latent_dims      = 12544,
    frames_per_video = 15,
    face_crop        = False,
)

IMAGENET_MEAN    = (0.485, 0.456, 0.406)
IMAGENET_STD     = (0.229, 0.224, 0.225)
WEIGHT_PATH      = Path("weights/genconvit_ed_final.pth")
MAX_VIDEO_FRAMES = 15

# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (exact copy from training notebook)
# ══════════════════════════════════════════════════════════════════════════════

class ConvEncoder(nn.Module):
    """224×224×3 → 7×7×256"""
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.net = nn.Sequential(
            blk(3, 16), blk(16, 32), blk(32, 64),
            blk(64, 128), blk(128, 256),
        )
    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    """7×7×256 → 224×224×3"""
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.ConvTranspose2d(i, o, 4, stride=2, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )
        self.net = nn.Sequential(
            blk(256, 128), blk(128, 64), blk(64, 32), blk(32, 16),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
        )
    def forward(self, x):
        return self.net(x)


class GenConViTED(nn.Module):
    """
    GenConViT Encoder-Decoder variant — exactly as trained.
    ConvNeXt reads AE reconstruction; Swin reads original frame.
    Both fused → classifier.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder  = ConvEncoder()
        self.decoder  = ConvDecoder()
        self.convnext = timm.create_model(
            cfg["backbone"], pretrained=False, num_classes=0, global_pool="avg"
        )
        cvx_dim = self.convnext.num_features
        self.swin = timm.create_model(
            cfg["embedder"], pretrained=False, num_classes=0, global_pool="avg"
        )
        swn_dim = self.swin.num_features
        self.head = nn.Sequential(
            nn.Linear(cvx_dim + swn_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, cfg["num_classes"]),
        )

    def forward(self, x):
        z      = self.encoder(x)
        recon  = self.decoder(z)
        f_cvx  = self.convnext(recon)
        f_swn  = self.swin(x)
        logits = self.head(torch.cat([f_cvx, f_swn], dim=1))
        return logits, recon


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def _download_weights() -> None:
    if WEIGHT_PATH.exists():
        return
    WEIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_id = st.secrets["gdrive"]["deepfake_video"]
    except Exception:
        file_id = "1FisjFyODlh6KO7irXX9s6tML6nrqAAKa"

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        import gdown
        gdown.download(url, str(WEIGHT_PATH), quiet=True)
    except Exception as exc:
        raise RuntimeError(f"Weight download failed: {exc}") from exc

    if not WEIGHT_PATH.exists():
        raise RuntimeError("Weight file not found after download attempt.")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER  — cached at process level, never reloads
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    _download_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GenConViTED(CFG).to(device)
    state  = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (matches training eval_tfm exactly)
# ══════════════════════════════════════════════════════════════════════════════
_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)

def preprocess_frame(frame_rgb: np.ndarray) -> torch.Tensor:
    img_size = CFG["img_size"]
    frame = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - _MEAN) / _STD
    frame = frame.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(frame))


# ══════════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, n_frames: int = CFG["frames_per_video"]):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    if total <= 0:
        cap.release()
        return None, None

    target_set = set(np.linspace(0, max(total - 1, 0), n_frames, dtype=int).tolist())
    frames, timestamps, fidx = [], [], 0

    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fidx in target_set:
            fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(fr_rgb.astype(np.uint8))
            timestamps.append(fidx / fps)
        fidx += 1
        if len(frames) >= n_frames:
            break

    cap.release()
    while frames and len(frames) < n_frames:
        frames.append(frames[-1])
        timestamps.append(timestamps[-1])
    return (frames, timestamps) if frames else (None, None)


def get_video_meta(path: str) -> dict:
    cap  = cv2.VideoCapture(str(path))
    meta = {}
    if cap.isOpened():
        fps   = cap.get(cv2.CAP_PROP_FPS) or 1
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta  = {
            "duration"  : f"{total/fps:.1f}s",
            "fps"       : f"{fps:.0f}",
            "resolution": f"{w}×{h}",
            "frames"    : str(total),
        }
    cap.release()
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_image(pil_img: Image.Image, model: nn.Module, device: torch.device) -> dict:
    img_rgb = np.array(pil_img.convert("RGB"))
    tensor  = preprocess_frame(img_rgb).unsqueeze(0).to(device)
    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            logits, _ = model(tensor)
    else:
        logits, _ = model(tensor)
    probs  = F.softmax(logits.float(), dim=1)
    p_fake = float(probs[0, 1].item())
    p_real = 1.0 - p_fake
    return {
        "label"     : "Fake" if p_fake >= 0.5 else "Real",
        "confidence": round(max(p_fake, p_real) * 100, 1),
        "p_fake"    : round(p_fake * 100, 1),
        "p_real"    : round(p_real * 100, 1),
    }


@torch.no_grad()
def predict_frames(frames_rgb: list, model: nn.Module, device: torch.device) -> dict:
    tensors = [preprocess_frame(f) for f in frames_rgb]
    batch   = torch.stack(tensors).to(device)
    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            logits, _ = model(batch)
    else:
        logits, _ = model(batch)
    probs        = F.softmax(logits.float(), dim=1)
    frame_scores = probs[:, 1].cpu().numpy()
    p_fake       = float(frame_scores.mean())
    p_real       = 1.0 - p_fake
    return {
        "label"       : "Fake" if p_fake >= 0.5 else "Real",
        "confidence"  : round(max(p_fake, p_real) * 100, 1),
        "p_fake"      : round(p_fake * 100, 1),
        "p_real"      : round(p_real * 100, 1),
        "frame_scores": frame_scores.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:      #07090f;
    --surface: #0f1118;
    --card:    #13151f;
    --border:  #1c2035;
    --accent:  #4fffb0;
    --danger:  #ff4d6d;
    --warn:    #ffb300;
    --text:    #dde3f0;
    --muted:   #525d7a;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'Syne', sans-serif;
}

html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    font-family: var(--sans) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header,
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Top bar ── */
.topbar {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 1.1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.logo-wrap { display: flex; align-items: center; gap: 0.85rem; }
.logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #4fffb0 0%, #00c8ff 100%);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.logo-name {
    font-size: 1.1rem; font-weight: 800;
    letter-spacing: -0.02em; color: var(--text);
}
.logo-name span { color: var(--accent); }
.logo-sub {
    font-family: var(--mono) !important;
    font-size: 0.58rem; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-top: 2px;
}
.status-pill {
    font-family: var(--mono) !important;
    font-size: 0.62rem; color: var(--muted);
    display: flex; align-items: center; gap: 0.4rem;
}
.dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
}

/* ── Page body ── */
.page { padding: 2rem 2.5rem; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] section {
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}

/* ── Button ── */
.stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.65rem 1.5rem !important;
    margin-top: 0.5rem !important;
    transition: opacity .2s !important;
}
.stButton > button:disabled {
    opacity: 0.3 !important;
    cursor: not-allowed !important;
}
.stButton > button:hover:not(:disabled) { opacity: 0.82 !important; }

/* ── Verdict card ── */
.verdict-wrap {
    border-radius: 12px;
    padding: 1.8rem 2rem 1.5rem;
    margin-top: 0.5rem;
    position: relative;
    overflow: hidden;
}
.verdict-real { background: #001c10; border: 1.5px solid var(--accent); }
.verdict-fake { background: #1a0010; border: 1.5px solid var(--danger); }

.vdict-tag {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.vdict-label {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.vdict-real .vdict-label { color: var(--accent); }
.vdict-fake .vdict-label { color: var(--danger); }
.vdict-conf {
    font-family: var(--mono);
    font-size: 0.76rem;
    color: var(--muted);
}

/* ── Prob bars ── */
.bar-section { margin-top: 1.3rem; }
.bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 9px;
}
.bar-name {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    width: 40px;
    flex-shrink: 0;
}
.bar-track {
    flex: 1;
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 3px; transition: width .6s ease; }
.fill-real { background: var(--accent); }
.fill-fake { background: var(--danger); }
.bar-pct {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem; margin-top: 1rem; }
.stat-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.85rem 1rem;
}
.stat-lbl {
    font-family: var(--mono) !important;
    font-size: 0.58rem; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.25rem;
}
.stat-val {
    font-family: var(--mono) !important;
    font-size: 1.05rem; font-weight: 600; color: var(--text);
}

/* ── Frame grid ── */
.frames-header {
    font-family: var(--mono) !important;
    font-size: 0.62rem; color: var(--muted);
    letter-spacing: 0.14em; text-transform: uppercase;
    border-top: 1px solid var(--border);
    padding-top: 1.2rem; margin-top: 1.5rem; margin-bottom: 0.6rem;
}
.frame-label {
    text-align: center;
    font-family: var(--mono);
    font-size: 0.62rem;
    margin-top: 4px;
}

/* ── Image meta ── */
.img-meta {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 6px;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 1.5rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 7px !important;
    border: none !important;
    padding: 0.5rem 1.4rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent) !important;
}

/* ── Video player ── */
[data-testid="stVideo"] video {
    border-radius: 10px;
    border: 1px solid var(--border);
    max-height: 260px;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 9px;
    border: 1px solid var(--border);
}

/* ── Spinner ── */
div.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Alert ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 9px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}

/* ── Bottom bar ── */
.bottombar {
    border-top: 1px solid var(--border);
    padding: 0.85rem 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--card);
    margin-top: 3rem;
}
.bottom-note {
    font-family: var(--mono) !important;
    font-size: 0.6rem; color: var(--muted);
}

/* ── Empty state ── */
.empty-state {
    height: 300px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    gap: 0.8rem;
}
.empty-icon { font-size: 2.2rem; opacity: 0.18; }
.empty-text {
    font-family: var(--mono) !important;
    font-size: 0.7rem; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase;
    text-align: center; line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL ONCE
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Starting up…"):
    try:
        MODEL, DEVICE = load_model()
        _ready = True
    except Exception as _e:
        _ready = False
        _err   = str(_e)


# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
status_html = (
    '<span class="dot"></span>Ready'
    if _ready else
    '<span style="color:#ff4d6d">⚠ Error</span>'
)
st.markdown(f"""
<div class="topbar">
    <div class="logo-wrap">
        <div class="logo-icon">🛡️</div>
        <div>
            <div class="logo-name">Deep<span>Fake</span> Detector</div>
            <div class="logo-sub">Forensic Media Analysis</div>
        </div>
    </div>
    <div class="status-pill">{status_html}</div>
</div>
""", unsafe_allow_html=True)

if not _ready:
    st.error(f"Failed to load model: {_err}")
    st.stop()

st.markdown('<div class="page">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _score_color(p_fake_pct: float) -> str:
    if p_fake_pct >= 65:
        return "#ff4d6d"
    if p_fake_pct >= 38:
        return "#ffb300"
    return "#4fffb0"


def render_verdict(label: str, confidence: float, p_real: float, p_fake: float):
    css  = "verdict-real" if label == "Real" else "verdict-fake"
    vcss = "vdict-real"   if label == "Real" else "vdict-fake"
    st.markdown(f"""
    <div class="verdict-wrap {css}">
        <div class="{vcss}">
            <div class="vdict-tag">Verdict</div>
            <div class="vdict-label">{label.upper()}</div>
            <div class="vdict-conf">Confidence: {confidence:.1f}%</div>
        </div>
        <div class="bar-section">
            <div class="bar-row">
                <div class="bar-name">Real</div>
                <div class="bar-track">
                    <div class="bar-fill fill-real" style="width:{p_real:.1f}%"></div>
                </div>
                <div class="bar-pct">{p_real:.1f}%</div>
            </div>
            <div class="bar-row">
                <div class="bar-name">Fake</div>
                <div class="bar-track">
                    <div class="bar-fill fill-fake" style="width:{p_fake:.1f}%"></div>
                </div>
                <div class="bar-pct">{p_fake:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_video_verdict(result: dict):
    v    = result["label"]
    css  = "verdict-real" if v == "Real" else "verdict-fake"
    vcss = "vdict-real"   if v == "Real" else "vdict-fake"
    fake_frames = sum(1 for s in result["frame_scores"] if s >= 0.5)
    total       = len(result["frame_scores"])
    st.markdown(f"""
    <div class="verdict-wrap {css}">
        <div class="{vcss}">
            <div class="vdict-tag">Verdict</div>
            <div class="vdict-label">{v.upper()}</div>
            <div class="vdict-conf">
                {fake_frames} / {total} frames flagged fake
                &nbsp;·&nbsp; Avg P(Fake): {result['p_fake']:.1f}%
            </div>
        </div>
        <div class="bar-section">
            <div class="bar-row">
                <div class="bar-name">Real</div>
                <div class="bar-track">
                    <div class="bar-fill fill-real" style="width:{result['p_real']:.1f}%"></div>
                </div>
                <div class="bar-pct">{result['p_real']:.1f}%</div>
            </div>
            <div class="bar-row">
                <div class="bar-name">Fake</div>
                <div class="bar-track">
                    <div class="bar-fill fill-fake" style="width:{result['p_fake']:.1f}%"></div>
                </div>
                <div class="bar-pct">{result['p_fake']:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_frame_grid(frames: list, scores: list, timestamps: list):
    st.markdown('<div class="frames-header">// Sampled Frames</div>', unsafe_allow_html=True)
    n_show = min(5, len(frames))
    cols   = st.columns(n_show)
    for i in range(n_show):
        color = _score_color(scores[i] * 100)
        with cols[i]:
            st.image(Image.fromarray(frames[i]), use_container_width=True)
            st.markdown(
                f"<div class='frame-label' style='color:{color}'>"
                f"{scores[i]:.2f} · {timestamps[i]:.1f}s"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_img, tab_vid = st.tabs(["🖼️  Image Analysis", "🎬  Video Analysis"])


# ── IMAGE TAB ─────────────────────────────────────────────────────────────────
with tab_img:
    st.markdown(
        "<p style='color:var(--muted);font-size:0.83rem;margin-bottom:1.2rem;"
        "font-family:var(--mono)'>Upload an image to check if it's authentic or AI-generated.</p>",
        unsafe_allow_html=True,
    )

    uploaded_img = st.file_uploader(
        "Image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_uploader",
        label_visibility="collapsed",
    )

    img_btn = st.button("▶  ANALYZE IMAGE", key="img_btn", disabled=not bool(uploaded_img))

    if uploaded_img and img_btn:
        pil_img = Image.open(uploaded_img).convert("RGB")
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.image(pil_img, use_container_width=True)
            w, h = pil_img.size
            st.markdown(
                f"<div class='img-meta'>{uploaded_img.name} &nbsp;·&nbsp; "
                f"{w}×{h}px &nbsp;·&nbsp; {uploaded_img.size / 1024:.0f} KB</div>",
                unsafe_allow_html=True,
            )

        with col_res:
            with st.spinner("Running inference…"):
                t0  = time.time()
                res = predict_image(pil_img, MODEL, DEVICE)
                ms  = (time.time() - t0) * 1000

            render_verdict(res["label"], res["confidence"], res["p_real"], res["p_fake"])

            bar_color = _score_color(res["p_fake"])
            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-lbl">P(Fake)</div>
                    <div class="stat-val" style="color:{bar_color}">{res['p_fake']:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-lbl">P(Real)</div>
                    <div class="stat-val">{res['p_real']:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-lbl">Inference</div>
                    <div class="stat-val">{ms:.0f} ms</div>
                </div>
                <div class="stat-box">
                    <div class="stat-lbl">Resolution</div>
                    <div class="stat-val">{w}×{h}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif not uploaded_img:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🖼️</div>
            <div class="empty-text">Upload an image<br>and click Analyze</div>
        </div>
        """, unsafe_allow_html=True)


# ── VIDEO TAB ─────────────────────────────────────────────────────────────────
with tab_vid:
    st.markdown(
        f"<p style='color:var(--muted);font-size:0.83rem;margin-bottom:1.2rem;"
        f"font-family:var(--mono)'>Upload a video for frame-by-frame analysis. "
        f"Samples {MAX_VIDEO_FRAMES} evenly-spaced frames per video.</p>",
        unsafe_allow_html=True,
    )

    uploaded_vid = st.file_uploader(
        "Video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="vid_uploader",
        label_visibility="collapsed",
    )

    vid_btn = st.button("▶  ANALYZE VIDEO", key="vid_btn", disabled=not bool(uploaded_vid))

    if uploaded_vid:
        st.video(uploaded_vid)
        st.markdown(
            f"<p style='font-family:var(--mono);font-size:0.7rem;color:var(--muted);margin-top:4px'>"
            f"{uploaded_vid.name} &nbsp;·&nbsp; {uploaded_vid.size / 1e6:.1f} MB</p>",
            unsafe_allow_html=True,
        )

    if uploaded_vid and vid_btn:
        suffix = Path(uploaded_vid.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_vid.getvalue())
            tmp_path = tmp.name

        try:
            meta = get_video_meta(tmp_path)

            with st.spinner("Extracting frames…"):
                frames, timestamps = extract_frames(tmp_path, n_frames=MAX_VIDEO_FRAMES)

            if not frames:
                st.error("Could not read video. Try another file.")
            else:
                t0 = time.time()
                with st.spinner("Analysing frames…"):
                    result = predict_frames(frames, MODEL, DEVICE)
                elapsed = time.time() - t0

                col_info, col_res = st.columns([1, 1], gap="large")

                with col_info:
                    st.markdown(f"""
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-lbl">Duration</div>
                            <div class="stat-val">{meta.get('duration','—')}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">Resolution</div>
                            <div class="stat-val">{meta.get('resolution','—')}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">FPS</div>
                            <div class="stat-val">{meta.get('fps','—')}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">Frames Analysed</div>
                            <div class="stat-val">{len(result['frame_scores'])}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">Fake Frames</div>
                            <div class="stat-val">{sum(1 for s in result['frame_scores'] if s >= 0.5)}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">Inference</div>
                            <div class="stat-val">{elapsed:.1f}s</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_res:
                    render_video_verdict(result)

                render_frame_grid(frames, result["frame_scores"], timestamps)

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    elif not uploaded_vid:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <div class="empty-text">Upload a video<br>and click Analyze</div>
        </div>
        """, unsafe_allow_html=True)
