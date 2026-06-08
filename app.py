"""
DeepFake Detector — Streamlit App
GenConViT-ED | ConvNeXt-tiny + Swin-tiny + Autoencoder
Weights auto-downloaded from Google Drive on first run, then cached.
Model loaded ONCE via @st.cache_resource — never reloaded on rerun.
"""

# ── Streamlit page config MUST be first ─────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Standard imports ─────────────────────────────────────────────────────
import os
import io
import time
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
#  CONFIG  (must exactly match training notebook CFG)
# ─────────────────────────────────────────────────────────────────────────
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
WEIGHT_PATH   = Path("weights/genconvit_ed_final.pth")

# ─────────────────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURE  (exact copy from training notebook)
# ─────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────
#  WEIGHT DOWNLOAD  (gdown, Google Drive)
# ─────────────────────────────────────────────────────────────────────────

def _download_weights() -> None:
    """Download weights from Google Drive if not already on disk."""
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


# ─────────────────────────────────────────────────────────────────────────
#  MODEL LOADER  — @st.cache_resource ensures model is loaded EXACTLY ONCE
#  across all sessions and reruns. Streamlit never calls this again.
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Download weights (if needed) and load model. Called once per process."""
    _download_weights()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GenConViTED(CFG).to(device)
    state  = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


# ─────────────────────────────────────────────────────────────────────────
#  PREPROCESSING  (matches training eval_tfm exactly)
#  Training used: Albumentations Normalize(IMAGENET_MEAN, IMAGENET_STD)
#  + ToTensorV2()  →  we replicate manually to avoid albumentations dep
# ─────────────────────────────────────────────────────────────────────────
_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)

def preprocess_frame(frame_rgb: np.ndarray) -> torch.Tensor:
    """
    uint8 HWC RGB → float32 CHW tensor, ImageNet-normalised.
    Matches albumentations Normalize + ToTensorV2 pipeline exactly.
    """
    img_size = CFG["img_size"]
    frame = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - _MEAN) / _STD            # HWC float32
    frame = frame.transpose(2, 0, 1)          # CHW
    return torch.from_numpy(np.ascontiguousarray(frame))


# ─────────────────────────────────────────────────────────────────────────
#  FRAME EXTRACTION  (sequential read — matches extract_frames_fast)
# ─────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, n_frames: int = CFG["frames_per_video"]):
    """
    Uniformly sample n_frames from a video via sequential read.
    Returns list of uint8 RGB arrays (224×224×3), or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    target_set = set(np.linspace(0, max(total - 1, 0), n_frames, dtype=int).tolist())
    frames, fidx = [], 0

    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fidx in target_set:
            fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(fr_rgb.astype(np.uint8))
        fidx += 1
        if len(frames) >= n_frames:
            break

    cap.release()
    while frames and len(frames) < n_frames:
        frames.append(frames[-1])
    return frames if frames else None


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


# ─────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────

@contextmanager
def _amp_ctx(device):
    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            yield
    else:
        yield


@torch.no_grad()
def predict_frames(frames_rgb: list, model: nn.Module, device: torch.device) -> dict:
    """
    frames_rgb: list of uint8 HWC RGB arrays.
    Returns dict with p_fake, p_real, label, confidence, frame_scores.
    """
    tensors = [preprocess_frame(f) for f in frames_rgb]
    batch   = torch.stack(tensors).to(device)       # (N, 3, 224, 224)

    with _amp_ctx(device):
        logits, _ = model(batch)                    # (N, 2)

    probs        = F.softmax(logits.float(), dim=1) # (N, 2)
    frame_scores = probs[:, 1].cpu().numpy()         # P(fake) per frame
    p_fake       = float(frame_scores.mean())
    p_real       = 1.0 - p_fake

    return {
        "p_fake"      : round(p_fake, 4),
        "p_real"      : round(p_real, 4),
        "label"       : "FAKE" if p_fake >= 0.5 else "REAL",
        "confidence"  : round(max(p_fake, p_real) * 100, 1),
        "frame_scores": frame_scores.tolist(),
    }


@torch.no_grad()
def predict_image(pil_img: Image.Image, model: nn.Module, device: torch.device) -> dict:
    """Single image → treat as one-frame video."""
    img_rgb  = np.array(pil_img.convert("RGB"))
    tensor   = preprocess_frame(img_rgb).unsqueeze(0).to(device)

    with _amp_ctx(device):
        logits, _ = model(tensor)

    probs  = F.softmax(logits.float(), dim=1)
    p_fake = float(probs[0, 1].item())
    p_real = 1.0 - p_fake

    return {
        "p_fake"    : round(p_fake, 4),
        "p_real"    : round(p_real, 4),
        "label"     : "FAKE" if p_fake >= 0.5 else "REAL",
        "confidence": round(max(p_fake, p_real) * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 0.65:
        return "#ff3860"
    if score >= 0.38:
        return "#ffb300"
    return "#00d97e"


def _sparkline_svg(scores: list) -> str:
    """Minimal SVG bar sparkline for per-frame scores."""
    n     = len(scores)
    W, H  = 360, 52
    bw    = max(2, W // n - 2)
    gap   = (W - n * bw) // max(n - 1, 1)
    bars  = ""
    for i, s in enumerate(scores):
        x     = i * (bw + gap)
        bh    = max(2, int(s * (H - 8)))
        y     = H - bh
        color = _score_color(s)
        bars += (f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" '
                 f'rx="2" fill="{color}" opacity="0.85"/>'
                 f'<text x="{x + bw//2}" y="{H+10}" text-anchor="middle" '
                 f'font-size="7" fill="#888" font-family="monospace">{i+1}</text>')

    thresh_y = int((1 - 0.5) * (H - 8))
    return f"""
    <svg width="100%" viewBox="0 0 {W} {H+14}" xmlns="http://www.w3.org/2000/svg">
        <line x1="0" y1="{thresh_y}" x2="{W}" y2="{thresh_y}"
              stroke="#ffb300" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>
        {bars}
    </svg>"""


# ─────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:       #0b0c10;
    --card:     #13141a;
    --card2:    #1a1b23;
    --border:   #252633;
    --accent:   #5b6ef5;
    --accent-g: #38d9a9;
    --danger:   #ff3860;
    --warn:     #ffb300;
    --safe:     #00d97e;
    --text:     #e4e4f0;
    --muted:    #5c5c7a;
    --mono:     'DM Mono', monospace;
    --body:     'Outfit', sans-serif;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main { background: var(--bg) !important; color: var(--text) !important; }

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
footer { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

* { font-family: var(--body) !important; box-sizing: border-box; }
code, .mono { font-family: var(--mono) !important; }

/* ── Top bar ── */
.topbar {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 1.1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.logo {
    display: flex; align-items: center; gap: 0.75rem;
}
.logo-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-g) 100%);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.logo-text {
    font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em;
    color: var(--text);
}
.logo-sub {
    font-family: var(--mono) !important;
    font-size: 0.6rem; color: var(--muted);
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 1px;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--safe); display: inline-block;
    box-shadow: 0 0 6px var(--safe);
    margin-right: 0.4rem;
}

/* ── Page wrap ── */
.page { padding: 2rem 2.5rem; }

/* ── Upload area ── */
.upload-card {
    background: var(--card);
    border: 1.5px dashed var(--border);
    border-radius: 14px;
    padding: 2rem;
    transition: border-color .2s;
}
.upload-card:hover { border-color: var(--accent); }
.upload-title {
    font-size: 0.72rem; font-family: var(--mono) !important;
    color: var(--muted); letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 1rem;
}

/* ── Verdict ── */
.verdict-card {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    position: relative;
    overflow: hidden;
}
.verdict-card.fake { background: rgba(255,56,96,0.07); border: 1px solid rgba(255,56,96,0.25); }
.verdict-card.real { background: rgba(0,217,126,0.07); border: 1px solid rgba(0,217,126,0.2); }

.verdict-tag {
    font-family: var(--mono) !important;
    font-size: 0.6rem; letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.verdict-tag.fake { color: var(--danger); }
.verdict-tag.real { color: var(--safe); }

.verdict-label {
    font-size: 3.2rem; font-weight: 800;
    letter-spacing: -0.04em; line-height: 1;
    margin-bottom: 0.3rem;
}
.verdict-label.fake { color: var(--danger); }
.verdict-label.real { color: var(--safe); }

.verdict-conf {
    font-family: var(--mono) !important;
    font-size: 0.75rem; color: var(--muted);
}

/* ── Score bar ── */
.score-wrap {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.4rem 1.6rem;
}
.score-header {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 0.9rem;
}
.score-title {
    font-family: var(--mono) !important;
    font-size: 0.62rem; color: var(--muted);
    letter-spacing: 0.12em; text-transform: uppercase;
}
.score-num {
    font-family: var(--mono) !important;
    font-size: 1.3rem; font-weight: 500;
}
.bar-track {
    background: var(--card2); border-radius: 6px;
    height: 10px; width: 100%; overflow: hidden;
}
.bar-fill {
    height: 10px; border-radius: 6px;
    transition: width .6s ease;
}
.bar-labels {
    display: flex; justify-content: space-between;
    font-family: var(--mono) !important;
    font-size: 0.58rem; color: var(--muted); margin-top: 0.4rem;
}

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; }
.stat-box {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.9rem 1.1rem;
}
.stat-lbl {
    font-family: var(--mono) !important;
    font-size: 0.58rem; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.stat-val {
    font-family: var(--mono) !important;
    font-size: 1.1rem; font-weight: 500; color: var(--text);
}

/* ── Frames ── */
.frames-title {
    font-family: var(--mono) !important;
    font-size: 0.6rem; color: var(--muted);
    letter-spacing: 0.15em; text-transform: uppercase;
    margin: 1.2rem 0 0.6rem;
    border-top: 1px solid var(--border); padding-top: 1rem;
}
.frame-score {
    font-family: var(--mono) !important;
    font-size: 0.65rem; text-align: center; margin-top: 4px;
    padding: 1px 0;
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] section {
    background: var(--card2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}
[data-testid="stImage"] img {
    border-radius: 8px; border: 1px solid var(--border);
}
[data-testid="stVideo"] video {
    border-radius: 10px; border: 1px solid var(--border);
    max-height: 260px;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 7px !important;
    border: none !important;
    padding: 0.45rem 1.2rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--card2) !important;
    color: var(--text) !important;
}
div.stSpinner > div {
    border-top-color: var(--accent) !important;
}
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Empty state ── */
.empty-state {
    height: 320px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; gap: 0.9rem;
}
.empty-icon { font-size: 2.5rem; opacity: 0.2; }
.empty-text {
    font-family: var(--mono) !important;
    font-size: 0.72rem; color: var(--muted);
    letter-spacing: 0.12em; text-transform: uppercase;
    text-align: center; line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
#  LOAD MODEL ONCE  (cached at process level — never reloads)
# ─────────────────────────────────────────────────────────────────────────

with st.spinner("Starting up…"):
    try:
        MODEL, DEVICE = load_model()
        _ready = True
    except Exception as _e:
        _ready = False
        _err   = str(_e)


# ─────────────────────────────────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────────────────────────────────

status_html = (
    '<span class="status-dot"></span>Ready'
    if _ready else
    '<span style="color:#ff3860">⚠ Model Error</span>'
)
st.markdown(f"""
<div class="topbar">
    <div class="logo">
        <div class="logo-icon">🔍</div>
        <div>
            <div class="logo-text">DeepFake Detector</div>
            <div class="logo-sub">Forensic Media Analysis</div>
        </div>
    </div>
    <div style="font-family:var(--mono);font-size:0.68rem;color:var(--muted);">
        {status_html}
    </div>
</div>
""", unsafe_allow_html=True)

if not _ready:
    st.error(f"Failed to load model: {_err}")
    st.stop()

st.markdown('<div class="page">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
#  MAIN LAYOUT  — two columns
# ─────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.4], gap="large")

# ── LEFT: Upload ──────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="upload-title">// Upload Media</div>', unsafe_allow_html=True)

    tab_video, tab_image = st.tabs(["VIDEO", "IMAGE"])

    uploaded_video = None
    uploaded_image = None

    with tab_video:
        uploaded_video = st.file_uploader(
            "video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            label_visibility="collapsed",
            key="vid_up",
        )
        if uploaded_video:
            st.video(uploaded_video)

    with tab_image:
        uploaded_image = st.file_uploader(
            "image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
            key="img_up",
        )
        if uploaded_image:
            st.image(uploaded_image, use_container_width=True)

    # ── Analyse button ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    has_input = bool(uploaded_video or uploaded_image)

    analyse = st.button(
        "▶  Analyse",
        disabled=not has_input,
        use_container_width=True,
    )

    # ── Guidance ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:1.2rem;font-family:var(--mono);font-size:0.65rem;
                color:var(--muted);line-height:1.9;">
        Supported formats<br>
        <span style="color:var(--text);">Video</span> · mp4 / avi / mov / mkv / webm<br>
        <span style="color:var(--text);">Image</span> · jpg / png / bmp / webp<br><br>
        Threshold: ≥ 50% → FAKE
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT: Results ────────────────────────────────────────────────────────
with col_right:

    if not has_input:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <div class="empty-text">Upload a video or image<br>and click Analyse</div>
        </div>
        """, unsafe_allow_html=True)

    elif analyse:
        # ── VIDEO path ────────────────────────────────────────────────────
        if uploaded_video:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_video.getvalue())
                tmp_path = tmp.name

            meta = get_video_meta(tmp_path)

            with st.spinner("Extracting frames…"):
                frames = extract_frames(tmp_path, n_frames=CFG["frames_per_video"])

            if not frames:
                st.error("Could not read video. Try another file.")
            else:
                t0 = time.time()
                with st.spinner("Analysing…"):
                    result = predict_frames(frames, MODEL, DEVICE)
                elapsed = time.time() - t0

                p_fake = result["p_fake"]
                label  = result["label"]
                conf   = result["confidence"]
                scores = result["frame_scores"]
                is_fake = label == "FAKE"
                bar_color = _score_color(p_fake)
                vc = "fake" if is_fake else "real"
                tag_icon = "⚠" if is_fake else "✓"

                # Verdict
                st.markdown(f"""
                <div class="verdict-card {vc}">
                    <div class="verdict-tag {vc}">{tag_icon}&nbsp; Analysis Complete
                        <span style="margin-left:auto;opacity:.5">{elapsed:.1f}s</span>
                    </div>
                    <div class="verdict-label {vc}">{label}</div>
                    <div class="verdict-conf">Confidence {conf}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Score bar
                pct = int(p_fake * 100)
                st.markdown(f"""
                <div class="score-wrap">
                    <div class="score-header">
                        <span class="score-title">Deepfake Score</span>
                        <span class="score-num" style="color:{bar_color}">{pct}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{pct}%;background:{bar_color}"></div>
                    </div>
                    <div class="bar-labels">
                        <span>REAL</span><span>50%</span><span>FAKE</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Stats
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
                        <div class="stat-val">{len(scores)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Frame sparkline
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="score-wrap"><div class="score-title" '
                    f'style="margin-bottom:.7rem">Per-frame Score</div>'
                    + _sparkline_svg(scores)
                    + "</div>",
                    unsafe_allow_html=True,
                )

                # Sampled frames
                st.markdown('<div class="frames-title">// Sampled Frames</div>',
                            unsafe_allow_html=True)
                n_show = min(5, len(frames))
                f_cols = st.columns(n_show)
                for ci, (fc, fr, sc) in enumerate(
                    zip(f_cols, frames[:n_show], scores[:n_show])
                ):
                    with fc:
                        st.image(Image.fromarray(fr), use_container_width=True)
                        c = _score_color(sc)
                        st.markdown(
                            f'<div class="frame-score" style="color:{c}">'
                            f'{sc:.2f}</div>',
                            unsafe_allow_html=True,
                        )

                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # ── IMAGE path ────────────────────────────────────────────────────
        elif uploaded_image:
            pil_img = Image.open(uploaded_image)
            w, h    = pil_img.size

            t0 = time.time()
            with st.spinner("Analysing image…"):
                result = predict_image(pil_img, MODEL, DEVICE)
            elapsed = time.time() - t0

            p_fake  = result["p_fake"]
            label   = result["label"]
            conf    = result["confidence"]
            is_fake = label == "FAKE"
            bar_color = _score_color(p_fake)
            vc = "fake" if is_fake else "real"
            tag_icon = "⚠" if is_fake else "✓"
            pct = int(p_fake * 100)

            # Verdict
            st.markdown(f"""
            <div class="verdict-card {vc}">
                <div class="verdict-tag {vc}">{tag_icon}&nbsp; Analysis Complete
                    <span style="margin-left:auto;opacity:.5">{elapsed:.2f}s</span>
                </div>
                <div class="verdict-label {vc}">{label}</div>
                <div class="verdict-conf">Confidence {conf}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Score bar
            st.markdown(f"""
            <div class="score-wrap">
                <div class="score-header">
                    <span class="score-title">Deepfake Score</span>
                    <span class="score-num" style="color:{bar_color}">{pct}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct}%;background:{bar_color}"></div>
                </div>
                <div class="bar-labels">
                    <span>REAL</span><span>50%</span><span>FAKE</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Image stats
            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-lbl">Resolution</div>
                    <div class="stat-val">{w}×{h}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-lbl">P(Fake)</div>
                    <div class="stat-val" style="color:{bar_color}">{p_fake:.4f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-lbl">P(Real)</div>
                    <div class="stat-val">{result['p_real']:.4f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-lbl">Time</div>
                    <div class="stat-val">{elapsed:.2f}s</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # File uploaded but button not clicked yet
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">▶</div>
            <div class="empty-text">Click Analyse<br>to run detection</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close .page

# ── Bottom bar ────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid var(--border);padding:0.9rem 2.5rem;
            display:flex;justify-content:space-between;align-items:center;
            background:var(--card);">
    <span style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);">
        For research and educational use only
    </span>
    <span style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);">
        224×224 · 15 frames/video · Threshold 50%
    </span>
</div>
""", unsafe_allow_html=True)
