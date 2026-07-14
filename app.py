import os
import io
import gc
import base64
import requests

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import cv2
import pydicom

st.set_page_config(page_title="NeuroScan AI", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

# The MedSAM model now runs on a separate backend service (not in this
# Streamlit process), since loading it here was hitting Streamlit Cloud's
# RAM ceiling. Set MEDSAM_BACKEND_URL in Streamlit Cloud's "Secrets" panel
# (Settings -> Secrets) to your deployed backend's base URL, e.g.
# https://neuroscan-medsam.onrender.com
MEDSAM_BACKEND_URL = st.secrets.get("MEDSAM_BACKEND_URL", os.environ.get("MEDSAM_BACKEND_URL", ""))
MEDSAM_TIMEOUT_S = 45

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&family=DM+Serif+Display&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family:'DM Sans',sans-serif; background-color:#283359 !important; color:#d5d9e6;
}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"],
[data-testid="stToolbar"],[data-testid="stMain"] { background-color:#283359 !important; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:0 4rem 6rem 4rem; max-width:1200px; }
body {
    background-image:
        radial-gradient(ellipse 60% 40% at 15% -5%, rgba(124,158,255,.08), transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(155,107,255,.06), transparent 70%);
    background-attachment:fixed;
}
.ns-header { display:flex; align-items:flex-start; justify-content:space-between;
    padding:2.8rem 0; border-bottom:1px solid rgba(255,255,255,.06); margin-bottom:3rem; }
.ns-wordmark { font-family:'DM Serif Display',serif; font-size:2.2rem; color:#f0f2f8;
    letter-spacing:-.01em; line-height:1; }
.ns-wordmark .scan { color:#7c9eff; }
.ns-tagline { font-family:'Space Mono',monospace; font-size:.6rem; color:#8e97bd;
    letter-spacing:.14em; text-transform:uppercase; line-height:1.8; text-align:right; margin-top:.2rem; }
.ns-eyebrow { font-family:'Space Mono',monospace; font-size:.6rem; letter-spacing:.18em;
    text-transform:uppercase; color:#7c9eff; margin-bottom:.35rem; }
.ns-title { font-family:'DM Serif Display',serif; font-size:1.75rem; color:#f0f2f8;
    letter-spacing:-.015em; margin-bottom:1.4rem; line-height:1.1; }
.ns-upload-hint { font-family:'Space Mono',monospace; font-size:.6rem; color:#8e97bd;
    letter-spacing:.1em; text-transform:uppercase; margin-top:.5rem; }
[data-testid="stFileUploader"] section {
    background:linear-gradient(160deg,#35426f,#2e3a63) !important;
    border:1px solid rgba(124,158,255,.12) !important; border-radius:14px !important;
    padding:1.5rem !important; }
.ns-info-card { background:linear-gradient(160deg,#35426f,#2e3a63);
    border:1px solid rgba(124,158,255,.1); border-radius:14px; overflow:hidden; }
.ns-info-row { display:flex; justify-content:space-between; align-items:baseline;
    padding:.85rem 1.4rem; border-bottom:1px solid rgba(255,255,255,.04); }
.ns-info-row:last-child { border-bottom:none; }
.ns-info-key { font-family:'Space Mono',monospace; font-size:.62rem; color:#8e97bd;
    text-transform:uppercase; letter-spacing:.12em; }
.ns-info-val { font-size:.82rem; color:#c9cdd9; text-align:right; max-width:58%;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ns-img-label { font-family:'Space Mono',monospace; font-size:.58rem; color:#8e97bd;
    text-transform:uppercase; letter-spacing:.14em; margin-bottom:.5rem; }
.ns-img-label-active { font-family:'Space Mono',monospace; font-size:.58rem; color:#7c9eff;
    text-transform:uppercase; letter-spacing:.14em; margin-bottom:.5rem; }
.ns-img-label-active::before { content:"● "; }
.ns-img-wrap { background:#1b2340; border-radius:10px; overflow:hidden;
    border:1px solid rgba(255,255,255,.04); }
.ns-auto-badge { display:inline-flex; align-items:center; gap:.4rem;
    background:rgba(124,158,255,.07); border:1px solid rgba(124,158,255,.18);
    border-radius:20px; padding:.22rem .7rem; font-family:'Space Mono',monospace;
    font-size:.58rem; color:#7c9eff; letter-spacing:.12em; text-transform:uppercase;
    margin-bottom:.5rem; }
.ns-auto-badge::before { content:"●  "; }
.ns-applied { font-size:.78rem; color:#8e97bd; font-style:italic; margin-bottom:1.2rem; }
.ns-applied b { color:#8b91a3; font-style:normal; font-weight:400; }
.ns-manual-card { background:linear-gradient(160deg,#35426f,#2e3a63);
    border:1px solid rgba(124,158,255,.08); border-radius:12px;
    padding:1.2rem 1.6rem; margin-bottom:1.2rem; }
.stCheckbox label { color:#c9cdd9 !important; font-size:.88rem !important;
    font-family:'DM Sans',sans-serif !important; }
.stButton > button { background:linear-gradient(135deg,#7c9eff,#9b6bff) !important;
    color:#0c0f1a !important; font-family:'DM Sans',sans-serif !important;
    font-weight:500 !important; border:none !important; border-radius:8px !important;
    padding:.6rem 1.6rem !important; font-size:.88rem !important; letter-spacing:.02em !important; }
.stButton > button:hover { opacity:.88 !important; }
[data-testid="stDownloadButton"] button { background:rgba(124,158,255,.08) !important;
    color:#7c9eff !important; border:1px solid rgba(124,158,255,.2) !important;
    font-size:.84rem !important; border-radius:8px !important; padding:.5rem 1.2rem !important; }
.ns-result-tumor, .ns-result-clear { position:relative; overflow:hidden;
    border-radius:16px; padding:3rem 2rem; text-align:center; margin:1.4rem 0; }
.ns-result-tumor { background:linear-gradient(160deg,#35426f,#3d2f6e);
    border:1px solid rgba(155,107,255,.25); }
.ns-result-clear { background:linear-gradient(160deg,#2f5560,#2b4a63);
    border:1px solid rgba(100,220,180,.2); }
.ns-result-tumor::before, .ns-result-clear::before { content:''; position:absolute;
    top:-80px; right:-80px; width:280px; height:280px; pointer-events:none; }
.ns-result-tumor::before { background:radial-gradient(circle,rgba(155,107,255,.12),transparent 65%); }
.ns-result-clear::before { background:radial-gradient(circle,rgba(100,220,180,.08),transparent 65%); }
.ns-result-eyebrow { font-family:'Space Mono',monospace; font-size:.6rem;
    letter-spacing:.16em; text-transform:uppercase; margin-bottom:.8rem; }
.ns-eb-tumor { color:#9b6bff; } .ns-eb-clear { color:#64dcb4; }
.ns-result-name-tumor, .ns-result-name-clear { font-family:'DM Serif Display',serif;
    font-size:3rem; line-height:1; margin-bottom:.6rem; letter-spacing:-.02em; }
.ns-result-name-tumor { color:#f0f2f8; } .ns-result-name-clear { color:#64dcb4; }
.ns-conf-tumor, .ns-conf-clear { font-family:'Space Mono',monospace; font-size:.82rem;
    letter-spacing:.06em; }
.ns-conf-tumor { color:#9b6bff; } .ns-conf-clear { color:#64dcb4; }
.ns-rdiv { width:200px; height:1px; background:rgba(155,107,255,.2); margin:1.2rem auto; }
.ns-rdiv-clear { width:200px; height:1px; background:rgba(100,220,180,.15); margin:1.2rem auto; }
.ns-metrics { display:flex; gap:.8rem; margin-top:1.2rem; }
.ns-metric { flex:1; background:linear-gradient(160deg,#35426f,#2e3a63);
    border:1px solid rgba(124,158,255,.1); border-radius:12px;
    padding:1.1rem .8rem; text-align:center; }
.ns-metric-key { font-family:'Space Mono',monospace; font-size:.55rem; color:#8e97bd;
    text-transform:uppercase; letter-spacing:.14em; margin-bottom:.4rem; }
.ns-metric-val { font-family:'DM Serif Display',serif; font-size:1.5rem; color:#f0f2f8; line-height:1; }
.ns-metric-unit { font-family:'Space Mono',monospace; font-size:.6rem; color:#7c9eff; margin-left:.15rem; }
.ns-gradcam-note { font-size:.8rem; color:#8e97bd; font-style:italic; margin-bottom:1.2rem; }
.ns-cam-label { font-family:'Space Mono',monospace; font-size:.56rem; color:#8e97bd;
    text-transform:uppercase; letter-spacing:.14em; text-align:center; margin-top:.5rem; }
.ns-hr { border:none; border-top:1px solid rgba(255,255,255,.06); margin:3rem 0; }
.ns-empty { text-align:center; padding:6rem 0 4rem 0; }
.ns-empty-icon { font-size:3.5rem; opacity:.08; margin-bottom:1.2rem; }
.ns-empty-text { font-family:'DM Serif Display',serif; font-size:1rem; color:#5d6a9e; }
.stSlider label { color:#3d4460 !important; font-size:.78rem !important;
    font-family:'Space Mono',monospace !important; }
div[data-testid="stRadio"] { margin-bottom:1.2rem; }
div[data-testid="stRadio"] > label { display:none; }
div[data-testid="stRadio"] > div { display:flex !important; gap:.3rem !important;
    background:rgba(255,255,255,.03) !important; border:1px solid rgba(255,255,255,.06) !important;
    border-radius:10px !important; padding:.25rem !important; width:fit-content !important; }
div[data-testid="stRadio"] > div > label { display:block !important;
    font-family:'DM Sans',sans-serif !important; font-size:.84rem !important;
    color:#aab2cf !important; padding:.45rem 1.1rem !important; border-radius:7px !important;
    cursor:pointer !important; margin:0 !important; }
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background:#465391 !important; color:#f0f2f8 !important; }
div[data-testid="stRadio"] > div > label > div:first-child { display:none !important; }
div[data-testid="stTextInput"]:has(input[placeholder="__bbox__"]) {
    position:fixed !important; left:-9999px !important; top:0 !important;
    width:1px !important; height:1px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ns-header">
    <div class="ns-wordmark">Neuro<span class="scan">Scan</span> AI</div>
    <div class="ns-tagline">Brain Tumor Detection<br>Powered by Deep Learning</div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def load_nifti(uploaded):
    import nibabel as nib, tempfile
    suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name
    try:
        data = nib.load(path).get_fdata(dtype=np.float32)
    finally:
        os.unlink(path)
    if data.ndim == 4:
        data = data[:, :, :, 0]
    return np.rot90(data, k=1, axes=(0, 1))


def adaptive_preprocess(img):
    steps = []
    out = cv2.normalize(img.copy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps.append("normalisation")
    if out.mean() < 80:
        out = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(out)
        steps.append("CLAHE (dark scan)")
    else:
        m = out > 15
        if m.any():
            out[m] = cv2.equalizeHist(out[m].reshape(-1, 1)).ravel()
        steps.append("histogram eq.")
    lap = cv2.Laplacian(out, cv2.CV_64F).var()
    if lap > 500:
        out = cv2.fastNlMeansDenoising(out, h=15, templateWindowSize=7, searchWindowSize=21)
        steps.append("NLM denoising (noisy)")
    elif lap > 150:
        out = cv2.GaussianBlur(out, (3, 3), 0)
        steps.append("Gaussian 3×3")
    else:
        steps.append("no denoising")
    return out, " · ".join(steps)


def call_medsam_backend(img_rgb, bbox):
    """
    Sends the image + box prompt to the separate MedSAM backend service and
    returns a full-size uint8 mask (0/255). Replaces the old in-process
    download_medsam()/load_medsam()/run_medsam() — MedSAM itself now runs on
    its own service, not inside this Streamlit process.
    """
    if not MEDSAM_BACKEND_URL:
        raise RuntimeError(
            "MEDSAM_BACKEND_URL is not set. Add it in Streamlit Cloud's "
            "Settings -> Secrets, e.g. MEDSAM_BACKEND_URL = \"https://your-service.onrender.com\""
        )

    ok, png_bytes = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode image for upload")

    resp = requests.post(
        f"{MEDSAM_BACKEND_URL.rstrip('/')}/segment",
        files={"image": ("image.png", png_bytes.tobytes(), "image/png")},
        data={"bbox": ",".join(str(int(v)) for v in bbox)},
        timeout=MEDSAM_TIMEOUT_S,
    )
    resp.raise_for_status()
    payload = resp.json()

    mask_bytes = base64.b64decode(payload["mask_png_base64"])
    mask_arr = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    return (mask_arr > 0).astype(np.uint8)


def trace_otsu(gray, bbox):
    """Fallback if MedSAM can't run: Otsu inside the box."""
    x1, y1, x2, y2 = bbox
    h, w = gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((h, w), np.uint8)
    _, m = cv2.threshold(cv2.GaussianBlur(roi, (5, 5), 0), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((h, w), np.uint8)
    clean = np.zeros_like(m)
    cv2.drawContours(clean, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    full = np.zeros((h, w), np.uint8)
    full[y1:y2, x1:x2] = clean
    return full


# ═════════════════════════════════════════════════════════════════════════════
# 01 · Upload
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="ns-eyebrow">// 01</p><p class="ns-title">Upload MRI Scan</p>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload", type=["jpg", "jpeg", "png", "webp", "dcm", "nii", "gz"],
    label_visibility="collapsed")
st.markdown('<p class="ns-upload-hint">JPG · PNG · WebP · DICOM (.dcm) · NIfTI (.nii .nii.gz)</p>',
            unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown('<div class="ns-empty"><div class="ns-empty-icon">⬡</div>'
                '<p class="ns-empty-text">Upload a scan above to begin</p></div>',
                unsafe_allow_html=True)
    st.stop()

fname = uploaded_file.name.lower()
is_dicom = fname.endswith(".dcm")
is_nifti = fname.endswith(".nii") or fname.endswith(".nii.gz")
pixel_spacing_mm = 1.0
pixel_array = img_array = None

if is_dicom:
    dcm = pydicom.dcmread(uploaded_file)
    pixel_array = dcm.pixel_array.squeeze()
    try:
        pixel_spacing_mm = float(dcm.PixelSpacing[0])
    except Exception:
        pixel_spacing_mm = 1.0
elif is_nifti:
    pixel_array = load_nifti(uploaded_file)
else:
    img_array = np.array(Image.open(uploaded_file).convert("L")).squeeze()

if pixel_array is not None and pixel_array.ndim == 3:
    n = pixel_array.shape[2] if is_nifti else pixel_array.shape[0]
    st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="ns-eyebrow">Multi-Slice · {n} slices detected</p>',
                unsafe_allow_html=True)
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = n // 2
    c1, c2, c3 = st.columns([1, 12, 1])
    with c1:
        st.write("")
        if st.button("◀"):
            st.session_state.slice_idx = max(0, st.session_state.slice_idx - 1)
    with c2:
        st.session_state.slice_idx = st.slider(
            "slice", 0, n - 1, min(st.session_state.slice_idx, n - 1),
            label_visibility="collapsed")
    with c3:
        st.write("")
        if st.button("▶"):
            st.session_state.slice_idx = min(n - 1, st.session_state.slice_idx + 1)
    st.caption(f"Slice {st.session_state.slice_idx + 1} / {n}")
    img_array = (pixel_array[:, :, st.session_state.slice_idx] if is_nifti
                 else pixel_array[st.session_state.slice_idx])
elif pixel_array is not None:
    img_array = pixel_array

img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
ci, cf = st.columns([3, 2])
with ci:
    st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
    st.image(img_array, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with cf:
    h_o, w_o = img_array.shape[:2]
    fmt = "DICOM" if is_dicom else "NIfTI" if is_nifti else "Image"
    st.markdown(f"""
    <div class="ns-info-card">
      <div class="ns-info-row"><span class="ns-info-key">File</span>
        <span class="ns-info-val">{uploaded_file.name}</span></div>
      <div class="ns-info-row"><span class="ns-info-key">Format</span>
        <span class="ns-info-val">{fmt}</span></div>
      <div class="ns-info-row"><span class="ns-info-key">Dimensions</span>
        <span class="ns-info-val">{w_o} × {h_o} px</span></div>
      <div class="ns-info-row"><span class="ns-info-key">Mean Brightness</span>
        <span class="ns-info-val">{img_array.mean():.1f}</span></div>
      <div class="ns-info-row"><span class="ns-info-key">Noise (Laplacian)</span>
        <span class="ns-info-val">{cv2.Laplacian(img_array, cv2.CV_64F).var():.1f}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 02 · Preprocessing
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
st.markdown('<p class="ns-eyebrow">// 02</p><p class="ns-title">Preprocessing</p>',
            unsafe_allow_html=True)

mode = st.radio("mode", ["Auto", "Manual"], horizontal=True,
                label_visibility="collapsed", key="preprocess_radio")

if mode == "Auto":
    denoised, applied = adaptive_preprocess(img_array)
    st.markdown('<span class="ns-auto-badge">AUTO</span>', unsafe_allow_html=True)
    st.markdown(f'<p class="ns-applied"><b>Applied:</b> {applied}</p>', unsafe_allow_html=True)
else:
    st.markdown('<div class="ns-manual-card">', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        do_norm = st.checkbox("Normalize", value=True)
        do_eq = st.checkbox("Histogram equalization")
        do_med = st.checkbox("Median filter")
    with cb:
        do_clahe = st.checkbox("CLAHE contrast")
        do_gauss = st.checkbox("Gaussian blur")
        do_nlm = st.checkbox("Non-local means")
    st.markdown('</div>', unsafe_allow_html=True)

    p = img_array.copy()
    if do_norm:
        p = cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if do_clahe:
        p = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(p)
    if do_eq:
        m = p > 15
        if m.any():
            p[m] = cv2.equalizeHist(p[m].reshape(-1, 1)).ravel()
    denoised = p.copy()
    if do_gauss:
        denoised = cv2.GaussianBlur(denoised, (st.slider("Gaussian kernel", 1, 15, 3, 2),) * 2, 0)
    if do_med:
        denoised = cv2.medianBlur(denoised, st.slider("Median kernel", 1, 15, 3, 2))
    if do_nlm:
        denoised = cv2.fastNlMeansDenoising(denoised, h=st.slider("NLM strength", 1, 30, 10))

o1, o2 = st.columns(2)
with o1:
    st.markdown('<p class="ns-img-label">ORIGINAL</p>', unsafe_allow_html=True)
    st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
    st.image(img_array, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with o2:
    st.markdown('<p class="ns-img-label-active">PROCESSED</p>', unsafe_allow_html=True)
    st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
    st.image(denoised, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 03 · Diagnosis
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
st.markdown('<p class="ns-eyebrow">// 03</p><p class="ns-title">Diagnosis</p>',
            unsafe_allow_html=True)

if st.button("Run Diagnosis ▶"):
    st.session_state.diagnosis_done = True

if not st.session_state.get("diagnosis_done"):
    st.stop()

from predict import load_model, preprocess, predict, get_gradcam, overlay_gradcam

model = load_model("brain_tumor_detector.pt")

# The model was trained on RAW pixels (Grayscale -> Resize -> ToTensor).
# It has never seen a CLAHE'd / equalised / denoised image. Feeding the
# preprocessed version in is out-of-distribution input and measurably hurts
# accuracy. Preprocessing exists for the human eye and for Otsu — not for
# the CNN. So: classify the raw scan; display/segment the processed one.
x = preprocess(img_array)
gray128 = cv2.resize(img_array, (128, 128)).astype(np.float32) / 255.0

with st.spinner("Analysing…"):
    class_name, confidence, probs = predict(model, x)

detected = class_name.lower().replace(" ", "")

if detected == "notumor":
    st.markdown(f"""
    <div class="ns-result-clear">
      <p class="ns-result-eyebrow ns-eb-clear">// Clear Scan</p>
      <p class="ns-result-name-clear">No Tumor Detected</p>
      <div class="ns-rdiv-clear"></div>
      <p class="ns-conf-clear">{confidence*100:.1f}% confidence</p>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="ns-result-tumor">
      <p class="ns-result-eyebrow ns-eb-tumor">// Tumor Detected</p>
      <p class="ns-result-name-tumor">{class_name}</p>
      <div class="ns-rdiv"></div>
      <p class="ns-conf-tumor">{confidence*100:.1f}% confidence</p>
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 04 · Grad-CAM
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
st.markdown('<p class="ns-eyebrow">// 04</p><p class="ns-title">Grad-CAM Explanation</p>',
            unsafe_allow_html=True)
st.markdown("<p class='ns-gradcam-note'>Regions highlighted in red most influenced "
            "the model's prediction.</p>", unsafe_allow_html=True)

with st.spinner("Generating heatmap…"):
    heat = get_gradcam(model, x, int(np.argmax(probs)))
    overlaid = overlay_gradcam(gray128, heat)

g1, g2, g3 = st.columns(3)
for col, im, lbl in [(g1, gray128, "Original Scan"),
                     (g2, heat, "Activation Heatmap"),
                     (g3, overlaid, "Overlay")]:
    with col:
        st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
        st.image(im, clamp=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="ns-cam-label">{lbl}</p>', unsafe_allow_html=True)

buf = io.BytesIO()
Image.fromarray(overlaid).save(buf, format="PNG")
st.download_button("⬇  Download Grad-CAM Report", buf.getvalue(),
                   f"gradcam_{uploaded_file.name.rsplit('.', 1)[0]}.png", "image/png")

# ═════════════════════════════════════════════════════════════════════════════
# 05 · MedSAM segmentation
# ═════════════════════════════════════════════════════════════════════════════
if detected == "notumor":
    st.stop()

st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
st.markdown('<p class="ns-eyebrow">// 05</p><p class="ns-title">Tumor Segmentation</p>',
            unsafe_allow_html=True)
st.markdown("<p class='ns-gradcam-note'>Use the Grad-CAM heatmap as a guide. Draw a box "
            "around the tumor — MedSAM traces the boundary inside it.</p>",
            unsafe_allow_html=True)

img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
H, W = img_rgb.shape[:2]

DW = 520
DH = int(H * DW / W)
_b = io.BytesIO()
Image.fromarray(img_rgb).resize((DW, DH)).save(_b, format="PNG")
b64 = base64.b64encode(_b.getvalue()).decode()
sx, sy = round(W / DW, 6), round(H / DH, 6)

components.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=Space+Mono&display=swap');
#cv {{ border:1px solid rgba(124,158,255,.18); border-radius:10px;
      cursor:crosshair; display:block; background:#1b2340; }}
#co {{ font-family:'Space Mono',monospace; font-size:11px; color:#8e97bd;
      margin-top:8px; min-height:16px; letter-spacing:.06em; }}
#cb {{ margin-top:10px; padding:8px 20px; background:rgba(124,158,255,.08);
      color:#7c9eff; border:1px solid rgba(124,158,255,.25); border-radius:8px;
      cursor:pointer; font-family:'DM Sans',sans-serif; font-size:13px; font-weight:500; }}
#cb:hover {{ background:rgba(124,158,255,.16); }}
#dn {{ font-family:'Space Mono',monospace; font-size:11px; color:#64dcb4;
      margin-left:12px; letter-spacing:.06em; }}
</style>
<canvas id="cv" width="{DW}" height="{DH}"></canvas>
<div id="co"></div>
<button id="cb" onclick="cf()">Confirm Box</button><span id="dn"></span>
<script>
(function() {{
  const c=document.getElementById('cv'), x=c.getContext('2d'), im=new Image();
  let ax,ay,dr=false,bx=null;
  im.onload=()=>x.drawImage(im,0,0);
  im.src='data:image/png;base64,{b64}';
  c.addEventListener('mousedown',e=>{{const r=c.getBoundingClientRect();
    ax=e.clientX-r.left; ay=e.clientY-r.top; dr=true;}});
  c.addEventListener('mousemove',e=>{{ if(!dr) return;
    const r=c.getBoundingClientRect(), px=e.clientX-r.left, py=e.clientY-r.top;
    x.clearRect(0,0,c.width,c.height); x.drawImage(im,0,0);
    x.strokeStyle='#7c9eff'; x.lineWidth=2; x.setLineDash([5,3]);
    x.strokeRect(ax,ay,px-ax,py-ay);
    bx={{x1:Math.round(Math.min(ax,px)),y1:Math.round(Math.min(ay,py)),
         x2:Math.round(Math.max(ax,px)),y2:Math.round(Math.max(ay,py))}};
    document.getElementById('co').innerText=
      '('+bx.x1+', '+bx.y1+') → ('+bx.x2+', '+bx.y2+')';}});
  c.addEventListener('mouseup',()=>{{dr=false;}});
  window.cf=function() {{
    if(!bx) {{ alert('Draw a box first.'); return; }}
    const v=[Math.round(bx.x1*{sx}),Math.round(bx.y1*{sy}),
             Math.round(bx.x2*{sx}),Math.round(bx.y2*{sy})].join(',');
    for(const i of window.parent.document.querySelectorAll('input[type="text"]')) {{
      if(i.placeholder==='__bbox__') {{
        // React tracks the value internally, so set it through the native
        // setter or React will ignore the change.
        Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value')
          .set.call(i,v);
        i.dispatchEvent(new Event('input',{{bubbles:true}}));
        // Streamlit's text_input only COMMITS to the server on Enter or blur.
        // Without this, React updates but Python never sees the value.
        i.dispatchEvent(new KeyboardEvent('keydown',
          {{key:'Enter', code:'Enter', keyCode:13, which:13, bubbles:true}}));
        i.dispatchEvent(new Event('change',{{bubbles:true}}));
        i.blur();
        document.getElementById('dn').innerText='✓ CONFIRMED';
        return;
      }}
    }}
    document.getElementById('dn').innerText=v;
  }};
}})();
</script>
""", height=DH + 110)

raw = st.text_input("bbox", placeholder="__bbox__", key="bbox_receiver",
                    label_visibility="collapsed")

bbox = None
if raw and raw != "__bbox__":
    try:
        p = [int(v.strip()) for v in raw.split(",")]
        if len(p) == 4 and p[0] < p[2] and p[1] < p[3]:
            bbox = p
    except Exception:
        bbox = None

if not bbox:
    st.stop()

if st.button("Run Segmentation ▶", key="run_seg"):
    st.session_state.seg_done = True

if not st.session_state.get("seg_done"):
    st.stop()

engine = "MedSAM"
try:
    with st.spinner("Running MedSAM on the backend service…"):
        mask = call_medsam_backend(img_rgb, bbox)
except Exception as e:
    import traceback
    msg = str(e)

    # Show the FULL traceback on screen — same reasoning as before: a
    # one-line summary hides the actual cause. Screenshot this if it appears.
    with st.expander("⚠  MedSAM backend error — expand and screenshot this", expanded=True):
        st.caption(f"app.py build: backend-service-v1   |   {type(e).__name__}")
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)),
                language="text")

    if isinstance(e, requests.exceptions.Timeout):
        st.warning("MedSAM backend timed out (cold start or overload). "
                   "Falling back to Otsu thresholding.")
    elif isinstance(e, requests.exceptions.ConnectionError):
        st.warning("Couldn't reach the MedSAM backend service — check it's deployed "
                   "and MEDSAM_BACKEND_URL is set correctly. Falling back to Otsu thresholding.")
    else:
        st.warning(f"MedSAM backend call failed ({type(e).__name__}: {msg}). "
                   "Falling back to Otsu thresholding.")
    # Otsu benefits from the contrast enhancement, unlike the CNN.
    mask = trace_otsu(denoised, bbox)
    engine = "Otsu fallback"

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

box_img = img_rgb.copy()
cv2.rectangle(box_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (124, 158, 255), 2)

seg = img_rgb.copy()
if cnts:
    tint = np.zeros_like(seg)
    tint[mask > 0] = [155, 107, 255]
    seg = cv2.addWeighted(seg, 0.78, tint, 0.22, 0)
    cv2.drawContours(seg, cnts, -1, (100, 220, 180), 2)

s1, s2 = st.columns(2)
with s1:
    st.markdown('<p class="ns-img-label">BOX PROMPT</p>', unsafe_allow_html=True)
    st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
    st.image(box_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with s2:
    st.markdown(f'<p class="ns-img-label-active">{engine.upper()} SEGMENTATION</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
    st.image(seg, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if not cnts:
    st.warning("No region found inside the box. Try a tighter box around the tumor.")
    st.stop()

largest = max(cnts, key=cv2.contourArea)
area = int(np.count_nonzero(mask)) * (pixel_spacing_mm ** 2)
_, _, cw, ch = cv2.boundingRect(largest)
(_, _), radius = cv2.minEnclosingCircle(largest)
unit = "mm" if pixel_spacing_mm != 1.0 else "px"

st.markdown(f"""
<div class="ns-metrics">
  <div class="ns-metric"><p class="ns-metric-key">Tumor Area</p>
    <p class="ns-metric-val">{area:,.0f}<span class="ns-metric-unit">{unit}²</span></p></div>
  <div class="ns-metric"><p class="ns-metric-key">Width</p>
    <p class="ns-metric-val">{cw*pixel_spacing_mm:.0f}<span class="ns-metric-unit">{unit}</span></p></div>
  <div class="ns-metric"><p class="ns-metric-key">Height</p>
    <p class="ns-metric-val">{ch*pixel_spacing_mm:.0f}<span class="ns-metric-unit">{unit}</span></p></div>
  <div class="ns-metric"><p class="ns-metric-key">Max Diameter</p>
    <p class="ns-metric-val">{radius*2*pixel_spacing_mm:.0f}<span class="ns-metric-unit">{unit}</span></p></div>
</div>
""", unsafe_allow_html=True)

buf2 = io.BytesIO()
Image.fromarray(seg).save(buf2, format="PNG")
st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
st.download_button("⬇  Download Segmentation", buf2.getvalue(),
                   f"segmentation_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                   "image/png", key="dl_seg")

gc.collect()
