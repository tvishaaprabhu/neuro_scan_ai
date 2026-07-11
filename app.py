import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import pydicom

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&family=DM+Serif+Display&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0c0f1a;
    color: #c9cdd9;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 4rem 6rem 4rem;
    max-width: 1200px;
}

/* Background radial glows matching Lovable */
body {
    background-image:
        radial-gradient(ellipse 60% 40% at 15% -5%, rgba(124,158,255,0.08), transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(155,107,255,0.06), transparent 70%);
    background-attachment: fixed;
}

/* ── Header ── */
.ns-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding: 2.8rem 0 2.8rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 3rem;
}
.ns-wordmark {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    color: #f0f2f8;
    letter-spacing: -0.01em;
    line-height: 1;
}
.ns-wordmark .scan { color: #7c9eff; }
.ns-wordmark .ai { color: #f0f2f8; }
.ns-tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #3d4460;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    line-height: 1.8;
    text-align: right;
    margin-top: 0.2rem;
}

/* ── Step eyebrow + title ── */
.ns-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7c9eff;
    margin-bottom: 0.35rem;
}
.ns-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    font-weight: 400;
    color: #f0f2f8;
    letter-spacing: -0.015em;
    margin-bottom: 1.4rem;
    line-height: 1.1;
}

/* ── Upload zone ── */
.ns-upload-hint {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #3d4460;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
[data-testid="stFileUploader"] section {
    background: linear-gradient(160deg, #131829 0%, #0e1220 100%) !important;
    border: 1px solid rgba(124,158,255,0.12) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] section > div {
    color: #8b91a3 !important;
}

/* ── Info card ── */
.ns-info-card {
    background: linear-gradient(160deg, #131829 0%, #0e1220 100%);
    border: 1px solid rgba(124,158,255,0.1);
    border-radius: 14px;
    padding: 0;
    overflow: hidden;
    height: 100%;
}
.ns-info-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.85rem 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.ns-info-row:last-child { border-bottom: none; }
.ns-info-key {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #3d4460;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.ns-info-val {
    font-size: 0.82rem;
    color: #c9cdd9;
    text-align: right;
    max-width: 58%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Image panels ── */
.ns-img-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: #3d4460;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.5rem;
}
.ns-img-label-active {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: #7c9eff;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.5rem;
}
.ns-img-label-active::before { content: "● "; }
.ns-img-wrap {
    background: #090c14;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.04);
}

/* ── Preprocessing toggle ── */
.ns-toggle-wrap {
    display: flex;
    gap: 0.3rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.25rem;
    width: fit-content;
    margin-bottom: 1.4rem;
}
.ns-toggle-btn {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    padding: 0.45rem 1.1rem;
    border-radius: 7px;
    border: none;
    cursor: pointer;
    transition: all 0.15s ease;
}
.ns-toggle-active {
    background: #1e2542;
    color: #f0f2f8;
    font-weight: 500;
}
.ns-toggle-inactive {
    background: transparent;
    color: #3d4460;
}
.ns-toggle-rec {
    font-family: 'Space Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.1em;
    color: #7c9eff;
    text-transform: uppercase;
    vertical-align: super;
    margin-left: 0.3rem;
}

/* ── Auto badge ── */
.ns-auto-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(124,158,255,0.07);
    border: 1px solid rgba(124,158,255,0.18);
    border-radius: 20px;
    padding: 0.22rem 0.7rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: #7c9eff;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.ns-auto-badge::before { content: "●  "; }
.ns-applied {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #3d4460;
    font-style: italic;
    margin-bottom: 1.2rem;
}
.ns-applied b { color: #8b91a3; font-style: normal; font-weight: 400; }

/* ── Manual checkboxes card ── */
.ns-manual-card {
    background: linear-gradient(160deg, #131829 0%, #0e1220 100%);
    border: 1px solid rgba(124,158,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
}
.stCheckbox label {
    color: #c9cdd9 !important;
    font-size: 0.88rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stCheckbox [data-testid="stCheckbox"] {
    padding: 0.4rem 0 !important;
}

/* ── Diagnosis run button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c9eff 0%, #9b6bff 100%) !important;
    color: #0c0f1a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.6rem !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Result card — tumor ── */
.ns-result-tumor {
    position: relative;
    overflow: hidden;
    background: linear-gradient(160deg, #131829 0%, #160e28 100%);
    border: 1px solid rgba(155,107,255,0.25);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 1.4rem 0;
}
.ns-result-tumor::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(155,107,255,0.12) 0%, transparent 65%);
    pointer-events: none;
}
/* ── Result card — clear ── */
.ns-result-clear {
    position: relative;
    overflow: hidden;
    background: linear-gradient(160deg, #0d1a1a 0%, #0a1520 100%);
    border: 1px solid rgba(100,220,180,0.2);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 1.4rem 0;
}
.ns-result-clear::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(100,220,180,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.ns-result-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.ns-result-eyebrow-tumor { color: #9b6bff; }
.ns-result-eyebrow-clear { color: #64dcb4; }
.ns-result-name-tumor {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #f0f2f8;
    line-height: 1;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
.ns-result-name-clear {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #64dcb4;
    line-height: 1;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
.ns-result-conf-tumor {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #9b6bff;
    letter-spacing: 0.06em;
}
.ns-result-conf-clear {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #64dcb4;
    letter-spacing: 0.06em;
}
.ns-result-divider {
    width: 200px;
    height: 1px;
    background: rgba(155,107,255,0.2);
    margin: 1.2rem auto;
}
.ns-result-divider-clear {
    width: 200px;
    height: 1px;
    background: rgba(100,220,180,0.15);
    margin: 1.2rem auto;
}

/* ── Grad-CAM image labels ── */
.ns-gradcam-note {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: #3d4460;
    font-style: italic;
    margin-bottom: 1.2rem;
}
.ns-cam-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.56rem;
    color: #3d4460;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    text-align: center;
    margin-top: 0.5rem;
}

/* ── Divider ── */
.ns-hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 3rem 0;
}

/* ── Empty state ── */
.ns-empty {
    text-align: center;
    padding: 6rem 0 4rem 0;
}
.ns-empty-icon {
    font-size: 3.5rem;
    opacity: 0.08;
    margin-bottom: 1.2rem;
}
.ns-empty-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    color: #1e2542;
    letter-spacing: 0.01em;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: rgba(124,158,255,0.08) !important;
    color: #7c9eff !important;
    border: 1px solid rgba(124,158,255,0.2) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
}

/* Slider */
.stSlider label { color: #3d4460 !important; font-size: 0.78rem !important; font-family: 'Space Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-header">
    <div class="ns-wordmark">Neuro<span class="scan">Scan</span> <span class="ai">AI</span></div>
    <div class="ns-tagline">Brain Tumor Detection<br>Powered by Deep Learning</div>
</div>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_nifti(uploaded):
    import nibabel as nib
    import tempfile, os
    suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        data = img.get_fdata(dtype=np.float32)
    finally:
        os.unlink(tmp_path)
    if data.ndim == 4:
        data = data[:, :, :, 0]
    data = np.rot90(data, k=1, axes=(0, 1))
    return data


def adaptive_preprocess(img: np.ndarray):
    steps = []
    out = cv2.normalize(img.copy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps.append("normalisation")
    mean_b = out.mean()
    if mean_b < 80:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        out = clahe.apply(out)
        steps.append("CLAHE (dark scan)")
    else:
        mask = out > 15
        if mask.any():
            out[mask] = cv2.equalizeHist(out[mask].reshape(-1, 1)).ravel()
        steps.append("histogram eq.")
    lap_var = cv2.Laplacian(out, cv2.CV_64F).var()
    if lap_var > 500:
        out = cv2.fastNlMeansDenoising(out, h=15, templateWindowSize=7, searchWindowSize=21)
        steps.append("NLM denoising (noisy)")
    elif lap_var > 150:
        out = cv2.GaussianBlur(out, (3, 3), 0)
        steps.append("Gaussian 3×3")
    else:
        steps.append("no denoising")
    return out, " · ".join(steps)


# ── Step 01: Upload ───────────────────────────────────────────────────────────
st.markdown('<p class="ns-eyebrow">// 01</p><p class="ns-title">Upload MRI Scan</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload",
    type=["jpg", "jpeg", "png", "webp", "dcm", "nii", "nii.gz"],
    label_visibility="collapsed"
)
st.markdown('<p class="ns-upload-hint">JPG · PNG · WebP · DICOM (.dcm) · NIfTI (.nii .nii.gz)</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    fname = uploaded_file.name.lower()
    is_dicom = fname.endswith(".dcm")
    is_nifti = fname.endswith(".nii") or fname.endswith(".nii.gz")
    pixel_array = None
    img_array = None

    if is_dicom:
        dicom = pydicom.dcmread(uploaded_file)
        pixel_array = dicom.pixel_array.squeeze()
    elif is_nifti:
        pixel_array = load_nifti(uploaded_file)
    else:
        img = Image.open(uploaded_file)
        img_array = np.array(img.convert("L")).squeeze()

    # Multi-slice scroller
    if pixel_array is not None and len(pixel_array.shape) == 3:
        n_slices = pixel_array.shape[2] if is_nifti else pixel_array.shape[0]
        st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
        st.markdown(f'<p class="ns-eyebrow">Multi-Slice · {n_slices} slices detected</p>', unsafe_allow_html=True)
        if "slice_idx" not in st.session_state:
            st.session_state.slice_idx = n_slices // 2
        col_prev, col_slider, col_next = st.columns([1, 12, 1])
        with col_prev:
            st.write("")
            if st.button("◀"):
                st.session_state.slice_idx = max(0, st.session_state.slice_idx - 1)
        with col_slider:
            st.session_state.slice_idx = st.slider(
                "slice", 0, n_slices - 1,
                st.session_state.slice_idx,
                label_visibility="collapsed"
            )
        with col_next:
            st.write("")
            if st.button("▶"):
                st.session_state.slice_idx = min(n_slices - 1, st.session_state.slice_idx + 1)
        st.caption(f"Slice {st.session_state.slice_idx + 1} / {n_slices}")
        img_array = pixel_array[:, :, st.session_state.slice_idx] if is_nifti else pixel_array[st.session_state.slice_idx]
    elif pixel_array is not None:
        img_array = pixel_array

    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Scan preview + info card
    st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
    col_img, col_info = st.columns([3, 2])
    with col_img:
        st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
        st.image(img_array, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_info:
        h, w = img_array.shape[:2]
        fmt = "DICOM" if is_dicom else "NIfTI" if is_nifti else "Image"
        noise = round(float(cv2.Laplacian(img_array, cv2.CV_64F).var()), 1)
        brightness = round(float(img_array.mean()), 1)
        st.markdown(f"""
        <div class="ns-info-card">
            <div class="ns-info-row">
                <span class="ns-info-key">File</span>
                <span class="ns-info-val">{uploaded_file.name}</span>
            </div>
            <div class="ns-info-row">
                <span class="ns-info-key">Format</span>
                <span class="ns-info-val">{fmt}</span>
            </div>
            <div class="ns-info-row">
                <span class="ns-info-key">Dimensions</span>
                <span class="ns-info-val">{w} × {h} px</span>
            </div>
            <div class="ns-info-row">
                <span class="ns-info-key">Mean Brightness</span>
                <span class="ns-info-val">{brightness}</span>
            </div>
            <div class="ns-info-row">
                <span class="ns-info-key">Noise (Laplacian)</span>
                <span class="ns-info-val">{noise}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Step 02: Preprocessing ────────────────────────────────────────────────
    st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
    st.markdown('<p class="ns-eyebrow">// 02</p><p class="ns-title">Preprocessing</p>', unsafe_allow_html=True)

    # Auto/Manual toggle using session state
    if "preprocess_mode" not in st.session_state:
        st.session_state.preprocess_mode = "auto"

    col_t1, col_t2, col_rest = st.columns([1.2, 1, 8])
    with col_t1:
        if st.button("Auto  RECOMMENDED" if st.session_state.preprocess_mode != "auto" else "✓ Auto  RECOMMENDED",
                     key="btn_auto"):
            st.session_state.preprocess_mode = "auto"
            st.rerun()
    with col_t2:
        if st.button("Manual" if st.session_state.preprocess_mode != "manual" else "✓ Manual",
                     key="btn_manual"):
            st.session_state.preprocess_mode = "manual"
            st.rerun()

    if st.session_state.preprocess_mode == "auto":
        denoised, steps_applied = adaptive_preprocess(img_array)
        st.markdown(f'<span class="ns-auto-badge">AUTO</span>', unsafe_allow_html=True)
        st.markdown(f'<p class="ns-applied"><b>Applied:</b> {steps_applied}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ns-manual-card">', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            do_norm = st.checkbox("Normalize", value=True)
            do_eq = st.checkbox("Histogram equalization")
            do_med = st.checkbox("Median filter")
        with col_b:
            do_clahe = st.checkbox("CLAHE contrast")
            do_gauss = st.checkbox("Gaussian blur")
            do_nlm = st.checkbox("Non-local means")
        st.markdown('</div>', unsafe_allow_html=True)

        proc = img_array.copy()
        if do_norm:
            proc = cv2.normalize(proc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if do_clahe:
            proc = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(proc)
        if do_eq:
            mask = proc > 15
            if mask.any():
                proc[mask] = cv2.equalizeHist(proc[mask].reshape(-1, 1)).ravel()
        denoised = proc.copy()
        if do_gauss:
            gk = st.slider("Gaussian kernel", 1, 15, 3, step=2)
            denoised = cv2.GaussianBlur(denoised, (gk, gk), 0)
        if do_med:
            mk = st.slider("Median kernel", 1, 15, 3, step=2)
            denoised = cv2.medianBlur(denoised, mk)
        if do_nlm:
            h_s = st.slider("NLM strength", 1, 30, 10)
            denoised = cv2.fastNlMeansDenoising(denoised, h=h_s)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="ns-img-label">ORIGINAL</p>', unsafe_allow_html=True)
        st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
        st.image(img_array, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="ns-img-label-active">PROCESSED</p>', unsafe_allow_html=True)
        st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
        st.image(denoised, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 03: Diagnosis ────────────────────────────────────────────────────
    st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
    st.markdown('<p class="ns-eyebrow">// 03</p><p class="ns-title">Diagnosis</p>', unsafe_allow_html=True)

    run_btn = st.button("Run Diagnosis ▶")

    if run_btn:
        try:
            from predict import load_model, preprocess, predict, get_gradcam, overlay_gradcam, CLASS_NAMES
            model = load_model("brain_tumor_detector.keras")
            input_tensor = preprocess(denoised)
            gray_128 = cv2.resize(denoised, (128, 128)).astype(np.float32) / 255.0

            with st.spinner("Analysing..."):
                class_name, confidence, all_probs = predict(model, input_tensor)

            detected_class = class_name.lower().replace(" ", "")

            if detected_class == "notumor":
                st.markdown(f"""
                <div class="ns-result-clear">
                    <p class="ns-result-eyebrow ns-result-eyebrow-clear">// Clear Scan</p>
                    <p class="ns-result-name-clear">No Tumor Detected</p>
                    <div class="ns-result-divider-clear"></div>
                    <p class="ns-result-conf-clear">{confidence*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ns-result-tumor">
                    <p class="ns-result-eyebrow ns-result-eyebrow-tumor">// Tumor Detected</p>
                    <p class="ns-result-name-tumor">{class_name}</p>
                    <div class="ns-result-divider"></div>
                    <p class="ns-result-conf-tumor">{confidence*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)

            # ── Step 04: Grad-CAM ─────────────────────────────────────────────
            st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
            st.markdown('<p class="ns-eyebrow">// 04</p><p class="ns-title">Grad-CAM Explanation</p>', unsafe_allow_html=True)
            st.markdown('<p class="ns-gradcam-note">Regions highlighted in red most influenced the model\'s prediction.</p>', unsafe_allow_html=True)

            with st.spinner("Generating heatmap..."):
                class_idx = int(np.argmax(all_probs))
                heatmap = get_gradcam(model, input_tensor, class_idx)
                overlaid = overlay_gradcam(gray_128, heatmap)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
                st.image(gray_128, clamp=True, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<p class="ns-cam-label">Original Scan</p>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
                st.image(heatmap, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<p class="ns-cam-label">Activation Heatmap</p>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
                st.image(overlaid, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<p class="ns-cam-label">Overlay</p>', unsafe_allow_html=True)

            st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
            buf = io.BytesIO()
            Image.fromarray(overlaid).save(buf, format="PNG")
            st.download_button(
                "⬇  Download Grad-CAM Report",
                data=buf.getvalue(),
                file_name=f"gradcam_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Diagnosis failed: {e}")
            st.info("Make sure `brain_tumor_detector.keras` and `predict.py` are in the repo root.")

else:
    st.markdown("""
    <div class="ns-empty">
        <div class="ns-empty-icon">⬡</div>
        <p class="ns-empty-text">Upload a scan above to begin</p>
    </div>
    """, unsafe_allow_html=True)
