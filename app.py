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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0E1A;
    color: #E8EAF0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.8rem 0 1.8rem 0;
    border-bottom: 1px solid #1E2433;
    margin-bottom: 2.5rem;
}
.hero-left { display: flex; align-items: center; gap: 0.8rem; }
.hero-icon { font-size: 1.6rem; }
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-title span { color: #4ECDC4; }
.hero-right {
    font-size: 0.78rem;
    color: #4B5563;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-align: right;
}

.step-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4ECDC4;
    margin-bottom: 0.3rem;
}
.step-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 1rem;
}

.card {
    background: #111827;
    border: 1px solid #1E2433;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.result-card-tumor {
    background: linear-gradient(135deg, #111827, #0D1520);
    border: 1px solid #D97706;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-card-clear {
    background: linear-gradient(135deg, #111827, #0D1520);
    border: 1px solid #4ECDC4;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-eyebrow {
    font-size: 0.7rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
}
.result-class-tumor {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.3rem;
    line-height: 1.1;
}
.result-class-clear {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #4ECDC4;
    margin-bottom: 0.3rem;
    line-height: 1.1;
}
.result-conf-tumor { font-size: 1rem; color: #D97706; font-weight: 500; }
.result-conf-clear { font-size: 1rem; color: #4ECDC4; font-weight: 500; }

.auto-badge {
    display: inline-block;
    background: #0D2B2A;
    border: 1px solid #4ECDC4;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-size: 0.78rem;
    color: #4ECDC4;
    margin-bottom: 0.6rem;
}
.preprocess-note {
    font-size: 0.8rem;
    color: #6B7280;
    margin-top: 0.3rem;
}

.divider { border: none; border-top: 1px solid #1E2433; margin: 2rem 0; }

.stButton > button {
    background: #4ECDC4 !important;
    color: #0A0E1A !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 0.65rem 2rem !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em !important;
    width: auto !important;
}
.stButton > button:hover { background: #38B2AC !important; }
.stToggle label { color: #9CA3AF !important; font-size: 0.85rem !important; }
.stCheckbox label { color: #9CA3AF !important; font-size: 0.84rem !important; }
.stSlider label { color: #9CA3AF !important; font-size: 0.84rem !important; }
[data-testid="stFileUploader"] section {
    border: 2px dashed #1E2433 !important;
    border-radius: 10px !important;
    background: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <span class="hero-icon">🧠</span>
        <p class="hero-title">Neuro<span>Scan</span> AI</p>
    </div>
    <div class="hero-right">Brain Tumor Detection · Powered by Deep Learning</div>
</div>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_nifti(file_bytes):
    """Load a NIfTI file from bytes, return pixel array."""
    import nibabel as nib
    import tempfile, os
    suffix = ".nii.gz" if file_bytes.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes.read())
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        data = img.get_fdata(dtype=np.float32)
    finally:
        os.unlink(tmp_path)
    # Drop 4th dimension if present (take first volume)
    if data.ndim == 4:
        data = data[:, :, :, 0]
    # Reorient to standard (H, W, slices)
    data = np.rot90(data, k=1, axes=(0, 1))
    return data


def adaptive_preprocess(img: np.ndarray) -> tuple:
    """
    Adaptively choose and apply the best preprocessing pipeline for each image.
    Returns (processed_image, description_string).
    """
    steps = []
    out = img.copy().astype(np.float32)

    # 1. Normalize to 0-255
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps.append("normalisation")

    # 2. Assess brightness — if mean < 80 (dark scan), boost contrast more
    mean_brightness = out.mean()
    if mean_brightness < 80:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        out = clahe.apply(out)
        steps.append("CLAHE (high contrast boost — dark scan)")
    else:
        # Standard brain-masked histogram equalization
        mask = out > 15
        if mask.any():
            brain_pixels = out[mask]
            brain_eq = cv2.equalizeHist(brain_pixels.reshape(-1, 1))
            out[mask] = brain_eq.ravel()
        steps.append("brain-masked histogram equalisation")

    # 3. Assess noise level via Laplacian variance
    lap_var = cv2.Laplacian(out, cv2.CV_64F).var()
    if lap_var > 500:
        # High noise — apply stronger NLM denoising
        out = cv2.fastNlMeansDenoising(out, h=15, templateWindowSize=7, searchWindowSize=21)
        steps.append("NLM denoising (h=15 — high noise detected)")
    elif lap_var > 150:
        # Moderate noise — light Gaussian
        out = cv2.GaussianBlur(out, (3, 3), 0)
        steps.append("Gaussian blur 3×3 (moderate noise)")
    else:
        steps.append("no denoising (clean scan)")

    return out, " · ".join(steps)


# ── Step 01: Upload ───────────────────────────────────────────────────────────
st.markdown('<p class="step-eyebrow">Step 01</p><p class="step-title">Upload MRI Scan</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload",
    type=["jpg", "jpeg", "png", "webp", "dcm", "nii", "nii.gz"],
    label_visibility="collapsed"
)
st.markdown('<p class="preprocess-note">Supports JPG · PNG · WebP · DICOM (.dcm) · NIfTI (.nii, .nii.gz)</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    fname = uploaded_file.name.lower()
    is_dicom = fname.endswith(".dcm")
    is_nifti = fname.endswith(".nii") or fname.endswith(".nii.gz")
    pixel_array = None
    img_array = None

    # ── Parse file ────────────────────────────────────────────────────────────
    if is_dicom:
        dicom = pydicom.dcmread(uploaded_file)
        pixel_array = dicom.pixel_array.squeeze()
    elif is_nifti:
        pixel_array = load_nifti(uploaded_file)
    else:
        img = Image.open(uploaded_file)
        img_array = np.array(img.convert("L")).squeeze()

    # ── Slice selector for multi-slice ────────────────────────────────────────
    if pixel_array is not None and len(pixel_array.shape) == 3:
        n_slices = pixel_array.shape[2] if is_nifti else pixel_array.shape[0]
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="step-eyebrow">Multi-Slice File Detected</p>', unsafe_allow_html=True)
        st.caption(f"{n_slices} slices found — select the slice to analyse.")

        if "slice_idx" not in st.session_state:
            st.session_state.slice_idx = n_slices // 2

        col_prev, col_slider, col_next = st.columns([1, 10, 1])
        with col_prev:
            st.write("")
            if st.button("◀"):
                st.session_state.slice_idx = max(0, st.session_state.slice_idx - 1)
        with col_slider:
            st.session_state.slice_idx = st.slider(
                "Slice", 0, n_slices - 1,
                st.session_state.slice_idx,
                label_visibility="collapsed"
            )
        with col_next:
            st.write("")
            if st.button("▶"):
                st.session_state.slice_idx = min(n_slices - 1, st.session_state.slice_idx + 1)

        st.caption(f"Slice {st.session_state.slice_idx + 1} of {n_slices}")

        if is_nifti:
            img_array = pixel_array[:, :, st.session_state.slice_idx]
        else:
            img_array = pixel_array[st.session_state.slice_idx]

    elif pixel_array is not None:
        img_array = pixel_array

    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Scan preview ──────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(img_array, caption="Uploaded scan", use_container_width=True)
    with col_info:
        h, w = img_array.shape[:2]
        fmt = "DICOM" if is_dicom else "NIfTI" if is_nifti else uploaded_file.type
        st.markdown(f"""
        <div class="card">
            <p class="step-eyebrow">Scan Info</p>
            <p style="color:#9CA3AF;font-size:0.85rem;margin:0.3rem 0;">
                <span style="color:#E8EAF0;font-weight:500;">File</span>&nbsp;&nbsp;{uploaded_file.name}
            </p>
            <p style="color:#9CA3AF;font-size:0.85rem;margin:0.3rem 0;">
                <span style="color:#E8EAF0;font-weight:500;">Dimensions</span>&nbsp;&nbsp;{w} × {h} px
            </p>
            <p style="color:#9CA3AF;font-size:0.85rem;margin:0.3rem 0;">
                <span style="color:#E8EAF0;font-weight:500;">Format</span>&nbsp;&nbsp;{fmt}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Step 02: Preprocessing ────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-eyebrow">Step 02</p><p class="step-title">Image Preprocessing</p>', unsafe_allow_html=True)

    auto_mode = st.toggle("Auto preprocessing (recommended)", value=True)

    if auto_mode:
        denoised, applied_steps = adaptive_preprocess(img_array)
        st.markdown(f'<span class="auto-badge">Auto</span>', unsafe_allow_html=True)
        st.markdown(f'<p class="preprocess-note">Applied: {applied_steps}</p>', unsafe_allow_html=True)
    else:
        st.caption("Select your own preprocessing options.")
        col_a, col_b = st.columns(2)
        with col_a:
            do_normalize = st.checkbox("Normalize (0–255)", value=True)
            do_clahe = st.checkbox("CLAHE contrast enhancement")
            do_equalize = st.checkbox("Histogram equalization")
        with col_b:
            do_gaussian = st.checkbox("Gaussian blur")
            do_median = st.checkbox("Median filter")
            do_nlm = st.checkbox("Non-local means denoising")

        processed = img_array.copy()
        if do_normalize:
            processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if do_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        if do_equalize:
            mask = processed > 15
            if mask.any():
                brain_pixels = processed[mask]
                brain_eq = cv2.equalizeHist(brain_pixels.reshape(-1, 1))
                processed[mask] = brain_eq.ravel()

        denoised = processed.copy()
        if do_gaussian:
            gk = st.slider("Gaussian kernel", 1, 15, 3, step=2)
            denoised = cv2.GaussianBlur(denoised, (gk, gk), 0)
        if do_median:
            mk = st.slider("Median kernel", 1, 15, 3, step=2)
            denoised = cv2.medianBlur(denoised, mk)
        if do_nlm:
            h_val = st.slider("NLM strength", 1, 30, 10)
            denoised = cv2.fastNlMeansDenoising(denoised, h=h_val)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array, caption="Original", use_container_width=True)
    with col2:
        st.image(denoised, caption="Processed", use_container_width=True)

    # ── Step 03: Diagnosis ────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-eyebrow">Step 03</p><p class="step-title">Run Diagnosis</p>', unsafe_allow_html=True)

    run_btn = st.button("Run Diagnosis →")

    if run_btn:
        try:
            from predict import load_model, preprocess, predict, get_gradcam, overlay_gradcam, CLASS_NAMES
            model = load_model("brain_tumor_detector.keras")

            input_tensor = preprocess(denoised)
            gray_128 = cv2.resize(denoised, (128, 128)).astype(np.float32) / 255.0

            with st.spinner("Analysing scan..."):
                class_name, confidence, all_probs = predict(model, input_tensor)

            detected_class = class_name.lower().replace(" ", "")

            if detected_class == "notumor":
                st.markdown(f"""
                <div class="result-card-clear">
                    <p class="result-eyebrow">Diagnosis Result</p>
                    <p class="result-class-clear">No Tumor Detected</p>
                    <p class="result-conf-clear">{confidence*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-tumor">
                    <p class="result-eyebrow">Tumor Detected</p>
                    <p class="result-class-tumor">{class_name}</p>
                    <p class="result-conf-tumor">{confidence*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)

            # ── Step 04: Grad-CAM ─────────────────────────────────────────────
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<p class="step-eyebrow">Step 04</p><p class="step-title">Grad-CAM Explanation</p>', unsafe_allow_html=True)
            st.caption("Regions highlighted in red most influenced the model's prediction.")

            with st.spinner("Generating Grad-CAM..."):
                class_idx = int(np.argmax(all_probs))
                heatmap = get_gradcam(model, input_tensor, class_idx)
                overlaid = overlay_gradcam(gray_128, heatmap)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(gray_128, caption="Original scan", clamp=True, use_container_width=True)
            with col2:
                st.image(heatmap, caption="Activation heatmap", use_container_width=True)
            with col3:
                st.image(overlaid, caption="Overlay", use_container_width=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            buf = io.BytesIO()
            Image.fromarray(overlaid).save(buf, format="PNG")
            st.download_button(
                label="Download Grad-CAM Report",
                data=buf.getvalue(),
                file_name=f"gradcam_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Diagnosis failed: {e}")
            st.info("Make sure `brain_tumor_detector.keras` and `predict.py` are in the repo root.")

else:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:4rem 0;color:#1F2937;">
        <p style="font-size:3.5rem;margin:0;filter:grayscale(0.3);">🧠</p>
        <p style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
            color:#374151;margin-top:0.8rem;">Upload a scan above to begin</p>
    </div>
    """, unsafe_allow_html=True)
