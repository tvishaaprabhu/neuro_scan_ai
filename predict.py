"""
Inference + Grad-CAM in PyTorch.  Same public API as the old TF version,
so app.py barely changes:
    load_model, preprocess, predict, get_gradcam, overlay_gradcam, CLASS_NAMES
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import streamlit as st

from model import BrainTumorCNN, CLASS_NAMES, IMG_SIZE  # noqa: F401

torch.set_num_threads(1)   # keep CPU RAM/threads predictable on Streamlit Cloud


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str = "brain_tumor_detector.pt"):
    model = BrainTumorCNN()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image_array: np.ndarray) -> torch.Tensor:
    """uint8 HxW (or HxWxC) -> float tensor (1, 1, 128, 128) in [0,1]."""
    img = image_array
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    return torch.from_numpy(img)[None, None, :, :]


def predict(model, x: torch.Tensor):
    """-> (class_name, confidence, all_probs ndarray[4])"""
    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].numpy()
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx]), probs


def get_gradcam(model, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """
    Grad-CAM via forward/backward hooks on the last conv layer.
    Returns a JET-colourised RGB heatmap, uint8 (128, 128, 3).
    """
    acts, grads = {}, {}
    layer = model.gradcam_target_layer()

    h1 = layer.register_forward_hook(
        lambda m, i, o: acts.__setitem__("v", o.detach())
    )
    h2 = layer.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("v", go[0].detach())
    )

    try:
        model.zero_grad()
        out = model(x)                     # grad needed -> no inference_mode
        out[0, class_idx].backward()

        a = acts["v"][0]                   # (C, h, w)
        g = grads["v"][0]                  # (C, h, w)
        weights = g.mean(dim=(1, 2))       # channel importance
        cam = F.relu((weights[:, None, None] * a).sum(0))
        cam = cam.cpu().numpy()
    except Exception:
        cam = np.zeros((8, 8), np.float32)
    finally:
        h1.remove()
        h2.remove()
        model.zero_grad()

    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam.astype(np.float32), (IMG_SIZE, IMG_SIZE))

    colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_gradcam(gray_norm: np.ndarray, heatmap_rgb: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """gray_norm: float 128x128 in [0,1].  -> blended uint8 RGB."""
    base = cv2.cvtColor(np.uint8(gray_norm * 255), cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base, 1 - alpha, heatmap_rgb, alpha, 0)
