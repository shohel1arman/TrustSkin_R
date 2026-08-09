"""Perturbation families for Phase 5 robustness testing."""
import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

FAMILIES = ["gaussian_noise", "gaussian_blur", "brightness", "contrast",
            "jpeg", "occlusion", "rotation"]

_PARAMS = {
    "gaussian_noise": [None, 8, 16, 26, 40, 60],
    "gaussian_blur":  [None, 0.6, 1.1, 1.8, 2.6, 3.6],
    "brightness":     [None, 0.85, 0.72, 0.6, 0.5, 0.4],
    "contrast":       [None, 0.85, 0.72, 0.6, 0.5, 0.4],
    "jpeg":           [None, 50, 35, 25, 15, 8],
    "occlusion":      [None, 0.10, 0.18, 0.26, 0.35, 0.45],
    "rotation":       [None, 8, 15, 25, 35, 45],
}


def apply(img, family, severity, seed=0):
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family}")
    if severity < 1 or severity > 5:
        raise ValueError("severity must be 1..5")
    p = _PARAMS[family][severity]
    rng = np.random.default_rng(seed)

    if family == "gaussian_noise":
        arr = np.asarray(img).astype(np.float32)
        arr += rng.normal(0, p, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if family == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=p))
    if family == "brightness":
        return ImageEnhance.Brightness(img).enhance(p)
    if family == "contrast":
        return ImageEnhance.Contrast(img).enhance(p)
    if family == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(p))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if family == "occlusion":
        arr = np.asarray(img).copy()
        h, w = arr.shape[:2]
        side = int(min(h, w) * p)
        if side > 0:
            y = rng.integers(0, max(1, h - side))
            x = rng.integers(0, max(1, w - side))
            arr[y:y + side, x:x + side] = 0
        return Image.fromarray(arr)
    if family == "rotation":
        return img.rotate(p, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    raise AssertionError
