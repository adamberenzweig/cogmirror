"""Nuisance perturbations for the noise-floor check. Each maps a cropped PIL
image to a PIL image of the same size and should leave the expression
unchanged to a human viewer."""

from __future__ import annotations

import io


def _shift(dx: int, dy: int):
    def f(im):
        from PIL import Image

        return im.transform(im.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.BICUBIC,
                            fillcolor=(114, 114, 114))
    return f


def _brightness(factor: float):
    def f(im):
        from PIL import ImageEnhance

        return ImageEnhance.Brightness(im).enhance(factor)
    return f


def _jpeg(quality: int):
    def f(im):
        from PIL import Image

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return f


def _zoom(factor: float):
    def f(im):
        w, h = im.size
        cw, ch = round(w / factor), round(h / factor)
        left, top = (w - cw) // 2, (h - ch) // 2
        return im.crop((left, top, left + cw, top + ch)).resize((w, h))
    return f


# Shifts are in pixels of the 768-px model input; 16 px is about 2% of the side.
PERTURBATIONS = {
    "shift_r16": _shift(16, 0),
    "shift_d16": _shift(0, 16),
    "zoom_1.05": _zoom(1.05),
    "bright_0.8": _brightness(0.8),
    "bright_1.2": _brightness(1.2),
    "jpeg_40": _jpeg(40),
}


def get(name: str):
    return PERTURBATIONS[name]
