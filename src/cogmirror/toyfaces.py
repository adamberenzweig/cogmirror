"""Schematic faces for smoke tests. Not a substitute for real images: they
only show whether a readout moves at all between an obvious smile and an
obvious frown."""

from __future__ import annotations

from PIL import Image, ImageDraw

EXPRESSIONS = ("neutral", "happy", "sad", "angry")


def draw(expression: str, size: int = 768) -> Image.Image:
    s = size / 768
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    w = int(10 * s)
    d.ellipse([84 * s, 84 * s, 684 * s, 684 * s], fill=(255, 214, 170), outline=(60, 40, 30), width=w)
    for cx in (280, 488):  # eyes
        d.ellipse([(cx - 28) * s, 300 * s, (cx + 28) * s, 356 * s], fill=(30, 30, 30))
    box = [234 * s, 380 * s, 534 * s, 580 * s]
    if expression == "happy":
        d.arc(box, start=20, end=160, fill=(30, 30, 30), width=int(16 * s))
    elif expression in ("sad", "angry"):
        d.arc([234 * s, 480 * s, 534 * s, 680 * s], start=200, end=340, fill=(30, 30, 30), width=int(16 * s))
    else:
        d.line([284 * s, 520 * s, 484 * s, 520 * s], fill=(30, 30, 30), width=int(16 * s))
    if expression == "angry":
        d.line([220 * s, 230 * s, 330 * s, 280 * s], fill=(30, 30, 30), width=int(18 * s))
        d.line([548 * s, 230 * s, 438 * s, 280 * s], fill=(30, 30, 30), width=int(18 * s))
    elif expression == "sad":
        d.line([220 * s, 270 * s, 330 * s, 235 * s], fill=(30, 30, 30), width=int(14 * s))
        d.line([548 * s, 270 * s, 438 * s, 235 * s], fill=(30, 30, 30), width=int(14 * s))
    return img
