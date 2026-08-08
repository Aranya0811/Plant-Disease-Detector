"""OpenCV preprocessing applied before model inference (and optionally at train time)."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """Improve contrast under uneven lighting — common in field leaf photos."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_leaf_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Keep green-ish pixels; black out background. Use with care."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return masked


def preprocess_image(
    image_bgr: np.ndarray,
    image_size: int = 224,
    use_clahe: bool = True,
    use_leaf_mask: bool = False,
) -> np.ndarray:
    """Resize + optional OpenCV enhancements. Returns BGR uint8 image."""
    if use_clahe:
        image_bgr = apply_clahe(image_bgr)

    if use_leaf_mask:
        image_bgr = apply_leaf_mask(image_bgr)

    return cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
