"""OpenCV webcam wrapper (Phase 6) and the shared CMU image preprocessing.

Camera.read() returns 240x320 RGB uint8 (the stored demo format). The model sees
a 216x288 center crop (preprocess_image), matching CMU diff_foam.
"""

from __future__ import annotations

import numpy as np

STORE_HW = (240, 320)
CROP_HW = (216, 288)


def center_crop(img: np.ndarray, crop_hw: tuple[int, int] = CROP_HW) -> np.ndarray:
    """Center crop the last two axes of (..., H, W) or the first two of (H, W, C)."""
    ch, cw = crop_hw
    if img.ndim == 3 and img.shape[-1] in (1, 3):  # H W C
        h, w = img.shape[:2]
        top, left = (h - ch) // 2, (w - cw) // 2
        return img[top : top + ch, left : left + cw]
    h, w = img.shape[-2:]
    top, left = (h - ch) // 2, (w - cw) // 2
    return img[..., top : top + ch, left : left + cw]


def preprocess_image(img_hwc: np.ndarray, crop_hw: tuple[int, int] = CROP_HW) -> np.ndarray:
    """uint8 HxWx3 RGB -> float32 3xhxw in [0,1], center cropped."""
    chw = np.moveaxis(np.asarray(img_hwc), -1, 0).astype(np.float32) / 255.0
    return center_crop(chw, crop_hw)


class Camera:
    def __init__(self, index: int | str = 0, size_hw: tuple[int, int] = STORE_HW) -> None:
        import cv2

        self._cv2 = cv2
        self.size_hw = size_hw
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("camera read failed")
        h, w = self.size_hw
        frame = self._cv2.resize(frame, (w, h), interpolation=self._cv2.INTER_AREA)
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()
