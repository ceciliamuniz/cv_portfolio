import os
from typing import Tuple
import numpy as np
import cv2


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Ensure the image is single-channel grayscale (uint8)."""
    if img is None:
        raise ValueError("Input image is None")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        # Normalize to 0-255 and convert
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    return img


def gradient_mag_angle(gray: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gradient magnitude and angle using Sobel derivatives.

    Returns (mag, angle_deg) as uint8 images scaled for visualization.
    Angle is mapped to [0, 180) degrees (unsigned gradients).
    """
    gray = ensure_gray(gray)
    # Compute Sobel gradients in x and y (float64 for precision)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    mag = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0  # [0,180)

    # Normalize for visualization
    mag_viz = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle_viz = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag_viz, angle_viz


def laplacian_of_gaussian(gray: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Compute Laplacian of Gaussian filtered image for edge detection-like response.

    Returns a uint8 visualization image scaled to [0,255].
    """
    gray = ensure_gray(gray)
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    # Laplacian (float for precision)
    log_resp = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
    # For visualization, take absolute and normalize
    log_abs = np.abs(log_resp)
    log_viz = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return log_viz


def save_image(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(path, img)


def panel_compare(images: Tuple[np.ndarray, ...], titles: Tuple[str, ...]) -> np.ndarray:
    """Create a simple side-by-side comparison panel with titles.
    Titles are rendered using OpenCV putText on top padding.
    """
    # Normalize sizes to the smallest height
    heights = [im.shape[0] for im in images]
    target_h = min(heights)
    resized = []
    for im in images:
        scale = target_h / im.shape[0]
        new_w = int(im.shape[1] * scale)
        resized.append(cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA))

    # Convert to 3-channel for annotation
    colored = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im for im in resized]

    # Top title bar
    pad_h = 40
    panels = []
    for im, title in zip(colored, titles):
        pad = np.ones((pad_h, im.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(pad, title, (10, pad_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(np.vstack([pad, im]))

    # Concatenate horizontally
    panel = np.hstack(panels)
    return panel
