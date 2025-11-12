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
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    return img


def detect_edges_canny(gray: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    """
    Detect edge keypoints using Canny edge detection algorithm.
    
    Algorithm steps (from lecture):
    1. Gaussian blur to reduce noise
    2. Compute gradient magnitude and direction using Sobel
    3. Non-maximum suppression to thin edges
    4. Double threshold to classify strong/weak edges
    5. Edge tracking by hysteresis to connect edges
    
    Returns binary edge map (255 = edge, 0 = non-edge).
    """
    gray = ensure_gray(gray)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    # Canny edge detection
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return edges


def detect_corners_harris(gray: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04, 
                          threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect corner keypoints using Harris corner detection algorithm.
    
    Algorithm steps (from lecture):
    1. Compute image gradients Ix and Iy using Sobel
    2. Compute products: Ix², Iy², Ix*Iy
    3. Compute weighted sum (Gaussian window) to get structure tensor M
    4. Calculate corner response: R = det(M) - k*trace(M)²
    5. Apply threshold and non-maximum suppression
    
    Returns:
    - corner_response: Harris response map (float)
    - corner_keypoints: Binary map of detected corners (255 = corner, 0 = non-corner)
    """
    gray = ensure_gray(gray)
    
    # Compute Harris corner response
    # cv2.cornerHarris implements the Harris corner detection algorithm
    corner_response = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
    
    # Threshold: keep strong corners only
    threshold_value = threshold * corner_response.max()
    corner_keypoints = np.zeros_like(gray)
    corner_keypoints[corner_response > threshold_value] = 255
    
    # Optional: Apply non-maximum suppression to get cleaner corner points
    corner_keypoints = non_max_suppression(corner_response, threshold_value)
    
    return corner_response, corner_keypoints


def non_max_suppression(response: np.ndarray, threshold: float, window_size: int = 3) -> np.ndarray:
    """
    Apply non-maximum suppression to get local maxima.
    This ensures we only keep the strongest responses in a local neighborhood.
    """
    from scipy.ndimage import maximum_filter
    
    # Find local maxima
    local_max = maximum_filter(response, size=window_size)
    detected = (response == local_max) & (response > threshold)
    
    keypoints = np.zeros(response.shape, dtype=np.uint8)
    keypoints[detected] = 255
    return keypoints


def draw_keypoints_on_image(img: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int], 
                            marker_size: int = 3) -> np.ndarray:
    """
    Draw keypoints on an image as colored markers.
    
    Args:
        img: Original image (grayscale or BGR)
        keypoints: Binary keypoint map (255 = keypoint)
        color: BGR color tuple (e.g., (0, 255, 0) for green)
        marker_size: Size of the marker circle
    
    Returns:
        Image with keypoints drawn
    """
    if img.ndim == 2:
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        output = img.copy()
    
    # Find keypoint locations
    y_coords, x_coords = np.where(keypoints == 255)
    
    # Draw circles at keypoint locations
    for x, y in zip(x_coords, y_coords):
        cv2.circle(output, (x, y), marker_size, color, -1)
    
    return output


def save_image(path: str, img: np.ndarray) -> None:
    """Save image to disk, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(path, img)


def panel_compare(images: Tuple[np.ndarray, ...], titles: Tuple[str, ...]) -> np.ndarray:
    """Create a side-by-side comparison panel with titles."""
    # Normalize sizes to the smallest height
    heights = [im.shape[0] for im in images]
    target_h = min(heights)
    resized = []
    for im in images:
        scale = target_h / im.shape[0]
        new_w = int(im.shape[1] * scale)
        resized.append(cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA))

    # Convert to 3-channel for annotation
    colored = []
    for im in resized:
        if im.ndim == 2:
            colored.append(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
        else:
            colored.append(im)

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
