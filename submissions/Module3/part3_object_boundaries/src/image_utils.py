import os
from typing import Tuple, List
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


def find_boundaries_contour(img: np.ndarray, method: str = 'otsu') -> Tuple[np.ndarray, List, np.ndarray]:
    """
    Find object boundaries using contour detection.
    
    Classical CV approach:
    1. Convert to grayscale
    2. Apply thresholding (Otsu's method or adaptive)
    3. Morphological operations to clean up
    4. Find contours
    5. Filter contours by area
    
    Args:
        img: Input BGR image
        method: 'otsu', 'adaptive', or 'canny'
    
    Returns:
        binary_mask: Binary segmentation mask
        contours: List of detected contours
        hierarchy: Contour hierarchy
    """
    gray = ensure_gray(img)
    
    # Step 1: Thresholding to segment foreground/background
    if method == 'otsu':
        # Otsu's automatic threshold selection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        # Adaptive threshold for varying lighting
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    elif method == 'canny':
        # Edge-based approach
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.Canny(blur, 50, 150)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Step 2: Morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Step 3: Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return binary, contours, hierarchy


def find_boundaries_grabcut(img: np.ndarray, margin_percent: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find object boundaries using GrabCut algorithm.
    
    GrabCut is an iterative graph-cut based segmentation:
    1. Initialize rectangular ROI (assumes object is centered)
    2. Build Gaussian Mixture Models for foreground/background
    3. Iteratively refine segmentation using min-cut
    
    Args:
        img: Input BGR image
        margin_percent: Margin from edges for initial rectangle (0.0-0.5)
    
    Returns:
        mask: Segmentation mask (0=bg, 1=fg, 2=probable_bg, 3=probable_fg)
        fg_mask: Binary foreground mask (255=foreground, 0=background)
    """
    h, w = img.shape[:2]
    
    # Initialize rectangle (assume object is in center, not touching edges)
    margin_h = int(h * margin_percent)
    margin_w = int(w * margin_percent)
    rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
    
    # GrabCut requires float64 internally
    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    
    # Run GrabCut (5 iterations)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask: foreground = 1 or 3
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    return mask, fg_mask


def draw_contours_with_info(img: np.ndarray, contours: List, min_area: float = 100) -> np.ndarray:
    """
    Draw contours and their bounding shapes on the image.
    
    Visualizations:
    - Contour outline (green)
    - Bounding rectangle (blue)
    - Minimum enclosing circle (cyan)
    - Convex hull (yellow)
    
    Args:
        img: Input image (BGR)
        contours: List of contours
        min_area: Minimum contour area to draw
    
    Returns:
        Image with drawn contours and boundaries
    """
    output = img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # Draw contour outline (green)
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        
        # Bounding rectangle (blue)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Minimum enclosing circle (cyan)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(output, (int(cx), int(cy)), int(radius), (255, 255, 0), 2)
        
        # Convex hull (yellow)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(output, [hull], -1, (0, 255, 255), 2)
        
        # Add area text
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx_text = int(M["m10"] / M["m00"])
            cy_text = int(M["m01"] / M["m00"])
            cv2.putText(output, f"Area: {int(area)}", (cx_text - 40, cy_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output


def extract_largest_object(img: np.ndarray, contours: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the largest object from the image based on contours.
    
    Returns:
        segmented: Image with only the largest object
        mask: Binary mask of the largest object
    """
    if len(contours) == 0:
        return img.copy(), np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask for largest object
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # Extract object using mask
    segmented = cv2.bitwise_and(img, img, mask=mask)
    
    return segmented, mask


def find_precise_boundaries(mask: np.ndarray) -> Tuple[List, dict]:
    """
    Find precise boundary information from a binary mask.
    
    Returns:
        contours: Precise boundary contours
        info: Dictionary with boundary metrics (area, perimeter, bounding box, etc.)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return [], {}
    
    # Get largest contour
    cnt = max(contours, key=cv2.contourArea)
    
    # Calculate boundary metrics
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    
    # Fit ellipse if enough points
    ellipse = None
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
    
    info = {
        'area': area,
        'perimeter': perimeter,
        'bounding_box': (x, y, w, h),
        'min_enclosing_circle': ((int(cx), int(cy)), int(radius)),
        'centroid': (int(cx), int(cy)),
        'ellipse': ellipse,
        'num_points': len(cnt)
    }
    
    return contours, info


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
