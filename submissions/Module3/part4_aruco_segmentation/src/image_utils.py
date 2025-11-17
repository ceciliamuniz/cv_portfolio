import os
from typing import Tuple, List, Dict, Optional
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


def detect_aruco_markers(img: np.ndarray, dict_type=cv2.aruco.DICT_6X6_50) -> Tuple[List, List, List]:
    """
    Detect ArUco markers in an image using OpenCV's built-in detection.
    
    ArUco markers are binary square fiducial markers designed for fast detection.
    They're commonly used in AR, camera calibration, and pose estimation.
    
    Args:
        img: Input image (BGR or grayscale)
        dict_type: ArUco dictionary type (default: 4x4 with 50 markers)
    
    Returns:
        corners: List of detected marker corners (each is 4x2 array)
        ids: List of detected marker IDs
        rejected: List of rejected marker candidates
    """
    gray = ensure_gray(img)
    
    # Get ArUco dictionary and detection parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Create detector and detect markers (newer API)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    return corners, ids, rejected


def get_marker_centers(corners: List) -> np.ndarray:
    """
    Calculate center points of detected ArUco markers.
    
    Args:
        corners: List of marker corners from detectMarkers
    
    Returns:
        centers: Nx2 array of (x, y) center coordinates
    """
    if len(corners) == 0:
        return np.array([])
    
    centers = []
    for corner in corners:
        # Each corner is shape (1, 4, 2) - 4 corners with (x,y) each
        cx = np.mean(corner[0][:, 0])
        cy = np.mean(corner[0][:, 1])
        centers.append([cx, cy])
    
    return np.array(centers)


def segment_object_from_markers(img: np.ndarray, marker_centers: np.ndarray, 
                                 expand_ratio: float = 1.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment object using ArUco markers as boundary hints.
    
    Strategy:
    1. Create convex hull from marker centers (approximate boundary)
    2. Expand hull region slightly to ensure object coverage
    3. Use GrabCut or watershed within this region
    4. Refine boundary using edge detection
    
    Args:
        img: Input BGR image
        marker_centers: Nx2 array of marker center positions
        expand_ratio: How much to expand the hull region (>1.0)
    
    Returns:
        mask: Binary segmentation mask
        boundary_contour: Refined object boundary contour
    """
    if len(marker_centers) < 3:
        # Not enough markers, return empty mask
        return np.zeros(img.shape[:2], dtype=np.uint8), np.array([])
    
    h, w = img.shape[:2]
    
    # Step 1: Create convex hull from markers
    hull_indices = cv2.convexHull(marker_centers.astype(np.float32), returnPoints=False)
    hull_points = marker_centers[hull_indices.flatten()]
    
    # Step 2: Expand the hull
    centroid = np.mean(hull_points, axis=0)
    expanded_hull = []
    for point in hull_points:
        # Move point away from centroid
        direction = point - centroid
        expanded_point = centroid + direction * expand_ratio
        expanded_hull.append(expanded_point)
    expanded_hull = np.array(expanded_hull, dtype=np.int32)
    
    # Step 3: Create initial mask from expanded hull
    initial_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(initial_mask, [expanded_hull], 255)
    
    # Step 4: Refine using GrabCut
    try:
        # Create rectangle that bounds the hull
        x, y, rw, rh = cv2.boundingRect(expanded_hull)
        rect = (max(0, x), max(0, y), min(w-x, rw), min(h-y, rh))
        
        grabcut_mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        # Initialize with our hull-based mask
        grabcut_mask[initial_mask == 255] = cv2.GC_PR_FGD  # Probably foreground
        grabcut_mask[initial_mask == 0] = cv2.GC_PR_BGD    # Probably background
        
        # Run GrabCut
        cv2.grabCut(img, grabcut_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        
        # Extract foreground
        refined_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except:
        # If GrabCut fails, use initial mask
        refined_mask = initial_mask
    
    # Step 5: Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Step 6: Find final boundary contour
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        boundary_contour = max(contours, key=cv2.contourArea)
    else:
        boundary_contour = np.array([])
    
    return refined_mask, boundary_contour


def draw_markers(img: np.ndarray, corners: List, ids: Optional[List]) -> np.ndarray:
    """
    Draw detected ArUco markers on the image.
    
    Args:
        img: Input BGR image
        corners: Marker corners from detectMarkers
        ids: Marker IDs
    
    Returns:
        Image with markers drawn
    """
    output = img.copy()
    
    if len(corners) == 0:
        return output
    
    # Draw marker borders and IDs
    cv2.aruco.drawDetectedMarkers(output, corners, ids)
    
    # Draw centers
    centers = get_marker_centers(corners)
    for i, (cx, cy) in enumerate(centers):
        cv2.circle(output, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        if ids is not None and i < len(ids):
            cv2.putText(output, f"ID:{ids[i][0]}", (int(cx)+10, int(cy)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return output


def draw_segmentation_result(img: np.ndarray, mask: np.ndarray, boundary: np.ndarray,
                             marker_centers: np.ndarray) -> np.ndarray:
    """
    Visualize segmentation result with boundary and markers.
    
    Args:
        img: Original BGR image
        mask: Binary segmentation mask
        boundary: Boundary contour
        marker_centers: Nx2 array of marker positions
    
    Returns:
        Visualization image
    """
    output = img.copy()
    
    # Draw semi-transparent mask
    colored_mask = np.zeros_like(img)
    colored_mask[mask == 255] = [0, 255, 0]  # Green
    output = cv2.addWeighted(output, 0.7, colored_mask, 0.3, 0)
    
    # Draw boundary contour (thick green line)
    if len(boundary) > 0:
        cv2.drawContours(output, [boundary], -1, (0, 255, 0), 3)
        
        # Draw bounding shapes
        x, y, w, h = cv2.boundingRect(boundary)
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Calculate and draw centroid
        M = cv2.moments(boundary)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(output, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(output, "Centroid", (cx+10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw marker positions
    for cx, cy in marker_centers:
        cv2.circle(output, (int(cx), int(cy)), 6, (255, 255, 0), -1)
        cv2.circle(output, (int(cx), int(cy)), 8, (0, 0, 0), 2)
    
    # Draw convex hull of markers
    if len(marker_centers) >= 3:
        hull = cv2.convexHull(marker_centers.astype(np.float32))
        cv2.polylines(output, [hull.astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
    
    return output


def generate_aruco_marker(marker_id: int, size: int = 200, 
                          dict_type=cv2.aruco.DICT_6X6_50) -> np.ndarray:
    """
    Generate an ArUco marker image for printing.
    
    Args:
        marker_id: ID of the marker (0-49 for DICT_4X4_50)
        size: Size of the marker in pixels
        dict_type: ArUco dictionary type
    
    Returns:
        Marker image (grayscale, size x size)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    return marker_img


def save_image(path: str, img: np.ndarray) -> None:
    """Save image to disk, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(path, img)


def panel_compare(images: Tuple[np.ndarray, ...], titles: Tuple[str, ...]) -> np.ndarray:
    """Create a side-by-side comparison panel with titles."""
    heights = [im.shape[0] for im in images]
    target_h = min(heights)
    resized = []
    for im in images:
        scale = target_h / im.shape[0]
        new_w = int(im.shape[1] * scale)
        resized.append(cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA))

    colored = []
    for im in resized:
        if im.ndim == 2:
            colored.append(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
        else:
            colored.append(im)

    pad_h = 40
    panels = []
    for im, title in zip(colored, titles):
        pad = np.ones((pad_h, im.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(pad, title, (10, pad_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(np.vstack([pad, im]))

    panel = np.hstack(panels)
    return panel
