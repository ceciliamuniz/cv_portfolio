#!/usr/bin/env python3
"""
Smart ArUco detection with false positive filtering.
Finds the 4 real markers among many false positives.
"""

import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import os

def smart_marker_filtering(corners, ids, expected_count=4):
    """
    Filter detected markers to find the most likely real markers.
    Uses size consistency, spatial distribution, and marker quality.
    """
    if ids is None or len(ids) == 0:
        return [], []
    
    # Calculate marker properties
    marker_data = []
    for i, corner in enumerate(corners):
        # Get the 4 corners of the marker
        pts = corner[0]
        
        # Calculate area
        area = cv.contourArea(pts)
        
        # Calculate center
        center = np.mean(pts, axis=0)
        
        # Calculate aspect ratio and size consistency
        width = np.linalg.norm(pts[1] - pts[0])
        height = np.linalg.norm(pts[3] - pts[0])
        aspect_ratio = width / height if height > 0 else 0
        
        # Calculate marker "squareness" 
        # Perfect square has aspect ratio close to 1
        squareness = 1.0 - abs(1.0 - aspect_ratio)
        
        marker_data.append({
            'index': i,
            'id': ids[i][0],
            'center': center,
            'area': area,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'squareness': squareness
        })
    
    # Filter by area (remove too small/too large markers)
    areas = [m['area'] for m in marker_data]
    if len(areas) > 0:
        median_area = np.median(areas)
        area_std = np.std(areas)
        
        # Keep markers within reasonable area range
        filtered_markers = []
        for marker in marker_data:
            area_score = abs(marker['area'] - median_area) / (area_std + 1e-6)
            if area_score < 2.0 and marker['squareness'] > 0.5:  # Reasonably square
                filtered_markers.append(marker)
        
        marker_data = filtered_markers
    
    # If we have more than expected, use spatial distribution
    if len(marker_data) > expected_count:
        # Sort by area (prefer consistent sized markers)
        marker_data.sort(key=lambda x: -x['squareness'])
        
        # Take top candidates and ensure good spatial distribution
        selected = []
        min_distance = 50  # Minimum distance between markers
        
        for marker in marker_data:
            too_close = False
            for selected_marker in selected:
                dist = np.linalg.norm(marker['center'] - selected_marker['center'])
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(marker)
                
            if len(selected) >= expected_count:
                break
        
        marker_data = selected
    
    # Extract the filtered results
    if marker_data:
        filtered_indices = [m['index'] for m in marker_data]
        filtered_corners = [corners[i] for i in filtered_indices]
        filtered_ids = [ids[i] for i in filtered_indices]
        return filtered_corners, filtered_ids
    else:
        return [], []

def test_smart_detection():
    """Test smart detection on all images."""
    # Use permissive detection + smart filtering
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters()
    
    # Permissive parameters to catch all possible markers
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.minMarkerPerimeterRate = 0.005
    parameters.maxMarkerPerimeterRate = 8.0
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.minOtsuStdDev = 2.0
    parameters.errorCorrectionRate = 0.6
    
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    print("Smart ArUco Detection (Permissive + Intelligent Filtering)")
    print("=" * 70)
    
    total_raw = 0
    total_filtered = 0
    
    for i in range(10):
        img_path = f'images/IMG_378{i}.jpeg'
        if os.path.exists(img_path):
            image = cv.imread(img_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            
            # Raw detection
            corners, ids, rejected = detector.detectMarkers(gray)
            raw_count = len(ids) if ids is not None else 0
            
            # Smart filtering
            if ids is not None:
                filtered_corners, filtered_ids = smart_marker_filtering(corners, ids, expected_count=4)
                filtered_count = len(filtered_ids)
                filtered_ids_list = [id_arr[0] for id_arr in filtered_ids] if filtered_ids else []
            else:
                filtered_count = 0
                filtered_ids_list = []
            
            status = "✓" if filtered_count == 4 else "⚠" if filtered_count > 0 else "✗"
            print(f"{status} {os.path.basename(img_path)}: {raw_count} → {filtered_count} | IDs: {filtered_ids_list}")
            
            total_raw += raw_count
            total_filtered += filtered_count
    
    print("=" * 70)
    print(f"Total: {total_raw} raw detections → {total_filtered} filtered")
    print(f"Expected: 40 markers (4 per image)")
    print(f"Success rate: {(total_filtered/40)*100:.1f}%" if total_filtered > 0 else "0%")

if __name__ == "__main__":
    test_smart_detection()