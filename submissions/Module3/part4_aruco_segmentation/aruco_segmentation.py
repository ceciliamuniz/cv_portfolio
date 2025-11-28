"""
Part 4: ArUco Marker-Based Object Segmentation
Detect ArUco markers on non-rectangular object boundaries and segment the object.
"""

import cv2 as cv
import numpy as np
import cv2.aruco as aruco
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json


class ArucoSegmentation:
    """Simplified ArUco marker-based object segmentation using GrabCut."""
    
    def __init__(self, aruco_dict_type=cv.aruco.DICT_6X6_1000):
        """
        Initialize ArUco detector with optimized parameters.
        
        Args:
            aruco_dict_type: Type of ArUco dictionary to use (default: DICT_6X6_1000)
        """
        self.aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_type)
        
        # Initialize and configure the parameters for reliable detection
        params = cv.aruco.DetectorParameters()
        
        # Apply the best parameters for reliable detection
        params.adaptiveThreshWinSizeMax = 23
        params.minMarkerPerimeterRate = 0.005
        params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        params.errorCorrectionRate = 0.8
        
        # The detector MUST be initialized with these parameters
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, params)
        
    def detect_markers(self, image: np.ndarray):
        """Standard marker detection with optimized parameters"""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids
    
    def detect_markers_comprehensive(self, image: np.ndarray):
        """Comprehensive marker detection with multiple strategies to find all 4+ markers"""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
            
        best_corners = None
        best_ids = None
        max_markers = 0
        
        print(f"[DEBUG] Starting comprehensive marker detection on image {gray.shape}")
        
        # Strategy 1: Standard optimized detection
        corners1, ids1, rejected1 = self.detector.detectMarkers(gray)
        markers_found = len(ids1) if ids1 is not None else 0
        if markers_found > max_markers:
            best_corners, best_ids, max_markers = corners1, ids1, markers_found
        print(f"[DETECT-1] Standard: {markers_found} markers")
        
        # Strategy 2: Very aggressive parameters
        if max_markers < 4:
            aggressive_params = cv.aruco.DetectorParameters()
            aggressive_params.adaptiveThreshWinSizeMin = 3
            aggressive_params.adaptiveThreshWinSizeMax = 100
            aggressive_params.adaptiveThreshWinSizeStep = 2
            aggressive_params.minMarkerPerimeterRate = 0.0005  # Very lenient
            aggressive_params.maxMarkerPerimeterRate = 8.0
            aggressive_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
            aggressive_params.errorCorrectionRate = 0.3  # Very lenient
            aggressive_params.minCornerDistanceRate = 0.001
            aggressive_params.adaptiveThreshConstant = 5
            
            aggressive_detector = cv.aruco.ArucoDetector(self.aruco_dict, aggressive_params)
            corners2, ids2, rejected2 = aggressive_detector.detectMarkers(gray)
            markers_found = len(ids2) if ids2 is not None else 0
            if markers_found > max_markers:
                best_corners, best_ids, max_markers = corners2, ids2, markers_found
            print(f"[DETECT-2] Aggressive: {markers_found} markers")
        
        # Strategy 3: Multiple preprocessing approaches
        if max_markers < 4:
            preprocessing_methods = [
                ('CLAHE', self._apply_clahe),
                ('Gaussian Blur', lambda img: cv.GaussianBlur(img, (3, 3), 0)),
                ('Bilateral Filter', lambda img: cv.bilateralFilter(img, 9, 75, 75)),
                ('Histogram Equalization', lambda img: cv.equalizeHist(img)),
                ('Morphology Opening', lambda img: cv.morphologyEx(img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))),
            ]
            
            for name, preprocess_func in preprocessing_methods:
                try:
                    processed_gray = preprocess_func(gray.copy())
                    corners3, ids3, rejected3 = self.detector.detectMarkers(processed_gray)
                    markers_found = len(ids3) if ids3 is not None else 0
                    if markers_found > max_markers:
                        best_corners, best_ids, max_markers = corners3, ids3, markers_found
                    print(f"[DETECT-3] {name}: {markers_found} markers")
                    
                    if max_markers >= 4:
                        break  # Found enough markers
                except Exception as e:
                    print(f"[DETECT-3] {name} failed: {e}")
        
        # Strategy 4: Multi-scale detection
        if max_markers < 4:
            scales = [0.7, 0.8, 1.2, 1.5]  # Different scales
            for scale in scales:
                try:
                    h, w = gray.shape
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_gray = cv.resize(gray, (new_w, new_h))
                    
                    corners4, ids4, rejected4 = self.detector.detectMarkers(scaled_gray)
                    markers_found = len(ids4) if ids4 is not None else 0
                    
                    # Scale corners back to original size
                    if corners4 is not None and len(corners4) > 0:
                        scaled_corners = []
                        for corner in corners4:
                            scaled_corner = corner / scale
                            scaled_corners.append(scaled_corner)
                        corners4 = scaled_corners
                    
                    if markers_found > max_markers:
                        best_corners, best_ids, max_markers = corners4, ids4, markers_found
                    print(f"[DETECT-4] Scale {scale}: {markers_found} markers")
                    
                    if max_markers >= 4:
                        break
                except Exception as e:
                    print(f"[DETECT-4] Scale {scale} failed: {e}")
        
        # Strategy 5: Ultra-aggressive last resort
        if max_markers < 4:
            ultra_params = cv.aruco.DetectorParameters()
            ultra_params.adaptiveThreshWinSizeMin = 3
            ultra_params.adaptiveThreshWinSizeMax = 200
            ultra_params.adaptiveThreshWinSizeStep = 1
            ultra_params.minMarkerPerimeterRate = 0.0001  # Ultra lenient
            ultra_params.maxMarkerPerimeterRate = 10.0
            ultra_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_NONE  # Skip refinement
            ultra_params.errorCorrectionRate = 0.1  # Ultra lenient
            ultra_params.minCornerDistanceRate = 0.0005
            
            ultra_detector = cv.aruco.ArucoDetector(self.aruco_dict, ultra_params)
            corners5, ids5, rejected5 = ultra_detector.detectMarkers(gray)
            markers_found = len(ids5) if ids5 is not None else 0
            if markers_found > max_markers:
                best_corners, best_ids, max_markers = corners5, ids5, markers_found
            print(f"[DETECT-5] Ultra-aggressive: {markers_found} markers")
        
        print(f"[FINAL] Best detection found {max_markers} markers (target: 4+)")
        
        if max_markers < 4:
            print(f"[WARNING] Only found {max_markers} markers, assignment requires 4 minimum!")
        
        return best_corners, best_ids
    
    def _apply_clahe(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def get_all_marker_points(self, corners):
        all_points = []
        if corners:
            for corner in corners:
                all_points.extend(corner[0].astype(np.int32).tolist())
        return np.array(all_points)

    def simplified_grabcut_segmentation(self, image_bgr: np.ndarray):
        """
        Improved ArUco-based segmentation with better mask initialization for rounded objects.
        Uses markers to create precise initial mask rather than relying on bounding box.
        """
        corners, ids = self.detect_markers(image_bgr)
        
        if ids is None or len(ids) < 4:
            warning_msg = f"Found {len(ids) if ids is not None else 0} markers, assignment requires 4 minimum!"
            print(f"[WARNING] {warning_msg}")
            
            # Still attempt segmentation with fewer markers but show warning
            if ids is None or len(ids) < 2:
                return image_bgr, None, {"error": f"Insufficient markers for segmentation (found {len(ids) if ids is not None else 0}, need at least 2).", "warning": warning_msg}

        # 1. Gather marker center points for better object representation
        marker_centers = self.get_marker_centers(corners)
        all_corner_points = self.get_all_marker_points(corners)
        
        if len(marker_centers) < 2:
            return image_bgr, None, {"error": "Need at least 2 markers for segmentation."}
        
        print(f"[DEBUG] Found {len(ids)} markers with {len(marker_centers)} centers")
        
        # 2. Create improved initial mask using marker information
        mask = np.zeros(image_bgr.shape[:2], np.uint8)
        h, w = image_bgr.shape[:2]
        
        # Method A: Use convex hull of marker centers for inner core
        if len(marker_centers) >= 3:
            # Create convex hull from marker centers
            center_hull = cv.convexHull(marker_centers.astype(np.int32))
            # Fill inner area as definite foreground
            cv.fillPoly(mask, [center_hull], cv.GC_FGD)
            print(f"[DEBUG] Created center hull with {len(center_hull)} points")
        else:
            # For 2 markers, create ellipse between them
            if len(marker_centers) == 2:
                center = tuple(np.mean(marker_centers, axis=0).astype(int))
                # Calculate distance and create ellipse
                dist = np.linalg.norm(marker_centers[1] - marker_centers[0])
                axes = (int(dist * 0.3), int(dist * 0.2))  # Smaller ellipse for core
                cv.ellipse(mask, center, axes, 0, 0, 360, cv.GC_FGD, -1)
                print(f"[DEBUG] Created ellipse at {center} with axes {axes}")
        
        # Method B: Create expanded region using all marker corners
        corner_hull = cv.convexHull(all_corner_points)
        
        # Expand hull slightly to capture object boundary
        # Calculate centroid of hull
        M = cv.moments(corner_hull)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid = np.array([cx, cy])
            
            # Expand hull outward by 15-25% to capture full object
            expanded_hull = []
            for point in corner_hull:
                pt = point[0]
                direction = pt - centroid
                # Expand by 20%
                expanded_pt = centroid + direction * 1.2
                expanded_hull.append([expanded_pt.astype(np.int32)])
            
            expanded_hull = np.array(expanded_hull)
            
            # Create temporary mask for expanded region
            temp_mask = np.zeros(image_bgr.shape[:2], np.uint8)
            cv.fillPoly(temp_mask, [expanded_hull], 255)
            
            # Set expanded region as probable foreground (but not overriding definite FG)
            mask = np.where((temp_mask > 0) & (mask == 0), cv.GC_PR_FGD, mask)
            print(f"[DEBUG] Expanded hull by 20% around centroid ({cx}, {cy})")
        
        # Method C: Set background regions
        # Create a border region as definite background
        border_width = 20
        mask[:border_width, :] = cv.GC_BGD  # Top border
        mask[-border_width:, :] = cv.GC_BGD  # Bottom border  
        mask[:, :border_width] = cv.GC_BGD  # Left border
        mask[:, -border_width:] = cv.GC_BGD  # Right border
        
        # 3. Create minimal bounding box (just for GrabCut requirement)
        x, y, w, h = cv.boundingRect(corner_hull)
        # Small buffer of 5-10 pixels as recommended
        buffer = 8
        rect = (max(0, x - buffer), max(0, y - buffer), 
                min(image_bgr.shape[1] - (x - buffer), w + 2*buffer), 
                min(image_bgr.shape[0] - (y - buffer), h + 2*buffer))
        
        print(f"[DEBUG] Bounding rect: {rect}")
        
        # 4. Run GrabCut with improved initialization
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        try:
            # Use mask-based initialization for better results
            cv.grabCut(image_bgr, mask, rect, bgdModel, fgdModel, 8, cv.GC_INIT_WITH_MASK)
            
            # Run additional iterations to refine
            cv.grabCut(image_bgr, mask, rect, bgdModel, fgdModel, 3, cv.GC_EVAL)
            
            # 5. Extract and refine final mask
            final_mask = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')
            
            # Post-process mask to smooth boundaries
            # Apply morphological operations to clean up the mask
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            final_mask = cv.morphologyEx(final_mask, cv.MORPH_OPEN, kernel)
            final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)
            
            # Apply Gaussian blur to smooth edges for round objects
            final_mask = cv.GaussianBlur(final_mask, (3, 3), 0)
            final_mask = np.where(final_mask > 127, 255, 0).astype('uint8')
            
            print(f"[DEBUG] Final mask has {np.sum(final_mask > 0)} foreground pixels")
            
            # 6. Create visualization with improved annotations
            segmentation_viz = image_bgr.copy()
            
            # Draw detected markers with IDs
            cv.aruco.drawDetectedMarkers(segmentation_viz, corners, ids)
            
            # Draw marker centers
            for i, center in enumerate(marker_centers):
                cv.circle(segmentation_viz, tuple(center.astype(int)), 8, (0, 255, 255), 2)
                cv.putText(segmentation_viz, f'C{i}', tuple(center.astype(int) + [15, 15]), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw convex hulls
            if len(marker_centers) >= 3:
                center_hull = cv.convexHull(marker_centers.astype(np.int32))
                cv.polylines(segmentation_viz, [center_hull], True, (255, 255, 0), 2)
            
            cv.polylines(segmentation_viz, [corner_hull], True, (255, 0, 255), 2)
            
            # Apply segmentation overlay
            colored_mask = np.zeros_like(image_bgr)
            colored_mask[:, :, 1] = final_mask  # Green channel for mask
            segmentation_viz = cv.addWeighted(segmentation_viz, 0.7, colored_mask, 0.3, 0)
            
            # Calculate metrics
            area_pixels = int(np.sum(final_mask > 0))
            perimeter_pixels = int(cv.arcLength(corner_hull, True))
            
            return segmentation_viz, final_mask, {
                "markers_detected": len(ids),
                "marker_ids": ids.flatten().tolist() if ids is not None else [],
                "marker_centers": len(marker_centers),
                "area_pixels": area_pixels,
                "perimeter_pixels": perimeter_pixels,
                "method": "improved_grabcut",
                "status": f"Success: Detected {len(ids)} markers. Improved GrabCut segmentation completed."
            }
            
        except Exception as e:
            print(f"[ERROR] Improved GrabCut failed: {e}")
            import traceback
            traceback.print_exc()
            return image_bgr, None, {"error": f"Improved GrabCut segmentation failed: {e}"}
    
    def get_marker_centers(self, corners: List) -> np.ndarray:

        centers = []
        for corner in corners:
            # Each corner is a 4x2 array of marker corners
            center = corner[0].mean(axis=0)
            centers.append(center)
        return np.array(centers, dtype=np.float32)
    
    def segment_object(self, 
                       image: np.ndarray,
                       marker_centers: np.ndarray,
                       method: str = 'convex_hull') -> Tuple[np.ndarray, Dict]:
        """
        Segment object based on ArUco marker positions.
        
        Args:
            image: Input image
            marker_centers: Array of marker center points
            method: Segmentation method ('convex_hull', 'contour', 'alpha_shape')
            
        Returns:
            Tuple of (segmentation_mask, metrics_dict)
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(marker_centers) < 2:
            return mask, {"error": "Need at least 2 markers for segmentation"}
        
        # Special handling for 2 markers - create expanded bounding region
        if len(marker_centers) == 2:
            return self._segment_with_two_markers(image, marker_centers)
        
        metrics = {
            "num_markers": len(marker_centers),
            "method": method,
            "area_pixels": 0,
            "perimeter_pixels": 0
        }
        
        if method == 'convex_hull':
            # Create convex hull from marker centers
            hull = cv.convexHull(marker_centers.astype(np.int32))
            cv.fillPoly(mask, [hull], 255)
            
            # Calculate metrics
            metrics["area_pixels"] = cv.contourArea(hull)
            metrics["perimeter_pixels"] = cv.arcLength(hull, True)
            
        elif method == 'contour':
            # Find contours in the region containing markers
            # First, create a mask with marker positions
            marker_mask = np.zeros((h, w), dtype=np.uint8)
            for center in marker_centers:
                cv.circle(marker_mask, tuple(center.astype(int)), 20, 255, -1)
            
            # Dilate to connect markers
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
            dilated = cv.dilate(marker_mask, kernel, iterations=3)
            
            # Find external contour
            contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use the largest contour
                largest_contour = max(contours, key=cv.contourArea)
                cv.fillPoly(mask, [largest_contour], 255)
                
                metrics["area_pixels"] = cv.contourArea(largest_contour)
                metrics["perimeter_pixels"] = cv.arcLength(largest_contour, True)
                
        # Note: alpha_shape method removed - using simplified GrabCut approach instead
        
        return mask, metrics
    
    def aruco_segment_object_simple(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Streamlined ArUco segmentation using convex hull + GrabCut.
        Detects ArUco markers, finds the convex hull around them, 
        and segments the object using GrabCut within the convex hull ROI.
        
        Args:
            image_bgr: Input BGR image
            
        Returns:
            Tuple of (segmentation_visualization, status_message)
        """
        if image_bgr is None:
            return None, "Error: Image not loaded."

        # Convert to grayscale for marker detection and processing
        gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        
        # 1. Detect ArUco Markers using DICT_6X6_1000 for more robust detection
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        parameters = aruco.DetectorParameters()
        
        # Optimized parameters for DICT_6X6_1000 markers
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 2
        parameters.adaptiveThreshConstant = 7
        
        # More lenient contour filtering for better detection
        parameters.minMarkerPerimeterRate = 0.005  # More lenient
        parameters.maxMarkerPerimeterRate = 4.0     
        parameters.polygonalApproxAccuracyRate = 0.05
        parameters.minCornerDistanceRate = 0.01    # More lenient
        parameters.minDistanceToBorder = 1
        
        # Enable corner refinement for accuracy
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 30
        
        # More lenient quality thresholds
        parameters.minOtsuStdDev = 2.0             # More lenient
        parameters.errorCorrectionRate = 0.8
        
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        
        if ids is None or len(ids) < 2:
            return image_bgr, f"Not enough markers found (found {len(ids) if ids is not None else 0}, requires at least 2)."

        # 2. Define the Object's Convex Hull (Boundary ROI)
        all_points = []
        for corner in corners:
            # Add all four corner points of the marker to the list
            all_points.extend(corner[0].astype(int).tolist())
        
        all_points = np.array(all_points)
        
        # Find the convex hull of all marker points
        hull_points = cv.convexHull(all_points)

        # 3. Create a Bounding Box and Mask for GrabCut
        x, y, w, h = cv.boundingRect(hull_points)
        
        # Ensure the ROI is safe (within image bounds)
        x = max(0, x)
        y = max(0, y)
        w = min(image_bgr.shape[1] - x, w)
        h = min(image_bgr.shape[0] - y, h)
        
        # Initialize the GrabCut mask
        mask = np.zeros(image_bgr.shape[:2], np.uint8)
        
        # Define the GrabCut rectangle [x, y, width, height]
        rect = (x, y, w, h)
        
        # 4. Run GrabCut Segmentation
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        try:
            # Run GrabCut
            cv.grabCut(image_bgr, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
            
            # Create the final mask where only foreground and probable foreground are kept
            final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Apply the mask to the original image
            segmentation_viz = image_bgr * final_mask[:, :, np.newaxis]
            
            # Draw the convex hull boundary and detected markers
            cv.polylines(segmentation_viz, [hull_points], isClosed=True, color=(0, 255, 255), thickness=3)
            
            # Draw detected markers
            if len(corners) > 0:
                aruco.drawDetectedMarkers(segmentation_viz, corners, ids)
            
            return segmentation_viz, f"Success: Detected {len(ids)} markers. Segmented using GrabCut."
            
        except Exception as e:
            return image_bgr, f"GrabCut failed: {str(e)}"
    
    def _segment_with_two_markers(self, image: np.ndarray, marker_centers: np.ndarray):
        """
        Segment object using only 2 markers by creating expanded bounding region.
        
        Args:
            image: Input image
            marker_centers: Array of 2 marker center points
            
        Returns:
            Tuple of (segmentation_mask, metrics_dict)
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Calculate distance between markers and create expanded rectangle
        p1, p2 = marker_centers[0], marker_centers[1]
        center = (p1 + p2) / 2
        
        # Create rectangle perpendicular to marker line
        direction = p2 - p1
        perpendicular = np.array([-direction[1], direction[0]])
        
        # Normalize and scale
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        marker_distance = np.linalg.norm(direction)
        expansion = marker_distance * 0.8  # Expand by 80% of marker distance
        
        # Create 4 corners of expanded rectangle
        corners = np.array([
            center + direction * 0.6 + perpendicular * expansion,
            center + direction * 0.6 - perpendicular * expansion,
            center - direction * 0.6 - perpendicular * expansion,
            center - direction * 0.6 + perpendicular * expansion
        ], dtype=np.int32)
        
        # Ensure corners are within image bounds
        corners[:, 0] = np.clip(corners[:, 0], 0, w-1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h-1)
        
        # Fill the polygon
        cv.fillPoly(mask, [corners], 255)
        
        # Try to refine with GrabCut if possible
        try:
            x, y, rw, rh = cv.boundingRect(corners)
            rect = (x, y, rw, rh)
            
            grabcut_mask = np.zeros((h, w), dtype=np.uint8)
            grabcut_mask[mask == 255] = cv.GC_PR_FGD
            
            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)
            
            cv.grabCut(image, grabcut_mask, rect, bgd_model, fgd_model, 2, cv.GC_INIT_WITH_MASK)
            mask = np.where((grabcut_mask == cv.GC_FGD) | (grabcut_mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)
        except:
            pass  # Keep original mask if GrabCut fails
        
        metrics = {
            "num_markers": 2,
            "method": "two_marker_rectangle",
            "area_pixels": int(np.sum(mask == 255)),
            "perimeter_pixels": int(cv.arcLength(corners, True))
        }
        
        return mask, metrics
    
    def visualize_segmentation(self,
                              image: np.ndarray,
                              corners: List,
                              ids: List,
                              mask: np.ndarray,
                              marker_centers: np.ndarray) -> np.ndarray:
        """
        Create visualization of ArUco detection and segmentation.
        
        Args:
            image: Original input image
            corners: Detected marker corners
            ids: Detected marker IDs
            mask: Segmentation mask
            marker_centers: Marker center points
            
        Returns:
            Visualization image
        """
        # Create output image
        output = image.copy()
        
        # Draw detected markers
        if ids is not None and len(ids) > 0:
            cv.aruco.drawDetectedMarkers(output, corners, ids)
        
        # Draw marker centers
        for center in marker_centers:
            cv.circle(output, tuple(center.astype(int)), 5, (0, 255, 0), -1)
        
        # Overlay segmentation mask
        mask_colored = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        mask_colored[:, :, 1] = mask  # Green channel
        mask_colored[:, :, 0] = 0
        mask_colored[:, :, 2] = 0
        
        # Blend with original image
        output = cv.addWeighted(output, 0.7, mask_colored, 0.3, 0)
        
        # Draw contour of segmented region
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(output, contours, -1, (0, 255, 255), 3)
        
        return output
    
    def process_image(self, 
                     image_path: Path,
                     output_dir: Path,
                     method: str = 'simple') -> Dict:
        """
        Process a single image using simplified ArUco segmentation with GrabCut.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            method: Segmentation method (ignored - always uses GrabCut)
            
        Returns:
            Dictionary with processing results and all required web interface data
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read image
        image = cv.imread(str(image_path))
        if image is None:
            return {"error": f"Failed to read image: {image_path}"}
        
        # Use simplified GrabCut segmentation
        visualization, mask, metrics = self.simplified_grabcut_segmentation(image)
        
        if "error" in metrics:
            return {"error": metrics["error"]}
        
        # Save outputs
        vis_filename = f"{image_path.stem}_segmentation.jpg"
        mask_filename = f"{image_path.stem}_mask.jpg"
        
        vis_path = output_dir / vis_filename
        mask_path = output_dir / mask_filename
        
        cv.imwrite(str(vis_path), visualization)
        cv.imwrite(str(mask_path), mask)
        
        # Return complete result for web interface
        result = {
            "image": image_path.name,
            "markers_detected": metrics["markers_detected"],
            "marker_ids": metrics["marker_ids"],
            "area_pixels": metrics["area_pixels"],
            "perimeter_pixels": metrics["perimeter_pixels"],
            "method": "grabcut",
            "status": metrics["status"],
            "visualization_saved": str(vis_path),
            "mask_saved": str(mask_path),
            "output_path": str(vis_path)
        }
        
        return result
        
        return results


# Hardcoded image list for consistent processing - Updated to match actual uploaded images
IMAGE_FILE_NAMES = [
    "img1.png",  # Image 1 with 4 ArUco markers (6x6, 10mm)
    "img2.png",  # Image 2 with 4 ArUco markers (6x6, 10mm)
    "img3.png",  # Image 3 with 4 ArUco markers (6x6, 10mm)
    "img4.png",  # Image 4 with 4 ArUco markers (6x6, 10mm)
    "img5.png",  # Image 5 with 4 ArUco markers (6x6, 10mm)
    "img6.png",  # Image 6 with 4 ArUco markers (6x6, 10mm)
    "img7.png",  # Image 7 with 4 ArUco markers (6x6, 10mm)
    "img8.png",  # Image 8 with 4 ArUco markers (6x6, 10mm)
    "img9.png",  # Image 9 with 4 ArUco markers (6x6, 10mm)
    "img10.png",  # Image 10 with 4 ArUco markers (6x6, 10mm)
]

def process_all_images(images_dir: Path, 
                       output_dir: Path,
                       method: str = 'grabcut') -> List[Dict]:
    """
    Process hardcoded list of images for consistent results.
    
    Args:
        images_dir: Directory containing input images
        output_dir: Directory to save outputs
        method: Segmentation method to use (always uses GrabCut)
        
    Returns:
        List of result dictionaries for each image
    """
    segmenter = ArucoSegmentation()
    results = []
    
    # Use the hardcoded list instead of dynamic discovery
    image_files = [images_dir / name for name in IMAGE_FILE_NAMES]
    
    print(f"Processing {len(image_files)} images from hardcoded list.")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        if image_path.exists():
            result = segmenter.process_image(image_path, output_dir, method)
            results.append(result)
            
            # Print summary
            if "error" not in result:
                print(f"  ✓ Markers detected: {result['markers_detected']}")
                print(f"  ✓ Method: {result.get('method', 'grabcut')}")
                if 'output_path' in result:
                    print(f"  ✓ Saved: {result['output_path']}")
            else:
                print(f"  ✗ Error: {result['error']}")
        else:
            # Handle the case where a hardcoded file is missing
            error_result = {"error": f"Hardcoded file not found: {image_path.name}", "image": image_path.name}
            results.append(error_result)
            print(f"  ✗ Error: Hardcoded file not found: {image_path.name}")
    
    # Save summary JSON
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    return results



if __name__ == "__main__":
    # Setup paths
    script_dir = Path(__file__).parent
    images_dir = script_dir / "images"
    output_dir = script_dir / "outputs"
    markers_dir = script_dir / "aruco_markers"
    
    # Generate ArUco markers for printing (DISABLED - using user's custom markers)
    # print("=" * 80)
    # print("STEP 1: Generating ArUco Markers")
    # print("=" * 80)
    # generate_aruco_markers(markers_dir, marker_ids=list(range(20)))
    
    # Process images if they exist
    if images_dir.exists() and any(images_dir.iterdir()):
        print("\n" + "=" * 80)
        print("STEP 2: Processing Images with Optimized GrabCut Segmentation")
        print("=" * 80)
        
        # Only run process_all_images once with optimized GrabCut
        results = process_all_images(images_dir, output_dir)
        
        # Print statistics
        successful = [r for r in results if "error" not in r]
        if successful:
            total_markers = sum(r['markers_detected'] for r in successful)
            avg_markers = total_markers / len(successful)
            print(f"\n📊 Statistics:")
            print(f"  Total images processed: {len(results)}")
            print(f"  Successful segmentations: {len(successful)}")
            print(f"  Average markers per image: {avg_markers:.1f}")
    else:
        print("\n" + "=" * 80)
        print("INSTRUCTIONS FOR USE")
        print("=" * 80)
        print("\n1. Print the ArUco markers from: aruco_markers/")
        print("2. Stick markers on the boundary of a NON-RECTANGULAR object")
        print("3. Capture images from various distances and angles (min 10 images)")
        print("4. Save images to: images/")
        print("5. Run this script again to process the images")
        print("\nTips:")
        print("  - Use at least 4-6 markers around the object boundary")
        print("  - Space markers evenly for better segmentation")
        print("  - Ensure good lighting and marker visibility")
        print("  - Capture from different angles: front, side, top, oblique")
        print("  - Vary distances: close-up, medium, far")
