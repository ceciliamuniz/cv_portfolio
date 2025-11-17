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
    """ArUco marker-based object segmentation."""
    
    def __init__(self, aruco_dict_type=cv.aruco.DICT_4X4_50):
        """
        Initialize ArUco detector.
        
        Args:
            aruco_dict_type: Type of ArUco dictionary to use
        """
        # Initialize ArUco dictionary and parameters
        self.aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv.aruco.DetectorParameters()
        
        # Adjust parameters for better detection of various marker sizes
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 53
        self.aruco_params.minMarkerPerimeterRate = 0.01
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
    def detect_markers(self, image: np.ndarray) -> Tuple[List, List, List]:
        """
        Detect ArUco markers in the image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (corners, ids, rejected_points)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        return corners, ids, rejected
    
    def get_marker_centers(self, corners: List) -> np.ndarray:
        """
        Calculate center points of detected markers.
        
        Args:
            corners: List of marker corners from detection
            
        Returns:
            Array of center points (Nx2)
        """
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
                
        elif method == 'alpha_shape':
            # Alpha shape (concave hull) - more accurate for non-convex objects
            from scipy.spatial import Delaunay
            
            # Compute Delaunay triangulation
            tri = Delaunay(marker_centers)
            
            # Filter triangles by edge length (alpha parameter)
            alpha = 100.0  # Adjust based on marker spacing
            edges = set()
            
            for simplex in tri.simplices:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                    p1, p2 = marker_centers[edge[0]], marker_centers[edge[1]]
                    if np.linalg.norm(p1 - p2) < alpha:
                        edges.add(edge)
            
            # Create boundary from edges
            boundary_points = []
            for edge in edges:
                boundary_points.extend([marker_centers[edge[0]], marker_centers[edge[1]]])
            
            if boundary_points:
                boundary = np.array(boundary_points, dtype=np.int32)
                hull = cv.convexHull(boundary)
                cv.fillPoly(mask, [hull], 255)
                
                metrics["area_pixels"] = cv.contourArea(hull)
                metrics["perimeter_pixels"] = cv.arcLength(hull, True)
        
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
        
        # 1. Detect ArUco Markers using DICT_4X4_1000 (best performance from comprehensive test)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        parameters = aruco.DetectorParameters()
        
        # Optimized parameters for DICT_4X4_1000 to reduce false positives
        parameters.adaptiveThreshWinSizeMin = 5
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 4
        parameters.adaptiveThreshConstant = 7
        
        # More restrictive contour filtering
        parameters.minMarkerPerimeterRate = 0.02  # More restrictive
        parameters.maxMarkerPerimeterRate = 4.0   # More restrictive  
        parameters.polygonalApproxAccuracyRate = 0.03
        parameters.minCornerDistanceRate = 0.05   # More restrictive
        parameters.minDistanceToBorder = 3
        
        # Enable corner refinement for accuracy
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        
        # Higher quality thresholds
        parameters.minOtsuStdDev = 5.0           # More restrictive
        parameters.errorCorrectionRate = 0.6
        
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
        Process a single image using simplified ArUco segmentation.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            method: Segmentation method (now uses 'simple' by default)
            
        Returns:
            Dictionary with processing results and metrics
        """
        # Read image
        image = cv.imread(str(image_path))
        if image is None:
            return {"error": f"Failed to read image: {image_path}"}
        
        # Use simplified ArUco segmentation
        segmentation_viz, status = self.aruco_segment_object_simple(image)
        
        # Parse results
        if "Success" in status:
            # Extract number of markers from status message
            import re
            marker_match = re.search(r'Detected (\d+) markers', status)
            markers_detected = int(marker_match.group(1)) if marker_match else 0
            
            result = {
                "image": image_path.name,
                "markers_detected": markers_detected,
                "method": "simple_grabcut",
                "status": status
            }
        else:
            result = {
                "image": image_path.name,
                "markers_detected": 0,
                "error": status
            }
        
        # Save segmentation result if successful
        if "Success" in status:
            vis_path = output_dir / f"{image_path.stem}_segmentation.jpg"
            cv.imwrite(str(vis_path), segmentation_viz)
            result["output_path"] = str(vis_path)
        
        return result
        
        return results


def process_all_images(images_dir: Path, 
                       output_dir: Path,
                       method: str = 'convex_hull') -> List[Dict]:
    """
    Process all images in a directory.
    
    Args:
        images_dir: Directory containing input images
        output_dir: Directory to save outputs
        method: Segmentation method to use
        
    Returns:
        List of result dictionaries for each image
    """
    segmenter = ArucoSegmentation()
    results = []
    
    # Find all image files (remove duplicates from case-insensitive matching)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(ext))
        image_files.extend(images_dir.glob(ext.upper()))
    
    # Remove duplicates (Windows is case-insensitive)
    image_files = list(set(image_files))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        result = segmenter.process_image(image_path, output_dir, method)
        results.append(result)
        
        # Print summary
        if "error" not in result:
            print(f"  ✓ Markers detected: {result['markers_detected']}")
            print(f"  ✓ Method: {result.get('method', 'simple_grabcut')}")
            if 'output_path' in result:
                print(f"  ✓ Saved: {result['output_path']}")
        else:
            print(f"  ✗ Error: {result['error']}")
    
    # Save summary JSON
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    return results


def generate_aruco_markers(output_dir: Path, 
                          marker_ids: List[int] = None,
                          marker_size: int = 200,
                          dict_type=cv.aruco.DICT_4X4_50):
    """
    Generate printable ArUco markers for use in object segmentation.
    
    Args:
        output_dir: Directory to save marker images
        marker_ids: List of marker IDs to generate (default: 0-9)
        marker_size: Size of marker in pixels
        dict_type: ArUco dictionary type
    """
    if marker_ids is None:
        marker_ids = list(range(10))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    aruco_dict = cv.aruco.getPredefinedDictionary(dict_type)
    
    print(f"Generating {len(marker_ids)} ArUco markers...")
    
    for marker_id in marker_ids:
        marker_img = cv.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Add white border for easier printing
        bordered = cv.copyMakeBorder(marker_img, 20, 20, 20, 20, 
                                     cv.BORDER_CONSTANT, value=255)
        
        # Save marker
        marker_path = output_dir / f"aruco_marker_{marker_id:02d}.png"
        cv.imwrite(str(marker_path), bordered)
        print(f"  ✓ Generated marker {marker_id}: {marker_path}")
    
    print(f"\n✓ All markers saved to: {output_dir}")


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
        print("STEP 2: Processing Images with ArUco Detection")
        print("=" * 80)
        
        # Try different segmentation methods
        for method in ['convex_hull', 'contour']:
            method_output_dir = output_dir / method
            print(f"\n--- Using method: {method} ---")
            results = process_all_images(images_dir, method_output_dir, method)
            
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
