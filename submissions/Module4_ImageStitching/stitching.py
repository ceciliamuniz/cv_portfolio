"""Advanced Image Stitching Utilities

Provides comprehensive image stitching functionality including:
- Feature matching with custom and OpenCV SIFT
- Enhanced RANSAC homography estimation  
- Multi-band blending and seam optimization
- Panorama quality assessment
- Mobile panorama comparison

Author: Cecilia Muniz Siqueira
Module: CV_Module4_ImageStitching
"""
import numpy as np
import cv2 as cv
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time


def compute_features_and_matches(img1, img2, sift_impl, use_custom=True, comparison_mode=False):
    """Compute features and matches between two images.
    
    Args:
        img1, img2: Input images
        sift_impl: Custom SIFT implementation module
        use_custom: Whether to use custom SIFT (True) or OpenCV (False)
        comparison_mode: If True, return both custom and OpenCV results
    
    Returns:
        Dictionary with matching results and optionally comparison data
    """
    # Convert to grayscale
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    results = {'custom': None, 'opencv': None}
    
    if use_custom or comparison_mode:
        print("Computing custom SIFT features...")
        start_time = time.time()
        
        # Custom SIFT pipeline
        pyramid1 = sift_impl.build_gaussian_pyramid(gray1)
        pyramid2 = sift_impl.build_gaussian_pyramid(gray2)
        
        dog1 = sift_impl.compute_dog(pyramid1)
        dog2 = sift_impl.compute_dog(pyramid2)
        
        keypoints1 = sift_impl.detect_keypoints(dog1)
        keypoints2 = sift_impl.detect_keypoints(dog2)
        
        kps1_oriented = sift_impl.assign_orientations(keypoints1, pyramid1)
        kps2_oriented = sift_impl.assign_orientations(keypoints2, pyramid2)
        
        descriptors1 = sift_impl.compute_descriptors(kps1_oriented, pyramid1)
        descriptors2 = sift_impl.compute_descriptors(kps2_oriented, pyramid2)
        
        matches = sift_impl.match_descriptors(descriptors1, descriptors2)
        
        # Extract point coordinates
        pts1 = np.array([desc['pt'] for desc in descriptors1], dtype=np.float32)
        pts2 = np.array([desc['pt'] for desc in descriptors2], dtype=np.float32)
        
        # Get matched points
        matched_pts1 = []
        matched_pts2 = []
        for i1, i2, dist in matches:
            if i1 < len(pts1) and i2 < len(pts2):
                matched_pts1.append(pts1[i1])
                matched_pts2.append(pts2[i2])
        
        custom_time = time.time() - start_time
        
        results['custom'] = {
            'points1': matched_pts1,
            'points2': matched_pts2,
            'matches': matches,
            'keypoints1': len(descriptors1),
            'keypoints2': len(descriptors2),
            'processing_time': custom_time
        }
    
    if not use_custom or comparison_mode:
        print("Computing OpenCV SIFT features...")
        start_time = time.time()
        
        try:
            # OpenCV SIFT pipeline
            sift = cv.SIFT_create()
            kp1, desc1 = sift.detectAndCompute(gray1, None)
            kp2, desc2 = sift.detectAndCompute(gray2, None)
            
            # Match descriptors
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
            raw_matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            matched_pts1_cv = []
            matched_pts2_cv = []
            
            for match_pair in raw_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                        matched_pts1_cv.append(kp1[m.queryIdx].pt)
                        matched_pts2_cv.append(kp2[m.trainIdx].pt)
            
            opencv_time = time.time() - start_time
            
            results['opencv'] = {
                'points1': matched_pts1_cv,
                'points2': matched_pts2_cv,
                'matches': good_matches,
                'keypoints1': len(kp1),
                'keypoints2': len(kp2),
                'processing_time': opencv_time
            }
        
        except Exception as e:
            print(f"OpenCV SIFT failed: {e}")
            results['opencv'] = {'error': str(e)}
    
    # Return appropriate result
    if comparison_mode:
        return results
    else:
        return results['custom'] if use_custom else results['opencv']


def estimate_homography_ransac(points1, points2, threshold=4.0, max_iterations=5000, confidence=0.995):
    """Robust homography estimation using enhanced RANSAC.
    
    Args:
        points1: Source points (Nx2 array)
        points2: Destination points (Nx2 array)  
        threshold: Inlier distance threshold
        max_iterations: Maximum RANSAC iterations
        confidence: Desired confidence level
    
    Returns:
        Tuple of (homography_matrix, inlier_indices)
    """
    if len(points1) < 4 or len(points2) < 4:
        return None, []
    
    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)
    
    if len(points1) != len(points2):
        min_len = min(len(points1), len(points2))
        points1 = points1[:min_len]
        points2 = points2[:min_len]
    
    n_points = len(points1)
    best_homography = None
    best_inliers = []
    best_score = 0
    
    # Adaptive iteration calculation
    iterations = 0
    adaptive_max_iter = max_iterations
    
    print(f"Running RANSAC with {n_points} point pairs...")
    
    while iterations < adaptive_max_iter:
        # Randomly select 4 points
        if n_points < 4:
            break
            
        indices = np.random.choice(n_points, 4, replace=False)
        
        try:
            # Get corresponding points
            src_sample = points1[indices]
            dst_sample = points2[indices]
            
            # Compute homography using OpenCV for robustness
            H = cv.getPerspectiveTransform(src_sample, dst_sample)
            
            if H is None:
                continue
            
            # Transform all source points
            points1_hom = np.column_stack([points1, np.ones(len(points1))])
            transformed_hom = (H @ points1_hom.T).T
            
            # Convert from homogeneous coordinates
            transformed = transformed_hom[:, :2] / (transformed_hom[:, 2:3] + 1e-8)
            
            # Calculate distances to destination points
            distances = np.linalg.norm(transformed - points2, axis=1)
            
            # Find inliers
            inlier_mask = distances < threshold
            inliers = np.where(inlier_mask)[0]
            n_inliers = len(inliers)
            
            if n_inliers > best_score:
                best_score = n_inliers
                best_homography = H
                best_inliers = inliers
                
                # Update adaptive iteration count
                if n_inliers > 4:
                    inlier_ratio = n_inliers / n_points
                    if inlier_ratio > 0.1:  # Reasonable inlier ratio
                        adaptive_max_iter = min(max_iterations, 
                                              int(np.log(1 - confidence) / 
                                                  np.log(1 - inlier_ratio**4)))
        
        except Exception as e:
            # Skip this iteration if homography computation fails
            continue
        
        iterations += 1
        
        # Early termination if we have enough inliers
        if best_score > n_points * 0.6:  # 60% inliers is very good
            break
    
    # Refine homography using all inliers if we found a good one
    if best_homography is not None and len(best_inliers) >= 8:
        try:
            src_inliers = points1[best_inliers]
            dst_inliers = points2[best_inliers]
            
            # Use least squares refinement
            refined_H, _ = cv.findHomography(src_inliers, dst_inliers, 
                                           method=cv.RANSAC, 
                                           ransacReprojThreshold=threshold)
            if refined_H is not None:
                best_homography = refined_H
                
                # Recalculate inliers with refined homography
                points1_hom = np.column_stack([points1, np.ones(len(points1))])
                transformed_hom = (best_homography @ points1_hom.T).T
                transformed = transformed_hom[:, :2] / (transformed_hom[:, 2:3] + 1e-8)
                distances = np.linalg.norm(transformed - points2, axis=1)
                best_inliers = np.where(distances < threshold)[0]
        
        except Exception:
            pass  # Use original homography if refinement fails
    
    print(f"RANSAC completed: {len(best_inliers)}/{n_points} inliers after {iterations} iterations")
    
    return best_homography, best_inliers.tolist()


def advanced_warp_and_blend(images, homographies, reference=0, blend_mode='multiband'):
    """Advanced warping and blending with multiple blending options.
    
    Args:
        images: List of input images
        homographies: List of homography matrices
        reference: Index of reference image
        blend_mode: 'simple', 'feather', 'multiband', or 'seam'
    
    Returns:
        Stitched panorama image
    """
    if not images or len(images) != len(homographies):
        raise ValueError("Number of images must match number of homographies")
    
    print(f"Warping and blending {len(images)} images using {blend_mode} blending...")
    
    # Compute bounding box of all warped images
    corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        H = homographies[i]
        warped_corners = cv.perspectiveTransform(img_corners, H)
        corners.append(warped_corners.reshape(-1, 2))
    
    all_corners = np.vstack(corners)
    min_x, min_y = np.min(all_corners, axis=0)
    max_x, max_y = np.max(all_corners, axis=0)
    
    # Add padding to avoid edge artifacts
    padding = 50
    min_x, min_y = min_x - padding, min_y - padding
    max_x, max_y = max_x + padding, max_y + padding
    
    # Calculate canvas size
    canvas_w = int(max_x - min_x)
    canvas_h = int(max_y - min_y)
    
    # Translation matrix to ensure positive coordinates (define first)
    translation = np.array([[1, 0, -min_x],
                           [0, 1, -min_y],
                           [0, 0, 1]], dtype=np.float32)
    
    # Limit canvas size to prevent memory issues
    max_canvas_size = 8000  # Max width or height
    if canvas_w > max_canvas_size or canvas_h > max_canvas_size:
        scale = min(max_canvas_size / canvas_w, max_canvas_size / canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)
        print(f"[WARNING] Canvas too large, scaling down by {scale:.2f}x to {canvas_w}x{canvas_h}")
        
        # Scale the translation and homographies accordingly
        scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
        translation = scale_matrix @ translation
        homographies = [scale_matrix @ H for H in homographies]
    
    # Warp all images to the common coordinate system
    warped_images = []
    warped_masks = []
    
    for i, img in enumerate(images):
        # Apply translation to homography
        H_translated = translation @ homographies[i]
        
        # Warp image and create mask
        warped_img = cv.warpPerspective(img, H_translated, (canvas_w, canvas_h), 
                                       flags=cv.INTER_LINEAR, 
                                       borderMode=cv.BORDER_CONSTANT, 
                                       borderValue=(0, 0, 0))
        
        # Create mask for valid pixels
        mask = cv.warpPerspective(np.ones(img.shape[:2], dtype=np.uint8) * 255, 
                                 H_translated, (canvas_w, canvas_h))
        
        warped_images.append(warped_img)
        warped_masks.append(mask)
    
    # Apply selected blending method
    if blend_mode == 'simple':
        panorama = simple_blend(warped_images, warped_masks)
    elif blend_mode == 'feather':
        panorama = feather_blend(warped_images, warped_masks)
    elif blend_mode == 'multiband':
        panorama = multiband_blend(warped_images, warped_masks)
    else:  # seam blending
        panorama = seam_blend(warped_images, warped_masks)
    
    return panorama


def simple_blend(warped_images, masks):
    """Simple averaging blend where images overlap."""
    canvas_h, canvas_w = warped_images[0].shape[:2]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    for img, mask in zip(warped_images, masks):
        # Ensure consistent dimensions
        if img.shape[:2] != (canvas_h, canvas_w):
            img = cv.resize(img, (canvas_w, canvas_h))
        if mask.shape[:2] != (canvas_h, canvas_w):
            mask = cv.resize(mask, (canvas_w, canvas_h))
            
        mask_norm = (mask > 0).astype(np.float32)
        canvas += img.astype(np.float32) * mask_norm[..., None]
        weight_sum += mask_norm
    
    # Avoid division by zero
    weight_sum[weight_sum == 0] = 1
    canvas = canvas / weight_sum[..., None]
    
    return canvas.astype(np.uint8)


def feather_blend(warped_images, masks, feather_width=50):
    """Feathered blending with distance-based weights."""
    canvas_h, canvas_w = warped_images[0].shape[:2]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    for img, mask in zip(warped_images, masks):
        # Ensure consistent dimensions
        if img.shape[:2] != (canvas_h, canvas_w):
            img = cv.resize(img, (canvas_w, canvas_h))
        if mask.shape[:2] != (canvas_h, canvas_w):
            mask = cv.resize(mask, (canvas_w, canvas_h))
        # Compute distance transform for feathering
        mask_binary = (mask > 0).astype(np.uint8)
        distance = cv.distanceTransform(mask_binary, cv.DIST_L2, 5)
        
        # Create feathered weights
        weights = np.clip(distance / feather_width, 0, 1)
        
        canvas += img.astype(np.float32) * weights[..., None]
        weight_sum += weights
    
    # Normalize
    weight_sum[weight_sum == 0] = 1
    canvas = canvas / weight_sum[..., None]
    
    return canvas.astype(np.uint8)


def multiband_blend(warped_images, masks, num_bands=6):
    """Multi-band blending using Laplacian pyramids."""
    if len(warped_images) < 2:
        return warped_images[0] if warped_images else np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Ensure all images and masks have the same dimensions
    target_h, target_w = warped_images[0].shape[:2]
    
    fixed_images = []
    fixed_masks = []
    
    for i, (img, mask) in enumerate(zip(warped_images, masks)):
        # Resize image if needed
        if img.shape[:2] != (target_h, target_w):
            img = cv.resize(img, (target_w, target_h))
        
        # Resize mask if needed
        if mask.shape[:2] != (target_h, target_w):
            mask = cv.resize(mask, (target_w, target_h))
        
        fixed_images.append(img)
        fixed_masks.append(mask)
    
    # Start with first image
    result = fixed_images[0].astype(np.float32)
    result_mask = (fixed_masks[0] > 0).astype(np.float32)
    
    # Progressively blend each additional image
    for i in range(1, len(fixed_images)):
        img2 = fixed_images[i].astype(np.float32)
        mask2 = (fixed_masks[i] > 0).astype(np.float32)
        
        # Find overlap region
        overlap = (result_mask * mask2) > 0
        
        if not np.any(overlap):
            # No overlap, simple addition
            result = np.where(mask2[..., None] > 0, img2, result)
            result_mask = np.maximum(result_mask, mask2)
        else:
            # Multi-band blending in overlap region
            blended = pyramid_blend(result, img2, overlap, num_bands)
            
            # Combine with non-overlap regions
            result = np.where(mask2[..., None] > 0, 
                            np.where(result_mask[..., None] > 0, blended, img2),
                            result)
            result_mask = np.maximum(result_mask, mask2)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def pyramid_blend(img1, img2, overlap_mask, num_bands):
    """Blend two images using Laplacian pyramid."""
    # Create Gaussian pyramids
    G1 = [img1]
    G2 = [img2]
    
    for _ in range(num_bands - 1):
        G1.append(cv.pyrDown(G1[-1]))
        G2.append(cv.pyrDown(G2[-1]))
    
    # Create Laplacian pyramids
    L1 = [G1[-1]]  # Top level is the same
    L2 = [G2[-1]]
    
    for i in range(num_bands - 1, 0, -1):
        # Ensure pyrUp output matches the target size
        target_size = (G1[i-1].shape[1], G1[i-1].shape[0])
        
        # Upsample and resize to exact target dimensions
        up1 = cv.pyrUp(G1[i])
        up2 = cv.pyrUp(G2[i])
        
        # Resize to ensure exact match with target dimensions
        if up1.shape[:2] != G1[i-1].shape[:2]:
            up1 = cv.resize(up1, target_size)
        if up2.shape[:2] != G2[i-1].shape[:2]:
            up2 = cv.resize(up2, target_size)
            
        L1_level = G1[i-1] - up1
        L2_level = G2[i-1] - up2
        L1.insert(0, L1_level)
        L2.insert(0, L2_level)
    
    # Create mask pyramid
    mask_pyramid = [overlap_mask.astype(np.float32)]
    for _ in range(num_bands - 1):
        mask_pyramid.append(cv.pyrDown(mask_pyramid[-1]))
    mask_pyramid.reverse()
    
    # Blend Laplacian pyramids
    blended_pyramid = []
    for i in range(num_bands):
        if len(mask_pyramid[i].shape) == 2:
            mask_3d = mask_pyramid[i][..., None]
        else:
            mask_3d = mask_pyramid[i]
        
        # Smooth transition in overlap
        alpha = cv.GaussianBlur(mask_3d, (51, 51), 0)
        blended = L1[i] * (1 - alpha) + L2[i] * alpha
        blended_pyramid.append(blended)
    
    # Reconstruct image from pyramid
    result = blended_pyramid[-1]
    for i in range(num_bands - 2, -1, -1):
        target_size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        upsampled = cv.pyrUp(result)
        
        # Ensure dimensions match for addition
        if upsampled.shape[:2] != blended_pyramid[i].shape[:2]:
            upsampled = cv.resize(upsampled, target_size)
            
        result = upsampled + blended_pyramid[i]
    
    return result


def seam_blend(warped_images, masks):
    """Seam-based blending using graph cuts (simplified version)."""
    # For now, fall back to feather blending
    # A full graph-cut seam finding implementation would be quite complex
    return feather_blend(warped_images, masks)


# Maintain backward compatibility
def warp_and_blend(images, homographies, reference=0):
    """Legacy function for backward compatibility."""
    return advanced_warp_and_blend(images, homographies, reference, 'feather')


def compare_with_mobile_panorama(custom_panorama, mobile_panorama_path=None):
    """Compare custom stitching result with mobile device panorama.
    
    Args:
        custom_panorama: Result from custom stitching algorithm
        mobile_panorama_path: Path to mobile device panorama (optional)
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'custom_metrics': assess_panorama_quality(custom_panorama),
        'mobile_metrics': None,
        'comparison_metrics': None
    }
    
    if mobile_panorama_path and Path(mobile_panorama_path).exists():
        try:
            mobile_pano = cv.imread(str(mobile_panorama_path))
            if mobile_pano is not None:
                comparison['mobile_metrics'] = assess_panorama_quality(mobile_pano)
                
                # Resize images to same height for comparison
                h_custom = custom_panorama.shape[0]
                h_mobile = mobile_pano.shape[0]
                
                if h_mobile != h_custom:
                    aspect_ratio = mobile_pano.shape[1] / mobile_pano.shape[0]
                    new_width = int(h_custom * aspect_ratio)
                    mobile_pano_resized = cv.resize(mobile_pano, (new_width, h_custom))
                else:
                    mobile_pano_resized = mobile_pano
                
                # Compute comparison metrics
                comparison['comparison_metrics'] = {
                    'resolution_ratio': (custom_panorama.shape[0] * custom_panorama.shape[1]) / 
                                      (mobile_pano.shape[0] * mobile_pano.shape[1]),
                    'sharpness_ratio': comparison['custom_metrics']['sharpness'] / 
                                     comparison['mobile_metrics']['sharpness'],
                    'aspect_ratio_custom': custom_panorama.shape[1] / custom_panorama.shape[0],
                    'aspect_ratio_mobile': mobile_pano.shape[1] / mobile_pano.shape[0]
                }
        
        except Exception as e:
            comparison['mobile_error'] = str(e)
    
    return comparison


def assess_panorama_quality(panorama):
    """Comprehensive panorama quality assessment.
    
    Args:
        panorama: Input panorama image
    
    Returns:
        Dictionary with quality metrics
    """
    metrics = {}
    
    # Convert to grayscale for analysis
    if len(panorama.shape) == 3:
        gray = cv.cvtColor(panorama, cv.COLOR_BGR2GRAY)
    else:
        gray = panorama.copy()
    
    # 1. Sharpness (Laplacian variance)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    metrics['sharpness'] = laplacian.var()
    
    # 2. Contrast (standard deviation)
    metrics['contrast'] = gray.std()
    
    # 3. Brightness statistics
    metrics['mean_brightness'] = gray.mean()
    metrics['brightness_std'] = gray.std()
    
    # 4. Edge density (measure of detail preservation)
    edges = cv.Canny(gray, 50, 150)
    metrics['edge_density'] = np.sum(edges > 0) / edges.size
    
    # 5. Noise estimation (using high-frequency components)
    # Apply high-pass filter
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered = cv.filter2D(gray, cv.CV_32F, kernel)
    metrics['noise_estimate'] = np.std(filtered)
    
    # 6. Geometric distortion (aspect ratio analysis)
    h, w = panorama.shape[:2]
    metrics['aspect_ratio'] = w / h
    metrics['resolution'] = w * h
    
    # 7. Seam visibility assessment
    # Look for vertical discontinuities (assuming horizontal stitching)
    gradients_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    vertical_gradient_strength = np.mean(np.abs(gradients_x))
    metrics['seam_visibility'] = vertical_gradient_strength
    
    # 8. Exposure consistency
    # Divide image into regions and check brightness consistency
    regions = []
    h_regions = 3
    w_regions = min(6, w // 100)  # Adapt to image width
    
    for i in range(h_regions):
        for j in range(w_regions):
            y_start = i * h // h_regions
            y_end = (i + 1) * h // h_regions
            x_start = j * w // w_regions
            x_end = (j + 1) * w // w_regions
            
            region = gray[y_start:y_end, x_start:x_end]
            if region.size > 0:
                regions.append(region.mean())
    
    if regions:
        metrics['exposure_consistency'] = 1.0 / (1.0 + np.std(regions))
    else:
        metrics['exposure_consistency'] = 0.0
    
    # 9. Overall quality score (weighted combination)
    # Normalize metrics to 0-1 scale and combine
    normalized_sharpness = min(metrics['sharpness'] / 1000, 1.0)
    normalized_contrast = min(metrics['contrast'] / 100, 1.0)
    normalized_edge_density = metrics['edge_density']
    normalized_exposure = metrics['exposure_consistency']
    
    # Lower noise is better
    normalized_noise = max(0, 1.0 - metrics['noise_estimate'] / 50)
    
    # Combine with weights
    metrics['overall_quality'] = (
        0.25 * normalized_sharpness +
        0.20 * normalized_contrast +
        0.20 * normalized_edge_density +
        0.20 * normalized_exposure +
        0.15 * normalized_noise
    )
    
    return metrics


def create_stitching_report(images, panorama, processing_times, sift_comparison=None, mobile_comparison=None):
    """Generate comprehensive stitching report.
    
    Args:
        images: List of input images
        panorama: Final stitched panorama
        processing_times: Dictionary with timing information
        sift_comparison: SIFT comparison results (optional)
        mobile_comparison: Mobile panorama comparison (optional)
    
    Returns:
        Dictionary with comprehensive report
    """
    report = {
        'input_info': {
            'num_images': len(images),
            'input_resolutions': [f"{img.shape[1]}x{img.shape[0]}" for img in images],
            'total_input_pixels': sum(img.shape[0] * img.shape[1] for img in images)
        },
        'output_info': {
            'panorama_resolution': f"{panorama.shape[1]}x{panorama.shape[0]}",
            'panorama_pixels': panorama.shape[0] * panorama.shape[1],
            'compression_ratio': sum(img.shape[0] * img.shape[1] for img in images) / (panorama.shape[0] * panorama.shape[1])
        },
        'quality_metrics': assess_panorama_quality(panorama),
        'performance': processing_times
    }
    
    if sift_comparison:
        report['sift_comparison'] = sift_comparison
    
    if mobile_comparison:
        report['mobile_comparison'] = mobile_comparison
    
    # Add recommendations based on metrics
    recommendations = []
    
    quality = report['quality_metrics']['overall_quality']
    if quality < 0.3:
        recommendations.append("Consider using more images for better overlap")
        recommendations.append("Check image exposure and lighting conditions")
    elif quality < 0.6:
        recommendations.append("Consider adjusting SIFT parameters for better feature detection")
        recommendations.append("Try different blending modes for smoother transitions")
    else:
        recommendations.append("Good quality panorama achieved")
    
    if report['quality_metrics']['seam_visibility'] > 100:
        recommendations.append("Visible seams detected - consider multi-band blending")
    
    if report['quality_metrics']['exposure_consistency'] < 0.5:
        recommendations.append("Inconsistent exposure detected - consider exposure compensation")
    
    report['recommendations'] = recommendations
    
    return report
