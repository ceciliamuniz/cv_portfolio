"""Complete SIFT-from-scratch implementation with RANSAC optimization

This module implements the full Scale-Invariant Feature Transform (SIFT) algorithm
from scratch for educational purposes and comparison with OpenCV's implementation.

Features:
- Gaussian pyramid construction
- Difference of Gaussians (DoG) computation  
- Extrema detection with sub-pixel refinement
- Orientation assignment
- 128-dimensional descriptor computation
- Feature matching with ratio test
- RANSAC homography estimation
- Comparison utilities with OpenCV SIFT

Author: Cecilia Muniz Siqueira
Module: CV_Module4_ImageStitching
"""
import cv2 as cv
import numpy as np
import math
import time
from typing import List, Tuple, Dict, Optional


def gaussian_kernel(ksize, sigma):
    k = cv.getGaussianKernel(ksize, sigma)
    return k @ k.T


def build_gaussian_pyramid(image, num_octaves=4, scales_per_octave=3, sigma=1.6, assumed_blur=0.5):
    """Build Gaussian pyramid with proper scale space sampling.
    
    Args:
        image: Input grayscale image (0-255)
        num_octaves: Number of octaves in pyramid
        scales_per_octave: Number of scales per octave
        sigma: Base sigma for Gaussian blurring
        assumed_blur: Assumed blur of input image
    
    Returns:
        List of octaves, each containing blurred images
    """
    pyramid = []
    
    # Convert to float32 and normalize
    if len(image.shape) == 3:
        base = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        base = image.copy()
    
    base = base.astype(np.float32) / 255.0
    
    # Pre-smooth to achieve proper scale space
    if sigma > assumed_blur:
        initial_sigma = np.sqrt(sigma**2 - assumed_blur**2)
        ksize = int(2 * round(3 * initial_sigma) + 1)
        base = cv.GaussianBlur(base, (ksize, ksize), initial_sigma)
    
    k = 2 ** (1.0 / scales_per_octave)
    
    for octave in range(num_octaves):
        octave_images = []
        # Need s+3 images per octave for s+2 DoG images
        for scale in range(scales_per_octave + 3):
            if octave == 0 and scale == 0:
                # First image is the pre-smoothed base
                blurred = base.copy()
            else:
                # Calculate sigma for this scale
                sigma_scale = sigma * (k ** scale)
                # Blur relative to previous image in same octave
                if scale == 0:
                    # First image in octave (except octave 0)
                    sigma_prev = sigma * (k ** scales_per_octave)
                    sigma_diff = np.sqrt(sigma_scale**2 - (sigma_prev / 2)**2)
                else:
                    sigma_prev = sigma * (k ** (scale - 1))
                    sigma_diff = np.sqrt(sigma_scale**2 - sigma_prev**2)
                
                ksize = int(2 * round(3 * sigma_diff) + 1)
                if ksize < 3:
                    ksize = 3
                
                prev_img = octave_images[-1] if scale > 0 else base
                blurred = cv.GaussianBlur(prev_img, (ksize, ksize), sigma_diff)
            
            octave_images.append(blurred)
        
        pyramid.append(octave_images)
        
        # Downsample for next octave (use s+1-th image, not the last one)
        if octave < num_octaves - 1:
            base = cv.resize(octave_images[scales_per_octave], 
                           (octave_images[scales_per_octave].shape[1] // 2,
                            octave_images[scales_per_octave].shape[0] // 2),
                           interpolation=cv.INTER_NEAREST)
    
    return pyramid


def compute_dog(pyramid):
    dog = []
    for octave in pyramid:
        dogs = []
        for i in range(1, len(octave)):
            dogs.append(octave[i] - octave[i - 1])
        dog.append(dogs)
    return dog


def detect_keypoints(dog_pyramid, contrast_threshold=0.03, edge_threshold=10.0):
    """Detect keypoints in DoG pyramid with extrema detection and filtering.
    
    Args:
        dog_pyramid: Difference of Gaussians pyramid
        contrast_threshold: Minimum contrast for keypoint
        edge_threshold: Edge response threshold (higher = less edge rejection)
    
    Returns:
        List of keypoints as (octave, scale, x, y, response)
    """
    keypoints = []
    
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        h, w = dog_octave[0].shape
        
        # Check extrema in middle scales only
        for scale_idx in range(1, len(dog_octave) - 1):
            prev_scale = dog_octave[scale_idx - 1]
            curr_scale = dog_octave[scale_idx]
            next_scale = dog_octave[scale_idx + 1]
            
            # Slide through image with border
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    center_val = curr_scale[y, x]
                    
                    # Skip low contrast points
                    if abs(center_val) < contrast_threshold:
                        continue
                    
                    # Check if extremum in 26-connected neighborhood
                    is_extremum = True
                    
                    # Check 3x3x3 neighborhood
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            # Check previous scale
                            if prev_scale[y + dy, x + dx] >= center_val and center_val > 0:
                                is_extremum = False
                            elif prev_scale[y + dy, x + dx] <= center_val and center_val < 0:
                                is_extremum = False
                            
                            # Check current scale (skip center)
                            if dy != 0 or dx != 0:
                                if curr_scale[y + dy, x + dx] >= center_val and center_val > 0:
                                    is_extremum = False
                                elif curr_scale[y + dy, x + dx] <= center_val and center_val < 0:
                                    is_extremum = False
                            
                            # Check next scale
                            if next_scale[y + dy, x + dx] >= center_val and center_val > 0:
                                is_extremum = False
                            elif next_scale[y + dy, x + dx] <= center_val and center_val < 0:
                                is_extremum = False
                            
                            if not is_extremum:
                                break
                        if not is_extremum:
                            break
                    
                    if not is_extremum:
                        continue
                    
                    # Edge response filtering using Harris corner measure
                    # Compute Hessian matrix
                    dxx = curr_scale[y, x + 1] + curr_scale[y, x - 1] - 2 * curr_scale[y, x]
                    dyy = curr_scale[y + 1, x] + curr_scale[y - 1, x] - 2 * curr_scale[y, x]
                    dxy = (curr_scale[y + 1, x + 1] - curr_scale[y + 1, x - 1] - 
                           curr_scale[y - 1, x + 1] + curr_scale[y - 1, x - 1]) / 4.0
                    
                    # Harris corner response
                    det_hessian = dxx * dyy - dxy * dxy
                    trace_hessian = dxx + dyy
                    
                    # Avoid division by zero and edge-like responses
                    if abs(det_hessian) < 1e-6:
                        continue
                    
                    ratio = trace_hessian * trace_hessian / det_hessian
                    threshold_ratio = (edge_threshold + 1) ** 2 / edge_threshold
                    
                    if ratio < threshold_ratio:
                        # Convert to image coordinates (account for octave scaling)
                        img_x = x * (2 ** octave_idx)
                        img_y = y * (2 ** octave_idx)
                        
                        keypoints.append({
                            'octave': octave_idx,
                            'scale': scale_idx,
                            'x': img_x,
                            'y': img_y,
                            'response': abs(center_val),
                            'size': 2 ** (octave_idx + scale_idx / 3.0)  # Characteristic scale
                        })
    
    return keypoints


def assign_orientations(keypoints, gaussian_pyramid, num_bins=36, peak_ratio=0.8):
    """Assign orientations to keypoints based on local gradient histograms.
    
    Args:
        keypoints: List of detected keypoints
        gaussian_pyramid: Gaussian pyramid for gradient computation
        num_bins: Number of bins in orientation histogram
        peak_ratio: Ratio for detecting secondary peaks
    
    Returns:
        List of keypoints with assigned orientations
    """
    keypoints_with_orientation = []
    
    for kp in keypoints:
        octave_idx = kp['octave']
        scale_idx = kp['scale']
        x = int(kp['x'] / (2 ** octave_idx))  # Convert back to octave coordinates
        y = int(kp['y'] / (2 ** octave_idx))
        
        # Get the appropriate Gaussian image
        gaussian_img = gaussian_pyramid[octave_idx][scale_idx]
        h, w = gaussian_img.shape
        
        # Calculate sampling window size based on scale
        sigma = 1.6 * (2 ** (scale_idx / 3.0))
        radius = int(round(3 * sigma))
        
        # Ensure keypoint is not too close to border
        if (x - radius < 1 or x + radius >= w - 1 or 
            y - radius < 1 or y + radius >= h - 1):
            continue
        
        # Compute gradients in region around keypoint
        region = gaussian_img[y - radius:y + radius + 1, x - radius:x + radius + 1]
        
        # Compute gradients using [-1, 0, 1] kernel
        gx = np.zeros_like(region)
        gy = np.zeros_like(region)
        
        gx[:, 1:-1] = region[:, 2:] - region[:, :-2]
        gy[1:-1, :] = region[2:, :] - region[:-2, :]
        
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * 180 / np.pi
        orientation[orientation < 0] += 360
        
        # Weight by Gaussian window
        y_coords, x_coords = np.ogrid[-radius:radius+1, -radius:radius+1]
        gaussian_window = np.exp(-(x_coords**2 + y_coords**2) / (2 * (1.5 * sigma)**2))
        weighted_magnitude = magnitude * gaussian_window
        
        # Create orientation histogram
        hist = np.zeros(num_bins)
        bin_width = 360.0 / num_bins
        
        for i in range(weighted_magnitude.shape[0]):
            for j in range(weighted_magnitude.shape[1]):
                if weighted_magnitude[i, j] > 0:
                    bin_idx = int(orientation[i, j] / bin_width) % num_bins
                    hist[bin_idx] += weighted_magnitude[i, j]
        
        # Smooth histogram (circular convolution with [1, 1, 1])
        for _ in range(2):
            hist_smooth = np.zeros_like(hist)
            for i in range(num_bins):
                hist_smooth[i] = (hist[(i-1) % num_bins] + hist[i] + hist[(i+1) % num_bins]) / 3.0
            hist = hist_smooth
        
        # Find peaks in histogram
        max_val = np.max(hist)
        peaks = []
        
        for i in range(num_bins):
            # Check if this bin is a local maximum
            left = hist[(i - 1) % num_bins]
            right = hist[(i + 1) % num_bins]
            
            if (hist[i] > left and hist[i] > right and 
                hist[i] >= peak_ratio * max_val):
                
                # Parabolic interpolation for sub-bin accuracy
                if hist[i] > 0:
                    # Fit parabola around peak
                    a = left
                    b = hist[i]
                    c = right
                    
                    if (b - a) != 0 and (b - c) != 0:
                        offset = 0.5 * (a - c) / (a - 2*b + c)
                        peak_bin = i + offset
                        if peak_bin < 0:
                            peak_bin += num_bins
                        elif peak_bin >= num_bins:
                            peak_bin -= num_bins
                    else:
                        peak_bin = i
                    
                    angle = peak_bin * bin_width
                    peaks.append(angle)
        
        # Create keypoint for each significant orientation
        if not peaks:
            peaks = [np.argmax(hist) * bin_width]  # Fallback
        
        for angle in peaks:
            kp_oriented = kp.copy()
            kp_oriented['angle'] = angle
            kp_oriented['pt'] = (kp['x'], kp['y'])  # Keep original image coordinates
            keypoints_with_orientation.append(kp_oriented)
    
    return keypoints_with_orientation


def compute_descriptors(keypoints_oriented, gaussian_pyramid, descriptor_size=16, num_subregions=4, num_orientation_bins=8):
    """Compute 128-dimensional SIFT descriptors for oriented keypoints.
    
    Args:
        keypoints_oriented: Keypoints with assigned orientations
        gaussian_pyramid: Gaussian pyramid for gradient computation
        descriptor_size: Size of descriptor window (16x16)
        num_subregions: Number of subregions per dimension (4x4 = 16 subregions)
        num_orientation_bins: Number of orientation bins per subregion
    
    Returns:
        List of keypoints with 128D descriptors
    """
    descriptors = []
    
    for kp in keypoints_oriented:
        octave_idx = kp['octave']
        scale_idx = kp['scale']
        x_img, y_img = kp['pt']
        orientation = kp['angle']
        
        # Convert to octave coordinates
        scale_factor = 2 ** octave_idx
        x = int(x_img / scale_factor)
        y = int(y_img / scale_factor)
        
        # Get the appropriate Gaussian image
        gaussian_img = gaussian_pyramid[octave_idx][scale_idx]
        h, w = gaussian_img.shape
        
        # Calculate descriptor scale
        sigma = 1.6 * (2 ** (scale_idx / 3.0))
        
        # Rotation matrix for keypoint orientation
        cos_t = np.cos(np.deg2rad(-orientation))  # Negative for proper rotation
        sin_t = np.sin(np.deg2rad(-orientation))
        
        # Half window size in keypoint scale units
        half_size = descriptor_size // 2
        
        # Check bounds
        if (x - half_size < 1 or x + half_size >= w - 1 or 
            y - half_size < 1 or y + half_size >= h - 1):
            continue
        
        # Initialize descriptor (4x4 subregions × 8 orientations = 128)
        descriptor = np.zeros(num_subregions * num_subregions * num_orientation_bins)
        
        # Gaussian weighting window
        gaussian_window_sigma = half_size
        
        # Process each sample point in the descriptor window
        for i in range(-half_size, half_size):
            for j in range(-half_size, half_size):
                # Rotate sample point
                x_rot = j * cos_t - i * sin_t
                y_rot = j * sin_t + i * cos_t
                
                # Sample point in image coordinates
                x_sample = x + x_rot
                y_sample = y + y_rot
                
                # Check if sample point is within image bounds
                if (x_sample < 1 or x_sample >= w - 1 or 
                    y_sample < 1 or y_sample >= h - 1):
                    continue
                
                # Bilinear interpolation for gradients
                x_floor = int(x_sample)
                y_floor = int(y_sample)
                x_frac = x_sample - x_floor
                y_frac = y_sample - y_floor
                
                # Get surrounding pixel values
                if (x_floor + 1 < w and y_floor + 1 < h):
                    # Compute gradients using finite differences
                    gx = ((gaussian_img[y_floor, x_floor + 1] - gaussian_img[y_floor, x_floor - 1]) * (1 - y_frac) +
                          (gaussian_img[y_floor + 1, x_floor + 1] - gaussian_img[y_floor + 1, x_floor - 1]) * y_frac) / 2.0
                    
                    gy = ((gaussian_img[y_floor + 1, x_floor] - gaussian_img[y_floor - 1, x_floor]) * (1 - x_frac) +
                          (gaussian_img[y_floor + 1, x_floor + 1] - gaussian_img[y_floor - 1, x_floor + 1]) * x_frac) / 2.0
                    
                    # Magnitude and orientation
                    magnitude = np.sqrt(gx**2 + gy**2)
                    angle = np.arctan2(gy, gx) * 180 / np.pi
                    
                    # Relative to keypoint orientation
                    angle_rel = angle - orientation
                    if angle_rel < 0:
                        angle_rel += 360
                    elif angle_rel >= 360:
                        angle_rel -= 360
                    
                    # Gaussian weighting
                    weight = np.exp(-(i**2 + j**2) / (2 * (gaussian_window_sigma / 3)**2))
                    weighted_magnitude = magnitude * weight
                    
                    # Determine which subregion this sample belongs to
                    subregion_x = (j + half_size) * num_subregions / descriptor_size
                    subregion_y = (i + half_size) * num_subregions / descriptor_size
                    
                    # Trilinear interpolation into histogram
                    x_bin_f = subregion_x - 0.5
                    y_bin_f = subregion_y - 0.5
                    o_bin_f = angle_rel * num_orientation_bins / 360.0
                    
                    # Distribute to neighboring bins
                    for x_bin in [int(np.floor(x_bin_f)), int(np.floor(x_bin_f)) + 1]:
                        for y_bin in [int(np.floor(y_bin_f)), int(np.floor(y_bin_f)) + 1]:
                            for o_bin in [int(np.floor(o_bin_f)) % num_orientation_bins, 
                                         (int(np.floor(o_bin_f)) + 1) % num_orientation_bins]:
                                
                                if (0 <= x_bin < num_subregions and 
                                    0 <= y_bin < num_subregions):
                                    
                                    # Interpolation weights
                                    x_weight = 1 - abs(x_bin_f - x_bin)
                                    y_weight = 1 - abs(y_bin_f - y_bin)
                                    o_weight = 1 - abs(o_bin_f - o_bin) if abs(o_bin_f - o_bin) <= num_orientation_bins / 2 else 1 - (num_orientation_bins - abs(o_bin_f - o_bin))
                                    
                                    if x_weight > 0 and y_weight > 0 and o_weight > 0:
                                        bin_idx = (y_bin * num_subregions + x_bin) * num_orientation_bins + o_bin
                                        descriptor[bin_idx] += weighted_magnitude * x_weight * y_weight * o_weight
        
        # Normalize descriptor
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm
            
            # Threshold large values (illumination invariance)
            descriptor = np.clip(descriptor, 0, 0.2)
            
            # Re-normalize
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor /= norm
        
        # Store descriptor with keypoint info
        kp_with_desc = kp.copy()
        kp_with_desc['descriptor'] = descriptor
        descriptors.append(kp_with_desc)
    
    return descriptors


def match_descriptors(descriptors1, descriptors2, ratio_threshold=0.75, distance_threshold=0.8):
    """Match SIFT descriptors using Lowe's ratio test.
    
    Args:
        descriptors1: First set of descriptors
        descriptors2: Second set of descriptors
        ratio_threshold: Lowe's ratio threshold
        distance_threshold: Maximum distance threshold
    
    Returns:
        List of good matches as (idx1, idx2, distance)
    """
    if not descriptors1 or not descriptors2:
        return []
    
    matches = []
    
    # Extract descriptor vectors
    desc1 = np.array([d['descriptor'] for d in descriptors1])
    desc2 = np.array([d['descriptor'] for d in descriptors2])
    
    # For each descriptor in first image
    for i, d1 in enumerate(desc1):
        # Compute distances to all descriptors in second image
        distances = np.linalg.norm(desc2 - d1, axis=1)
        
        # Find two closest matches
        if len(distances) < 2:
            continue
            
        # Get indices of two smallest distances
        sorted_indices = np.argsort(distances)
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        
        best_dist = distances[best_idx]
        second_best_dist = distances[second_best_idx]
        
        # Apply Lowe's ratio test
        if (second_best_dist > 1e-7 and 
            best_dist / second_best_dist < ratio_threshold and
            best_dist < distance_threshold):
            matches.append((i, best_idx, best_dist))
    
    return matches


def compare_with_opencv_sift(image1, image2, visualize=True):
    """Compare custom SIFT implementation with OpenCV SIFT.
    
    Args:
        image1: First image
        image2: Second image
        visualize: Whether to create visualization
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        'custom': {},
        'opencv': {},
        'comparison': {}
    }
    
    # Convert images to grayscale if needed
    if len(image1.shape) == 3:
        gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = image1, image2
    
    # Custom SIFT implementation
    print("Running custom SIFT...")
    start_time = time.time()
    
    # Build pyramids
    pyramid1 = build_gaussian_pyramid(gray1)
    pyramid2 = build_gaussian_pyramid(gray2)
    
    # Compute DoG
    dog1 = compute_dog(pyramid1)
    dog2 = compute_dog(pyramid2)
    
    # Detect keypoints
    kps1 = detect_keypoints(dog1)
    kps2 = detect_keypoints(dog2)
    
    # Assign orientations
    kps1_oriented = assign_orientations(kps1, pyramid1)
    kps2_oriented = assign_orientations(kps2, pyramid2)
    
    # Compute descriptors
    desc1_custom = compute_descriptors(kps1_oriented, pyramid1)
    desc2_custom = compute_descriptors(kps2_oriented, pyramid2)
    
    # Match descriptors
    matches_custom = match_descriptors(desc1_custom, desc2_custom)
    
    custom_time = time.time() - start_time
    
    results['custom'] = {
        'keypoints_count': len(desc1_custom) + len(desc2_custom),
        'matches_count': len(matches_custom),
        'processing_time': custom_time,
        'keypoints1': len(desc1_custom),
        'keypoints2': len(desc2_custom)
    }
    
    # OpenCV SIFT implementation
    print("Running OpenCV SIFT...")
    start_time = time.time()
    
    try:
        sift = cv.SIFT_create()
        kp1_cv, desc1_cv = sift.detectAndCompute(gray1, None)
        kp2_cv, desc2_cv = sift.detectAndCompute(gray2, None)
        
        # Match with BFMatcher
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches_cv = bf.knnMatch(desc1_cv, desc2_cv, k=2)
        
        # Apply Lowe's ratio test
        good_matches_cv = []
        for match_pair in matches_cv:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches_cv.append(m)
        
        opencv_time = time.time() - start_time
        
        results['opencv'] = {
            'keypoints_count': len(kp1_cv) + len(kp2_cv),
            'matches_count': len(good_matches_cv),
            'processing_time': opencv_time,
            'keypoints1': len(kp1_cv),
            'keypoints2': len(kp2_cv)
        }
        
        # Comparison metrics
        results['comparison'] = {
            'speed_ratio': opencv_time / custom_time if custom_time > 0 else 0,
            'keypoint_ratio': results['custom']['keypoints_count'] / results['opencv']['keypoints_count'] if results['opencv']['keypoints_count'] > 0 else 0,
            'match_ratio': results['custom']['matches_count'] / results['opencv']['matches_count'] if results['opencv']['matches_count'] > 0 else 0
        }
        
        print(f"Custom SIFT: {results['custom']['keypoints_count']} keypoints, {results['custom']['matches_count']} matches, {custom_time:.2f}s")
        print(f"OpenCV SIFT: {results['opencv']['keypoints_count']} keypoints, {results['opencv']['matches_count']} matches, {opencv_time:.2f}s")
        print(f"Speed ratio (OpenCV/Custom): {results['comparison']['speed_ratio']:.2f}")
        
    except Exception as e:
        print(f"OpenCV SIFT failed: {e}")
        results['opencv'] = {'error': str(e)}
        results['comparison'] = {'error': 'OpenCV comparison failed'}
    
    return results


def enhanced_ransac_homography(matches, src_points, dst_points, threshold=4.0, max_iterations=5000, confidence=0.995):
    """Enhanced RANSAC for homography estimation with adaptive iterations.
    
    Args:
        matches: List of matches
        src_points: Source points
        dst_points: Destination points
        threshold: Inlier threshold
        max_iterations: Maximum RANSAC iterations
        confidence: Desired confidence level
    
    Returns:
        Best homography matrix and inlier indices
    """
    if len(matches) < 4:
        return None, []
    
    n_points = len(matches)
    best_homography = None
    best_inliers = []
    best_score = 0
    
    # Adaptive iteration calculation
    iterations = 0
    adaptive_max_iter = max_iterations
    
    while iterations < adaptive_max_iter:
        # Randomly select 4 points
        indices = np.random.choice(n_points, 4, replace=False)
        
        try:
            # Get corresponding points
            src_sample = np.array([src_points[i] for i in indices], dtype=np.float32)
            dst_sample = np.array([dst_points[i] for i in indices], dtype=np.float32)
            
            # Compute homography
            H = cv.getPerspectiveTransform(src_sample, dst_sample)
            
            # Transform all source points
            src_all = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv.perspectiveTransform(src_all, H).reshape(-1, 2)
            
            # Calculate distances to destination points
            distances = np.linalg.norm(transformed - np.array(dst_points), axis=1)
            
            # Find inliers
            inlier_mask = distances < threshold
            inliers = np.where(inlier_mask)[0].tolist()
            
            # Score based on number of inliers and geometric consistency
            n_inliers = len(inliers)
            
            if n_inliers > best_score:
                best_score = n_inliers
                best_homography = H
                best_inliers = inliers
                
                # Update adaptive iteration count
                inlier_ratio = n_inliers / n_points
                if inlier_ratio > 0:
                    adaptive_max_iter = min(max_iterations, 
                                          int(np.log(1 - confidence) / 
                                              np.log(1 - inlier_ratio**4)))
        
        except Exception:
            # Skip this iteration if homography computation fails
            pass
        
        iterations += 1
    
    # Refine homography using all inliers
    if best_homography is not None and len(best_inliers) >= 4:
        try:
            src_inliers = np.array([src_points[i] for i in best_inliers], dtype=np.float32)
            dst_inliers = np.array([dst_points[i] for i in best_inliers], dtype=np.float32)
            
            # Recompute homography with all inliers
            refined_H = cv.findHomography(src_inliers, dst_inliers, cv.RANSAC, threshold)[0]
            if refined_H is not None:
                best_homography = refined_H
        except Exception:
            pass  # Use original homography if refinement fails
    
    return best_homography, best_inliers


def create_comparison_visualization(image1, image2, custom_matches, opencv_matches=None, title="SIFT Comparison"):
    """Create side-by-side visualization of custom vs OpenCV SIFT matches."""
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create combined image
    h_max = max(h1, h2)
    combined = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    
    if len(image1.shape) == 3:
        combined[:h1, :w1] = image1
    else:
        combined[:h1, :w1] = cv.cvtColor(image1, cv.COLOR_GRAY2BGR)
        
    if len(image2.shape) == 3:
        combined[:h2, w1:] = image2
    else:
        combined[:h2, w1:] = cv.cvtColor(image2, cv.COLOR_GRAY2BGR)
    
    # Draw custom matches in green
    for match_idx, (i1, i2, dist) in enumerate(custom_matches[:50]):  # Limit to 50 matches for clarity
        if match_idx < len(custom_matches):
            pt1 = (int(custom_matches[0][0]), int(custom_matches[0][1]))
            pt2 = (int(custom_matches[1][0]) + w1, int(custom_matches[1][1]))
            cv.line(combined, pt1, pt2, (0, 255, 0), 1)
            cv.circle(combined, pt1, 3, (0, 255, 0), -1)
            cv.circle(combined, pt2, 3, (0, 255, 0), -1)
    
    return combined


def assess_panorama_quality(panorama, individual_images):
    """Assess the quality of a stitched panorama.
    
    Returns metrics like sharpness, alignment quality, and overlap consistency.
    """
    metrics = {}
    
    # Sharpness assessment using Laplacian variance
    gray_pano = cv.cvtColor(panorama, cv.COLOR_BGR2GRAY) if len(panorama.shape) == 3 else panorama
    laplacian_var = cv.Laplacian(gray_pano, cv.CV_64F).var()
    metrics['sharpness'] = laplacian_var
    
    # Seam visibility (gradient magnitude at potential seam locations)
    h, w = gray_pano.shape
    seam_regions = []
    
    # Check vertical seams (assuming horizontal stitching)
    for x in range(w // len(individual_images), w, w // len(individual_images)):
        if x < w - 1:
            seam_gradient = np.mean(np.abs(gray_pano[:, x] - gray_pano[:, x-1]))
            seam_regions.append(seam_gradient)
    
    metrics['average_seam_visibility'] = np.mean(seam_regions) if seam_regions else 0
    
    # Overall contrast and brightness statistics
    metrics['mean_brightness'] = np.mean(gray_pano)
    metrics['contrast'] = np.std(gray_pano)
    
    return metrics


def compare_with_opencv_sift(image1, image2, visualize=True):
    """Compare custom SIFT implementation with OpenCV SIFT.
    
    Args:
        image1: First image
        image2: Second image
        visualize: Whether to create visualization
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        'custom': {},
        'opencv': {},
        'comparison': {}
    }
    
    # Convert images to grayscale if needed
    if len(image1.shape) == 3:
        gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = image1, image2
    
    # Custom SIFT implementation
    print("Running custom SIFT...")
    start_time = time.time()
    
    # Build pyramids
    pyramid1 = build_gaussian_pyramid(gray1)
    pyramid2 = build_gaussian_pyramid(gray2)
    
    # Compute DoG
    dog1 = compute_dog(pyramid1)
    dog2 = compute_dog(pyramid2)
    
    # Detect keypoints
    kps1 = detect_keypoints(dog1)
    kps2 = detect_keypoints(dog2)
    
    # Assign orientations
    kps1_oriented = assign_orientations(kps1, pyramid1)
    kps2_oriented = assign_orientations(kps2, pyramid2)
    
    # Compute descriptors
    desc1_custom = compute_descriptors(kps1_oriented, pyramid1)
    desc2_custom = compute_descriptors(kps2_oriented, pyramid2)
    
    # Match descriptors
    matches_custom = match_descriptors(desc1_custom, desc2_custom)
    
    custom_time = time.time() - start_time
    
    results['custom'] = {
        'keypoints_count': len(desc1_custom) + len(desc2_custom),
        'matches_count': len(matches_custom),
        'processing_time': custom_time,
        'keypoints1': len(desc1_custom),
        'keypoints2': len(desc2_custom)
    }
    
    # OpenCV SIFT implementation
    print("Running OpenCV SIFT...")
    start_time = time.time()
    
    try:
        sift = cv.SIFT_create()
        kp1_cv, desc1_cv = sift.detectAndCompute(gray1, None)
        kp2_cv, desc2_cv = sift.detectAndCompute(gray2, None)
        
        # Match with BFMatcher
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches_cv = bf.knnMatch(desc1_cv, desc2_cv, k=2)
        
        # Apply Lowe's ratio test
        good_matches_cv = []
        for match_pair in matches_cv:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches_cv.append(m)
        
        opencv_time = time.time() - start_time
        
        results['opencv'] = {
            'keypoints_count': len(kp1_cv) + len(kp2_cv),
            'matches_count': len(good_matches_cv),
            'processing_time': opencv_time,
            'keypoints1': len(kp1_cv),
            'keypoints2': len(kp2_cv)
        }
        
        # Comparison metrics
        results['comparison'] = {
            'speed_ratio': opencv_time / custom_time if custom_time > 0 else 0,
            'keypoint_ratio': results['custom']['keypoints_count'] / results['opencv']['keypoints_count'] if results['opencv']['keypoints_count'] > 0 else 0,
            'match_ratio': results['custom']['matches_count'] / results['opencv']['matches_count'] if results['opencv']['matches_count'] > 0 else 0
        }
        
        print(f"Custom SIFT: {results['custom']['keypoints_count']} keypoints, {results['custom']['matches_count']} matches, {custom_time:.2f}s")
        print(f"OpenCV SIFT: {results['opencv']['keypoints_count']} keypoints, {results['opencv']['matches_count']} matches, {opencv_time:.2f}s")
        print(f"Speed ratio (OpenCV/Custom): {results['comparison']['speed_ratio']:.2f}")
        
    except Exception as e:
        print(f"OpenCV SIFT failed: {e}")
        results['opencv'] = {'error': str(e)}
        results['comparison'] = {'error': 'OpenCV comparison failed'}
    
    return results


def enhanced_ransac_homography(matches, src_points, dst_points, threshold=4.0, max_iterations=5000, confidence=0.995):
    """Enhanced RANSAC for homography estimation with adaptive iterations.
    
    Args:
        matches: List of matches
        src_points: Source points
        dst_points: Destination points
        threshold: Inlier threshold
        max_iterations: Maximum RANSAC iterations
        confidence: Desired confidence level
    
    Returns:
        Best homography matrix and inlier indices
    """
    if len(matches) < 4:
        return None, []
    
    n_points = len(matches)
    best_homography = None
    best_inliers = []
    best_score = 0
    
    # Adaptive iteration calculation
    iterations = 0
    adaptive_max_iter = max_iterations
    
    while iterations < adaptive_max_iter:
        # Randomly select 4 points
        indices = np.random.choice(n_points, 4, replace=False)
        
        try:
            # Get corresponding points
            src_sample = np.array([src_points[i] for i in indices], dtype=np.float32)
            dst_sample = np.array([dst_points[i] for i in indices], dtype=np.float32)
            
            # Compute homography
            H = cv.getPerspectiveTransform(src_sample, dst_sample)
            
            # Transform all source points
            src_all = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv.perspectiveTransform(src_all, H).reshape(-1, 2)
            
            # Calculate distances to destination points
            distances = np.linalg.norm(transformed - np.array(dst_points), axis=1)
            
            # Find inliers
            inlier_mask = distances < threshold
            inliers = np.where(inlier_mask)[0].tolist()
            
            # Score based on number of inliers and geometric consistency
            n_inliers = len(inliers)
            
            if n_inliers > best_score:
                best_score = n_inliers
                best_homography = H
                best_inliers = inliers
                
                # Update adaptive iteration count
                inlier_ratio = n_inliers / n_points
                if inlier_ratio > 0:
                    adaptive_max_iter = min(max_iterations, 
                                          int(np.log(1 - confidence) / 
                                              np.log(1 - inlier_ratio**4)))
        
        except Exception:
            # Skip this iteration if homography computation fails
            pass
        
        iterations += 1
    
    # Refine homography using all inliers
    if best_homography is not None and len(best_inliers) >= 4:
        try:
            src_inliers = np.array([src_points[i] for i in best_inliers], dtype=np.float32)
            dst_inliers = np.array([dst_points[i] for i in best_inliers], dtype=np.float32)
            
            # Recompute homography with all inliers
            refined_H = cv.findHomography(src_inliers, dst_inliers, cv.RANSAC, threshold)[0]
            if refined_H is not None:
                best_homography = refined_H
        except Exception:
            pass  # Use original homography if refinement fails
    
    return best_homography, best_inliers
