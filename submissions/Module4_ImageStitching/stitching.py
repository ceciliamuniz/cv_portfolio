"""Stitching utilities: feature matching, RANSAC homography, warping and blending"""
import numpy as np
import cv2 as cv
from math import ceil


def pairwise_matches(img1, img2, sift_impl, use_opencv=False):
    # sift_impl: module providing build_gaussian_pyramid, detect_keypoints, etc.
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # custom SIFT
    pyr1 = sift_impl.build_gaussian_pyramid(gray1)
    pyr2 = sift_impl.build_gaussian_pyramid(gray2)
    dog1 = sift_impl.compute_dog(pyr1)
    dog2 = sift_impl.compute_dog(pyr2)
    kps1 = sift_impl.detect_keypoints(dog1)
    kps2 = sift_impl.detect_keypoints(dog2)
    kps1_o = sift_impl.assign_orientations(kps1, pyr1)
    kps2_o = sift_impl.assign_orientations(kps2, pyr2)
    desc1 = sift_impl.compute_descriptors(kps1_o, pyr1)
    desc2 = sift_impl.compute_descriptors(kps2_o, pyr2)

    # build keypoint coordinate lists
    pts1 = [d['pt'] for d in desc1]
    pts2 = [d['pt'] for d in desc2]

    matches = sift_impl.match_descriptors(desc1, desc2)
    good = []
    for i1, i2, _ in matches:
        if i1 < len(pts1) and i2 < len(pts2):
            good.append((pts1[i1], pts2[i2]))
    return good


def estimate_homography_ransac(matches, thresh=4.0, max_iters=2000):
    if len(matches) < 4:
        return None, []
    src = np.array([m[0] for m in matches], dtype=np.float32)
    dst = np.array([m[1] for m in matches], dtype=np.float32)
    bestH = None
    best_inliers = []
    n = len(matches)
    for _ in range(max_iters):
        idx = np.random.choice(n, 4, replace=False)
        src_sample = src[idx]
        dst_sample = dst[idx]
        H, _ = cv.findHomography(src_sample, dst_sample, 0)
        if H is None:
            continue
        proj = cv.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        dists = np.linalg.norm(proj - dst, axis=1)
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            bestH = H
    if bestH is None:
        return None, []
    return bestH, best_inliers.tolist()


def warp_and_blend(images, homographies, reference=0):
    # compute bounding box of all warped images
    h_ref, w_ref = images[reference].shape[:2]
    corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        H = homographies[i]
        warped = cv.perspectiveTransform(pts, H)
        corners.append(warped.reshape(-1, 2))
    all_pts = np.vstack(corners)
    min_x, min_y = np.min(all_pts, axis=0).astype(int)
    max_x, max_y = np.max(all_pts, axis=0).astype(int)

    # translation to keep positive
    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for i, img in enumerate(images):
        H = homographies[i].copy()
        H[0, 2] += tx
        H[1, 2] += ty
        warped = cv.warpPerspective(img, H, (canvas_w, canvas_h))
        mask = (cv.cvtColor(warped, cv.COLOR_BGR2GRAY) > 0).astype(np.float32)
        # simple feather blending
        canvas = (canvas.astype(np.float32) * weight[..., None] + warped.astype(np.float32) * mask[..., None])
        weight = weight + mask
        # avoid division by zero later
        weight[weight == 0] = 1.0
        canvas = (canvas / weight[..., None]).astype(np.uint8)

    return canvas
