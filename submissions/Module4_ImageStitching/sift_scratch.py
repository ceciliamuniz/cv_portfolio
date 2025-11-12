"""Minimal SIFT-from-scratch implementation (educational, simplified)
Provides: build_gaussian_pyramid, compute_dog, detect_keypoints, assign_orientations,
compute_descriptors, match_descriptors

This implementation is simplified for teaching and comparison. It is not optimized for speed.
"""
import cv2 as cv
import numpy as np
import math


def gaussian_kernel(ksize, sigma):
    k = cv.getGaussianKernel(ksize, sigma)
    return k @ k.T


def build_gaussian_pyramid(image, num_octaves=3, scales_per_octave=3, sigma=1.6):
    pyram = []
    base = image.astype(np.float32) / 255.0
    k = 2 ** (1.0 / scales_per_octave)
    for o in range(num_octaves):
        octave_imgs = []
        for s in range(scales_per_octave + 3):  # extra for DoG
            sigma_total = sigma * (k ** s) * (2 ** o)
            ksize = int(2 * round(3 * sigma_total) + 1)
            blurred = cv.GaussianBlur(base, (ksize, ksize), sigma_total)
            octave_imgs.append(blurred)
        pyram.append(octave_imgs)
        # downsample base by factor 2 for next octave
        base = cv.resize(base, (base.shape[1] // 2, base.shape[0] // 2), interpolation=cv.INTER_LINEAR)
    return pyram


def compute_dog(pyramid):
    dog = []
    for octave in pyramid:
        dogs = []
        for i in range(1, len(octave)):
            dogs.append(octave[i] - octave[i - 1])
        dog.append(dogs)
    return dog


def detect_keypoints(dog_pyr, contrast_threshold=0.03):
    keypoints = []
    for oi, dogs in enumerate(dog_pyr):
        h, w = dogs[0].shape
        for si in range(1, len(dogs) - 1):
            prev = dogs[si - 1]
            curr = dogs[si]
            nex = dogs[si + 1]
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    val = curr[y, x]
                    # 26-neighborhood
                    patch = np.concatenate([prev[y - 1:y + 2, x - 1:x + 2].ravel(),
                                            curr[y - 1:y + 2, x - 1:x + 2].ravel(),
                                            nex[y - 1:y + 2, x - 1:x + 2].ravel()])
                    if val == np.max(patch) or val == np.min(patch):
                        if abs(val) > contrast_threshold:
                            # store octave, scale, x, y
                            keypoints.append((oi, si, x, y))
    return keypoints


def assign_orientations(keypoints, gaussian_pyr, radius_factor=3, num_bins=36):
    kps_with_orient = []
    for oi, si, x, y in keypoints:
        img = gaussian_pyr[oi][si]
        h, w = img.shape
        # compute gradients
        gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx * gx + gy * gy)
        angle = (np.arctan2(gy, gx) * 180 / np.pi) % 360

        radius = int(radius_factor * 1.6)
        x0, y0 = int(x), int(y)
        x1, x2 = max(0, x0 - radius), min(w, x0 + radius + 1)
        y1, y2 = max(0, y0 - radius), min(h, y0 + radius + 1)

        mag_patch = magnitude[y1:y2, x1:x2]
        ang_patch = angle[y1:y2, x1:x2]
        # histogram
        hist, _ = np.histogram(ang_patch, bins=num_bins, range=(0, 360), weights=mag_patch)
        max_bin = np.argmax(hist)
        main_angle = (max_bin + 0.5) * (360.0 / num_bins)
        kps_with_orient.append({'octave': oi, 'scale': si, 'pt': (x0, y0), 'angle': main_angle})
    return kps_with_orient


def compute_descriptors(kps_orient, gaussian_pyr, window_width=16, num_cells=4, num_bins=8):
    descriptors = []
    for kp in kps_orient:
        oi = kp['octave']
        si = kp['scale']
        x, y = kp['pt']
        angle = np.deg2rad(kp['angle'])
        img = gaussian_pyr[oi][si]
        h, w = img.shape
        half = window_width // 2
        x1, x2 = x - half, x + half
        y1, y2 = y - half, y + half
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue
        patch = img[y1:y2, x1:x2]
        # gradients
        gx = cv.Sobel(patch, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(patch, cv.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) - angle) * 180 / np.pi
        ang = (ang + 360) % 360

        cell_w = window_width // num_cells
        desc = []
        for i in range(num_cells):
            for j in range(num_cells):
                ys = i * cell_w
                xs = j * cell_w
                mag_cell = mag[ys:ys + cell_w, xs:xs + cell_w]
                ang_cell = ang[ys:ys + cell_w, xs:xs + cell_w]
                hist, _ = np.histogram(ang_cell, bins=num_bins, range=(0, 360), weights=mag_cell)
                desc.extend(hist)
        desc = np.array(desc, dtype=np.float32)
        # normalize
        desc /= (np.linalg.norm(desc) + 1e-7)
        # threshold and renormalize
        desc = np.clip(desc, 0, 0.2)
        desc /= (np.linalg.norm(desc) + 1e-7)
        descriptors.append({'octave': oi, 'scale': si, 'pt': (x, y), 'angle': kp['angle'], 'desc': desc})
    return descriptors


def match_descriptors(descs1, descs2, ratio_thresh=0.75):
    matches = []
    feats1 = np.array([d['desc'] for d in descs1])
    feats2 = np.array([d['desc'] for d in descs2])
    if feats1.size == 0 or feats2.size == 0:
        return matches
    for i, f in enumerate(feats1):
        dists = np.linalg.norm(feats2 - f, axis=1)
        if len(dists) < 2:
            continue
        idx = np.argpartition(dists, 2)[:2]
        if dists[idx[0]] < 1e-9:
            ratio = 0.0
        else:
            ratio = dists[idx[0]] / (dists[idx[1]] + 1e-9)
        if ratio < ratio_thresh:
            matches.append((i, int(idx[0]), float(dists[idx[0]])))
    return matches
