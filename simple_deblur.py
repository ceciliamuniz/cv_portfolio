"""
Simplified Gaussian Blur and Recovery Demo
==========================================

This script demonstrates the basic concept:
1. Load image L
2. Apply Gaussian blur to get L_b  
3. Attempt to recover L from L_b using FFT
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load an image
print("Loading image...")
scenes_path = Path("images/scenes")
if scenes_path.exists():
    image_files = list(scenes_path.glob("*.jpg")) + list(scenes_path.glob("*.png"))
    if image_files:
        img_path = image_files[0]
        print(f"Using: {img_path.name}")
    else:
        print("No images found in scenes folder")
        exit()
else:
    print("Please add an image to images/scenes/ folder")
    exit()

# Load and prepare image
L = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
L = L.astype(np.float64) / 255.0  # Normalize to [0,1]

print(f"Image shape: {L.shape}")

# Step 1: Apply Gaussian Blur
print("\nStep 1: Applying Gaussian Blur")
kernel_size = 15
sigma = 2.0

# Create Gaussian kernel
gaussian_kernel = cv.getGaussianKernel(kernel_size, sigma)
gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel)

# Apply blur
L_b = cv.filter2D(L, -1, gaussian_kernel_2d)

print(f"Applied Gaussian blur with kernel size {kernel_size} and sigma={sigma}")

# Step 2: Attempt Recovery using FFT
print("\nStep 2: Attempting recovery using FFT")

# Pad kernel to image size for FFT
padded_kernel = np.zeros_like(L)
kh, kw = gaussian_kernel_2d.shape
pad_h = (L.shape[0] - kh) // 2
pad_w = (L.shape[1] - kw) // 2
padded_kernel[pad_h:pad_h+kh, pad_w:pad_w+kw] = gaussian_kernel_2d

# Shift kernel for FFT (move center to corner)
padded_kernel = np.fft.fftshift(padded_kernel)

# Compute FFTs
L_b_fft = np.fft.fft2(L_b)
kernel_fft = np.fft.fft2(padded_kernel)

# Simple deconvolution: divide in frequency domain
# Add small epsilon to avoid division by zero
epsilon = 1e-10
kernel_fft_safe = kernel_fft + epsilon * (np.abs(kernel_fft) < epsilon)

L_recovered_fft = L_b_fft / kernel_fft_safe
L_recovered = np.real(np.fft.ifft2(L_recovered_fft))

# Clip to valid range
L_recovered = np.clip(L_recovered, 0, 1)

# Calculate quality metric (PSNR)
mse = np.mean((L - L_recovered) ** 2)
psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

print(f"Recovery PSNR: {psnr:.2f} dB")

# Step 3: Display Results
print("\nStep 3: Displaying results")

plt.figure(figsize=(15, 10))

# Original vs Blurred vs Recovered
plt.subplot(2, 3, 1)
plt.imshow(L, cmap='gray')
plt.title('Original Image L')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(L_b, cmap='gray')
plt.title(f'Blurred Image L_b\n(sigma={sigma})')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(L_recovered, cmap='gray')
plt.title(f'Recovered Image\nPSNR: {psnr:.1f} dB')
plt.axis('off')

# Show difference (error) images
plt.subplot(2, 3, 4)
blur_diff = np.abs(L - L_b)
plt.imshow(blur_diff, cmap='hot')
plt.title('Difference: Original vs Blurred')
plt.axis('off')

plt.subplot(2, 3, 5)
recovery_diff = np.abs(L - L_recovered)
plt.imshow(recovery_diff, cmap='hot')
plt.title('Difference: Original vs Recovered')
plt.axis('off')

# Show Gaussian kernel
plt.subplot(2, 3, 6)
plt.imshow(gaussian_kernel_2d, cmap='gray')
plt.title(f'Gaussian Kernel\n{kernel_size}x{kernel_size}, sigma={sigma}')
plt.axis('off')

plt.tight_layout()
plt.suptitle('Gaussian Blur and FFT Recovery', fontsize=16, y=1.02)
plt.show()

# Show frequency domain
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(L))) + 1), cmap='hot')
plt.title('FFT of Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(np.log(np.abs(np.fft.fftshift(L_b_fft)) + 1), cmap='hot')
plt.title('FFT of Blurred Image')
plt.axis('off')

plt.subplot(133)
plt.imshow(np.log(np.abs(np.fft.fftshift(kernel_fft)) + 1), cmap='hot')
plt.title('FFT of Gaussian Kernel')
plt.axis('off')

plt.tight_layout()
plt.suptitle('Frequency Domain Analysis', fontsize=14)
plt.show()

# Analysis
print("\n" + "="*50)
print("ANALYSIS:")
print("="*50)
print(f"• Original image: {L.shape}")
print(f"• Gaussian blur: kernel={kernel_size}x{kernel_size}, sigma={sigma}")
print(f"• Recovery PSNR: {psnr:.2f} dB")

if psnr > 20:
    print("✅ Good recovery quality")
elif psnr > 15:
    print("⚠️ Moderate recovery quality") 
else:
    print("❌ Poor recovery quality")

print("\nKey Observations:")
print("• Blurring removes high-frequency details")
print("• FFT converts convolution to multiplication")
print("• Deconvolution = division in frequency domain")
print("• Perfect recovery is impossible due to information loss")
print("• Noise amplification is a major challenge")

print("\nMathematical relationship:")
print("L_b = L ⊛ G  (convolution in spatial domain)")
print("FFT(L_b) = FFT(L) × FFT(G)  (multiplication in frequency domain)")
print("FFT(L) = FFT(L_b) ÷ FFT(G)  (deconvolution)")
print("L_recovered = IFFT(FFT(L_b) ÷ FFT(G))")

print(f"\n✅ Experiment completed! Recovery PSNR: {psnr:.2f} dB")