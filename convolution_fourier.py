"""
Module 2 Part 2: Convolution and Fourier Transform
==================================================

Problem: 
1. Apply Gaussian Blurring filter on image L to get L_b
2. Retrieve image L back from L_b using Fourier transform

Theory:
- Convolution in spatial domain = Multiplication in frequency domain
- If L_b = L * G (where G is Gaussian kernel), then:
- FFT(L_b) = FFT(L) * FFT(G)
- Therefore: FFT(L) = FFT(L_b) / FFT(G)
- L_recovered = IFFT(FFT(L_b) / FFT(G))
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel"""
    kernel = cv.getGaussianKernel(size, sigma)
    return np.outer(kernel, kernel)

def pad_kernel_to_image_size(kernel, img_shape):
    """Pad kernel to match image size for FFT operations"""
    padded_kernel = np.zeros(img_shape)
    
    # Calculate padding
    kh, kw = kernel.shape
    pad_h = (img_shape[0] - kh) // 2
    pad_w = (img_shape[1] - kw) // 2
    
    # Place kernel in center
    padded_kernel[pad_h:pad_h+kh, pad_w:pad_w+kw] = kernel
    
    # Shift to move origin to corner (for FFT)
    padded_kernel = np.fft.fftshift(padded_kernel)
    
    return padded_kernel

def wiener_deconvolution(blurred_fft, kernel_fft, noise_level=0.01):
    """
    Wiener deconvolution - more stable than direct division
    
    H_wiener = conj(K) / (|K|^2 + noise_level)
    where K is the kernel in frequency domain
    """
    kernel_conj = np.conj(kernel_fft)
    kernel_mag_sq = np.abs(kernel_fft) ** 2
    
    # Wiener filter
    wiener_filter = kernel_conj / (kernel_mag_sq + noise_level)
    
    # Apply filter
    recovered_fft = blurred_fft * wiener_filter
    
    return recovered_fft

def gaussian_blur_and_recovery():
    """Main function to demonstrate Gaussian blur and recovery"""
    
    # Step 1: Load original image L
    print("Step 1: Loading original image L")
    
    # Try to load from scenes folder first, then fallback to any available image
    scenes_path = Path("images/scenes")
    if scenes_path.exists():
        image_files = list(scenes_path.glob("*.jpg")) + list(scenes_path.glob("*.png"))
        if image_files:
            img_path = image_files[0]
            print(f"Using image: {img_path.name}")
        else:
            print("No images found in scenes folder, please add an image")
            return
    else:
        print("Please create images/scenes folder and add an image, or update the path below")
        return
    
    # Load image in grayscale
    L = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if L is None:
        print(f"Could not load image: {img_path}")
        return
    
    # Convert to float for better precision
    L = L.astype(np.float64) / 255.0
    
    print(f"Original image shape: {L.shape}")
    
    # Step 2: Create Gaussian kernel and apply blur
    print("\nStep 2: Applying Gaussian blur")
    
    # Parameters for Gaussian blur
    kernel_size = 15  # Should be odd
    sigma = 3.0
    
    print(f"Gaussian kernel: size={kernel_size}, sigma={sigma}")
    
    # Method 1: Using OpenCV (for comparison)
    L_b_cv = cv.GaussianBlur((L * 255).astype(np.uint8), (kernel_size, kernel_size), sigma)
    L_b_cv = L_b_cv.astype(np.float64) / 255.0
    
    # Method 2: Manual convolution for exact kernel knowledge
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # Apply convolution manually
    L_b_manual = cv.filter2D(L, -1, gaussian_kernel)
    
    # Use manual blur for recovery (since we know exact kernel)
    L_b = L_b_manual
    
    # Step 3: Recover using Fourier Transform
    print("\nStep 3: Recovering image using Fourier Transform")
    
    # Pad kernel to image size for FFT
    padded_kernel = pad_kernel_to_image_size(gaussian_kernel, L.shape)
    
    # Compute FFTs
    L_b_fft = np.fft.fft2(L_b)
    kernel_fft = np.fft.fft2(padded_kernel)
    
    print("FFT computation complete")
    
    # Method 1: Direct deconvolution (can be unstable)
    print("Attempting direct deconvolution...")
    
    # Avoid division by very small numbers
    epsilon = 1e-10
    kernel_fft_safe = kernel_fft + epsilon * (np.abs(kernel_fft) < epsilon)
    
    L_recovered_direct_fft = L_b_fft / kernel_fft_safe
    L_recovered_direct = np.real(np.fft.ifft2(L_recovered_direct_fft))
    
    # Method 2: Wiener deconvolution (more stable)
    print("Applying Wiener deconvolution...")
    
    noise_levels = [0.001, 0.01, 0.1]
    L_recovered_wiener = {}
    
    for noise_level in noise_levels:
        L_recovered_wiener_fft = wiener_deconvolution(L_b_fft, kernel_fft, noise_level)
        L_recovered_wiener[noise_level] = np.real(np.fft.ifft2(L_recovered_wiener_fft))
    
    # Step 4: Display results
    print("\nStep 4: Displaying results")
    
    # Calculate metrics
    def calculate_psnr(original, recovered):
        mse = np.mean((original - recovered) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0  # Since we normalized to [0,1]
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim_simple(img1, img2):
        """Simplified SSIM calculation"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim
    
    # Calculate quality metrics
    psnr_direct = calculate_psnr(L, np.clip(L_recovered_direct, 0, 1))
    ssim_direct = calculate_ssim_simple(L, np.clip(L_recovered_direct, 0, 1))
    
    print(f"\nQuality Metrics:")
    print(f"Direct deconvolution - PSNR: {psnr_direct:.2f} dB, SSIM: {ssim_direct:.3f}")
    
    for noise_level in noise_levels:
        recovered = np.clip(L_recovered_wiener[noise_level], 0, 1)
        psnr = calculate_psnr(L, recovered)
        ssim = calculate_ssim_simple(L, recovered)
        print(f"Wiener (noise={noise_level}) - PSNR: {psnr:.2f} dB, SSIM: {ssim:.3f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Original process
    axes[0,0].imshow(L, cmap='gray')
    axes[0,0].set_title('Original Image L')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(L_b, cmap='gray')
    axes[0,1].set_title(f'Blurred Image L_b\n(sigma={sigma})')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(np.log(np.abs(L_b_fft) + 1), cmap='hot')
    axes[0,2].set_title('FFT Magnitude of L_b')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(np.log(np.abs(kernel_fft) + 1), cmap='hot')
    axes[0,3].set_title('FFT of Gaussian Kernel')
    axes[0,3].axis('off')
    
    # Row 2: Recovery results
    axes[1,0].imshow(np.clip(L_recovered_direct, 0, 1), cmap='gray')
    axes[1,0].set_title(f'Direct Deconvolution\nPSNR: {psnr_direct:.1f} dB')
    axes[1,0].axis('off')
    
    # Show best Wiener result
    best_noise_level = min(noise_levels, key=lambda x: abs(calculate_psnr(L, np.clip(L_recovered_wiener[x], 0, 1)) - psnr_direct))
    best_wiener = np.clip(L_recovered_wiener[best_noise_level], 0, 1)
    best_psnr = calculate_psnr(L, best_wiener)
    
    axes[1,1].imshow(best_wiener, cmap='gray')
    axes[1,1].set_title(f'Wiener Deconvolution\n(noise={best_noise_level}, PSNR: {best_psnr:.1f} dB)')
    axes[1,1].axis('off')
    
    # Show difference images
    diff_direct = np.abs(L - np.clip(L_recovered_direct, 0, 1))
    diff_wiener = np.abs(L - best_wiener)
    
    axes[1,2].imshow(diff_direct, cmap='hot')
    axes[1,2].set_title('Error: Direct Method')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(diff_wiener, cmap='hot')
    axes[1,3].set_title('Error: Wiener Method')
    axes[1,3].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Gaussian Blur and Fourier Transform Recovery', fontsize=16, y=1.02)
    plt.show()
    
    # Show frequency domain analysis
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(np.fft.fftshift(np.log(np.abs(np.fft.fft2(L)) + 1)), cmap='hot')
    plt.title('FFT of Original Image L')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(np.fft.fftshift(np.log(np.abs(L_b_fft) + 1)), cmap='hot')
    plt.title('FFT of Blurred Image L_b')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(np.fft.fftshift(np.log(np.abs(kernel_fft) + 1)), cmap='hot')
    plt.title('FFT of Gaussian Kernel')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Frequency Domain Analysis', fontsize=14)
    plt.show()
    
    # Analysis and conclusions
    print("\n" + "="*60)
    print("ANALYSIS AND CONCLUSIONS:")
    print("="*60)
    print(f"1. Original image shape: {L.shape}")
    print(f"2. Gaussian kernel: {kernel_size}x{kernel_size}, sigma={sigma}")
    print(f"3. Best recovery method: {'Wiener' if best_psnr > psnr_direct else 'Direct'}")
    print(f"4. Best PSNR achieved: {max(best_psnr, psnr_direct):.2f} dB")
    
    print("\nLimitations of deblurring:")
    print("- High frequency information is lost during blurring")
    print("- Noise gets amplified during deconvolution")
    print("- Perfect recovery is theoretically impossible")
    print("- Wiener filter provides better stability than direct division")
    
    print("\nKey insights:")
    print("- Convolution in spatial domain = Multiplication in frequency domain")
    print("- Deconvolution = Division in frequency domain")
    print("- Regularization (Wiener filter) is essential for stable results")
    
    return {
        'original': L,
        'blurred': L_b,
        'recovered_direct': L_recovered_direct,
        'recovered_wiener': L_recovered_wiener,
        'gaussian_kernel': gaussian_kernel,
        'metrics': {
            'psnr_direct': psnr_direct,
            'psnr_wiener': {nl: calculate_psnr(L, np.clip(L_recovered_wiener[nl], 0, 1)) for nl in noise_levels}
        }
    }

if __name__ == "__main__":
    print("Module 2 Part 2: Gaussian Blur and Fourier Transform Recovery")
    print("=" * 60)
    
    results = gaussian_blur_and_recovery()
    
    if results:
        print("\n✅ Experiment completed successfully!")
        print("📊 Results saved in returned dictionary")
    else:
        print("\n❌ Experiment failed - check image path")