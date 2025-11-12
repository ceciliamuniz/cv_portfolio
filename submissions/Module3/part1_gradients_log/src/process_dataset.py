import os
from glob import glob
import cv2
import numpy as np
from image_utils import gradient_mag_angle, laplacian_of_gaussian, save_image, panel_compare

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def list_images(folder):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    files.sort()
    return files


def maybe_generate_synthetic(folder, count=10):
    os.makedirs(folder, exist_ok=True)
    images = list_images(folder)
    if len(images) >= count:
        return images
    # Generate simple synthetic images if not enough provided
    for i in range(count - len(images)):
        img = np.zeros((360, 480, 3), dtype=np.uint8)
        # Draw shapes at varying positions/angles
        center = (np.random.randint(80, 400), np.random.randint(80, 280))
        color = (255, 255, 255)
        angle = np.random.uniform(0, 180)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Rectangle
        rect = np.zeros_like(img)
        cv2.rectangle(rect, (150, 120), (330, 240), (255, 255, 255), -1)
        img = cv2.warpAffine(rect, M, (img.shape[1], img.shape[0]))
        # Add circle
        cv2.circle(img, (np.random.randint(60, 420), np.random.randint(60, 300)), np.random.randint(20, 60), color, 2)
        # Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        path = os.path.join(folder, f'synth_{i+1:02d}.png')
        cv2.imwrite(path, noisy)
    return list_images(folder)


def process_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    images = maybe_generate_synthetic(DATA_DIR, count=10)
    if len(images) < 10:
        print(f"Warning: found only {len(images)} images; synthetic fillers added.")

    for idx, path in enumerate(images):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing {idx+1}/{len(images)}: {name}")
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  Skipped (failed to read): {path}")
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Gradients
        mag, ang = gradient_mag_angle(gray)
        save_image(os.path.join(OUT_DIR, 'grad_mag', f'{name}_grad_mag.png'), mag)
        save_image(os.path.join(OUT_DIR, 'grad_angle', f'{name}_grad_angle.png'), ang)

        # LoG
        log_viz = laplacian_of_gaussian(gray)
        save_image(os.path.join(OUT_DIR, 'log', f'{name}_log.png'), log_viz)

        # Comparison panel
        panel = panel_compare(
            (gray, mag, ang, log_viz),
            ("Original (gray)", "Gradient Magnitude", "Gradient Angle", "Laplacian of Gaussian")
        )
        save_image(os.path.join(OUT_DIR, 'comparison', f'{name}_comparison.png'), panel)

    print("Done. Outputs saved to:")
    print(f"  {os.path.abspath(OUT_DIR)}")


if __name__ == '__main__':
    process_all()
