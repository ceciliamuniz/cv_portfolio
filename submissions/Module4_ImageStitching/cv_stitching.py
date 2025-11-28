import cv2
import numpy as np
import imutils

def stitch_and_postprocess(images, output_prefix="StitchedOutput"):
    """
    Stitch images using OpenCV's Stitcher_create and post-process the result to crop borders.
    Args:
        images: List of images (numpy arrays)
        output_prefix: Prefix for output files
    Returns:
        Cropped stitched image or None if failed
    """
    if not images or len(images) < 2:
        print("[ERROR] Need at least 2 images for stitching.")
        return None
    imageStitcher = cv2.Stitcher_create()
    error, stitched_img = imageStitcher.stitch(images)
    if error == cv2.Stitcher_OK:
        print("[INFO] OpenCV Stitcher succeeded.")
        print(f"[INFO] Saving initial result as {output_prefix}.jpg")
        cv2.imwrite(f"{output_prefix}.jpg", stitched_img)
        
        # Add border for cropping
        stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros(thresh_img.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        minRect = mask.copy()
        sub = mask.copy()
        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh_img)
        contours = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        stitched_img = stitched_img[y:y + h, x:x + w]
        print(f"[INFO] Saving processed result as {output_prefix}_Processed.jpg")
        cv2.imwrite(f"{output_prefix}_Processed.jpg", stitched_img)
        return stitched_img
    else:
        print(f"[ERROR] OpenCV Stitcher failed with status {error}.")
        return None
    
if __name__ == "__main__":
    # Example usage - load images from file paths
    import glob
    image_paths = glob.glob("*.jpg") + glob.glob("*.png")
    if image_paths:
        images = [cv2.imread(path) for path in image_paths[:4]]  # Use first 4 images
        images = [img for img in images if img is not None]  # Filter valid images
        if len(images) >= 2:
            print(f"[INFO] Found {len(images)} images for stitching")
            stitched = stitch_and_postprocess(images)
            if stitched is not None:
                print("[SUCCESS] Stitching completed!")
            else:
                print("[ERROR] Stitching failed!")
        else:
            print("[ERROR] Need at least 2 valid images")
    else:
        print("[INFO] No images found in current directory. Place .jpg or .png files here to test.")