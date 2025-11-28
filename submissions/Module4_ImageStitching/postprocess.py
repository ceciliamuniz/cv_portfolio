import cv2 as cv

__all__ = ["crop_black_borders", "improve_exposure"]

def crop_black_borders(img):
    """Remove black borders from stitched image"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)
        return img[y:y+h, x:x+w]
    return img

def improve_exposure(img):
    """Apply histogram equalization for better exposure"""
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv.merge([l, a, b])
    result = cv.cvtColor(enhanced, cv.COLOR_LAB2BGR)
    return result
