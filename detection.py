import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

# Get all template images from folder
templates_path = Path("images/templates")
template_files = list(templates_path.glob("*.jpg")) + list(templates_path.glob("*.png"))

# Get scene images from folder  
scenes_path = Path("images/scenes")
scene_files = list(scenes_path.glob("*.jpg")) + list(scenes_path.glob("*.png"))

if len(template_files) == 0:
    print("No template images found in 'images/templates' folder")
    exit()

if len(scene_files) == 0:
    print("No scene images found in 'images/scenes' folder")  
    exit()

print(f"Found {len(scene_files)} scene images and {len(template_files)} template images")

# All the 3 normalized methods
methods = ['cv.TM_CCOEFF_NORMED', 
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF_NORMED']

count = 0

# Process each scene with each template
for scene_file in scene_files:
    img = cv.imread(str(scene_file), cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read scene: {scene_file}")
        continue
    img2 = img.copy()
    
    for template_file in template_files:
        count += 1
        print(f"\n{count}. Processing: {template_file.name} in {scene_file.name}")
        
        template = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Could not read template: {template_file}")
            continue
            
        # Skip if template is too large
        if template.shape[0] >= img.shape[0] or template.shape[1] >= img.shape[1]:
            print("Template too large, skipping...")
            continue
            
        w, h = template.shape[::-1]
        for meth in methods:
            img = img2.copy()
            method = eval(meth)
            # Apply template Matching
            res = cv.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(img,top_left, bottom_right, 255, 2)
            
            plt.figure(figsize=(12, 4))
            plt.subplot(131),plt.imshow(img2,cmap = 'gray')
            plt.title(f'Scene: {scene_file.name}'), plt.xticks([]), plt.yticks([])
            plt.subplot(132),plt.imshow(template,cmap = 'gray')
            plt.title(f'Template: {template_file.name}'), plt.xticks([]), plt.yticks([])
            plt.subplot(133),plt.imshow(img,cmap = 'gray')
            plt.title(f'Detection: {meth}'), plt.xticks([]), plt.yticks([])
            plt.tight_layout()
            plt.show()

print(f"\nCompleted! Processed {count} combinations.")