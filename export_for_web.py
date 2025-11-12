import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import json
import base64
from io import BytesIO
from PIL import Image

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
methods = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF_NORMED']

results = []
count = 0

def image_to_base64(image_path):
    """Convert image to base64 string for web display"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def cv_image_to_base64(cv_image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv.imencode('.png', cv_image)
    return base64.b64encode(buffer).decode('utf-8')

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
        
        # Store results for this combination
        combination_result = {
            "id": count,
            "scene_name": scene_file.name,
            "template_name": template_file.name,
            "scene_image": image_to_base64(scene_file),
            "template_image": image_to_base64(template_file),
            "methods": []
        }
        
        for meth in methods:
            img_copy = img2.copy()
            method = eval(meth)
            
            # Apply template Matching
            res = cv.matchTemplate(img_copy, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            
            # Get confidence score and location
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
                confidence = 1 - min_val
            else:
                top_left = max_loc
                confidence = max_val
            
            # Draw detection rectangle
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(img_copy, top_left, bottom_right, 255, 2)
            
            # Quality assessment
            match_quality = "EXCELLENT" if confidence > 0.8 else \
                           "GOOD" if confidence > 0.6 else \
                           "FAIR" if confidence > 0.4 else \
                           "POOR" if confidence > 0.2 else "VERY POOR"
            
            # Store method result
            method_result = {
                "method": meth.replace('cv.', ''),
                "confidence": round(confidence, 3),
                "quality": match_quality,
                "location": {
                    "x": int(top_left[0]),
                    "y": int(top_left[1]),
                    "width": int(w),
                    "height": int(h)
                },
                "result_image": cv_image_to_base64(img_copy)
            }
            
            combination_result["methods"].append(method_result)
            
            print(f"  {meth.replace('cv.', '')}: {confidence:.3f} ({match_quality})")
        
        results.append(combination_result)

# Save results to JSON file for React website
output_data = {
    "metadata": {
        "total_combinations": count,
        "scene_count": len(scene_files),
        "template_count": len(template_files),
        "methods_used": [m.replace('cv.', '') for m in methods],
        "generated_date": "2025-10-10"
    },
    "results": results
}

# Save to JSON file
with open("template_matching_results.json", "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Results saved to 'template_matching_results.json'")
print(f"📊 Total combinations: {count}")
print(f"💾 File size: {len(json.dumps(output_data)) / 1024:.1f} KB")
print("\n🌐 Ready for React website integration!")