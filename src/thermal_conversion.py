# src/thermal_conversion.py
import cv2
import os

def thermal_conversion(image_paths):
    thermal_dir = "data/thermal"

    # If thermal_dir exists and is NOT empty → skip generation
    if os.path.exists(thermal_dir) and len(os.listdir(thermal_dir)) > 0:
        print("[INFO] Thermal images already exist — skipping generation.")
        return
    
    # Else → generate
    os.makedirs(thermal_dir, exist_ok=True)
    print("[INFO] Generating thermal images...")

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] Loading failed for {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        img_name = os.path.basename(path)
        thermal_path = os.path.join(thermal_dir, img_name)
        cv2.imwrite(thermal_path, thermal)

    print("[INFO] Thermal image generation complete.")

