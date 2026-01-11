import cv2
import numpy as np
from pathlib import Path

def detect_dice_scores(image_path: str, output_path: str = "result_final.jpg"):
    # 1. Load Image
    if not Path(image_path).exists():
        print(f"Error: File {image_path} not found.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # Clone for display
    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Illumination Correction (CLAHE)
    # This acts like a "flat field correction". It boosts local contrast in dark areas
    # and limits it in bright areas, effectively removing the shadow gradient.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # 3. Blurring & Thresholding
    # Gaussian blur to smooth the fabric texture
    #blurred = cv2.GaussianBlur(gray_eq, (7, 7), 0)
    blurred = cv2.medianBlur(gray_eq, 9)

    # Global Binary Threshold
    # Since we equalized the histogram, the background is now a uniform mid-grey
    # and dice are bright white. We can safely pick a high threshold (e.g., 180-200).
    #_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)


    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 8)

    # Morphological Cleanup
    # Open: removes small white noise points
    # Dilate: closes small gaps in the dice faces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)

    # Save debug mask to verify we now have SOLID blobs
    cv2.imwrite("debug_mask_solid.jpg", thresh)

    # 4. Dice Detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dice_results = []
    
    # Sort contours by area (descending) to debug easier
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    print(f"DEBUG: Found {len(contours)} initial contours.")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter 1: Area
        # Dice are significant objects. Ignore tiny noise.
        if area < 1000: 
            continue

        # Filter 2: Shape (Square-ish)
        ### TO-DO: It can be the source of the problems: we can also use cv2.minAreaRect()
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Dice are roughly 1:1. We allow 0.6 to 1.5 for perspective/rotation.
        if aspect_ratio < 0.6 or aspect_ratio > 1.5:
            continue
        
        # Filter 3: Solidity (optional but helps)
        # Ratio of contour area to its convex hull area. 
        # A square is very solid (~1.0). Noise is usually jagged (<0.8).
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity < 0.85:
            continue

        # --- ROI Extraction for Pips ---
        # We look inside the bounding box + a small margin
        # TODO:
        # 1. maskowanie 
        # 2. filtrowanie oczek po powierzchni i kształcie
        # 3. prostowanie kostki (deskewing)
        roi = gray[y:y+h, x:x+w]
        
        # Local threshold for pips
        # Inside the white die, pips are black.
        # We use Otsu on the INVERTED roi to find the dark spots.
        _, pip_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Filter pips
        pip_contours, _ = cv2.findContours(pip_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_pips = []
        roi_area = w * h
        
        for p_cnt in pip_contours:
            p_area = cv2.contourArea(p_cnt)
            
            # Pip Filter: Size relative to the die face
            # Pips are usually 0.5% to 5% of the die face area
            if (roi_area * 0.005) < p_area < (roi_area * 0.1):
                
                # Check circularity
                perimeter = cv2.arcLength(p_cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (p_area / (perimeter * perimeter))
                
                # Pips are circular (allow > 0.6)
                if circularity > 0.6:
                    # Calculate center in global coordinates
                    M = cv2.moments(p_cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + x
                        cy = int(M["m01"] / M["m00"]) + y
                        valid_pips.append((cx, cy))

        # Dice must have 1-6 pips. 
        # Note: Sometimes a "1" is detected as 0 if the threshold is too aggressive,
        # or noise is detected as a 7th pip. We clamp or filter here.
        if 1 <= len(valid_pips) <= 6:
            dice_results.append({
                "score": len(valid_pips),
                "box": (x, y, w, h),
                "pips": valid_pips
            })
        elif len(valid_pips) > 6:
             # Fallback: if we found too many, take the largest 6 (likely noise included)
             # Or simply ignore this contour if it's garbage.
             pass

    # 5. Visualization
    total_score = 0
    print(f"Detected {len(dice_results)} valid dice.")

    for die in dice_results:
        score = die["score"]
        total_score += score
        x, y, w, h = die["box"]

        # Green Box
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Red Score
        label_size, _ = cv2.getTextSize(str(score), cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
        text_x = x + w//2 - label_size[0]//2
        text_y = y - 10
        cv2.putText(output_img, str(score), (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)

        # Blue Dots on Pips
        for px, py in die["pips"]:
            cv2.circle(output_img, (px, py), 3, (255, 0, 0), -1)

    print(f"Total Score: {total_score}")
    
    # Preparing a summary image
    def prepare_for_stack(image, title):
        if len(image.shape) == 2:
            display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display = image.copy()
        cv2.putText(display, title, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 3)
        return display

    s1 = prepare_for_stack(img, "1. Original")
    s2 = prepare_for_stack(gray, "2. Grayscale")
    s3 = prepare_for_stack(gray_eq, "3. CLAHE")
    s4 = prepare_for_stack(blurred, "4. Gaussian Blur")
    s5 = prepare_for_stack(thresh, "5. Threshold/Morph")
    s6 = prepare_for_stack(output_img, "6. Final Result")

    # Stacking all 6 stages images
    combined_debug = np.hstack((s1, s2, s3, s4, s5, s6))
    
    cv2.imwrite("processing_steps_all.jpg", combined_debug)
    cv2.imwrite(output_path, output_img)
    print(f"Saved debug collage to processing_steps_all.jpg")
    print(f"Saved final result to {output_path}")

if __name__ == "__main__":
    detect_dice_scores("data/img_1(dices_on_grey_background_with_shadows).jpg")