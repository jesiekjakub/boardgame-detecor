import cv2
import numpy as np

def stack_images(scale, img_list, cols):
    """
    Helper function to stack images into a grid for visualization.
    """
    rows = (len(img_list) + cols - 1) // cols
    
    # Resize images
    resized_imgs = []
    for img in img_list:
        if img.ndim == 2: # Convert grayscale to BGR for stacking
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized_imgs.append(cv2.resize(img, (0, 0), None, scale, scale))
    
    # Add blank images to fill the grid
    while len(resized_imgs) < rows * cols:
        resized_imgs.append(np.zeros_like(resized_imgs[0]))

    # Create rows
    horiz_rows = []
    for i in range(rows):
        horiz_rows.append(np.hstack(resized_imgs[i*cols : (i+1)*cols]))
        
    # Combine rows vertically
    veritcal_stack = np.vstack(horiz_rows)
    return veritcal_stack

def add_label(img, label):
    """Helper to add a text label to an image."""
    labeled_img = img.copy()
    cv2.putText(labeled_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled_img

def detect_game_elements_visualized(image):
    """
    Detects game elements and returns a combined visualization of the process.
    """
    visuals = [] # List to hold images for the final dashboard
    results = {"papers": [], "wheel": None, "hollow_square": None, "circle": None}
    
    output_img = image.copy()
    visuals.append(add_label(image, "Original"))

    # === 1. Preprocessing ===
    # GaussianBlur chosen to smooth out background texture grain
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    visuals.append(add_label(blur, "1. Gaussian Blur"))

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Kernel for morphological operations
    kernel_clean = np.ones((5, 5), np.uint8)
    
    # === 2. Detect White Papers ===
    # Improved for shadows: Lowered Value threshold, rely on low Saturation.
    # H: Any, S: 0-60 (low color), V: 80-255 (allows for shaded white)
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 15, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Clean noise
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_clean)
    visuals.append(add_label(mask_white, "2. White Mask (Papers)"))

    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Lowered area threshold to catch papers positioned further away or partially visible
        if area > 3000: 
            # Check for rectangular shape
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # Calculate solidity to reject non-compact shapes
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area if hull_area > 0 else 0

            # If it has 4 corners and is solid, it's likely a paper
            if len(approx) == 4 and solidity > 0.9:
                x, y, w, h = cv2.boundingRect(cnt)
                results["papers"].append((x, y, w, h))
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(output_img, "Paper", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # === 3. Detect Wheel of Fortune ===
    # Strategy: Detect Yellow & Blue anchors, combine, and dilate heavily.
    lower_yellow = np.array([20, 70, 80])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_blue = np.array([95, 20, 80])
    upper_blue = np.array([125, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    visuals.append(add_label(mask_yellow, "3a. Yellow Mask"))
    visuals.append(add_label(mask_blue, "3b. Blue Mask"))

    # Combine and Dilate
    mask_wheel_combined = cv2.bitwise_or(mask_yellow, mask_blue)
    kernel_dilate = np.ones((3, 3), np.uint8)
    # 3 iterations of dilation to merge the separate triangles into one big blob
    mask_wheel_dilated = cv2.dilate(mask_wheel_combined, kernel_dilate, iterations=3)
    visuals.append(add_label(mask_wheel_dilated, "3c. Wheel Mask (Dilated)"))

    contours, _ = cv2.findContours(mask_wheel_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # The largest blob is the wheel
        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) > 5000:
            x, y, w, h = cv2.boundingRect(largest_cnt)
            results["wheel"] = (x, y, w, h)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 255), 3)
            cv2.putText(output_img, "Wheel", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # === 4. Detect Enclosures (Hollow Square & Circle) ===
    # Use Adaptive Threshold to find edges, robust to lighting across the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 29, 3)
    # Clean up the thresholded image
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    visuals.append(add_label(thresh, "4a. Adaptive Thresh (Edges)"))

    # Use RETR_TREE to find nested contours (containers)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    potential_enclosures = []
    if hierarchy is not None:
        hierarchy = hierarchy[0] # get the first level
        for i, cnt in enumerate(contours):
            # Look for contours that have a parent (are inside something)
            # The parent is the outer rim, the current contour is the inner floor.
            parent_idx = hierarchy[i][3]
            
            if parent_idx != -1: # This contour is inside another
                area = cv2.contourArea(cnt)
                if area > 15000: # Filter out small things like dice
                    peri = cv2.arcLength(cnt, True)
                    # Circularity: 1.0 is a perfect circle, a square is ~0.785
                    circularity = (4 * np.pi * area) / (peri ** 2)
                    
                    shape_type = "unknown"
                    if circularity > 0.85:
                        shape_type = "circle"
                    elif 0.65 < circularity < 0.85:
                        shape_type = "square"
                    
                    if shape_type != "unknown":
                        # We found an inner floor. Let's get its parent (the outer rim).
                        outer_cnt = contours[parent_idx]
                        potential_enclosures.append({
                            'cnt': outer_cnt, 
                            'type': shape_type, 
                            'area': cv2.contourArea(outer_cnt)
                        })

    # Sort by area to process largest first
    potential_enclosures.sort(key=lambda x: x['area'], reverse=True)

    # Assign the best candidates
    for cand in potential_enclosures:
        bbox = cv2.boundingRect(cand['cnt'])
        x, y, w, h = bbox
        
        if cand['type'] == "square" and results["hollow_square"] is None:
            # Ensure it doesn't overlap with the wheel
            if results["wheel"]:
                wx, wy, ww, wh = results["wheel"]
                # Simple center-point check to avoid major overlap
                cx, cy = x + w//2, y + h//2
                if wx < cx < wx + ww and wy < cy < wy + wh:
                    continue # It's likely the wheel's bounding box
            
            results["hollow_square"] = bbox
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(output_img, "Hollow Square", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        elif cand['type'] == "circle" and results["circle"] is None:
            results["circle"] = bbox
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 100, 0), 3)
            cv2.putText(output_img, "Circle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    visuals.append(add_label(output_img, "Final Result"))

    # Combine all visualization images into one grid
    # Scale 0.5 to fit on screen, 4 columns
    combined_image = stack_images(0.5, visuals, cols=4) 
    
    return combined_image

def main():
    image_path = '/home/jakub/Artificial Intelligence/Studies/Term 5/[CV] Computer Vision/boardgame-detecor/data/img_2(main_elements).png' 
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print("Processing image...")
    combined_result_img = detect_game_elements_visualized(img)

    # Show the combined visualization
    cv2.imshow("Game Element Detection Pipeline", combined_result_img)
    
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()