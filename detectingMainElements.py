import cv2
import numpy as np

# --- Helper Functions (Unchanged) ---
def stack_images(scale, img_list, cols):
    unit_h, unit_w = 400, 400 
    rows = (len(img_list) + cols - 1) // cols
    processed_imgs = []
    for img in img_list:
        if img is None or img.size == 0: continue
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = unit_w; new_h = int(unit_w / aspect_ratio)
        else:
            new_h = unit_h; new_w = int(unit_h * aspect_ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top = (unit_h - new_h) // 2; bottom = unit_h - new_h - top
        left = (unit_w - new_w) // 2; right = unit_w - new_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        processed_imgs.append(cv2.resize(padded, (0, 0), None, fx=scale, fy=scale))
    while len(processed_imgs) < rows * cols: processed_imgs.append(np.zeros_like(processed_imgs[0]))
    horiz_rows = []
    for i in range(rows): horiz_rows.append(np.hstack(processed_imgs[i*cols : (i+1)*cols]))
    return np.vstack(horiz_rows)

def add_label(img, label):
    labeled_img = img.copy()
    if labeled_img.ndim == 2: labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(labeled_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
    return labeled_img

def get_warped_rect(image, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect); box = np.intp(box)
    pts = box.astype("float32")
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    src = np.array([tl, tr, br, bl], dtype="float32")
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (width, height))
    if warped.shape[1] * 0.9 > warped.shape[0]: warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped, box

def warp_from_points(image, pts):
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (width, height))
    return warped, rect.astype(int)

# --- Main Detection Function ---

def detect_game_elements_visualized(image):
    visuals = [] 
    results = {"papers": [], "wheel": None, "hollow_square": None, "circle": None}
    output_img = image.copy()
    visuals.append(add_label(image, "1. Original"))
    
    occupied_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # === Preprocessing ===
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    # Edge Detection
    v = np.median(blur)
    lower_canny = int(max(0, (1.0 - 0.33) * v))
    upper_canny = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower_canny, upper_canny)
    kernel_small = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel_small, iterations=2)
    visuals.append(add_label(edges_dilated, "2. Edges (Papers)"))

    # ============================================
    # STAGE 1: PAPERS (Unchanged)
    # ============================================
    paper_contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in paper_contours:
        area = cv2.contourArea(cnt)
        if area > 5000: 
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                d1 = np.linalg.norm(pts[0] - pts[1])
                d2 = np.linalg.norm(pts[1] - pts[2])
                aspect_ratio = max(d1, d2) / min(d1, d2) if min(d1, d2) > 0 else 0
                if 1.3 < aspect_ratio < 1.6:
                    warped_paper, box_points = get_warped_rect(image, hull)
                    results["papers"].append(warped_paper)
                    cv2.drawContours(output_img, [np.intp(box_points)], 0, (0, 0, 255), 2)
                    cv2.drawContours(occupied_mask, [hull], -1, 255, -1)

    # ============================================
    # STAGE 2: WHEEL OF FORTUNE (Unchanged)
    # ============================================
    lower_yellow = np.array([20, 70, 80]); upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_blue = np.array([95, 20, 80]); upper_blue = np.array([125, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_wheel_raw = cv2.bitwise_or(mask_yellow, mask_blue)
    
    mask_wheel_no_papers = cv2.bitwise_and(mask_wheel_raw, mask_wheel_raw, mask=cv2.bitwise_not(occupied_mask))
    mask_wheel_cleaned = cv2.morphologyEx(mask_wheel_no_papers, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    wheel_pixels = cv2.findNonZero(mask_wheel_cleaned)
    if wheel_pixels is not None:
        wheel_pixels = wheel_pixels.squeeze()
        center = np.median(wheel_pixels, axis=0)
        distances = np.linalg.norm(wheel_pixels - center, axis=1)
        q1, q3 = np.percentile(distances, [25, 75])
        limit = q3 + 1.5 * (q3 - q1)
        final_points = wheel_pixels[distances < limit]
        
        if len(final_points) >= 4:
            hull = cv2.convexHull(final_points.astype(np.int32))
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
            if len(approx) == 4:
                warped_wheel, box_points = warp_from_points(image, approx)
                results["wheel"] = warped_wheel
                cv2.drawContours(output_img, [box_points], 0, (0, 255, 255), 3)
            else:
                warped_wheel, box_points = get_warped_rect(image, hull)
                results["wheel"] = warped_wheel
                cv2.drawContours(output_img, [np.intp(box_points)], 0, (0, 255, 255), 3)
            cv2.drawContours(occupied_mask, [hull], -1, 255, -1)

    # Expand Occupied Mask slightly
    kernel_expansion = np.ones((15, 15), np.uint8)
    occupied_mask = cv2.dilate(occupied_mask, kernel_expansion, iterations=2)
    visuals.append(add_label(occupied_mask, "3. Occupied Mask (Pre-Enclosure)"))

    # ============================================
    # STAGE 3: ENCLOSURES (Circle First, then Square)
    # ============================================
    
    # --- PART A: HOUGH CIRCLE TRANSFORM ---
    # 1. Prepare Image: Median blur is crucial to reduce salt-and-pepper noise
    #    that confuses Hough accumulator.
    gray_blurred = cv2.medianBlur(gray, 7)
    
    # 2. Detect Circles
    #    dp=1.5: Lower resolution accumulator (faster, robust)
    #    minDist=100: Avoid multiple circles on the same object
    #    param1=50: Canny high threshold (passed to internal Canny)
    #    param2=30: Accumulator threshold (Lower = more circles, Higher = fewer/perfect circles)
    #    minRadius/maxRadius: Tuned to expected size of the container
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=100,
                               param1=50, param2=30, minRadius=40, maxRadius=150)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # Check if this circle is already in an occupied area (e.g. inside the wheel/paper?)
            if occupied_mask[center[1], center[0]] == 255:
                continue

            # We found the Hollow Circle!
            x, y = int(center[0] - radius), int(center[1] - radius)
            d = int(2 * radius)
            
            # Extract Crop (ensure bounds)
            y1, y2 = max(0, y), min(image.shape[0], y+d)
            x1, x2 = max(0, x), min(image.shape[1], x+d)
            
            if results["circle"] is None:
                results["circle"] = image[y1:y2, x1:x2]
                
                # Draw Visualization
                cv2.circle(output_img, center, radius, (0, 165, 255), 3) # Outline
                cv2.circle(output_img, center, 2, (0, 0, 255), 3)      # Center
                
                # IMPORTANT: Mask this area out so the Square search doesn't find it
                # We draw a filled circle on the occupied mask
                cv2.circle(occupied_mask, center, int(radius + 10), 255, -1)

    visuals.append(add_label(occupied_mask, "4. Mask after Circle"))

    # --- PART B: HOLLOW SQUARE (The Leftover Object) ---
    # Now that Papers, Wheel, and Circle are masked, the Square is the only thing left.
    
    # 1. Adaptive Threshold (Good for walls)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 19, 3)
    
    # 2. Masking
    thresh_masked = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(occupied_mask))
    
    # 3. Clean & Cluster
    kernel_clean = np.ones((3, 3), np.uint8)
    kernel_cluster = np.ones((9, 9), np.uint8) # Dilation to merge square walls
    
    opened = cv2.morphologyEx(thresh_masked, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    clusters = cv2.dilate(opened, kernel_cluster, iterations=3)
    
    visuals.append(add_label(thresh_masked, "5. Thresh (Square Search)"))
    visuals.append(add_label(clusters, "6. Clusters"))
    
    contours, _ = cv2.findContours(clusters, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest remaining contour (that isn't noise)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000: break # Too small, stop checking
        
        hull = cv2.convexHull(cnt)
        
        # We don't need complex shape checks anymore because the circle is gone.
        # Just check if it's roughly square-ish to be safe.
        x, y, w, h = cv2.boundingRect(hull)
        aspect_ratio = float(w) / h
        
        if 0.7 < aspect_ratio < 1.3: # Loose check for the square container
            if results["hollow_square"] is None:
                results["hollow_square"] = image[y:y+h, x:x+w]
                
                # Visualization
                cv2.drawContours(output_img, [hull], -1, (255, 0, 0), 2)
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
                break # We found the square, stop.

    visuals.append(add_label(output_img, "7. Final Detection"))
    
    crops_visuals = []
    if results["wheel"] is not None:
        crops_visuals.append(add_label(results["wheel"], "Extracted Wheel"))
    if results["hollow_square"] is not None:
        crops_visuals.append(add_label(results["hollow_square"], "Hollow Square"))
    if results["circle"] is not None:
        crops_visuals.append(add_label(results["circle"], "Hollow Circle"))
    for i, paper in enumerate(results["papers"]):
        crops_visuals.append(add_label(paper, f"Paper {i+1}"))
        
    main_grid = stack_images(0.5, visuals, cols=4) # Increased cols
    return main_grid, crops_visuals

def main():
    image_path = '/home/jakub/Artificial Intelligence/Studies/Term 5/[CV] Computer Vision/boardgame-detecor/data/img_3(main_elements).png'
    img = cv2.imread(image_path)
    if img is None: print("Error loading image"); return
    combined_img, crops = detect_game_elements_visualized(img)
    cv2.imshow("Detection Pipeline", combined_img)
    if crops: cv2.imshow("Extracted Elements", stack_images(0.8, crops, cols=4))
    cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()