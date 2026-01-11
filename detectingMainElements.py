import cv2
import numpy as np

# --- Helper Functions ---

def stack_images(scale, img_list, cols):
    """
    Stacks multiple images into a grid layout while maintaining their 
    original aspect ratios using padding instead of stretching.
    """
    unit_h, unit_w = 400, 400 
    rows = (len(img_list) + cols - 1) // cols
    processed_imgs = []
    
    for img in img_list:
        if img is None or img.size == 0:
            continue
            
        if img.ndim == 2: 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 1: # Landscape
            new_w = unit_w
            new_h = int(unit_w / aspect_ratio)
        else: # Portrait
            new_h = unit_h
            new_w = int(unit_h * aspect_ratio)
            
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        top = (unit_h - new_h) // 2
        bottom = unit_h - new_h - top
        left = (unit_w - new_w) // 2
        right = unit_w - new_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        final_img = cv2.resize(padded, (0, 0), None, fx=scale, fy=scale)
        processed_imgs.append(final_img)
        
    while len(processed_imgs) < rows * cols:
        processed_imgs.append(np.zeros_like(processed_imgs[0]))
        
    horiz_rows = []
    for i in range(rows):
        horiz_rows.append(np.hstack(processed_imgs[i*cols : (i+1)*cols]))
        
    return np.vstack(horiz_rows)

def add_label(img, label):
    """Helper to add a text label to an image."""
    labeled_img = img.copy()
    if labeled_img.ndim == 2:
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(labeled_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (50, 255, 50), 2, cv2.LINE_AA)
    return labeled_img

def get_warped_rect(image, contour, padding=0):
    """
    Takes a contour (or set of points), finds the minimum area rotated rectangle,
    and warps the image to extract that rectangle axis-aligned.
    """
    # 1. Get rotated rectangle
    rect = cv2.minAreaRect(contour)
    center, (w, h), angle = rect

    # 2. Get the 4 corners of the rotated rect
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # 3. Reorder points to: top-left, top-right, bottom-right, bottom-left
    pts = box.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    # 4. Compute width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 5. Perspective transform
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    src = np.array([tl, tr, br, bl], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    if warped.shape[1]*0.9 > warped.shape[0]: 
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped, box

# --- Main Detection Function ---

def detect_game_elements_visualized(image):
    visuals = [] 
    results = {"papers": [], "wheel": None, "hollow_square": None, "circle": None}
    
    output_img = image.copy()
    visuals.append(add_label(image, "1. Original"))

    # === Preprocessing ===
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)

    # ============================================
    # STAGE 1: DETECT PAPERS
    # ============================================
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 15, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    mask_white_closed = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    visuals.append(add_label(mask_white_closed, "2. Paper Mask"))
    
    paper_occupancy_mask = np.zeros_like(mask_white)

    contours, _ = cv2.findContours(mask_white_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000: 
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            hull = cv2.convexHull(cnt)
            solidity = float(area) / cv2.contourArea(hull)

            if len(approx) == 4 and solidity > 0.90:
                warped_paper, box_points = get_warped_rect(image, cnt)
                results["papers"].append(warped_paper)
                cv2.drawContours(output_img, [box_points], 0, (0, 255, 0), 2)
                cv2.drawContours(paper_occupancy_mask, [cnt], -1, 255, -1)

    # ============================================
    # STAGE 2: WHEEL OF FORTUNE (Smart Distance Filtering)
    # ============================================
    # 1. Color Masking (Strict)
    lower_yellow = np.array([20, 70, 80])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_blue = np.array([95, 20, 80])
    upper_blue = np.array([125, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    mask_wheel_raw = cv2.bitwise_or(mask_yellow, mask_blue)
    
    # 2. Subtract Papers
    mask_wheel_no_papers = cv2.bitwise_and(mask_wheel_raw, mask_wheel_raw, mask=cv2.bitwise_not(paper_occupancy_mask))
    
    # 3. Minimal Cleanup (Only Open to remove sparkles, NO Dilation)
    mask_wheel_cleaned = cv2.morphologyEx(mask_wheel_no_papers, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 4. Smart Point Analysis
    wheel_pixels = cv2.findNonZero(mask_wheel_cleaned)
    vis_wheel = np.zeros_like(mask_wheel_cleaned)

    if wheel_pixels is not None:
        wheel_pixels = wheel_pixels.squeeze() # (N, 2)
        
        # A. Calculate Median Center (More robust than Mean)
        # We use median to avoid outliers pulling the center point away
        median_x = np.median(wheel_pixels[:, 0])
        median_y = np.median(wheel_pixels[:, 1])
        center = np.array([median_x, median_y])
        
        # B. Calculate Euclidean Distances
        # shape (N,)
        distances = np.linalg.norm(wheel_pixels - center, axis=1)
        
        # C. Sort points by distance
        # We want to keep the "core" points that rise steadily in distance
        sorted_indices = np.argsort(distances)
        sorted_dists = distances[sorted_indices]
        
        # D. Detect "Knee" or "Gap" in the distance curve
        # If we encounter a large jump in distance, everything after is likely noise
        # Calculate the difference between consecutive points' distances
        diffs = np.diff(sorted_dists)
        
        # Threshold: We look for a gap larger than X pixels. 
        # Since pixels in a filled shape are contiguous, gaps > 5-10px usually mean a separate object.
        gap_threshold = 10.0 
        
        # Find the first index where the gap is too large
        cutoff_index = len(sorted_dists) # Default: keep all
        
        # We search from the 'middle' outwards to avoid being triggered by small gaps in the core
        start_search = int(len(diffs) * 0.5) 
        for i in range(start_search, len(diffs)):
            if diffs[i] > gap_threshold:
                cutoff_index = i + 1 # +1 because diff array is 1 shorter
                break
        
        # Alternative robustness: IQR Clip
        # If the gap logic fails (e.g. noise is uniform), use statistics
        q1 = np.percentile(sorted_dists, 25)
        q3 = np.percentile(sorted_dists, 75)
        iqr = q3 - q1
        iqr_limit = q3 + 1.5 * iqr
        
        # Combine strategies: take the stricter of the two limits
        valid_indices = sorted_indices[:cutoff_index]
        
        # Re-filter based on IQR just in case the gap wasn't found but points are huge
        final_points = []
        for idx in valid_indices:
            if distances[idx] < iqr_limit:
                final_points.append(wheel_pixels[idx])
        
        final_points = np.array(final_points)

        # Visualization of filtered points
        for p in final_points:
             cv2.circle(vis_wheel, (int(p[0]), int(p[1])), 1, 255, -1)
        visuals.append(add_label(vis_wheel, "3. Wheel (Smart Filter)"))

        # 5. Extract the "Outermost 4 Points" (Board Corners)
        if len(final_points) >= 4:
            # We calculate the MinAreaRect of the CLEANED cluster.
            # This implicitly finds the 4 best corners that enclose the board.
            hull = cv2.convexHull(final_points.astype(np.int32))
            
            # Check aspect ratio to ensure it's somewhat square-ish (board)
            rect = cv2.minAreaRect(hull)
            (w, h) = rect[1]
            if w > 0 and h > 0:
                aspect = min(w,h) / max(w,h)
                if aspect > 0.8: # Loose check, boards are usually squares
                    warped_wheel, box_points = get_warped_rect(image, hull)
                    results["wheel"] = warped_wheel
                    cv2.drawContours(output_img, [np.intp(box_points)], 0, (0, 255, 255), 2)

    # ============================================
    # STAGE 3: ENCLOSURES (Hollow Objects)
    # ============================================
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    v = np.median(blur)
    lower_canny = int(max(0, (1.0 - 0.33) * v))
    upper_canny = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower_canny, upper_canny)
    edges_dilated = cv2.dilate(edges, kernel_small, iterations=1)
    visuals.append(add_label(edges_dilated, "4. Edges"))

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    potential_enclosures = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 15000: 
            peri = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (peri ** 2) if peri > 0 else 0
            
            hull = cv2.convexHull(cnt)
            solidity = float(area)/cv2.contourArea(hull)
            
            shape_type = "unknown"
            if circularity > 0.80: shape_type = "circle"
            elif 0.6 < circularity < 0.85 and solidity < 0.5: shape_type = "square"
            
            if shape_type != "unknown":
                potential_enclosures.append({'cnt': cnt, 'type': shape_type, 'area': area})

    potential_enclosures.sort(key=lambda x: x['area'], reverse=True)

    for cand in potential_enclosures:
        x, y, w, h = cv2.boundingRect(cand['cnt'])
        
        if cand['type'] == "square" and results["hollow_square"] is None:
            results["hollow_square"] = image[y:y+h, x:x+w]
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
        elif cand['type'] == "circle" and results["circle"] is None:
            results["circle"] = image[y:y+h, x:x+w]
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 100, 0), 3)

    visuals.append(add_label(output_img, "5. Final Detection"))
    
    # === Add Extracted Crops ===
    crops_visuals = []
    if results["wheel"] is not None:
        crops_visuals.append(add_label(results["wheel"], "Extracted Wheel"))
    
    for i, paper in enumerate(results["papers"]):
        crops_visuals.append(add_label(paper, f"Paper {i+1}"))
        
    main_grid = stack_images(0.5, visuals, cols=3)
    return main_grid, crops_visuals

def main():
    # Update this path to your local image
    image_path = '/home/jakub/Artificial Intelligence/Studies/Term 5/[CV] Computer Vision/boardgame-detecor/data/img_2(main_elements).png'
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image from {image_path}")
        return

    print("Processing...")
    combined_img, crops = detect_game_elements_visualized(img)

    cv2.imshow("Detection Pipeline", combined_img)
    
    if crops:
        crops_img = stack_images(0.8, crops, cols=3)
        cv2.imshow("Extracted Elements (Warped)", crops_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()