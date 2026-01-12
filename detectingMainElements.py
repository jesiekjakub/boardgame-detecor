import cv2
import numpy as np

# --- Helper Functions (Unchanged) ---

def stack_images(scale, img_list, cols):
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
    labeled_img = img.copy()
    if labeled_img.ndim == 2:
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(labeled_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (50, 255, 50), 2, cv2.LINE_AA)
    return labeled_img

def get_warped_rect(image, contour, padding=0):
    rect = cv2.minAreaRect(contour)
    center, (w, h), angle = rect
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    pts = box.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    src = np.array([tl, tr, br, bl], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Ensure portrait orientation for consistency
    if warped.shape[1] * 0.9  > warped.shape[0]: 
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped, box

def warp_from_points(image, pts):
    """
    Warps the image using exactly 4 points (pts). 
    Unlike minAreaRect, this handles perspective distortion (trapezoids).
    """
    # Reshape to (4, 2)
    pts = pts.reshape(4, 2).astype("float32")
    
    # Order points: tl, tr, br, bl
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Top-Left
    rect[2] = pts[np.argmax(s)]      # Bottom-Right
    rect[1] = pts[np.argmin(diff)]   # Top-Right
    rect[3] = pts[np.argmax(diff)]   # Bottom-Left
    
    (tl, tr, br, bl) = rect
    
    # Compute width (max of top and bottom widths)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height (max of left and right heights)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points (Square/Rectangle)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the Perspective Transform Matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped, rect.astype(int)

# --- Main Detection Function ---

def detect_game_elements_visualized(image):
    visuals = [] 
    results = {"papers": [], "wheel": None, "hollow_square": None, "circle": None}
    
    output_img = image.copy()
    visuals.append(add_label(image, "1. Original"))

    # === Preprocessing ===
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # --- EDGE DETECTION (Moved up for Paper Detection) ---
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    v = np.median(blur)
    lower_canny = int(max(0, (1.0 - 0.33) * v))
    upper_canny = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower_canny, upper_canny)
    
    kernel_small = np.ones((3, 3), np.uint8)
    # Dilate edges to close small gaps in the paper border
    edges_dilated = cv2.dilate(edges, kernel_small, iterations=2)
    visuals.append(add_label(edges_dilated, "2. Edges (Dilated)"))

    # ============================================
    # STAGE 1: DETECT PAPERS (Canny + Hull Strategy)
    # ============================================
    paper_contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    paper_occupancy_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for cnt in paper_contours:
        area = cv2.contourArea(cnt)
        
        # 1. Filter by Area (Must be large enough to be a paper)
        if area > 5000: 
            # 2. Convex Hull (The key to your idea: simplifies the boundary)
            hull = cv2.convexHull(cnt)
            
            # 3. Approximate Polygon (Find the corners of the Hull)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
            
            # 4. Check for 4 Corners
            if len(approx) == 4:
                # 5. Aspect Ratio Check (1.3 to 1.5)
                # Get lengths of adjacent sides to calculate ratio
                pts = approx.reshape(4, 2)
                
                # Calculate distances: (pt0-pt1), (pt1-pt2)
                d1 = np.linalg.norm(pts[0] - pts[1])
                d2 = np.linalg.norm(pts[1] - pts[2])
                
                # Aspect ratio is always > 1 (Long / Short)
                aspect_ratio = max(d1, d2) / min(d1, d2) if min(d1, d2) > 0 else 0
                
                if 1.3 < aspect_ratio < 1.6: # Expanded slightly to 1.6 to be safe
                    # Correct! This is a paper
                    warped_paper, box_points = get_warped_rect(image, hull)
                    results["papers"].append(warped_paper)
                    
                    # Draw Visualization
                    cv2.drawContours(output_img, [hull], -1, (0, 255, 0), 2) # Green hull
                    cv2.drawContours(output_img, [np.intp(box_points)], 0, (0, 0, 255), 2) # Red rect
                    
                    # Mark this area as occupied so Wheel/Other stages ignore it
                    cv2.drawContours(paper_occupancy_mask, [hull], -1, 255, -1)

    # ============================================
    # STAGE 2: WHEEL OF FORTUNE (Rotation Fixed)
    # ============================================
    # 1. Color Masking
    lower_yellow = np.array([20, 70, 80])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_blue = np.array([95, 20, 80])
    upper_blue = np.array([125, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    mask_wheel_raw = cv2.bitwise_or(mask_yellow, mask_blue)
    
    # 2. Subtract Papers
    mask_wheel_no_papers = cv2.bitwise_and(mask_wheel_raw, mask_wheel_raw, mask=cv2.bitwise_not(paper_occupancy_mask))
    mask_wheel_cleaned = cv2.morphologyEx(mask_wheel_no_papers, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 3. Smart Point Analysis
    wheel_pixels = cv2.findNonZero(mask_wheel_cleaned)
    vis_wheel = np.zeros_like(mask_wheel_cleaned)

    if wheel_pixels is not None:
        wheel_pixels = wheel_pixels.squeeze() 
        
        median_x = np.median(wheel_pixels[:, 0])
        median_y = np.median(wheel_pixels[:, 1])
        center = np.array([median_x, median_y])
        
        distances = np.linalg.norm(wheel_pixels - center, axis=1)
        sorted_indices = np.argsort(distances)
        sorted_dists = distances[sorted_indices]
        
        # Robust IQR Filtering
        q1 = np.percentile(sorted_dists, 25)
        q3 = np.percentile(sorted_dists, 75)
        iqr = q3 - q1
        limit = q3 + 1.5 * iqr
        
        valid_indices = sorted_indices[sorted_dists < limit]
        final_points = wheel_pixels[valid_indices]

        # Visualization
        for p in final_points:
             cv2.circle(vis_wheel, (int(p[0]), int(p[1])), 1, 255, -1)
        visuals.append(add_label(vis_wheel, "3. Wheel (Smart Filter)"))

        if len(final_points) >= 4:
            # A. Get Convex Hull of the filtered cluster
            hull = cv2.convexHull(final_points.astype(np.int32))
            
            # B. Approximate Polygon to find corners
            peri = cv2.arcLength(hull, True)
            # 0.04 is a good standard, but for "round" corners sometimes 0.05 is safer
            approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
            
            # C. Check if we found 4 corners
            if len(approx) == 4:
                # Perfect! We have 4 corners. Use them directly.
                warped_wheel, box_points = warp_from_points(image, approx)
                results["wheel"] = warped_wheel
                
                # Draw the detected polygon (Yellow)
                cv2.drawContours(output_img, [box_points], 0, (0, 255, 255), 3)
            
            else:
                # Fallback: If approxPolyDP fails (e.g. returns 5 points due to noise),
                # we fall back to minAreaRect on the Hull.
                warped_wheel, box_points = get_warped_rect(image, hull)
                results["wheel"] = warped_wheel
                cv2.drawContours(output_img, [np.intp(box_points)], 0, (0, 255, 255), 3)

    # ============================================
    # STAGE 3: ENCLOSURES (Reuse Edges)
    # ============================================
    # We reuse the dilated edges from the start
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    potential_enclosures = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Avoid re-detecting the papers we already found
        # (A simple check: is the center of this contour inside our paper mask?)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if paper_occupancy_mask[cY, cX] > 0:
                continue

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
    image_path = '/home/jakub/Artificial Intelligence/Studies/Term 5/[CV] Computer Vision/boardgame-detecor/data/img_3(main_elements).png'
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