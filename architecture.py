import cv2
import numpy as np
from collections import deque
import os

# ==========================================
# 1. CONFIGURATION (Hyperparameters)
# ==========================================
class GameConfig:
    """
    Central configuration for all game detection parameters.
    Adjust these values based on your lighting and camera setup.
    """
    # --- Color Calibration (HSV: H=0-179, S=0-255, V=0-255) ---
    COLOR_RANGES = {
        "Red_1":   ((0, 20, 70), (10, 255, 255)),     # Lower Red
        "Red_2":   ((170, 20, 70), (180, 255, 255)),  # Upper Red (wraps around)
        "Yellow":  ((20, 20, 70), (35, 255, 255)),    # Yellow
        "Purple":  ((120, 10, 50), (145, 255, 255)),  # Purple (Replaces Green)
        "Blue":    ((85, 20, 70), (115, 255, 255)),   # Blue
    }

    # --- Stabilizer Settings ---
    STABILIZER_HISTORY_LEN = 60       # Frames to keep in buffer for smoothing
    STABILIZER_THRESHOLD = 4.0        # Max pixel deviation allowed to "lock" the view

    # --- Wheel Analysis Settings ---
    WHEEL_HISTORY_LEN = 15            # Frames to track for state logic (Spinning vs Stopped)
    NEEDLE_VALUE_THRESH = 110         # Max Value (brightness) to consider "Black"
    ROI_RADIUS = 20                   # Radius of the voting circle around the needle tip
    MIN_VOTES = 5                     # Min valid pixels required to declare a color found

    # --- General Detection ---
    MIN_CONTOUR_AREA = 5000           # Min area for Papers/Containers
    MIN_NEEDLE_AREA = 50              # Min area for the Needle

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def order_points(pts):
    """Orders coordinates: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def warp_from_points(image, pts):
    """Warps image based on 4 points."""
    sorted_pts = order_points(pts)
    (tl, tr, br, bl) = sorted_pts
    width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(sorted_pts, dst), (width, height))
    return warped, sorted_pts.astype(int)

def get_warped_rect(image, contour):
    """Fallback warp using minAreaRect if 4 corners aren't found cleanly."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect); box = np.intp(box)
    pts = order_points(box)
    (tl, tr, br, bl) = pts
    width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(pts, dst), (width, height))
    if warped.shape[1] * 0.9 > warped.shape[0]: warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped, box

# ==========================================
# 3. STABILIZER
# ==========================================

class CornerStabilizer:
    def __init__(self):
        self.history = deque(maxlen=GameConfig.STABILIZER_HISTORY_LEN)
        self.locked_coords = None
        self.threshold = GameConfig.STABILIZER_THRESHOLD
        self.history_len = GameConfig.STABILIZER_HISTORY_LEN

    def update(self, new_coords):
        sorted_coords = order_points(new_coords)
        
        # Instant fill on first detection to avoid lag
        if not self.history:
            for _ in range(self.history_len): self.history.append(sorted_coords)
        else:
            self.history.append(sorted_coords)
        
        # Calculate stability (max deviation from mean)
        history_arr = np.array(self.history)
        mean_coords = np.mean(history_arr, axis=0)
        max_deviation = np.max(np.linalg.norm(history_arr - mean_coords, axis=2))
        
        # If stable, update the lock. If unstable, keep the old lock.
        if max_deviation < self.threshold:
            self.locked_coords = mean_coords.astype(np.float32)
            
        return self.locked_coords if self.locked_coords is not None else sorted_coords

# ==========================================
# 4. WHEEL ANALYZER
# ==========================================

class WheelAnalyzer:
    def __init__(self):
        self.history_len = GameConfig.WHEEL_HISTORY_LEN
        self.color_history = deque(maxlen=self.history_len)
        self.current_state = "IDLE" 

    def find_outermost_tip(self, hsv_crop):
        h, w = hsv_crop.shape[:2]
        center = np.array([w // 2, h // 2])
        
        # 1. Mask Black (Needle) using Config Threshold
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, GameConfig.NEEDLE_VALUE_THRESH]) 
        mask = cv2.inRange(hsv_crop, lower_black, upper_black)
        
        # Use Close to preserve thin tips
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None, None
        
        # 2. Filter contours: Must be somewhat central and large enough
        valid_cnts = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < GameConfig.MIN_NEEDLE_AREA: continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
                # Must be within 30% of image center
                if np.linalg.norm(np.array([cX, cY]) - center) < w * 0.3:
                    valid_cnts.append(cnt)
        
        needle_cnt = max(valid_cnts, key=cv2.contourArea) if valid_cnts else max(cnts, key=cv2.contourArea)
        if cv2.contourArea(needle_cnt) < GameConfig.MIN_NEEDLE_AREA: return None, None

        # 3. Find point furthest from center
        points = needle_cnt.reshape(-1, 2)
        dists = np.linalg.norm(points - center, axis=1)
        outermost_tip = points[np.argmax(dists)]
        
        return outermost_tip, needle_cnt

    def get_color_by_voting(self, hsv_crop, tip, debug_img=None):
        h, w = hsv_crop.shape[:2]
        radius = GameConfig.ROI_RADIUS
        
        # ROI Mask
        mask_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_roi, tuple(tip), radius, 255, -1)
        
        # Exclusion Masks (Black Needle & Grey Background)
        mask_black = cv2.inRange(hsv_crop, np.array([0,0,0]), np.array([180, 255, GameConfig.NEEDLE_VALUE_THRESH]))
        mask_grey = cv2.inRange(hsv_crop, np.array([0,0,0]), np.array([180, 50, 255]))
        mask_invalid = cv2.bitwise_or(mask_black, mask_grey)
        
        # Final Valid Mask
        mask_valid = cv2.bitwise_and(mask_roi, cv2.bitwise_not(mask_invalid))
        
        # --- DEBUG VISUALIZATION ---
        if debug_img is not None:
            # Draw ROI and Valid Pixels
            cv2.circle(debug_img, tuple(tip), radius, (255, 255, 255), 1)
            
            # Compute stats of valid pixels
            valid_pixels = hsv_crop[mask_valid > 0]
            if valid_pixels.size > 0:
                mean_hsv = np.mean(valid_pixels, axis=0)
                stat_txt = f"H:{int(mean_hsv[0])} S:{int(mean_hsv[1])} V:{int(mean_hsv[2])}"
                # Background box for text
                cv2.rectangle(debug_img, (0, h-25), (160, h), (0,0,0), -1)
                cv2.putText(debug_img, stat_txt, (5, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # Voting
        votes = {k: 0 for k in GameConfig.COLOR_RANGES.keys() if k != "Red_2"}
        # Map "Red_2" votes to "Red_1" (simply "Red") logic
        
        for color_name, (lower, upper) in GameConfig.COLOR_RANGES.items():
            mask_color = cv2.inRange(hsv_crop, np.array(lower), np.array(upper))
            mask_final = cv2.bitwise_and(mask_valid, mask_color)
            count = cv2.countNonZero(mask_final)
            
            # Aggregate Red
            key = "Red_1" if "Red" in color_name else color_name
            if key in votes:
                votes[key] += count

        total_votes = sum(votes.values())
        if total_votes < GameConfig.MIN_VOTES: return "Unknown"
        
        # Normalize name (Red_1 -> Red)
        winner = max(votes, key=votes.get)
        return "Red" if "Red" in winner else winner

    def analyze(self, wheel_crop, debug_vis_img=None):
        if wheel_crop is None or wheel_crop.size == 0: return "No Wheel", "N/A"

        hsv = cv2.cvtColor(wheel_crop, cv2.COLOR_BGR2HSV)
        
        # 1. Find Tip
        tip, needle_cnt = self.find_outermost_tip(hsv)
        current_color = "Unknown"
        
        if tip is not None:
            if debug_vis_img is not None:
                cv2.drawContours(debug_vis_img, [needle_cnt], -1, (255, 0, 255), 2)
                cv2.circle(debug_vis_img, tuple(tip), 3, (0, 0, 255), -1)
            
            # 2. Vote for Color
            current_color = self.get_color_by_voting(hsv, tip, debug_vis_img)
            
            if debug_vis_img is not None:
                cv2.putText(debug_vis_img, f"Res: {current_color}", (5, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 3. Update State History
        self.color_history.append(current_color)

        if len(self.color_history) < self.history_len:
            return "Initializing...", "N/A"

        # Filter valid readings
        valid = [c for c in self.color_history if c not in ["Unknown"]]
        
        if len(valid) < 5: return "Searching...", "N/A"
            
        unique = set(valid)
        
        # Logic: Changing colors = Spinning. Constant color = Stopped.
        if len(unique) > 1:
            self.current_state = "SPINNING"
            return "SPINNING", "..."
        else:
            stable_col = valid[-1]
            if self.current_state == "SPINNING":
                self.current_state = "STOPPED"
                return "STOPPED", stable_col
            else:
                return "IDLE", stable_col

# ==========================================
# 5. GAME STATE MANAGER
# ==========================================

class GameStateManager:
    def __init__(self):
        self.scores = {"Paper_1": 0, "Paper_2": 0}
        self.wheel_analyzer = WheelAnalyzer()

    def handle_paper(self, crop, paper_index):
        # 0 -> Paper 1, 1 -> Paper 2
        return f"Paper {paper_index + 1}: 0"

    def handle_wheel(self, crop):
        debug_vis = crop.copy()
        state, color = self.wheel_analyzer.analyze(crop, debug_vis)
        
        # Show Debug Window
        vis_large = cv2.resize(debug_vis, (300, 300))
        cv2.imshow("Wheel Analysis Debug", vis_large)
        
        txt = f"Wheel: {state}"
        if state in ["STOPPED", "IDLE"]: txt += f" -> {color}"
        return txt

    def handle_container(self, crop, type_name):
        return f"{type_name}: 0"

# ==========================================
# 6. BOARD DETECTOR (2-Player Logic)
# ==========================================

class BoardDetector:
    def __init__(self):
        self.colors = {
            "paper": (0, 0, 255), "wheel": (0, 255, 255),
            "circle": (0, 165, 255), "square": (255, 0, 0)
        }
        self.wheel_stabilizer = CornerStabilizer()
        self.circle_stabilizer = CornerStabilizer()
        self.square_stabilizer = CornerStabilizer()
        
        # Fixed stabilizers for exactly 2 papers
        self.paper_stabilizers = {0: CornerStabilizer(), 1: CornerStabilizer()}

    def detect_elements(self, frame):
        output_img = frame.copy()
        # Initialize dictionary with fixed keys for 2 players
        crops = {"papers": [None, None], "wheel": None, "hollow_square": None, "circle": None}
        
        occupied_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        # --- STAGE 1: PAPERS (Top-Down Neighbors) ---
        # HYBRID APPROACH: Color Thresholding + Canny Edge Detection
        
        # === METHOD 1: White Color Masking (Good for occlusion/hands) ===
        # Adjust lower_white V value (160) if papers are not detected
        lower_white = np.array([0, 0, 160]) 
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((5,5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.dilate(mask_white, kernel, iterations=1)
        cnts_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # === METHOD 2: Canny Edge Detection (Good for contrast boundaries) ===
        v = np.median(blur)
        lower_canny = int(max(0, 0.67 * v))
        upper_canny = int(min(255, 1.33 * v))
        edges = cv2.Canny(blur, lower_canny, upper_canny)
        edges = cv2.dilate(edges, kernel, iterations=2)
        cnts_canny, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # === MERGE & PROCESS ===
        cnts = list(cnts_white) + list(cnts_canny)
        
        # Collect valid candidates with metadata
        paper_candidates = []
        for cnt in cnts:
            if cv2.contourArea(cnt) > GameConfig.MIN_CONTOUR_AREA:
                hull = cv2.convexHull(cnt)
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
                
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    
                    # Aspect Ratio Check
                    d1 = np.linalg.norm(pts[0]-pts[1]); d2 = np.linalg.norm(pts[1]-pts[2])
                    ar = max(d1, d2) / min(d1, d2) if min(d1, d2) > 0 else 0
                    
                    # Solidity Check (Rectangle fullness)
                    min_rect = cv2.minAreaRect(hull)
                    min_rect_area = min_rect[1][0] * min_rect[1][1]
                    hull_area = cv2.contourArea(hull)
                    solidity = (hull_area / min_rect_area) if min_rect_area > 0 else 0

                    if 1.2 < ar < 1.7 and solidity > 0.90:
                        x, y, w, h = cv2.boundingRect(hull)
                        paper_candidates.append({
                            'pts': pts,
                            'area': hull_area,
                            'center': (x + w//2, y + h//2)
                        })

        # --- SMART FILTERING (Non-Maximum Suppression) ---
        
        # 1. Sort by Area Descending (Largest = Best).
        paper_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        unique_papers = []
        min_dist_sq = (frame.shape[0] * 0.1) ** 2  # Threshold: ~10% of screen height squared
        
        for cand in paper_candidates:
            if len(unique_papers) >= 2: break 
            
            is_duplicate = False
            c1 = np.array(cand['center'])
            for existing in unique_papers:
                c2 = np.array(existing['center'])
                if np.sum((c1 - c2) ** 2) < min_dist_sq:
                    is_duplicate = True
                    break 
            
            if not is_duplicate:
                unique_papers.append(cand)

        # 2. Assign Top/Bottom Slots (Vertical Split)
        # CHANGED: Use Y-coordinate (height) to split instead of X
        screen_center_y = frame.shape[0] // 2
        final_slots = [None, None] # [Top_Paper, Bottom_Paper]

        for paper in unique_papers:
            # If center Y is higher (smaller value) than middle -> Top (Slot 0)
            # Else -> Bottom (Slot 1)
            idx = 0 if paper['center'][1] < screen_center_y else 1
            
            if final_slots[idx] is None:
                final_slots[idx] = paper['pts']

        # 3. Process Stabilizers
        detected_papers = [] 
        for i in range(2): # 0 = Top, 1 = Bottom
            raw_corners = final_slots[i]
            stabilizer = self.paper_stabilizers[i]
            
            stable_corners = None
            if raw_corners is not None:
                stable_corners = stabilizer.update(raw_corners)
            elif stabilizer.locked_coords is not None:
                stable_corners = stabilizer.locked_coords
            
            if stable_corners is not None:
                warped, box = warp_from_points(frame, stable_corners)
                detected_papers.append(warped)
                
                cv2.drawContours(output_img, [box], 0, self.colors["paper"], 2)
                cv2.drawContours(occupied_mask, [box], -1, 255, -1)
                
                label = "P1 (Top)" if i == 0 else "P2 (Bot)"
                cv2.putText(output_img, label, (box[0][0], box[0][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["paper"], 2)
            else:
                detected_papers.append(None)

        crops["papers"] = detected_papers

        # --- STAGE 2: WHEEL (Contour Solidity Strategy) ---
        
        # 1. Color Masking (Same as before)
        mask_wheel = np.zeros(frame.shape[:2], dtype=np.uint8)
        target_colors = ["Yellow", "Purple", "Blue"]
        
        for color_name in target_colors:
            if color_name in GameConfig.COLOR_RANGES:
                lower, upper = GameConfig.COLOR_RANGES[color_name]
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask_wheel = cv2.bitwise_or(mask_wheel, mask)
        
        # 2. Subtract Papers & Clean
        mask_wheel = cv2.bitwise_and(mask_wheel, mask_wheel, mask=cv2.bitwise_not(occupied_mask))
        mask_wheel = cv2.morphologyEx(mask_wheel, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        #cv2.imshow("Wheel Mask Debug", cv2.resize(mask_wheel, (0,0), fx=0.5, fy=0.5))

        # 3. Solidity Filtering
        # Find all individual blobs (triangles, noise, arm parts)
        cnts_wheel, _ = cv2.findContours(mask_wheel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_points = []
        
        for cnt in cnts_wheel:
            area = cv2.contourArea(cnt)
            if area < 500: continue # Ignore tiny noise
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = float(area) / hull_area
                
                # CRITICAL FILTER: 
                # The wheel's colored triangles are solid geometric shapes (Solidity ~ 1.0).
                # An arm/hand is irregular and has lower solidity.
                if solidity > 0.85: 
                    # This contour is likely a valid part of the wheel. Add its points.
                    # We add the points from the convex hull to be safe/clean.
                    valid_points.extend(hull.reshape(-1, 2))

        # 4. Global Geometry Check
        raw_corners = None
        
        if len(valid_points) > 0:
            # Create a single array of all valid points
            all_pts = np.array(valid_points, dtype=np.int32)
            
            # Find the Convex Hull of the ENTIRE set of valid pieces
            # This wraps all the triangles into one big shape (the Square of Fortune)
            master_hull = cv2.convexHull(all_pts)
            
            # Check if this master hull forms a Square
            peri = cv2.arcLength(master_hull, True)
            approx = cv2.approxPolyDP(master_hull, 0.04 * peri, True)
            
            # Check for 4 corners
            if len(approx) == 4:
                hull_area = cv2.contourArea(master_hull)
            
                # 2. Find the smallest possible rotated rectangle that fits the hull
                # This is more robust than approxPolyDP for noisy shapes
                rect = cv2.minAreaRect(master_hull)
                (cx, cy), (w, h), angle = rect
                rect_area = w * h
                
                if rect_area > 0:
                    # 3. Solidity Check (Intersection over Union-like ratio)
                    # How much of the rectangle is actually filled by the hull?
                    solidity = hull_area / rect_area
                    
                    # 4. Squareness Check (Aspect Ratio of the rotated box)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    
                    # Validation: High solidity (>0.85) and nearly equal sides (<1.3)
                    if solidity > 0.85 and aspect_ratio < 1.3:
                        raw_corners = approx.reshape(4, 2).astype(np.float32)
                    else:
                        # Convert rotated rect to 4 corner points
                        box = cv2.boxPoints(rect)
                        raw_corners = np.array(box, dtype=np.float32)
            else:
                # Fallback: If approxPolyDP fails (e.g. 5 points due to a clipped corner),
                # force a rectangle fit, but only if the shape is substantial.
                rect = cv2.minAreaRect(master_hull)
                w, h = rect[1]
                if w > 0 and h > 0:
                    ar = min(w,h) / max(w,h)
                    if 0.7 < ar < 1.3: # Still must be somewhat square
                        raw_corners = cv2.boxPoints(rect)

        # 5. Stabilization
        stable_corners = None
        if raw_corners is not None:
            stable_corners = self.wheel_stabilizer.update(raw_corners)
        elif self.wheel_stabilizer.locked_coords is not None:
            stable_corners = self.wheel_stabilizer.locked_coords
            
        # 6. Extract & Update Mask
        if stable_corners is not None:
            warped, box = warp_from_points(frame, stable_corners)
            crops["wheel"] = warped
            
            is_locked = (self.wheel_stabilizer.locked_coords is not None)
            color = self.colors["wheel"] if is_locked else (0, 255, 255) 
            cv2.drawContours(output_img, [box], 0, color, 3 if is_locked else 1)
            cv2.drawContours(occupied_mask, [box], -1, 255, -1)
            
        occupied_mask = cv2.dilate(occupied_mask, np.ones((15,15), np.uint8), iterations=2)
        #cv2.imshow("Occupied Mask Debug", cv2.resize(occupied_mask, (0,0), fx=0.5, fy=0.5))
        
        # --- STAGE 3: CIRCLE ---
        gray_med = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(gray_med, cv2.HOUGH_GRADIENT, dp=1.5, minDist=100,
                                   param1=50, param2=30, minRadius=120, maxRadius=250)
        raw_circle_box = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                if occupied_mask[cy, cx] == 0:
                    x, y, w, h = cx-r, cy-r, 2*r, 2*r
                    raw_circle_box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
                    break 
        
        # Stabilize Circle
        if raw_circle_box is not None: stable_circle = self.circle_stabilizer.update(raw_circle_box)
        elif self.circle_stabilizer.locked_coords is not None: stable_circle = self.circle_stabilizer.locked_coords
        else: stable_circle = None
        
        if stable_circle is not None:
            x, y, w, h = cv2.boundingRect(stable_circle.astype(int))
            # Safe crop bounds
            x, y = max(0, x), max(0, y)
            crops["circle"] = frame[y:y+h, x:x+w]
            cx, cy, r = x + w//2, y + h//2, max(w, h)//2
            cv2.circle(output_img, (cx, cy), r, self.colors["circle"], 3)
            cv2.circle(occupied_mask, (cx, cy), int(r+20), 255, -1)

        # --- STAGE 4: SQUARE ---
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 3)
        thresh = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(occupied_mask))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))  
        clusters = cv2.dilate(thresh, np.ones((9,9),np.uint8), iterations=3)     
        clusters = cv2.erode(clusters, np.ones((9,9),np.uint8), iterations=3)
        cv2.imshow("Occupied Mask Debug", cv2.resize(clusters, (0,0), fx=0.5, fy=0.5)) 
        cnts, _ = cv2.findContours(clusters, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        raw_sq_corners = None
        for cnt in cnts:
            if cv2.contourArea(cnt) < 5000: break
            hull = cv2.convexHull(cnt)
            x, y, w, h = cv2.boundingRect(hull)
            if 0.7 < float(w)/h < 1.3:
                raw_sq_corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
                break
        
        # Stabilize Square
        if raw_sq_corners is not None: stable_sq = self.square_stabilizer.update(raw_sq_corners)
        elif self.square_stabilizer.locked_coords is not None: stable_sq = self.square_stabilizer.locked_coords
        else: stable_sq = None
        
        if stable_sq is not None:
            warped, box = warp_from_points(frame, stable_sq)
            crops["hollow_square"] = warped
            cv2.drawContours(output_img, [box], 0, self.colors["square"], 3)

        return output_img, crops

# ==========================================
# 7. MAIN LOOP
# ==========================================
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): print(f"Error: {input_path}"); return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    detector = BoardDetector()
    manager = GameStateManager()
    print(f"Processing: {width}x{height} @ {fps}fps")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        detected_frame, crops = detector.detect_elements(frame)
        status_texts = []
        
        # Only iterate over actual detected papers (up to 2)
        for i, paper_crop in enumerate(crops["papers"]):
            status_texts.append(manager.handle_paper(paper_crop, i))
            
        if crops["wheel"] is not None:
            status_texts.append(manager.handle_wheel(crops["wheel"]))
        if crops["hollow_square"] is not None:
            status_texts.append(manager.handle_container(crops["hollow_square"], "Square"))
        if crops["circle"] is not None:
            status_texts.append(manager.handle_container(crops["circle"], "Circle"))

        y_offset = 40
        for text in status_texts:
            cv2.putText(detected_frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 40

        final_output = np.hstack((frame, detected_frame))
        out.write(final_output)
        
        preview = cv2.resize(final_output, (0,0), fx=0.4, fy=0.4)
        cv2.imshow("Game Tracker", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); out.release(); cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    video_path = '/home/jakub/Artificial Intelligence/Studies/Term 5/[CV] Computer Vision/boardgame-detecor/data/vid_4.MOV'
    output_path = 'game_output.MOV'
    if os.path.exists(video_path): process_video(video_path, output_path)
    else: print(f"File not found: {video_path}")