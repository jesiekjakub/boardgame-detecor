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
    """
    # --- Color Calibration (HSV: H=0-179, S=0-255, V=0-255) ---
    COLOR_RANGES = {
        "Red_1":   ((0, 20, 70), (10, 255, 255)),     
        "Red_2":   ((170, 20, 70), (180, 255, 255)),  
        "Yellow":  ((20, 20, 70), (35, 255, 255)),    
        "Purple":  ((120, 10, 50), (145, 255, 255)),  
        "Blue":    ((85, 20, 70), (115, 255, 255)),   
    }

    # --- Token Colors & Points ---
    TOKEN_COLORS = {
        "Blue":   {"range": ((37, 1, 70), (60, 255, 255)), "points": 1},
        "Green":  {"range": ((27, 20, 50), (37, 255, 255)),  "points": 5},
        "Purple": {"range": ((154, 10, 40), (174, 255, 255)), "points": 10},
        "Orange": {"range": ((10, 90, 100), (27, 255, 255)), "points": 50} 
    }
    MIN_TOKEN_AREA_PCT = 0.02 

    # --- Stabilizer Settings ---
    STABILIZER_HISTORY_LEN = 60       
    STABILIZER_THRESHOLD = 4.0        

    # --- Wheel Analysis Settings ---
    WHEEL_HISTORY_LEN = 15            
    NEEDLE_VALUE_THRESH = 110         
    ROI_RADIUS = 20                   
    MIN_VOTES = 5                     

    # --- General Detection ---
    MIN_CONTOUR_AREA = 5000           
    MIN_NEEDLE_AREA = 50 
    
    # --- DICE DETECTION SETTINGS (NEW) ---
    DICE_MIN_AREA = 100       # Min area for a die candidate
    DICE_MAX_AREA = 1500      # Max area (relative to crop)
    DICE_PIP_THRESH = 100      # Threshold for pip detection in warped image
    DICE_PIP_MIN_AREA = 10     # Min area for a pip
    DICE_PIP_MAX_AREA = 400    # Max area for a pip

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
    
    # Safety check for zero dimension
    if width == 0: width = 1
    if height == 0: height = 1
        
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(sorted_pts, dst), (width, height))
    return warped, sorted_pts.astype(int)

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
        if not self.history:
            for _ in range(self.history_len): self.history.append(sorted_coords)
        else:
            self.history.append(sorted_coords)
        
        history_arr = np.array(self.history)
        mean_coords = np.mean(history_arr, axis=0)
        max_deviation = np.max(np.linalg.norm(history_arr - mean_coords, axis=2))
        
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
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, GameConfig.NEEDLE_VALUE_THRESH]) 
        mask = cv2.inRange(hsv_crop, lower_black, upper_black)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None, None
        
        valid_cnts = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < GameConfig.MIN_NEEDLE_AREA: continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
                if np.linalg.norm(np.array([cX, cY]) - center) < w * 0.3:
                    valid_cnts.append(cnt)
        
        needle_cnt = max(valid_cnts, key=cv2.contourArea) if valid_cnts else max(cnts, key=cv2.contourArea)
        if cv2.contourArea(needle_cnt) < GameConfig.MIN_NEEDLE_AREA: return None, None

        points = needle_cnt.reshape(-1, 2)
        dists = np.linalg.norm(points - center, axis=1)
        outermost_tip = points[np.argmax(dists)]
        return outermost_tip, needle_cnt

    def get_color_by_voting(self, hsv_crop, tip, debug_img=None):
        h, w = hsv_crop.shape[:2]
        radius = GameConfig.ROI_RADIUS
        mask_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_roi, tuple(tip), radius, 255, -1)
        
        mask_black = cv2.inRange(hsv_crop, np.array([0,0,0]), np.array([180, 255, GameConfig.NEEDLE_VALUE_THRESH]))
        mask_grey = cv2.inRange(hsv_crop, np.array([0,0,0]), np.array([180, 50, 255]))
        mask_invalid = cv2.bitwise_or(mask_black, mask_grey)
        mask_valid = cv2.bitwise_and(mask_roi, cv2.bitwise_not(mask_invalid))
        
        votes = {k: 0 for k in GameConfig.COLOR_RANGES.keys() if k != "Red_2"}
        for color_name, (lower, upper) in GameConfig.COLOR_RANGES.items():
            mask_color = cv2.inRange(hsv_crop, np.array(lower), np.array(upper))
            mask_final = cv2.bitwise_and(mask_valid, mask_color)
            count = cv2.countNonZero(mask_final)
            key = "Red_1" if "Red" in color_name else color_name
            if key in votes: votes[key] += count

        total_votes = sum(votes.values())
        if total_votes < GameConfig.MIN_VOTES: return "Unknown"
        winner = max(votes, key=votes.get)
        return "Red" if "Red" in winner else winner

    def analyze(self, wheel_crop, debug_vis_img=None):
        if wheel_crop is None or wheel_crop.size == 0: return "No Wheel", "N/A"
        hsv = cv2.cvtColor(wheel_crop, cv2.COLOR_BGR2HSV)
        tip, needle_cnt = self.find_outermost_tip(hsv)
        current_color = "Unknown"
        
        if tip is not None:
            if debug_vis_img is not None:
                cv2.drawContours(debug_vis_img, [needle_cnt], -1, (255, 0, 255), 2)
                cv2.circle(debug_vis_img, tuple(tip), 3, (0, 0, 255), -1)
            current_color = self.get_color_by_voting(hsv, tip, debug_vis_img)

        self.color_history.append(current_color)
        if len(self.color_history) < self.history_len: return "Initializing...", "N/A"
        valid = [c for c in self.color_history if c not in ["Unknown"]]
        if len(valid) < 5: return "Searching...", "N/A"
        unique = set(valid)
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
# 5. PAPER ANALYZER
# ==========================================

class PaperAnalyzer:
    def __init__(self):
        self.token_config = GameConfig.TOKEN_COLORS
        self.min_area_pct = GameConfig.MIN_TOKEN_AREA_PCT
        self.max_area_pct = 0.20 
        
        self.target_ratio = 2.2
        self.ratio_tolerance = 0.3 
        self.min_solidity = 0.85

    def analyze(self, paper_crop, debug_img=None):
        if paper_crop is None:
            return 0, None

        total_score = 0
        h, w = paper_crop.shape[:2]
        paper_area = h * w
        viz_crop = paper_crop.copy()
        
        hsv = cv2.cvtColor(paper_crop, cv2.COLOR_BGR2HSV)
        
        for color_name, info in self.token_config.items():
            lower, upper = info["range"]
            points = info["points"]
            
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1) 
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                area_pct = area / paper_area
                
                if self.min_area_pct <= area_pct <= self.max_area_pct:
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (rw, rh), angle = rect
                    box = np.intp(cv2.boxPoints(rect))
                    
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    short_side = min(rw, rh)
                    long_side = max(rw, rh)
                    ar = long_side / short_side if short_side > 0 else 0
                    
                    valid_ratio = (self.target_ratio - self.ratio_tolerance) < ar < (self.target_ratio + self.ratio_tolerance)
                    valid_solidity = solidity > self.min_solidity

                    if valid_solidity and valid_ratio:
                        total_score += points
                        cv2.drawContours(viz_crop, [box], 0, (0, 255, 0), 2)
                        cv2.putText(viz_crop, f"{points}pts", (box[1][0], box[1][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    elif solidity > self.min_solidity*0.9:
                        mask_roi = np.zeros_like(mask)
                        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
                        dist = cv2.distanceTransform(mask_roi, cv2.DIST_L2, 5)
                        _, peak_mask = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
                        peak_mask = peak_mask.astype(np.uint8)
                        peak_cnts, _ = cv2.findContours(peak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        num_peaks = len(peak_cnts)
                        if num_peaks < 1: num_peaks = 1 
                        
                        score_add = points * num_peaks
                        total_score += score_add
                        cv2.drawContours(viz_crop, [box], 0, (0, 165, 255), 2)
                        cv2.putText(viz_crop, f"x{num_peaks} ({score_add}pts)", 
                                    (box[1][0], box[1][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return total_score, viz_crop

# ==========================================
# 6. DICE ANALYZER (No Resize & King Pip Filter)
# ==========================================

class DiceAnalyzer:
    def __init__(self):
        # HSV Settings for "White" Dice
        # Lower: Any Hue, Very Low Saturation, High Value (Brightness)
        self.lower_white = np.array([0, 0, 200]) 
        self.upper_white = np.array([180, 60, 255])
        
        # Kernels
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_erode = np.ones((3, 3), np.uint8) 

    def count_pips_in_warped(self, die_crop):
        """
        Counts pips using a 2-Pass Bounding Box Filter.
        1. Upscale.
        2. Detect all potential pips.
        3. Identify "Major Pips" (Area >= 64% of Max Pip).
        4. Create Bounding Box around Major Pips.
        5. Mask out everything outside this box (removing boundary artifacts).
        6. Re-count inside the box.
        """
        if die_crop is None or die_crop.size == 0: return 0, die_crop
        
        # 0. Debug Info
        print(f"Pip Analysis Input Shape: {die_crop.shape}")

        # 1. Upscale (4x) for sub-pixel accuracy
        scale = 4
        src = cv2.resize(die_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 2. Grayscale & Threshold (No Blur)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 4. BOUNDARY CLEANUP (The "Frame" Killer)
        #    Set a margin of pixels around the entire image to Black (0).
        #    This slices off the white "rim" artifacts caused by the die's edge shadows.
        h, w = thresh.shape
        margin = int(0.075*(np.sum(thresh.shape)))
        cv2.rectangle(thresh, (0, 0), (w, margin), 0, -1)          # Top
        cv2.rectangle(thresh, (0, h-margin), (w, h), 0, -1)        # Bottom
        cv2.rectangle(thresh, (0, 0), (margin, h), 0, -1)          # Left
        cv2.rectangle(thresh, (w-margin, 0), (w, h), 0, -1)        # Right
        thresh_copy = thresh.copy()

        # 3. Find Candidates (Pass 1)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        face_area = gray.shape[0] * gray.shape[1]
        candidates = []
        
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # Loose filter
            if (face_area * 0.002) < area < (face_area * 0.25):
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                circ = 4 * np.pi * (area / (peri ** 2))
                
                if circ > 0.5: # Loose circularity
                    candidates.append(cnt)
        
        # Fallback if nothing found
        if not candidates: return 0, src
        
        # 4. Find Major Pips (King Filter)
        max_area = max([cv2.contourArea(c) for c in candidates])
        major_thresh = 0.50 * max_area
        
        # These are the pips we are CONFIDENT belong to the face
        major_pips = [c for c in candidates if cv2.contourArea(c) >= major_thresh]
        
        if not major_pips: return 0, src

        # 5. Define "Active Region" (Smallest Bounding Box)
        # We assume Major Pips define the "face area".
        all_points = np.vstack(major_pips)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # 6. Masking (The Boundary Killer)
        # Create a mask that is BLACK everywhere ...
        mask_roi = np.zeros_like(thresh)
        # ... except inside the bounding box (WHITE)
        cv2.rectangle(mask_roi, (x, y), (x + w, y + h), 255, -1)
        
        # Apply mask: This keeps pixels inside the box, effectively
        # removing any "white border artifacts" that are outside this cluster.
        thresh_clean = cv2.bitwise_and(thresh, thresh, mask=mask_roi)
        
        # 7. Final Count (Pass 2)
        cnts_final, _ = cv2.findContours(thresh_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        final_count = 0
        viz = src.copy()
        
        for cnt in cnts_final:
            area = cv2.contourArea(cnt)
            # Basic noise check (min area)
            if area > (face_area * 0.002):
                 peri = cv2.arcLength(cnt, True)
                 if peri == 0: continue
                 circ = 4 * np.pi * (area / (peri ** 2))
                 
                 # Accept any circular shape inside the valid box
                 if circ > 0.5:
                     final_count += 1
                     cv2.drawContours(viz, [cnt], -1, (0, 255, 0), 2)
        if final_count > 6:
            final_count = 6

        return final_count, viz

    def analyze(self, crop, container_type, debug_img=None):
        """
        Main detection pipeline.
        """
        if crop is None or crop.size == 0: return []
        scores = []
        h, w = crop.shape[:2]
        
        # --- STAGE 1: COLOR SEGMENTATION (Find White Objects) ---
        # Median blur is great for ignoring fabric texture while keeping edges sharp
        blurred_crop = cv2.medianBlur(crop, 7)
        hsv = cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2HSV)
        
        # Create White Mask
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # --- STAGE 2: GEOMETRY CLEANUP (The Fix) ---
        
        # 1. Circle ROI (Still useful to mask strictly outside the circle)
        if container_type == "Circle":
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = min(w, h) // 2
            cv2.circle(roi_mask, center, int(radius * 0.85), 255, -1)
            white_mask = cv2.bitwise_and(white_mask, white_mask, mask=roi_mask)
            
        # 2. Square: No harsh masking, just a tiny 2px border clean
        #    This prevents infinite lines if the crop hits the very edge of the image.
        elif container_type == "Square":
            cv2.rectangle(white_mask, (0,0), (w, 2), 0, -1) # Top
            cv2.rectangle(white_mask, (0,h-2), (w, h), 0, -1) # Bot
            cv2.rectangle(white_mask, (0,0), (2, h), 0, -1) # Left
            cv2.rectangle(white_mask, (w-2,0), (w, h), 0, -1) # Right
        
        # 3. Erosion (Critical):
        #    If a die is touching the bright metal rim, they will merge into one blob.
        #    Eroding breaks this thin connection.
        white_mask = cv2.erode(white_mask, self.kernel_erode, iterations=2)
        
        # 4. Open (Noise Removal)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        
        # 5. Dilate (Restore Size)
        #    We dilate back to get the original die shape, but the connection to the rim is now broken.
        white_mask = cv2.dilate(white_mask, self.kernel_erode, iterations=2)

        # Visualize Mask
        if debug_img is not None:
             mask_vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
             mask_vis = cv2.resize(mask_vis, (80, 80))
             debug_img[0:80, 0:80] = mask_vis

        # --- STAGE 3: CONTOUR ANALYSIS ---
        cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            
            # Filter 1: Area
            # Dice are usually 1000-15000px depending on camera height.
            # Rims are usually HUGE (perimeter of image) or tiny (noise).
            if area < GameConfig.DICE_MIN_AREA or area > GameConfig.DICE_MAX_AREA:
                continue
            
            # Filter 2: Solidity (The Rim Killer)
            # A die is a solid square (Solidity ~0.95).
            # A metal rim is a hollow loop (Solidity < 0.3) or a thin line.
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            if solidity < 0.80: continue 
            
            # Filter 3: Aspect Ratio
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            
            if min(rw, rh) == 0: continue
            ar = max(rw, rh) / min(rw, rh)
            
            # Dice are square-ish. Rims are often long and thin.
            if ar > 1.35: continue 

            # Get Detection Box
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # --- STAGE 4: WARP & COUNT ---
            warped_die, _ = warp_from_points(crop, box.astype(np.float32))
            
            # REMOVED: Resizing step to keep original detail for pip counting
            # warped_die = cv2.resize(warped_die, (100, 100))
            
            pips, pip_viz = self.count_pips_in_warped(warped_die)
            
            if True:#1 <= pips:
                #if pips > 6:
                #    pips = 6
                scores.append(pips)
                
                if debug_img is not None:
                    # Draw Green Box
                    cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
                    # Draw Score
                    cv2.putText(debug_img, str(pips), (int(cx)-10, int(cy)+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return scores

# ==========================================
# 7. GAME STATE MANAGER
# ==========================================

class GameStateManager:
    def __init__(self):
        self.scores = {"Paper_1": 0, "Paper_2": 0}
        self.wheel_analyzer = WheelAnalyzer()
        self.paper_analyzer = PaperAnalyzer()
        self.dice_analyzer = DiceAnalyzer() # Initialize the new DiceAnalyzer

    def handle_paper(self, crop, paper_index):
        if crop is None: 
            return f"Paper {paper_index + 1}: Not Found"
            
        score, viz_crop = self.paper_analyzer.analyze(crop)
        self.scores[f"Paper_{paper_index+1}"] = score
        cv2.imshow(f"Paper {paper_index + 1} Analysis", viz_crop)
        return f"Paper {paper_index + 1}: {score}"

    def handle_wheel(self, crop):
        debug_vis = crop.copy()
        state, color = self.wheel_analyzer.analyze(crop, debug_vis)
        
        vis_large = cv2.resize(debug_vis, (300, 300))
        cv2.imshow("Wheel Analysis Debug", vis_large)
        
        txt = f"Wheel: {state}"
        if state in ["STOPPED", "IDLE"]: txt += f" -> {color}"
        return txt

    def handle_container(self, crop, type_name):
        """
        Handles Circle and Square containers to detect dice.
        type_name: "Circle" or "Square"
        """
        if crop is None or crop.size == 0: return f"{type_name}: No Crop"
        
        debug_vis = crop.copy()
        
        # Call the new dice analyzer
        # type_name determines the masking strategy
        dice_scores = self.dice_analyzer.analyze(crop, type_name, debug_vis)
        
        # Show debug view
        cv2.imshow(f"{type_name} Analysis", debug_vis)
        
        if not dice_scores:
            return f"{type_name}: Empty"
        else:
            return f"{type_name} Dice: {dice_scores}"

# ==========================================
# 8. BOARD DETECTOR (2-Player Logic)
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
        self.paper_stabilizers = {0: CornerStabilizer(), 1: CornerStabilizer()}

    def detect_elements(self, frame):
        output_img = frame.copy()
        crops = {"papers": [None, None], "wheel": None, "hollow_square": None, "circle": None}
        
        occupied_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        # --- STAGE 1: PAPERS ---
        lower_white = np.array([0, 0, 160]) 
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((5,5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.dilate(mask_white, kernel, iterations=1)
        cnts_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        v = np.median(blur)
        lower_canny = int(max(0, 0.67 * v))
        upper_canny = int(min(255, 1.33 * v))
        edges = cv2.Canny(blur, lower_canny, upper_canny)
        edges = cv2.dilate(edges, kernel, iterations=2)
        cnts_canny, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = list(cnts_white) + list(cnts_canny)
        
        paper_candidates = []
        for cnt in cnts:
            if cv2.contourArea(cnt) > GameConfig.MIN_CONTOUR_AREA:
                hull = cv2.convexHull(cnt)
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
                
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    d1 = np.linalg.norm(pts[0]-pts[1]); d2 = np.linalg.norm(pts[1]-pts[2])
                    ar = max(d1, d2) / min(d1, d2) if min(d1, d2) > 0 else 0
                    
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

        paper_candidates.sort(key=lambda x: x['area'], reverse=True)
        unique_papers = []
        min_dist_sq = (frame.shape[0] * 0.1) ** 2  
        
        for cand in paper_candidates:
            if len(unique_papers) >= 2: break 
            is_duplicate = False
            c1 = np.array(cand['center'])
            for existing in unique_papers:
                c2 = np.array(existing['center'])
                if np.sum((c1 - c2) ** 2) < min_dist_sq:
                    is_duplicate = True; break 
            if not is_duplicate: unique_papers.append(cand)

        screen_center_y = frame.shape[0] // 2
        final_slots = [None, None] 

        for paper in unique_papers:
            idx = 0 if paper['center'][1] < screen_center_y else 1
            if final_slots[idx] is None: final_slots[idx] = paper['pts']

        detected_papers = [] 
        for i in range(2): 
            raw_corners = final_slots[i]
            stabilizer = self.paper_stabilizers[i]
            
            stable_corners = None
            if raw_corners is not None: stable_corners = stabilizer.update(raw_corners)
            elif stabilizer.locked_coords is not None: stable_corners = stabilizer.locked_coords
            
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

        # --- STAGE 2: WHEEL ---
        mask_wheel = np.zeros(frame.shape[:2], dtype=np.uint8)
        target_colors = ["Yellow", "Purple", "Blue"]
        
        for color_name in target_colors:
            if color_name in GameConfig.COLOR_RANGES:
                lower, upper = GameConfig.COLOR_RANGES[color_name]
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask_wheel = cv2.bitwise_or(mask_wheel, mask)
        
        mask_wheel = cv2.bitwise_and(mask_wheel, mask_wheel, mask=cv2.bitwise_not(occupied_mask))
        mask_wheel = cv2.morphologyEx(mask_wheel, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        
        cnts_wheel, _ = cv2.findContours(mask_wheel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_points = []
        for cnt in cnts_wheel:
            area = cv2.contourArea(cnt)
            if area < 500: continue 
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity > 0.85: valid_points.extend(hull.reshape(-1, 2))

        raw_corners = None
        if len(valid_points) > 0:
            all_pts = np.array(valid_points, dtype=np.int32)
            master_hull = cv2.convexHull(all_pts)
            peri = cv2.arcLength(master_hull, True)
            approx = cv2.approxPolyDP(master_hull, 0.04 * peri, True)
            
            if len(approx) == 4:
                hull_area = cv2.contourArea(master_hull)
                rect = cv2.minAreaRect(master_hull)
                (cx, cy), (w, h), angle = rect
                rect_area = w * h
                if rect_area > 0:
                    solidity = hull_area / rect_area
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    if solidity > 0.80 and aspect_ratio < 1.3:
                        raw_corners = approx.reshape(4, 2).astype(np.float32)
                    else:
                        box = cv2.boxPoints(rect)
                        raw_corners = np.array(box, dtype=np.float32)
            else:
                rect = cv2.minAreaRect(master_hull)
                w, h = rect[1]
                if w > 0 and h > 0:
                    ar = min(w,h) / max(w,h)
                    if 0.7 < ar < 1.3: raw_corners = cv2.boxPoints(rect)

        stable_corners = None
        if raw_corners is not None: stable_corners = self.wheel_stabilizer.update(raw_corners)
        elif self.wheel_stabilizer.locked_coords is not None: stable_corners = self.wheel_stabilizer.locked_coords
            
        if stable_corners is not None:
            warped, box = warp_from_points(frame, stable_corners)
            crops["wheel"] = warped
            is_locked = (self.wheel_stabilizer.locked_coords is not None)
            color = self.colors["wheel"] if is_locked else (0, 255, 255) 
            cv2.drawContours(output_img, [box], 0, color, 3 if is_locked else 1)
            cv2.drawContours(occupied_mask, [box], -1, 255, -1)
            
        occupied_mask = cv2.dilate(occupied_mask, np.ones((15,15), np.uint8), iterations=2)
        
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
        
        if raw_circle_box is not None: stable_circle = self.circle_stabilizer.update(raw_circle_box)
        elif self.circle_stabilizer.locked_coords is not None: stable_circle = self.circle_stabilizer.locked_coords
        else: stable_circle = None
        
        if stable_circle is not None:
            x, y, w, h = cv2.boundingRect(stable_circle.astype(int))
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
        
        if raw_sq_corners is not None: stable_sq = self.square_stabilizer.update(raw_sq_corners)
        elif self.square_stabilizer.locked_coords is not None: stable_sq = self.square_stabilizer.locked_coords
        else: stable_sq = None
        
        if stable_sq is not None:
            warped, box = warp_from_points(frame, stable_sq)
            crops["hollow_square"] = warped
            cv2.drawContours(output_img, [box], 0, self.colors["square"], 3)

        return output_img, crops

# ==========================================
# 9. MAIN LOOP
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
        
        for i, paper_crop in enumerate(crops["papers"]):
            status_texts.append(manager.handle_paper(paper_crop, i))
            
        if crops["wheel"] is not None:
            status_texts.append(manager.handle_wheel(crops["wheel"]))
            
        # Updated to process dice in Square
        if crops["hollow_square"] is not None:
            status_texts.append(manager.handle_container(crops["hollow_square"], "Square"))
            
        # Updated to process dice in Circle
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
    video_path = '/home/jakub/Artificial Intelligence/Studies/Term 5/[CV] Computer Vision/boardgame-detecor/data/vid_4.MOV' # Replace with your local path
    output_path = 'game_output.MOV'
    if os.path.exists(video_path): process_video(video_path, output_path)
    else: print(f"File not found: {video_path}")