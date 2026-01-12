import cv2
import numpy as np
from collections import deque
import os

# ==========================================
# 1. HELPER FUNCTIONS (Geometry & Sorting)
# ==========================================

def order_points(pts):
    """
    Orders coordinates: top-left, top-right, bottom-right, bottom-left.
    Crucial for consistent frame-to-frame comparison.
    """
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Top-Left
    rect[2] = pts[np.argmax(s)]      # Bottom-Right
    rect[1] = pts[np.argmin(diff)]   # Top-Right
    rect[3] = pts[np.argmax(diff)]   # Bottom-Left
    
    return rect

def get_warped_rect(image, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    pts = order_points(box)
    (tl, tr, br, bl) = pts
    
    width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    # src is already ordered by order_points
    
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(pts, dst), (width, height))
    if warped.shape[1] * 0.9 > warped.shape[0]: 
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        
    return warped, box

def warp_from_points(image, pts):
    sorted_pts = order_points(pts)
    (tl, tr, br, bl) = sorted_pts
    
    width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(sorted_pts, dst), (width, height))
    
    return warped, sorted_pts.astype(int)

# ==========================================
# 2. STABILIZER (Updated with Instant Lock)
# ==========================================

class CornerStabilizer:
    def __init__(self, history_len=30, stable_threshold=5.0):
        self.history = deque(maxlen=history_len)
        self.locked_coords = None
        self.threshold = stable_threshold
        self.history_len = history_len

    def update(self, new_coords):
        """
        Input: (4, 2) numpy array of corners.
        Output: The STABLE (locked) coordinates to use.
        """
        # 1. Always sort points first
        sorted_coords = order_points(new_coords)
        
        # 2. INSTANT LOCK: If history is empty, fill it immediately
        # This prevents the "settling" delay on the first detection.
        if not self.history:
            for _ in range(self.history_len):
                self.history.append(sorted_coords)
        else:
            self.history.append(sorted_coords)
        
        # 3. Check Stability
        history_arr = np.array(self.history)
        mean_coords = np.mean(history_arr, axis=0)
        
        # Calculate max deviation from mean
        diffs = np.linalg.norm(history_arr - mean_coords, axis=2)
        max_deviation = np.max(diffs)
        
        # 4. Logic: Only update lock if the buffer is stable
        if max_deviation < self.threshold:
            self.locked_coords = mean_coords.astype(np.float32)
            
        # If unstable, return OLD lock. If no lock yet, return current mean or sorted_coords.
        return self.locked_coords if self.locked_coords is not None else sorted_coords

# ==========================================
# 3. WHEEL ANALYZER (Unchanged)
# ==========================================

class WheelAnalyzer:
    def __init__(self):
        self.color_ranges = {
            "Red_1":   ((0, 100, 100), (10, 255, 255)),   
            "Red_2":   ((160, 100, 100), (179, 255, 255)), 
            "Yellow":  ((22, 100, 120), (38, 255, 255)),
            "Green":   ((40, 80, 80), (85, 255, 255)),
            "Blue":    ((90, 100, 100), (130, 255, 255)),
        }
        self.blur_threshold = 150 

    def get_state(self, gray_image):
        score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        if score < self.blur_threshold:
            return "SPINNING", score
        else:
            return "READABLE", score

    def find_needle_tip(self, crop_hsv, img_center):
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([179, 255, 70]) 
        mask_needle = cv2.inRange(crop_hsv, lower_black, upper_black)

        kernel = np.ones((3,3), np.uint8)
        mask_needle = cv2.morphologyEx(mask_needle, cv2.MORPH_OPEN, kernel)
        mask_needle = cv2.morphologyEx(mask_needle, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(mask_needle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None, None

        needle_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(needle_cnt) < 100: return None, None

        rect = cv2.minAreaRect(needle_cnt)
        box = cv2.boxPoints(rect); box = np.intp(box)

        max_dist = 0
        tipA = box[0]; tipB = box[1]
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(box[i] - box[j])
                if dist > max_dist:
                    max_dist = dist; tipA = box[i]; tipB = box[j]

        distA = np.linalg.norm(tipA - img_center)
        distB = np.linalg.norm(tipB - img_center)
        long_tip = tipA if distA > distB else tipB
        
        return long_tip, needle_cnt

    def determine_color(self, crop_hsv, tip, img_center):
        h, w = crop_hsv.shape[:2]
        vec = np.array([tip[0] - img_center[0], tip[1] - img_center[1]], dtype=np.float32)
        vec_len = np.linalg.norm(vec)
        if vec_len == 0: return "Error", (0,0,0)
        
        offset = 20 
        sample_point = tip + (vec / vec_len * offset)
        sx, sy = int(sample_point[0]), int(sample_point[1])
        sx = np.clip(sx, 0, w - 1); sy = np.clip(sy, 0, h - 1)

        roi = crop_hsv[max(0, sy-1):min(h, sy+2), max(0, sx-1):min(w, sx+2)]
        if roi.size == 0: return "Edge", (sx, sy)
        
        avg_hsv = np.mean(roi, axis=(0,1))
        hue, sat, val = avg_hsv

        if val < 70: return "Shadow", (sx, sy)

        color = "Unknown"
        if (self.color_ranges["Red_1"][0][0] <= hue <= self.color_ranges["Red_1"][1][0]) or \
           (self.color_ranges["Red_2"][0][0] <= hue <= self.color_ranges["Red_2"][1][0]):
             color = "Red"
        elif self.color_ranges["Yellow"][0][0] <= hue <= self.color_ranges["Yellow"][1][0] and sat > 80:
            color = "Yellow"
        elif self.color_ranges["Green"][0][0] <= hue <= self.color_ranges["Green"][1][0] and sat > 80:
            color = "Green"
        elif self.color_ranges["Blue"][0][0] <= hue <= self.color_ranges["Blue"][1][0] and sat > 80:
            color = "Blue"
            
        return color, (sx, sy)

    def analyze(self, wheel_crop, debug_img=None):
        if wheel_crop is None or wheel_crop.size == 0: return "Not Detected", "N/A", 0

        h, w = wheel_crop.shape[:2]
        img_center = np.array([w // 2, h // 2])
        gray = cv2.cvtColor(wheel_crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(wheel_crop, cv2.COLOR_BGR2HSV)

        state, score = self.get_state(gray)
        result_color = "N/A"
        
        if state == "READABLE":
            long_tip, needle_cnt = self.find_needle_tip(hsv, img_center)
            if long_tip is not None:
                result_color, sample_pt = self.determine_color(hsv, long_tip, img_center)
                if debug_img is not None:
                    cv2.drawContours(debug_img, [needle_cnt], -1, (255,0,255), 2)
                    cv2.line(debug_img, tuple(img_center), tuple(long_tip), (0,255,0), 2)
                    cv2.circle(debug_img, sample_pt, 4, (255,255,255), -1)
                    cv2.circle(debug_img, sample_pt, 2, (0,0,0), -1)

        return state, result_color, score

# ==========================================
# 4. GAME STATE MANAGER
# ==========================================

class GameStateManager:
    def __init__(self):
        self.scores = {}
        self.wheel_analyzer = WheelAnalyzer()

    def handle_paper(self, crop, paper_id):
        return f"P{paper_id}: 0"

    def handle_wheel(self, crop):
        debug_vis = crop.copy()
        state, color, sharpness = self.wheel_analyzer.analyze(crop, debug_vis)
        cv2.imshow("Wheel Debug", debug_vis)
        
        status_text = f"Wheel: {state}"
        if state == "READABLE":
            status_text += f" -> {color}"
        return status_text

    def handle_container(self, crop, type_name):
        return f"{type_name}: 0"

# ==========================================
# 5. MAIN DETECTOR (Full Stabilization)
# ==========================================

class BoardDetector:
    def __init__(self):
        self.colors = {
            "paper": (0, 0, 255), "wheel": (0, 255, 255),
            "circle": (0, 165, 255), "square": (255, 0, 0)
        }
        
        # --- STABILIZERS ---
        # 1. Wheel Stabilizer
        self.wheel_stabilizer = CornerStabilizer(history_len=10, stable_threshold=4.0)
        
        # 2. Container Stabilizers
        self.circle_stabilizer = CornerStabilizer(history_len=30, stable_threshold=4.0)
        self.square_stabilizer = CornerStabilizer(history_len=30, stable_threshold=4.0)
        
        # 3. Paper Stabilizers (Dict because # of papers varies)
        self.paper_stabilizers = {} 

    def detect_elements(self, frame):
        output_img = frame.copy()
        crops = {"papers": [], "wheel": None, "hollow_square": None, "circle": None}
        
        occupied_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        # --- STAGE 1: PAPERS ---
        v = np.median(blur)
        lower = int(max(0, 0.67 * v)); upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blur, lower, upper)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
        
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Track which stabilizers were used this frame (to handle disappearing papers)
        used_paper_ids = []
        
        # Sort contours by x-coordinate to consistently assign IDs to Stabilizers (Left -> Right)
        # This prevents "Paper 1" swapping with "Paper 2" if they are close.
        paper_candidates = []
        for cnt in cnts:
            if cv2.contourArea(cnt) > 5000:
                hull = cv2.convexHull(cnt)
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    d1 = np.linalg.norm(pts[0]-pts[1]); d2 = np.linalg.norm(pts[1]-pts[2])
                    ar = max(d1, d2) / min(d1, d2) if min(d1, d2) > 0 else 0
                    if 1.3 < ar < 1.6:
                        # Store candidate
                        x, y, _, _ = cv2.boundingRect(hull)
                        paper_candidates.append((x, hull))
        
        # Sort by X coordinate
        paper_candidates.sort(key=lambda x: x[0])
        
        for i, (_, hull) in enumerate(paper_candidates):
            paper_id = i
            if paper_id not in self.paper_stabilizers:
                self.paper_stabilizers[paper_id] = CornerStabilizer(history_len=30, stable_threshold=5.0)
            
            # Get raw corners
            rect = cv2.minAreaRect(hull)
            raw_corners = cv2.boxPoints(rect)
            
            # Stabilize
            stable_corners = self.paper_stabilizers[paper_id].update(raw_corners)
            
            # Warp & Draw
            warped, box = warp_from_points(frame, stable_corners)
            crops["papers"].append(warped)
            cv2.drawContours(output_img, [box], 0, self.colors["paper"], 2)
            cv2.drawContours(occupied_mask, [box], -1, 255, -1)


        # --- STAGE 2: WHEEL (Stabilized) ---
        mask_wheel = cv2.inRange(hsv, (20, 70, 80), (35, 255, 255)) | cv2.inRange(hsv, (95, 20, 80), (125, 255, 255))
        mask_wheel = cv2.bitwise_and(mask_wheel, mask_wheel, mask=cv2.bitwise_not(occupied_mask))
        mask_wheel = cv2.morphologyEx(mask_wheel, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        
        pts = cv2.findNonZero(mask_wheel)
        raw_corners = None
        
        if pts is not None:
            pts = pts.squeeze()
            if pts.ndim == 1: pts = pts.reshape(-1, 2)
            center = np.median(pts, axis=0)
            dists = np.linalg.norm(pts - center, axis=1)
            q1, q3 = np.percentile(dists, [25, 75])
            pts = pts[dists < (q3 + 1.5*(q3-q1))]
            
            if len(pts) > 4:
                hull = cv2.convexHull(pts.astype(np.int32))
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.04*peri, True)
                if len(approx) == 4:
                    raw_corners = approx.reshape(4, 2)
                else:
                    rect = cv2.minAreaRect(hull)
                    raw_corners = cv2.boxPoints(rect)

        # Update Stabilizer
        if raw_corners is not None:
            stable_corners = self.wheel_stabilizer.update(raw_corners)
        elif self.wheel_stabilizer.locked_coords is not None:
            stable_corners = self.wheel_stabilizer.locked_coords
        else:
            stable_corners = None

        if stable_corners is not None:
            warped, box = warp_from_points(frame, stable_corners)
            crops["wheel"] = warped
            is_locked = (self.wheel_stabilizer.locked_coords is not None)
            color = self.colors["wheel"] if is_locked else (0, 255, 255) 
            thickness = 3 if is_locked else 1
            cv2.drawContours(output_img, [box], 0, color, thickness)
            cv2.drawContours(occupied_mask, [box], -1, 255, -1)

        occupied_mask = cv2.dilate(occupied_mask, np.ones((15,15),np.uint8), iterations=2)

        # --- STAGE 3: HOLLOW CIRCLE (Stabilized) ---
        gray_med = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(gray_med, cv2.HOUGH_GRADIENT, dp=1.5, minDist=100,
                                   param1=50, param2=30, minRadius=40, maxRadius=150)
        
        raw_circle_box = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                if occupied_mask[cy, cx] == 0:
                    # Convert Circle to Box Points for Stabilizer consistency
                    # (x, y, w, h) -> Box Points
                    x, y, w, h = cx-r, cy-r, 2*r, 2*r
                    raw_circle_box = np.array([
                        [x, y], [x+w, y], [x+w, y+h], [x, y+h]
                    ], dtype=np.float32)
                    break 
        
        # Update Stabilizer (Circle)
        if raw_circle_box is not None:
            stable_circle = self.circle_stabilizer.update(raw_circle_box)
        elif self.circle_stabilizer.locked_coords is not None:
            stable_circle = self.circle_stabilizer.locked_coords
        else:
            stable_circle = None
            
        if stable_circle is not None:
            # Reconstruct circle from stabilized box
            # x_min = min x, y_min = min y
            tl = stable_circle[0] # roughly top-left after ordering
            # To be safe, re-calculate bbox from stable corners
            x, y, w, h = cv2.boundingRect(stable_circle.astype(int))
            cx, cy = x + w//2, y + h//2
            r = max(w, h) // 2
            
            x1, y1 = max(0, cx-r), max(0, cy-r)
            x2, y2 = min(frame.shape[1], cx+r), min(frame.shape[0], cy+r)
            crops["circle"] = frame[y1:y2, x1:x2]
            cv2.circle(output_img, (cx, cy), r, self.colors["circle"], 3)
            cv2.circle(occupied_mask, (cx, cy), int(r+5), 255, -1)

        # --- STAGE 4: HOLLOW SQUARE (Stabilized) ---
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 19, 3)
        thresh = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(occupied_mask))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        clusters = cv2.dilate(thresh, np.ones((9,9),np.uint8), iterations=3)
        
        cnts, _ = cv2.findContours(clusters, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        raw_sq_corners = None
        for cnt in cnts:
            if cv2.contourArea(cnt) < 5000: break
            hull = cv2.convexHull(cnt)
            x, y, w, h = cv2.boundingRect(hull)
            if 0.7 < float(w)/h < 1.3:
                # Convert BoundingRect to Box Points
                raw_sq_corners = np.array([
                    [x, y], [x+w, y], [x+w, y+h], [x, y+h]
                ], dtype=np.float32)
                break
        
        # Update Stabilizer (Square)
        if raw_sq_corners is not None:
            stable_sq = self.square_stabilizer.update(raw_sq_corners)
        elif self.square_stabilizer.locked_coords is not None:
            stable_sq = self.square_stabilizer.locked_coords
        else:
            stable_sq = None

        if stable_sq is not None:
            # Warp/Crop square
            warped, box = warp_from_points(frame, stable_sq)
            crops["hollow_square"] = warped
            cv2.drawContours(output_img, [box], 0, self.colors["square"], 3)

        return output_img, crops

# ==========================================
# 6. MAIN LOOP
# ==========================================

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: {input_path}")
        return

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
            status = manager.handle_paper(paper_crop, i+1)
            status_texts.append(status)
            
        if crops["wheel"] is not None:
            status_texts.append(manager.handle_wheel(crops["wheel"]))
            
        if crops["hollow_square"] is not None:
            status_texts.append(manager.handle_container(crops["hollow_square"], "Square"))
            
        if crops["circle"] is not None:
            status_texts.append(manager.handle_container(crops["circle"], "Circle"))

        y_offset = 40
        for text in status_texts:
            cv2.putText(detected_frame, text, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 40

        final_output = np.hstack((frame, detected_frame))
        out.write(final_output)

        preview = cv2.resize(final_output, (0,0), fx=0.4, fy=0.4)
        cv2.imshow("Game Tracker", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    video_path = '/home/jakub/Artificial Intelligence/Studies/Term 5/[CV] Computer Vision/boardgame-detecor/data/vid_2.MOV'
    output_path = 'game_output.MOV'
    
    if os.path.exists(video_path):
        process_video(video_path, output_path)
    else:
        print(f"File not found: {video_path}")