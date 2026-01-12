import cv2
import numpy as np
from collections import deque
import os

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def get_warped_rect(image, contour):
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

def warp_from_points(image, pts):
    sorted_pts = order_points(pts)
    (tl, tr, br, bl) = sorted_pts
    width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(sorted_pts, dst), (width, height))
    return warped, sorted_pts.astype(int)

# ==========================================
# 2. STABILIZER
# ==========================================

class CornerStabilizer:
    def __init__(self, history_len=30, stable_threshold=5.0):
        self.history = deque(maxlen=history_len)
        self.locked_coords = None
        self.threshold = stable_threshold
        self.history_len = history_len

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
# 3. WHEEL ANALYZER (Improved Needle Detection)
# ==========================================

class WheelAnalyzer:
    def __init__(self):
        # Color Ranges (Hue 0-179)
        self.color_ranges = {
            "Red_1":   ((0, 70, 70), (10, 255, 255)),   
            "Red_2":   ((170, 70, 70), (180, 255, 255)), 
            "Yellow":  ((20, 70, 70), (35, 255, 255)), 
            "Green":   ((36, 70, 70), (90, 255, 255)),
            "Blue":    ((95, 70, 70), (130, 255, 255)),
        }
        
        self.history_len = 15
        self.color_history = deque(maxlen=self.history_len)
        self.current_state = "IDLE" 

    def find_outermost_tip(self, hsv_crop):
        h, w = hsv_crop.shape[:2]
        center = np.array([w // 2, h // 2])
        
        # 1. Mask Black (Needle)
        # Increased Value limit to 110 to catch reflective black plastic
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 150]) 
        mask = cv2.inRange(hsv_crop, lower_black, upper_black)
        
        # FIX: Replaced OPEN with CLOSE. 
        # OPEN erodes (shrinks) which kills the thin tip. CLOSE fills gaps.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None, None
        
        # 2. Smart Contour Selection
        # Filter for contours that actually cross the center area (avoid border shadows)
        valid_cnts = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < 50: continue
            # Check if contour is somewhat close to center
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                dist_to_center = np.linalg.norm(np.array([cX, cY]) - center)
                # If the blob is within 30% of the center, it's likely the needle
                if dist_to_center < w * 0.3:
                    valid_cnts.append(cnt)
        
        if not valid_cnts:
            # Fallback: Just take largest
            needle_cnt = max(cnts, key=cv2.contourArea)
        else:
            needle_cnt = max(valid_cnts, key=cv2.contourArea)

        if cv2.contourArea(needle_cnt) < 50: return None, None 

        # 3. Find Outermost Point
        points = needle_cnt.squeeze()
        if points.ndim == 1: points = points.reshape(-1, 2)
        
        dists = np.linalg.norm(points - center, axis=1)
        max_idx = np.argmax(dists)
        outermost_tip = points[max_idx]
        
        return outermost_tip, needle_cnt

    def get_color_by_voting(self, hsv_crop, tip, debug_img=None):
        h, w = hsv_crop.shape[:2]
        
        # Define ROI
        radius = 20
        mask_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_roi, tuple(tip), radius, 255, -1)
        
        # Invalid pixels (Needle + Grey Background)
        lower_black = np.array([0, 0, 0]); upper_black = np.array([180, 255, 120]) 
        mask_black = cv2.inRange(hsv_crop, lower_black, upper_black)
        
        lower_grey = np.array([0, 0, 0]); upper_grey = np.array([180, 50, 255])
        mask_grey = cv2.inRange(hsv_crop, lower_grey, upper_grey)
        
        mask_invalid = cv2.bitwise_or(mask_black, mask_grey)
        mask_valid = cv2.bitwise_and(mask_roi, cv2.bitwise_not(mask_invalid))
        
        # --- DEBUG: Show HSV Stats ---
        if debug_img is not None:
            # Calculate mean HSV of VALID pixels only
            valid_pixels = hsv_crop[mask_valid > 0]
            
            if valid_pixels.size > 0:
                mean_hsv = np.mean(valid_pixels, axis=0)
                h_val, s_val, v_val = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])
                stat_text = f"HSV: {h_val} {s_val} {v_val}"
                
                # Draw Box and Text for visibility
                cv2.rectangle(debug_img, (0, h-30), (160, h), (0,0,0), -1)
                cv2.putText(debug_img, stat_text, (5, h - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(debug_img, "HSV: No Valid Pixels", (5, h - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw ROI circle
            cv2.circle(debug_img, tuple(tip), radius, (255, 255, 255), 1)

        # Vote
        votes = {"Red": 0, "Yellow": 0, "Green": 0, "Blue": 0}
        
        for color_name, (lower, upper) in self.color_ranges.items():
            if color_name == "Red_2": continue 
            target_name = "Red" if "Red" in color_name else color_name
            mask_color = cv2.inRange(hsv_crop, np.array(lower), np.array(upper))
            mask_final = cv2.bitwise_and(mask_valid, mask_color)
            votes[target_name] += cv2.countNonZero(mask_final)

        total_votes = sum(votes.values())
        if total_votes < 5: return "Unknown"
        
        return max(votes, key=votes.get)

    def analyze(self, wheel_crop, debug_vis_img=None):
        if wheel_crop is None or wheel_crop.size == 0: return "No Wheel", "N/A"

        hsv = cv2.cvtColor(wheel_crop, cv2.COLOR_BGR2HSV)
        
        tip, needle_cnt = self.find_outermost_tip(hsv)
        current_color = "Unknown"
        
        if tip is not None:
            # Draw Needle Contour on Debug
            if debug_vis_img is not None:
                cv2.drawContours(debug_vis_img, [needle_cnt], -1, (255, 0, 255), 2)
                # Draw Tip
                cv2.circle(debug_vis_img, tuple(tip), 3, (0, 0, 255), -1)
            
            current_color = self.get_color_by_voting(hsv, tip, debug_vis_img)
            
            if debug_vis_img is not None:
                cv2.putText(debug_vis_img, f"Res: {current_color}", (5, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        self.color_history.append(current_color)

        if len(self.color_history) < self.history_len:
            return "Initializing...", "N/A"

        valid_colors = [c for c in self.color_history if c not in ["Unknown"]]
        if len(valid_colors) < 5: return "Searching...", "N/A"
            
        unique_colors = set(valid_colors)
        if len(unique_colors) > 1:
            self.current_state = "SPINNING"
            return "SPINNING", "..."
        else:
            stable_col = valid_colors[-1]
            if self.current_state == "SPINNING":
                self.current_state = "STOPPED"
                return "STOPPED", stable_col
            else:
                return "IDLE", stable_col

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
        state, color = self.wheel_analyzer.analyze(crop, debug_vis)
        
        # --- SEPARATE WINDOW ---
        vis_large = cv2.resize(debug_vis, (300, 300))
        cv2.imshow("Wheel Analysis Debug", vis_large)
        
        status_text = f"Wheel: {state}"
        if state in ["STOPPED", "IDLE"]:
            status_text += f": {color}"
        return status_text

    def handle_container(self, crop, type_name):
        return f"{type_name}: 0"

# ==========================================
# 5. BOARD DETECTOR (Unchanged)
# ==========================================

class BoardDetector:
    def __init__(self):
        self.colors = {"paper": (0, 0, 255), "wheel": (0, 255, 255), "circle": (0, 165, 255), "square": (255, 0, 0)}
        self.wheel_stabilizer = CornerStabilizer(history_len=30, stable_threshold=4.0)
        self.circle_stabilizer = CornerStabilizer(history_len=30, stable_threshold=4.0)
        self.square_stabilizer = CornerStabilizer(history_len=30, stable_threshold=4.0)
        self.paper_stabilizers = {} 

    def detect_elements(self, frame):
        output_img = frame.copy()
        crops = {"papers": [], "wheel": None, "hollow_square": None, "circle": None}
        occupied_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        v = np.median(blur)
        lower = int(max(0, 0.67 * v)); upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blur, lower, upper)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                        x, y, _, _ = cv2.boundingRect(hull)
                        paper_candidates.append((x, hull))
        paper_candidates.sort(key=lambda x: x[0])
        for i, (_, hull) in enumerate(paper_candidates):
            if i not in self.paper_stabilizers: self.paper_stabilizers[i] = CornerStabilizer()
            rect = cv2.minAreaRect(hull); raw_corners = cv2.boxPoints(rect)
            stable_corners = self.paper_stabilizers[i].update(raw_corners)
            warped, box = warp_from_points(frame, stable_corners)
            crops["papers"].append(warped)
            cv2.drawContours(output_img, [box], 0, self.colors["paper"], 2)
            cv2.drawContours(occupied_mask, [box], -1, 255, -1)

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
                if len(approx) == 4: raw_corners = approx.reshape(4, 2)
                else: rect = cv2.minAreaRect(hull); raw_corners = cv2.boxPoints(rect)
        if raw_corners is not None: stable_corners = self.wheel_stabilizer.update(raw_corners)
        elif self.wheel_stabilizer.locked_coords is not None: stable_corners = self.wheel_stabilizer.locked_coords
        else: stable_corners = None
        if stable_corners is not None:
            warped, box = warp_from_points(frame, stable_corners)
            crops["wheel"] = warped
            is_locked = (self.wheel_stabilizer.locked_coords is not None)
            color = self.colors["wheel"] if is_locked else (0, 255, 255) 
            cv2.drawContours(output_img, [box], 0, color, 3 if is_locked else 1)
            cv2.drawContours(occupied_mask, [box], -1, 255, -1)
        occupied_mask = cv2.dilate(occupied_mask, np.ones((15,15),np.uint8), iterations=2)

        gray_med = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(gray_med, cv2.HOUGH_GRADIENT, dp=1.5, minDist=100,
                                   param1=50, param2=30, minRadius=40, maxRadius=150)
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
            cx, cy, r = x + w//2, y + h//2, max(w, h) // 2
            x1, y1 = max(0, cx-r), max(0, cy-r)
            x2, y2 = min(frame.shape[1], cx+r), min(frame.shape[0], cy+r)
            crops["circle"] = frame[y1:y2, x1:x2]
            cv2.circle(output_img, (cx, cy), r, self.colors["circle"], 3)
            cv2.circle(occupied_mask, (cx, cy), int(r+5), 255, -1)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 3)
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
# 6. MAIN LOOP
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
            status_texts.append(manager.handle_paper(paper_crop, i+1))
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
    # Update this path to your local video
    video_path = '/home/jakub/Artificial Intelligence/Studies/Term 5/[CV] Computer Vision/boardgame-detecor/data/vid_3.MOV'
    output_path = 'game_output.MOV'
    if os.path.exists(video_path): process_video(video_path, output_path)
    else: print(f"File not found: {video_path}")