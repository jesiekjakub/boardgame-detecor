import cv2
import numpy as np

# ==========================================
# 1. HELPER FUNCTIONS (Standard Geometry)
# ==========================================

def get_warped_rect(image, contour):
    """
    Warps an image based on minAreaRect.
    Works directly on the full resolution image.
    """
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
    if warped.shape[1] * 0.9 > warped.shape[0]: 
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        
    return warped, box

def warp_from_points(image, pts):
    """Warps image based on 4 points (approxPolyDP)."""
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

# ==========================================
# 2. GAME STATE MANAGER
# ==========================================

class GameStateManager:
    def __init__(self):
        self.scores = {} 

    def handle_paper(self, crop, paper_id):
        # TODO: Implement token/money counting logic here
        return f"P{paper_id}: 0"

    def handle_wheel(self, crop):
        return "Wheel: Active"

    def handle_container(self, crop, type_name):
        return f"{type_name}: 0"

# ==========================================
# 3. MAIN DETECTOR (Full Resolution)
# ==========================================

class BoardDetector:
    def __init__(self):
        self.colors = {
            "paper": (0, 0, 255), "wheel": (0, 255, 255),
            "circle": (0, 165, 255), "square": (255, 0, 0)
        }

    def detect_elements(self, frame):
        """
        Runs detection on the FULL resolution frame.
        Maximum accuracy, lower speed.
        """
        output_img = frame.copy()
        crops = {"papers": [], "wheel": None, "hollow_square": None, "circle": None}
        
        # Working on the original frame directly
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
        for cnt in cnts:
            if cv2.contourArea(cnt) > 5000: # Threshold for full resolution
                hull = cv2.convexHull(cnt)
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    d1 = np.linalg.norm(pts[0]-pts[1]); d2 = np.linalg.norm(pts[1]-pts[2])
                    ar = max(d1, d2) / min(d1, d2) if min(d1, d2) > 0 else 0
                    if 1.3 < ar < 1.6:
                        warped, box = get_warped_rect(frame, hull)
                        crops["papers"].append(warped)
                        cv2.drawContours(output_img, [box], 0, self.colors["paper"], 2)
                        cv2.drawContours(occupied_mask, [hull], -1, 255, -1)

        # --- STAGE 2: WHEEL ---
        mask_wheel = cv2.inRange(hsv, (20, 70, 80), (35, 255, 255)) | cv2.inRange(hsv, (95, 20, 80), (125, 255, 255))
        mask_wheel = cv2.bitwise_and(mask_wheel, mask_wheel, mask=cv2.bitwise_not(occupied_mask))
        mask_wheel = cv2.morphologyEx(mask_wheel, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        
        pts = cv2.findNonZero(mask_wheel)
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
                    warped, box = warp_from_points(frame, approx)
                    cv2.drawContours(output_img, [box], 0, self.colors["wheel"], 3)
                else:
                    warped, box = get_warped_rect(frame, hull)
                    cv2.drawContours(output_img, [box], 0, self.colors["wheel"], 3)
                crops["wheel"] = warped
                cv2.drawContours(occupied_mask, [hull], -1, 255, -1)

        occupied_mask = cv2.dilate(occupied_mask, np.ones((15,15),np.uint8), iterations=2)

        # --- STAGE 3: HOLLOW CIRCLE (Hough) ---
        gray_med = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(gray_med, cv2.HOUGH_GRADIENT, dp=1.5, minDist=100,
                                   param1=50, param2=30, minRadius=40, maxRadius=150)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                if occupied_mask[cy, cx] == 0:
                    x1, y1 = max(0, cx-r), max(0, cy-r)
                    x2, y2 = min(frame.shape[1], cx+r), min(frame.shape[0], cy+r)
                    
                    crops["circle"] = frame[y1:y2, x1:x2]
                    cv2.circle(output_img, (cx, cy), r, self.colors["circle"], 3)
                    cv2.circle(occupied_mask, (cx, cy), int(r+5), 255, -1) 
                    break 

        # --- STAGE 4: HOLLOW SQUARE ---
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 19, 3)
        thresh = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(occupied_mask))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        clusters = cv2.dilate(thresh, np.ones((9,9),np.uint8), iterations=3)
        
        cnts, _ = cv2.findContours(clusters, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        for cnt in cnts:
            if cv2.contourArea(cnt) < 5000: break
            hull = cv2.convexHull(cnt)
            x, y, w, h = cv2.boundingRect(hull)
            if 0.7 < float(w)/h < 1.3:
                crops["hollow_square"] = frame[y:y+h, x:x+w]
                cv2.rectangle(output_img, (x, y), (x+w, y+h), self.colors["square"], 3)
                break

        return output_img, crops

# ==========================================
# 4. MAIN VIDEO LOOP
# ==========================================

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # .MOV Codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    # NO optimization params passed. Runs at full scale.
    detector = BoardDetector()
    manager = GameStateManager()
    
    print(f"Processing video: {width}x{height} @ {fps}fps")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detect Elements (Full Res)
        detected_frame, crops = detector.detect_elements(frame)
        
        # 2. Analyze Elements
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

        # 3. Draw Results
        y_offset = 40
        for text in status_texts:
            cv2.putText(detected_frame, text, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 40

        # 4. Save & Display
        final_output = np.hstack((frame, detected_frame))
        out.write(final_output)

        preview = cv2.resize(final_output, (0,0), fx=0.4, fy=0.4)
        cv2.imshow("Game Tracker", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

# --- Run ---
if __name__ == "__main__":
    # Update to your .MOV file path
    video_path = '/home/jakub/Artificial Intelligence/Studies/Term 5/[CV] Computer Vision/boardgame-detecor/data/vid_1.MOV'
    output_path = 'game_output.MOV'
    
    import os
    if os.path.exists(video_path):
        process_video(video_path, output_path)
    else:
        print(f"File not found: {video_path}")