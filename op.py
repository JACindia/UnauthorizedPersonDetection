#!/usr/bin/env python3
"""
===============================================================================
Professional Employee Face Verification System v4.1 - GENDER/AGE REMOVED
NPY-ONLY EMBEDDINGS | FIXED PERSON BBOX | DETAILED QUALITY FILTERS | FACE BOX ALERTS
Production-Ready with Robust Error Handling, Professional Logging & Tracking
===============================================================================
"""

import asyncio
import io
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Generator

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from telegram import Bot
from telegram.error import RetryAfter, TelegramError
from scipy.spatial import distance as dist

# =============================================================================
# TRACKING CONSTANTS
# =============================================================================
MAX_DISAPPEARED = 50
MIN_HITS = 3
IOU_THRESH = 0.3
DIST_THRESH = 100
MAX_MISSED_FRAMES = 100

# =============================================================================
# PROFESSIONAL CONFIGURATION
# =============================================================================

# Environment Setup
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1|probesize;32|analyzeduration;0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Core Paths and Credentials
EMPLOYEE_EMBEDDINGS_FOLDER = Path("embeddings")
TELEGRAM_TOKEN = "8204810755:AAFy1qt1ap44Nq6CjibuSuXMJXYBUtkkxeA"
TELEGRAM_CHAT_ID = "-5160513799"

# Operational Modes
TEST_MODE = True  # Set False for production deployment
RTSP_STREAMS = [
    {"name": "Camera 1", "url": "rtsp://admin:Jezt@884@192.168.74.114/cam/realmonitor?channel=1&subtype=0"},
    {"name": "Camera 2", "url": "rtsp://admin:Jezt@884@192.168.74.112/cam/realmonitor?channel=1&subtype=0"},
]

# Detection Thresholds (Production-Optimized)
SIMILARITY_THRESHOLD = 0.42
FACE_DET_TH = 0.50
PERSON_CONF_TH = 0.45
PERSON_CLASS_ID = 0  # YOLO 'person' class
MODEL_NAME = "buffalo_l"
FACE_CTX_ID = 0
FACE_DET_SIZE = (640, 640)

# Processing Parameters
FRAME_SKIP = 5
ALERT_COOLDOWN = 120  # 2 minutes between alerts per stream
SEND_DELAY = 1.0
STATS_INTERVAL = 300  # 5 minutes
STATUS_UPDATE_INTERVAL = 1800  # 30 minutes
WEBCAM_WARMUP = 3.0

# Visual Configuration
ALERT_FACE_COLOR = (0, 0, 255)      # Red for unauthorized face
ALERT_PERSON_COLOR = (0, 165, 255)  # Orange for person bbox
CONFIRMED_TRACK_COLOR = (0, 255, 0) # Green for confirmed tracks
BOX_THICKNESS = 3

# =============================================================================
# SIMPLIFIED TRACK CLASS (GENDER/AGE REMOVED)
# =============================================================================
class Track:
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.disappeared = 0
        self.hits = 1
        self.confirmed = False
        self.best = None  # Best face crop
        self.area = 0
        self.missed = 0

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.box
        return int((x1+x2)/2), int((y1+y2)/2)

    def update_crop(self, frm):
        x1, y1, x2, y2 = map(int, self.box)
        h, w = frm.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        a = (x2 - x1) * (y2 - y1)
        if a > self.area:
            self.area = a
            self.best = frm[y1:y2, x1:x2].copy()

# =============================================================================
# SIMPLIFIED TRACKER CLASS (GENDER/AGE REMOVED)
# =============================================================================
class ImprovedTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED, min_hits=MIN_HITS,
                 iou_thresh=IOU_THRESH, dist_thresh=DIST_THRESH):
        self.nextID = 0
        self.tracks = OrderedDict()
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh

    def register(self, box):
        self.tracks[self.nextID] = Track(self.nextID, box)
        self.nextID += 1

    def deregister(self, tid):
        if tid in self.tracks:
            del self.tracks[tid]

    def iou(self, boxA, boxB):
        """Compute Intersection over Union (IoU)"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, frm, dets, fid, fps, video_name):
        if len(dets) == 0:
            for t in list(self.tracks.values()):
                t.disappeared += 1
                if t.disappeared > self.max_disappeared:
                    yield t  # Yield for finalize
                    self.deregister(t.track_id)
            return

        if len(self.tracks) == 0:
            for det in dets:
                self.register(det["box"])
            return

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].box for tid in track_ids]

        # IoU matrix
        iou_mat = np.zeros((len(track_boxes), len(dets)), dtype=np.float32)
        for i, tb in enumerate(track_boxes):
            for j, d in enumerate(dets):
                iou_mat[i, j] = self.iou(tb, d["box"])

        assigned_tracks = set()
        assigned_dets = set()

        # First pass: IoU assignment
        for i in range(len(track_boxes)):
            j = np.argmax(iou_mat[i])
            if iou_mat[i, j] >= self.iou_thresh and j not in assigned_dets:
                t = self.tracks[track_ids[i]]
                t.box = dets[j]["box"]
                t.disappeared = 0
                t.hits += 1
                if t.hits >= self.min_hits:
                    t.confirmed = True

                t.update_crop(frm)
                t.missed = 0

                assigned_tracks.add(i)
                assigned_dets.add(j)

        # Second pass: Distance for unassigned tracks
        for i in range(len(track_boxes)):
            if i in assigned_tracks:
                continue
            t = self.tracks[track_ids[i]]
            t_cent = t.centroid
            best_j = -1
            best_d = self.dist_thresh
            for j, d in enumerate(dets):
                if j in assigned_dets:
                    continue
                cX = (d["box"][0]+d["box"][2])//2
                cY = (d["box"][1]+d["box"][3])//2
                d_dist = dist.euclidean(t_cent, (cX, cY))
                if d_dist < best_d:
                    best_d = d_dist
                    best_j = j

            if best_j != -1:
                d = dets[best_j]
                t.box = d["box"]
                t.disappeared = 0
                t.hits += 1
                if t.hits >= self.min_hits:
                    t.confirmed = True

                t.update_crop(frm)
                t.missed = 0

                assigned_tracks.add(i)
                assigned_dets.add(best_j)
            else:
                t.disappeared += 1
                t.missed += 1
                if t.disappeared > self.max_disappeared or t.missed > MAX_MISSED_FRAMES:
                    yield t
                    self.deregister(t.track_id)

        # Register new detections
        for j, d in enumerate(dets):
            if j not in assigned_dets:
                t = Track(self.nextID, d["box"])
                t.update_crop(frm)
                self.tracks[self.nextID] = t
                self.nextID += 1

        yield from self.tracks.values()

# =============================================================================
# PROFESSIONAL LOGGING SETUP
# =============================================================================
def setup_logging():
    """Initialize professional-grade logging with file rotation and UTF-8 support."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s",
        handlers=[
            logging.FileHandler("face_recognition.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# ROBUST QUALITY ASSESSMENT ENGINE
# =============================================================================
def safe_getattr(obj: Any, attr: str, default: float = 1000.0) -> float:
    """Safely extract float attributes with comprehensive error handling."""
    try:
        value = getattr(obj, attr, default)
        return float(value) if value is not None else default
    except (ValueError, TypeError, AttributeError):
        return default

def quality_pass(face_metrics: Dict[str, float]) -> Tuple[bool, str]:
    """
    Professional quality validation with detailed rejection reasons.
    All checks use validated float values with fallback defaults.
    """
    f = face_metrics
    
    # 1. High blur threshold with compensation checks
    if f["blur"] > 2656:
        if not (f["face_brightness"] >= 90 and 
                abs(f["yaw"]) <= 20 and abs(f["pitch"]) <= 20 and abs(f["roll"]) <= 15):
            return False, f"High blur ({f['blur']:.0f}) - requires bright frontal face"
        return True, "High blur compensated"
    
    # 2. Low brightness with size/pose compensation
    if f["face_brightness"] < 56:
        if not (f["face_size_ratio"] >= 0.01 and abs(f["yaw"]) <= 15 and f["brightness_diff"] >= -15):
            return False, f"Low brightness ({f['face_brightness']:.0f}) - needs larger frontal face"
        return True, "Low brightness compensated"
    
    # 3. Brightness difference validation
    if f["brightness_diff"] < -23:
        if not (f["blur"] <= 1500 and f["det_score"] >= 0.85):
            return False, f"Poor brightness diff ({f['brightness_diff']:.1f}) - needs sharp detection"
        return True, "Brightness diff compensated"
    
    # 4. Head pose limits with brightness compensation
    if abs(f["yaw"]) > 25 or abs(f["pitch"]) > 25:
        if f["face_brightness"] < 110:
            return False, f"Extreme pose (Yaw:{f['yaw']:.1f}, Pitch:{f['pitch']:.1f}) - needs bright face"
        return True, "Pose compensated by brightness"
    
    # 5. Minimum face size validation
    if f["face_size_ratio"] < 0.004:
        if not (f["blur"] <= 1200 and f["face_brightness"] >= 100):
            return False, f"Face too small ({f['face_size_ratio']:.4f}) - needs sharp bright face"
        return True, "Small face compensated"
    
    # 6. Detection confidence (relaxed for production reliability)
    if f["det_score"] < 0.77:
        if not (f["face_brightness"] > 80 and f["blur"] < 2000):
            return False, f"Low confidence ({f['det_score']:.3f}) - needs brighter sharper image"
        return True, "Low confidence compensated"
    
    return True, "All quality checks passed"

# =============================================================================
# PRODUCTION MODEL INITIALIZATION
# =============================================================================
class YOLOPersonDetector:
    """Robust YOLO person detector with GPU acceleration and error recovery."""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            self.model = YOLO("yolov8n.pt").to('cuda')
            logger.info("✓ YOLOv8 person detector loaded (GPU)")
        except Exception as e:
            logger.error(f"✗ YOLO initialization failed: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect persons with expanded bounding boxes for reliable face capture."""
        if not self.model:
            return []
        
        try:
            h, w = frame.shape[:2]
            results = self.model(frame, verbose=False, conf=PERSON_CONF_TH)
            
            persons = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if int(box.cls[0]) == PERSON_CLASS_ID:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf[0].cpu().numpy())
                            
                            # Professional expansion for complete face coverage
                            expand_factor = 0.25
                            persons.append((
                                max(0, int(x1 * (1 - expand_factor))),
                                max(0, int(y1 * (1 - expand_factor))),
                                min(w, int(x2 * (1 + expand_factor))),
                                min(h, int(y2 * (1 + expand_factor))),
                                conf
                            ))
            return persons
        except Exception as e:
            logger.warning(f"YOLO detection error: {e}")
            return []

class FaceRecognitionEngine:
    """Production-grade face recognition with NPY-only embedding support."""
    
    def __init__(self):
        self.app = None
        self._initialize_face_model()
    
    def _initialize_face_model(self):
        """Initialize InsightFace with GPU providers."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(f"CUDAExecutionProvider unavailable: {providers}")
            
            self.app = FaceAnalysis(
                name=MODEL_NAME,
                providers=["CUDAExecutionProvider"],
                allowed_modules=["detection", "recognition"]
            )
            self.app.prepare(ctx_id=FACE_CTX_ID, det_size=FACE_DET_SIZE)
            logger.info("✓ InsightFace loaded (GPU) - Ready for recognition")
        except Exception as e:
            logger.error(f"✗ Face recognition initialization failed: {e}")
            self.app = None
    
    def detect_faces_in_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List:
        """Detect faces within person bounding box region."""
        if not self.app:
            return []
        
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return []
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        
        try:
            faces = self.app.get(crop, max_num=2)
            return [f for f in faces if safe_getattr(f, "det_score", 0.0) >= FACE_DET_TH]
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            return []
    
    def load_employees_from_npy(self, folder: Path) -> List[np.ndarray]:
        """Load employee embeddings exclusively from .npy files (512-dim vectors)."""
        if not folder.exists():
            logger.warning(f"No embeddings folder: {folder}")
            return []
        
        embeddings = []
        npy_files = list(folder.glob("*.npy")) + list(folder.glob("*.NPY"))
        
        if not npy_files:
            logger.warning(f"No .npy embedding files found in: {folder}")
            return []
        
        for npy_path in npy_files:
            try:
                emb_array = np.load(str(npy_path))
                # Validate InsightFace 512-dim embedding
                if emb_array.ndim == 1 and len(emb_array) == 512:
                    embeddings.append(emb_array)
                    logger.info(f"✓ Loaded: {npy_path.name}")
                else:
                    logger.warning(f"Invalid shape {npy_path.name}: {emb_array.shape}")
            except Exception as e:
                logger.error(f"Failed to load {npy_path.name}: {e}")
        
        logger.info(f"✅ Loaded {len(embeddings)} employee embeddings")
        return embeddings
    
    def load_employees(self, npy_folder: Path) -> List[np.ndarray]:
        """Primary employee loading method - NPY only."""
        return self.load_employees_from_npy(npy_folder)
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def is_authorized(self, embedding: np.ndarray, employees: List[np.ndarray]) -> Tuple[bool, float]:
        """Check if face embedding matches any employee."""
        if not employees:
            return False, 0.0
        
        similarities = [self.cosine_similarity(embedding, emp) for emp in employees]
        max_sim = max(similarities)
        return max_sim >= SIMILARITY_THRESHOLD, max_sim

# =============================================================================
# ENHANCED FACE VALIDATION WITH PERSON BBOX CONSTRAINTS
# =============================================================================
def validate_face(frame: np.ndarray, face: Any, person_bbox: Tuple[int, int, int, int]) -> Tuple[bool, str]:
    """
    Professional face validation with strict person bbox containment check.
    """
    # Extract and validate face bbox
    bbox = getattr(face, "bbox", None)
    if bbox is None or len(bbox) != 4:
        return False, "No valid face bbox"
    
    try:
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            return False, "Invalid face bbox dimensions"
    except (ValueError, TypeError):
        return False, "Malformed face bbox"
    
    h_img, w_img = frame.shape[:2]
    
    # Minimum face size check
    face_w, face_h = x2 - x1, y2 - y1
    if face_w < 35 or face_h < 35 or face_w * face_h < 1200:
        return False, f"Face too small ({face_w}x{face_h})"
    
    # FIXED PERSON BBOX VALIDATION - Proper expansion and bounds checking
    px1, py1, px2, py2 = person_bbox
    expand_x1, expand_y1 = px1 * 0.75, py1 * 0.75  # 25% inward expansion tolerance
    expand_x2, expand_y2 = px2 * 1.25, py2 * 1.25  # 25% outward expansion tolerance
    
    # Ensure face bbox is fully contained within expanded person bbox
    if not (expand_x1 <= x1 <= x2 <= expand_x2 and 
            expand_y1 <= y1 <= y2 <= expand_y2):
        return False, "Face outside person bounding box"
    
    # Comprehensive quality metrics
    metrics = {
        'blur': safe_getattr(face, 'blur', 1000),
        'face_brightness': safe_getattr(face, 'brightness', 100),
        'brightness_diff': safe_getattr(face, 'brightness_diff', 0),
        'yaw': safe_getattr(face, 'yaw', 0),
        'pitch': safe_getattr(face, 'pitch', 0),
        'roll': safe_getattr(face, 'roll', 0),
        'face_size_ratio': (face_w * face_h) / (w_img * h_img),
        'det_score': safe_getattr(face, 'det_score', 0.0)
    }
    
    return quality_pass(metrics)

# =============================================================================
# PROFESSIONAL ALERT VISUALIZATION WITH TRACKING
# =============================================================================
def draw_alert_frame(frame: np.ndarray, person_bbox: Tuple[int, int, int, int], 
                    face_bbox: Tuple[int, int, int, int], sim_score: float, 
                    track_id: Optional[int] = None) -> np.ndarray:
    """Create professional alert frame with clear color-coded bounding boxes."""
    alert_frame = frame.copy()
    
    # Person bounding box (Orange, thick)
    px1, py1, px2, py2 = map(int, person_bbox)
    cv2.rectangle(alert_frame, (px1, py1), (px2, py2), ALERT_PERSON_COLOR, BOX_THICKNESS)
    track_text = f"TRACK #{track_id}" if track_id else "PERSON DETECTED"
    cv2.putText(alert_frame, track_text, (px1, py1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ALERT_PERSON_COLOR, 2)
    
    # Unauthorized face bounding box (Red, extra thick)
    fx1, fy1, fx2, fy2 = map(int, face_bbox)
    cv2.rectangle(alert_frame, (fx1, fy1), (fx2, fy2), ALERT_FACE_COLOR, BOX_THICKNESS+1)
    cv2.putText(alert_frame, f"UNAUTHORIZED ({sim_score:.3f})", (fx1, fy1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ALERT_FACE_COLOR, 2)
    
    # Alert timestamp and status
    h, w = alert_frame.shape[:2]
    cv2.putText(alert_frame, f"🚨 SECURITY ALERT - {time.strftime('%H:%M:%S IST')}", 
                (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return alert_frame

def draw_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    """Draw all tracks with different colors based on status."""
    display_frame = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.box)
        
        # Color based on track status
        if track.confirmed:
            color = CONFIRMED_TRACK_COLOR  # Green for confirmed
        else:
            color = (255, 0, 255)  # Magenta for new tracks
        
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"ID:{track.track_id} H:{track.hits}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return display_frame

# =============================================================================
# PRODUCTION TELEGRAM NOTIFIER
# =============================================================================
class TelegramNotifier:
    """Reliable Telegram integration with rate limiting and retry logic."""
    
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self.last_status_time = 0
        self.last_important_time = 0
    
    async def _safe_send(self, method, **kwargs):
        """Send message with comprehensive error handling and rate limiting."""
        try:
            if method == self.bot.send_message:
                await method(chat_id=self.chat_id, **kwargs)
            elif method == self.bot.send_photo:
                await method(chat_id=self.chat_id, **kwargs)
            await asyncio.sleep(SEND_DELAY)
        except RetryAfter as e:
            logger.warning(f"Telegram rate limit: {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
            await method(chat_id=self.chat_id, **kwargs)
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
        except Exception as e:
            logger.error(f"Notification failed: {e}")
    
    async def send_status(self, message: str, is_important: bool = False):
        current_time = time.time()
        cooldown = 300 if not is_important else 60
        
        if not is_important and (current_time - self.last_status_time) < cooldown:
            return
        
        try:
            await self._safe_send(self.bot.send_message, text=message)
            self.last_status_time = current_time if not is_important else self.last_important_time
            logger.info(f"📱 Status sent: {message[:50]}...")
        except Exception as e:
            logger.error(f"Status send failed: {e}")
    
    async def send_client_status(self, message: str):
        await self.send_status(f"🤖 CLIENT STATUS: {message}", True)
    
    async def send_periodic_summary(self, streams_stats: List[Dict]):
        summary = "📊 SYSTEM SUMMARY\n\n"
        summary += "".join([f"📹 {s['name']}: 👥{s['persons']:3d} 👤{s['faces']:3d} ✅{s['auth']:2d} ❌{s['unauth']:2d} ⚠️{s['reject']:2d}\n" 
                           for s in streams_stats])
        summary += f"\n⏰ {time.strftime('%Y-%m-%d %H:%M:%S IST')}"
        await self.send_status(summary)
    
    async def send_alert(self, frame: np.ndarray, stream_info: Dict[str, str], 
                        person_bbox: Tuple[int, int, int, int], face_bbox: Tuple[int, int, int, int], 
                        sim_score: float, track_id: Optional[int] = None):
        """Send professional security alert with annotated image."""
        try:
            alert_frame = draw_alert_frame(frame, person_bbox, face_bbox, sim_score, track_id)
            
            _, buffer = cv2.imencode('.jpg', alert_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            bio = io.BytesIO(buffer.tobytes())
            bio.name = 'security_alert.jpg'
            
            caption = (f"🚨 UNAUTHORIZED ACCESS DETECTED\n\n"
                      f"📹 {stream_info['name']}\n"
                      f"👤 Similarity Score: {sim_score:.3f} (Threshold: {SIMILARITY_THRESHOLD})\n")
            if track_id:
                caption += f"🎯 Track ID: #{track_id}\n"
            caption += f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"
            caption += f"🔴 Red Box = Unauthorized Face\n🟠 Orange Box = Person Detection"
            
            await self._safe_send(self.bot.send_photo, photo=bio, caption=caption)
            logger.info(f"🚨 Alert sent for {stream_info['name']} (sim: {sim_score:.3f}) {'[Track '+str(track_id)+']' if track_id else ''}")
            
        except Exception as e:
            logger.error(f"Alert transmission failed: {e}")

# =============================================================================
# PRODUCTION STREAM PROCESSOR WITH TRACKING
# =============================================================================
class UltimateStreamProcessor:
    """Production-grade RTSP/Webcam stream processor with tracking and reconnection logic."""
    
    def __init__(self, config: Dict[str, str]):
        self.name = config["name"]
        self.original_url = config["url"]
        self.effective_url = config["url"]
        self.is_webcam = False
        self.cap = None
        self.frame_count = 0
        self.process_count = 0
        self.last_alert = 0.0
        self.stats = {
            'persons_detected': 0, 'faces_detected': 0, 
            'authorized': 0, 'unauthorized': 0, 'quality_rejected': 0,
            'tracks_created': 0, 'tracks_lost': 0
        }
        self.tracker = ImprovedTracker()
        self.frame_id = 0
        self.fps = 30.0
    
    def find_webcam(self) -> Optional[int]:
        """Auto-detect available webcam."""
        logger.info(f"🔍 {self.name}: Scanning webcams...")
        for cam_id in range(6):
            try:
                cap = cv2.VideoCapture(cam_id)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None and frame.size > 0:
                    logger.info(f"✅ {self.name}: Webcam {cam_id} ({frame.shape})")
                    return cam_id
            except Exception:
                continue
        logger.warning(f"⚠️ {self.name}: No webcam found")
        return None
    
    def connect_with_warmup(self) -> bool:
        """Establish robust stream connection with warmup validation."""
        if self.cap:
            self.cap.release()
        
        # Handle webcam auto-detection
        if self.original_url == "0":
            webcam_id = self.find_webcam()
            if webcam_id is None:
                return False
            self.effective_url = str(webcam_id)
            self.is_webcam = True
        
        # Initialize capture
        self.cap = cv2.VideoCapture(int(self.effective_url) if self.is_webcam else self.effective_url)
        if not self.cap.isOpened():
            return False
        
        # Production capture settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        if self.is_webcam:
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Connection warmup
        warmup_start = time.time()
        warmup_frames = 0
        while time.time() - warmup_start < WEBCAM_WARMUP:
            ret, _ = self.cap.read()
            if ret:
                warmup_frames += 1
            time.sleep(0.05)
        
        success = self.cap.isOpened() and warmup_frames >= 3
        status = "✅" if success else "❌"
        logger.info(f"{status} {self.name} connected | Warmup: {warmup_frames} frames")
        return success
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame with automatic reconnection."""
        if not self.cap or not self.cap.isOpened():
            if not self.connect_with_warmup():
                return None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.connect_with_warmup()
            return None
        
        self.frame_count += 1
        
        # TEST_MODE visualization
        if TEST_MODE:
            cv2.imshow(f"{self.name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)
        
        return frame
    
    def should_process(self) -> bool:
        """Frame skip logic for performance optimization."""
        self.process_count += 1
        return self.process_count % FRAME_SKIP == 0
    
    def can_alert(self) -> bool:
        """Alert cooldown enforcement."""
        return time.time() - self.last_alert >= ALERT_COOLDOWN
    
    def log_person_detection(self, person_count: int, conf_scores: List[float], tracks_count: int):
        """Log person detection statistics with tracking info."""
        conf_avg = np.mean(conf_scores) if conf_scores else 0.0
        logger.info(f"{self.name:12s} | 👥 Persons: {person_count} | Tracks: {tracks_count} | Avg Conf: {conf_avg:.2f}")
    
    def log_face_result(self, result: str, sim_score: float = 0.0):
        """Log comprehensive face recognition results."""
        self.stats['faces_detected'] += 1
        if result == "AUTHORIZED":
            self.stats['authorized'] += 1
        elif result == "UNAUTHORIZED":
            self.stats['unauthorized'] += 1
        elif result == "QUALITY_REJECT":
            self.stats['quality_rejected'] += 1
        
        status_icon = "✅" if result == "AUTHORIZED" else "❌" if result == "UNAUTHORIZED" else "⚠️"
        logger.info(f"{self.name:12s} | {status_icon} {result} | Similarity: {sim_score:.3f}")
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Generate JSON-compatible statistics summary."""
        return {
            'name': self.name,
            'persons': self.stats['persons_detected'],
            'faces': self.stats['faces_detected'],
            'auth': self.stats['authorized'],
            'unauth': self.stats['unauthorized'],
            'reject': self.stats['quality_rejected'],
            'tracks': self.stats['tracks_created'],
            'lost': self.stats['tracks_lost']
        }
    
    def get_log_summary(self) -> str:
        """Compact log summary for console display."""
        return (f"{self.name:12s} | 👥{self.stats['persons_detected']:3d} 👤{self.stats['faces_detected']:3d} "
                f"✅{self.stats['authorized']:2d} ❌{self.stats['unauthorized']:2d} ⚠️{self.stats['quality_rejected']:2d} "
                f"📍{self.stats['tracks_created']:2d}")
    
    def cleanup(self):
        """Graceful resource cleanup."""
        if self.cap:
            self.cap.release()
        if TEST_MODE:
            cv2.destroyAllWindows()
        logger.info(f"🧹 {self.name} cleaned up")

# =============================================================================
# PRODUCTION MAIN ORCHESTRATOR WITH TRACKING
# =============================================================================
async def main():
    """Production main loop with comprehensive error handling, monitoring & tracking."""
    logger.info("🚀 Professional Face Verification System v4.1 (NO GENDER/AGE) - Starting...")
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    start_time = time.time()
    
    try:
        # System initialization sequence
        await notifier.send_client_status("🔄 Initializing AI models...")
        
        # Load detection models
        yolo_detector = YOLOPersonDetector()
        face_engine = FaceRecognitionEngine()
        
        # Load employee database
        await notifier.send_client_status("👥 Loading employee database...")
        employees = face_engine.load_employees(EMPLOYEE_EMBEDDINGS_FOLDER)
        if not employees:
            await notifier.send_client_status("❌ No employee embeddings found in 'embeddings/' folder")
            logger.error("No .npy embedding files found. Add 512-dim embeddings to embeddings/ folder.")
            return
        
        # Initialize streams
        await notifier.send_client_status("📹 Connecting to cameras...")
        streams = [UltimateStreamProcessor(config) for config in RTSP_STREAMS]
        active_streams = [s for s in streams if s.connect_with_warmup()]
        
        if not active_streams:
            await notifier.send_client_status("⚠️ No active cameras detected - Enable TEST_MODE for webcam testing")
            return
        
        # System ready notification
        uptime = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        status_msg = f"✅ SYSTEM LIVE | Employees: {len(employees)} | Active Cameras: {len(active_streams)} | Tracking: Enabled | NO GENDER/AGE"
        await notifier.send_client_status(status_msg)
        logger.info(f"✅ PRODUCTION READY | Employees: {len(employees)} | Cameras: {len(active_streams)}")
        logger.info(f"  TEST_MODE: {TEST_MODE} | Press 'q' in preview windows to exit")
        
        # Main processing loop
        frame_counter = 0
        last_status_update = time.time()
        
        while True:
            for stream in streams:
                # Read and process frames
                frame = stream.read_frame()
                if frame is None or not stream.should_process():
                    continue
                
                stream.frame_id += 1
                
                # Person detection
                raw_persons = yolo_detector.detect(frame)
                
                # Prepare detections for tracker (simplified - no gender/age)
                dets = []
                for person in raw_persons:
                    x1, y1, x2, y2, conf = person
                    dets.append({
                        "box": [x1, y1, x2, y2],
                        "conf": conf
                    })
                
                if dets:
                    stream.stats['persons_detected'] += len(dets)
                
                # Update tracker
                tracks = list(stream.tracker.update(frame, dets, stream.frame_id, stream.fps, stream.name))
                lost_tracks = []
                for track in tracks:
                    if track.track_id not in stream.tracker.tracks:
                        lost_tracks.append(track)
                        stream.stats['tracks_lost'] += 1
                
                stream.stats['tracks_created'] = len(stream.tracker.tracks)
                
                if tracks:
                    stream.log_person_detection(len(dets), [d["conf"] for d in dets], len(stream.tracker.tracks))
                
                # TEST_MODE: Draw tracks
                if TEST_MODE and tracks:
                    display_frame = draw_tracks(frame, tracks)
                    cv2.imshow(f"{stream.name} - Tracks", display_frame)
                
                # Face recognition per confirmed track
                for track in tracks:
                    if not track.confirmed:
                        continue
                    
                    person_bbox = track.box
                    faces = face_engine.detect_faces_in_region(frame, tuple(map(int, person_bbox)))
                    
                    for face in faces:
                        # Quality validation
                        is_valid, reject_reason = validate_face(frame, face, tuple(map(int, person_bbox)))
                        if not is_valid:
                            stream.log_face_result("QUALITY_REJECT", 0.0)
                            logger.debug(f"{stream.name} | Track#{track.track_id} | Quality reject: {reject_reason}")
                            continue
                        
                        # Extract embedding for recognition
                        embedding = getattr(face, "embedding", None)
                        if embedding is None:
                            stream.log_face_result("QUALITY_REJECT", 0.0)
                            logger.debug(f"{stream.name} | Track#{track.track_id} | No embedding generated")
                            continue
                        
                        # Authorization check
                        is_auth, max_sim = face_engine.is_authorized(embedding, employees)
                        result = "AUTHORIZED" if is_auth else "UNAUTHORIZED"
                        stream.log_face_result(result, max_sim)
                        
                        # Security alert for unauthorized access (per track)
                        if not is_auth and stream.can_alert():
                            face_bbox = tuple(map(int, getattr(face, "bbox", [0, 0, 0, 0])))
                            await notifier.send_alert(
                                frame, {"name": stream.name, "url": stream.effective_url},
                                tuple(map(int, person_bbox)), face_bbox, max_sim, track.track_id
                            )
                            stream.last_alert = time.time()
                            break  # One alert per cooldown period per stream
            
            # Periodic reporting
            frame_counter += 1
            if frame_counter % STATS_INTERVAL == 0:
                logger.info("\n📈 5-MINUTE SUMMARY:")
                for stream in streams:
                    logger.info(stream.get_log_summary())
                logger.info("-" * 80)
            
            # Status updates
            if time.time() - last_status_update > STATUS_UPDATE_INTERVAL:
                active_summaries = [s.get_stats_summary() for s in streams if s.frame_count > 0]
                if active_summaries:
                    await notifier.send_periodic_summary(active_summaries)
                last_status_update = time.time()
            
            await asyncio.sleep(0.01)  # Yield control
            
    except KeyboardInterrupt:
        logger.info("⏹️  Graceful shutdown initiated...")
    except Exception as e:
        logger.error(f"💥 Critical error: {e}", exc_info=True)
        await notifier.send_client_status(f"💥 CRITICAL ERROR: {str(e)}")
    finally:
        # Cleanup and final report
        uptime = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        await notifier.send_client_status(f"🛑 System stopped | Uptime: {uptime}")
        
        for stream in streams:
            stream.cleanup()
        
        logger.info("✅ System shutdown complete")

if __name__ == "__main__":
    import onnxruntime as ort  # Import here for main scope
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Terminated by user")
    except Exception as e:
        logger.error(f"💥 Fatal startup error: {e}", exc_info=True)
# =============================================================================
# END OF FILE