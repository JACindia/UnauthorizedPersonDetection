#!/usr/bin/env python3
"""
===============================================================================
Employee Face Verification System 
FULL GPU-ONLY + ADAPTIVE FACE DETECTION v4.3 (FIXED)
FIXES: SyntaxError + Detects REAL FACES, rejects walls/non-faces
===============================================================================
"""

import asyncio
import io
import logging
import os
import sys
import time
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial import distance as dist

import cv2
import cupy as cp
import onnxruntime as ort
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from telegram import Bot
from telegram.error import RetryAfter, TelegramError

# =============================================================================
# FULL GPU-ONLY CONFIGURATION
# =============================================================================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1|probesize;32|analyzeduration;0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

providers = ort.get_available_providers()
if not providers or "CUDAExecutionProvider" not in providers:
    print("❌ FULL GPU REQUIRED!")
    sys.exit(1)

cp.get_default_memory_pool().free_all_blocks()
print("✓ CuPy memory pool initialized")

EMPLOYEE_EMBEDDINGS_FOLDER = Path("embeddings")
TELEGRAM_TOKEN = "8204810755:AAFy1qt1ap44Nq6CjibuSuXMJXYBUtkkxeA"
TELEGRAM_CHAT_ID = "-5160513799"

TEST_MODE = True
RTSP_STREAMS = [
    {"name": "Camera 1", "url": "rtsp://admin:Jezt@884@192.168.74.114/cam/realmonitor?channel=1&subtype=0"},
    {"name": "Camera 2", "url": "rtsp://admin:Jezt@884@192.168.74.112/cam/realmonitor?channel=1&subtype=0"},
]

# 🔧 v4.3: RELAXED FACE DETECTION
SIMILARITY_THRESHOLD = 0.42
FACE_DET_TH = 0.55
PERSON_CONF_TH = 0.45
PERSON_CLASS_ID = 0
MODEL_NAME = "buffalo_l"
FACE_CTX_ID = 0
FACE_DET_SIZE = (640, 640)
FRAME_SKIP = 3
ALERT_COOLDOWN = 30
QUALITY_REJECT_COOLDOWN = 10
SEND_DELAY = 1.0
STATS_INTERVAL = 100
STATUS_UPDATE_INTERVAL = 1800

FACE_SIZE_MIN_PX = 40
FACE_ASPECT_RATIO_MIN = 0.65
FACE_ASPECT_RATIO_MAX = 1.5
HEAD_RATIO_MIN = 0.20

MAX_DISAPPEARED = 20
MIN_HITS = 2
IOU_THRESH = 0.25
DIST_THRESH = 70

# Logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.FileHandler("face_recognition_v4.3.log", encoding='utf-8'), logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# =============================================================================
# 🔧 PERSON TRACKING
# =============================================================================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(boxAArea + boxBArea - inter)

class Track:
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.centroid = ((box[0]+box[2])//2, (box[1]+box[3])//2)
        self.disappeared = 0
        self.hits = 1
        self.confirmed = False
        self.last_alert_time = 0
        self.last_face_time = 0
        self.recent_faces = 0
        
    def can_process_face(self):
        return time.time() - self.last_face_time > 3.0
    
    def update_face_detected(self):
        self.last_face_time = time.time()
        self.recent_faces += 1

class ImprovedTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED, min_hits=MIN_HITS):
        self.nextID = 0
        self.tracks = OrderedDict()
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits

    def update(self, detections):
        if not detections:
            for track in list(self.tracks.values()):
                track.disappeared += 1
                if track.disappeared > self.max_disappeared:
                    del self.tracks[track.track_id]
            return self.tracks

        boxes = [det[:4] for det in detections]
        
        if not self.tracks:
            for box in boxes:
                self.tracks[self.nextID] = Track(self.nextID, box)
                self.nextID += 1
            return self.tracks

        track_ids = list(self.tracks.keys())
        matched_tracks = {}
        matched_dets = []

        for i, track_id in enumerate(track_ids):
            best_iou = 0
            best_det_idx = -1
            for j, box in enumerate(boxes):
                if j in matched_dets: continue
                iou_score = iou(self.tracks[track_id].box, box)
                if iou_score > best_iou and iou_score > IOU_THRESH:
                    best_iou = iou_score
                    best_det_idx = j
            
            if best_det_idx != -1:
                matched_tracks[track_id] = best_det_idx
                matched_dets.append(best_det_idx)

        for track_id, det_idx in matched_tracks.items():
            track = self.tracks[track_id]
            track.box = boxes[det_idx]
            track.centroid = ((boxes[det_idx][0]+boxes[det_idx][2])//2, 
                            (boxes[det_idx][1]+boxes[det_idx][3])//2)
            track.disappeared = 0
            track.hits += 1
            if track.hits >= self.min_hits:
                track.confirmed = True

        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    del self.tracks[track_id]

        for j, box in enumerate(boxes):
            if j not in matched_dets:
                self.tracks[self.nextID] = Track(self.nextID, box)
                self.nextID += 1

        return self.tracks

# =============================================================================
# 🔧 v4.3 SMART FACE VALIDATOR
# =============================================================================

def safe_float(value, default=1000.0):
    if value is None: return default
    try: return float(value)
    except: return default

def safe_getattr(obj, attr, default=1000.0):
    try:
        return safe_float(getattr(obj, attr, default))
    except: return default

def get_face_bbox_ultra_safe(face):
    try:
        bbox = getattr(face, "bbox", None)
        if bbox and len(bbox) >= 4:
            return tuple(int(x) for x in bbox[:4])
    except: pass
    try:
        kps = getattr(face, "kps", None) or getattr(face, "landmarks", None)
        if kps and len(kps) >= 2:
            x_coords = [safe_float(kp[0]) for kp in kps if len(kp) >= 2]
            y_coords = [safe_float(kp[1]) for kp in kps if len(kp) >= 2]
            if len(x_coords) >= 2:
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                return (int(x1-15), int(y1-15), int(x2+15), int(y2+15))
    except: pass
    return None

def is_real_face_candidate(face_bbox, person_bbox, frame_shape):
    x1, y1, x2, y2 = face_bbox
    px1, py1, px2, py2 = person_bbox
    
    face_w, face_h = x2-x1, y2-y1
    person_h = py2-py1
    
    if face_w < FACE_SIZE_MIN_PX or face_h < FACE_SIZE_MIN_PX:
        return False, "too_small"
    
    if not (px1 < x2 and px2 > x1 and py1 < y2 and py2 > y1):
        return False, "outside_person"
    
    head_ratio = face_h / person_h
    if head_ratio < HEAD_RATIO_MIN or head_ratio > 0.55:
        return False, f"bad_head_ratio_{head_ratio:.2f}"
    
    return True, "geometry_ok"

def has_quality_embedding(face):
    embedding = getattr(face, "embedding", None)
    if embedding is None: 
        return False, "no_embedding"
    if not hasattr(embedding, 'shape') or embedding.shape[0] != 512:
        return False, "bad_embedding_shape"
    det_score = safe_getattr(face, 'det_score', 0)
    return det_score >= FACE_DET_TH, "det_ok"

def final_quality_check(face_metrics):
    f = face_metrics
    if f.get("blur", 0) > 3000:
        return False, "extreme_blur"
    if f.get("face_brightness", 0) < 30:
        return False, "too_dark"
    if abs(f.get("yaw", 0)) > 45 or abs(f.get("pitch", 0)) > 45:
        return False, "extreme_pose"
    return True, "quality_ok"

def validate_face_v4_3(frame, face, person_bbox):
    bbox = get_face_bbox_ultra_safe(face)
    if not bbox:
        return False, {'error': 'no_bbox'}
    
    h, w = frame.shape[:2]
    
    geo_ok, geo_reason = is_real_face_candidate(bbox, person_bbox, (h, w))
    if not geo_ok:
        return False, {'error': f'geo_{geo_reason}', 'bbox': bbox}
    
    emb_ok, emb_reason = has_quality_embedding(face)
    if not emb_ok:
        return False, {'error': f'emb_{emb_reason}', 'bbox': bbox}
    
    x1, y1, x2, y2 = bbox
    face_w, face_h = x2-x1, y2-y1
    
    metrics = {
        'blur': safe_getattr(face, 'blur'),
        'face_brightness': safe_getattr(face, 'brightness'),
        'yaw': safe_getattr(face, 'yaw'),
        'pitch': safe_getattr(face, 'pitch'),
        'det_score': safe_getattr(face, 'det_score'),
        'face_w': face_w,
        'face_h': face_h,
        'bbox': bbox,
        'error': None
    }
    
    quality_ok, quality_reason = final_quality_check(metrics)
    if not quality_ok:
        return False, {'error': f'quality_{quality_reason}', **metrics}
    
    return True, {'status': 'VALID_FACE', **metrics}

# =============================================================================
# GPU MODELS
# =============================================================================

class YOLOPersonDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.model.to('cuda')
        self.model.fuse()
        logger.info("✓ YOLOv8n GPU loaded")
    
    def detect(self, frame):
        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False, conf=PERSON_CONF_TH, device='cuda')
        persons = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == PERSON_CLASS_ID:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        persons.append((
                            max(0, int(x1*0.95)), max(0, int(y1*0.95)),
                            min(w, int(x2*1.05)), min(h, int(y2*1.05)),
                            conf
                        ))
        return persons

class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(name=MODEL_NAME, providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
        logger.info("✓ InsightFace buffalo_l GPU loaded")
    
    def detect_faces(self, frame, person_bbox):
        x1, y1, x2, y2 = person_bbox
        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
        if crop.size == 0:
            return []
        
        faces = self.app.get(crop)
        return [f for f in faces if safe_getattr(f, 'det_score', 0) >= FACE_DET_TH]
    
    def load_embeddings(self, folder):
        if not folder.exists():
            logger.warning(f"No embeddings folder: {folder}")
            return []
        embeddings = []
        for npy_file in folder.glob("*.npy"):
            try:
                emb = np.load(str(npy_file))
                if emb.shape == (512,):
                    embeddings.append(emb)
                elif emb.shape == (1, 512):
                    embeddings.append(emb[0])
            except Exception as e:
                logger.error(f"Failed to load {npy_file}: {e}")
        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings
    
    @staticmethod
    def cosine_sim(a, b):
        try:
            a_gpu, b_gpu = cp.asarray(a), cp.asarray(b)
            sim = float(cp.dot(a_gpu, b_gpu) / (cp.linalg.norm(a_gpu) * cp.linalg.norm(b_gpu)))
            cp.get_default_memory_pool().free_all_blocks()
            return sim
        except:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def is_authorized(self, embedding, employees):
        if not employees:
            return False, 0.0
        max_sim = max(self.cosine_sim(embedding, emp) for emp in employees)
        return max_sim >= SIMILARITY_THRESHOLD, max_sim

# =============================================================================
# FIXED Telegram Notifier
# =============================================================================

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self.last_reject_time = 0
    
    async def send_debug_reject(self, frame, stream_name, metrics):
        if time.time() - self.last_reject_time < QUALITY_REJECT_COOLDOWN:
            return
        try:
            debug_frame = frame.copy()
            h, w = debug_frame.shape[:2]
            
            error = metrics.get('error', 'unknown')
            bbox = metrics.get('bbox')
            
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            cv2.putText(debug_frame, f"REJECT: {error}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_frame, stream_name, (10, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            bio = io.BytesIO(buffer.tobytes())
            bio.name = 'face_reject_v4.3.jpg'
            
            caption = (f"⚠️ FACE REJECT v4.3\n"
                      f"📹 {stream_name}\n"
                      f"❌ {error}\n"
                      f"⏰ {time.strftime('%H:%M:%S')}")
            
            await self.bot.send_photo(chat_id=self.chat_id, photo=bio, caption=caption)
            self.last_reject_time = time.time()
            logger.info(f"📤 REJECT sent: {error}")
        except Exception as e:
            logger.error(f"Telegram reject error: {e}")
    
    async def send_status(self, message):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=f"🤖 v4.3: {message}")
        except Exception as e:
            logger.error(f"Status error: {e}")
    
    async def send_alert(self, frame, stream_name, track_id):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            bio = io.BytesIO(buffer.tobytes())
            caption = f"🚨 UNAUTHORIZED v4.3!\n{stream_name} Track#{track_id}"
            await self.bot.send_photo(chat_id=self.chat_id, photo=bio, caption=caption)
        except Exception as e:
            logger.error(f"Alert error: {e}")

# =============================================================================
# FIXED Stream Processor (NOW ASYNC)
# =============================================================================

class StreamProcessor:
    def __init__(self, config):
        self.name = config["name"]
        self.url = config["url"]
        self.cap = None
        self.frame_count = 0
        self.process_count = 0
        self.tracker = ImprovedTracker()
        self.stats = {'tracks': 0, 'faces': 0, 'auth': 0, 'reject': 0}
    
    def connect(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()
    
    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            self.connect()
            return None
        ret, frame = self.cap.read()
        if not ret:
            self.connect()
            return None
        self.frame_count += 1
        if TEST_MODE:
            cv2.imshow(self.name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()
        return frame
    
    async def process_frame(self, frame, yolo_detector, face_engine, notifier, employees):  # ✅ FIXED: Now async
        persons = yolo_detector.detect(frame)
        tracks = self.tracker.update(persons)
        
        confirmed_tracks = [t for t in tracks.values() if t.confirmed]
        self.stats['tracks'] = len(confirmed_tracks)
        
        for track in confirmed_tracks:
            if not track.can_process_face():
                continue
                
            faces = face_engine.detect_faces(frame, track.box)
            self.stats['faces'] += len(faces)
            
            for face in faces:
                is_valid, metrics = validate_face_v4_3(frame, face, track.box)
                
                if not is_valid:
                    await notifier.send_debug_reject(frame, self.name, metrics)  # ✅ Now works!
                    self.stats['reject'] += 1
                    continue
                
                embedding = getattr(face, "embedding", None)
                if embedding is None:
                    continue
                
                is_auth, sim_score = face_engine.is_authorized(embedding, employees)
                
                if is_auth:
                    self.stats['auth'] += 1
                    logger.info(f"{self.name} Track#{track.track_id} ✅ AUTH {sim_score:.3f}")
                else:
                    logger.info(f"{self.name} Track#{track.track_id} ❌ UNAUTH {sim_score:.3f}")
                    if time.time() - track.last_alert_time > ALERT_COOLDOWN:
                        await notifier.send_alert(frame, self.name, track.track_id)
                        track.last_alert_time = time.time()
                
                track.update_face_detected()
                break
    
    def get_summary(self):
        return f"{self.name}: T{self.stats['tracks']} F{self.stats['faces']} A{self.stats['auth']} R{self.stats['reject']}"

# =============================================================================
# MAIN v4.3 - FULLY FIXED
# =============================================================================

async def main():
    logger.info("🚀 v4.3 - ADAPTIVE FACE DETECTION LIVE!")
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    
    await notifier.send_status("Starting v4.3...")
    
    yolo_detector = YOLOPersonDetector()
    face_engine = FaceEngine()
    employees = face_engine.load_embeddings(EMPLOYEE_EMBEDDINGS_FOLDER)
    
    if not employees:
        await notifier.send_status("❌ NO EMBEDDINGS - Add .npy files!")
        return
    
    streams = [StreamProcessor(s) for s in RTSP_STREAMS]
    
    await notifier.send_status(f"✅ v4.3 LIVE | {len(employees)} employees | {len(streams)} cams")
    
    frame_count = 0
    while True:
        for stream in streams:
            frame = stream.read_frame()
            if frame is None or stream.process_count % FRAME_SKIP != 0:
                stream.process_count += 1
                continue
            
            stream.process_count += 1
            await stream.process_frame(frame, yolo_detector, face_engine, notifier, employees)  # ✅ FIXED
            
        frame_count += 1
        if frame_count % STATS_INTERVAL == 0:
            logger.info("📊 " + " | ".join(s.get_summary() for s in streams))
        
        await asyncio.sleep(0.01)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Shutdown")
    except Exception as e:
        print(f"💥 Error: {e}")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)