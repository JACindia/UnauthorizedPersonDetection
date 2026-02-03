import cv2
import numpy as np
import os
from pathlib import Path
import time
import argparse
from PIL import Image
import io
import asyncio
from telegram import Bot
import insightface
import logging
from threading import Thread
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCameraFaceRecognitionSystem:
    def __init__(self, embeddings_dir="embeddings_adapted", threshold=0.22, 
                 rtsp_urls=None, telegram_token=None, telegram_chat_id=None):
        self.embeddings_dir = Path(embeddings_dir)
        self.threshold = threshold
        self.rtsp_urls = rtsp_urls or [
            0
        ]
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        # Initialize InsightFace (SHARED across all cameras)
        try:
            self.model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ InsightFace GPU loaded")
        except:
            self.model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ InsightFace CPU loaded")
        
        self.known_embeddings = {}
        self.load_known_faces()
        
        self.caps = {}
        self.bot = None
        self.last_alerts = {}
        self.alert_cooldown = 15
        self.telegram_tested = False
        
        # Telegram
        if self.telegram_token and self.telegram_chat_id:
            try:
                self.bot = Bot(token=self.telegram_token)
                logger.info("✅ Telegram ready")
            except Exception as e:
                logger.error(f"❌ Telegram: {e}")

    def test_telegram(self):
        if not self.bot:
            return False
        try:
            asyncio.run(self.bot.send_message(chat_id=self.telegram_chat_id, text="🤖 Multi-Camera Face Recog LIVE ✅"))
            logger.info("✅ Telegram OK")
            self.telegram_tested = True
            return True
        except Exception as e:
            logger.error(f"❌ Telegram: {e}")
            return False

    def load_known_faces(self):
        if not self.embeddings_dir.exists():
            os.makedirs(self.embeddings_dir, exist_ok=True)
            return
        
        for npy_file in self.embeddings_dir.glob("*.npy"):
            name = npy_file.stem
            embedding = np.load(npy_file)
            embedding = embedding / np.linalg.norm(embedding)
            self.known_embeddings[name] = embedding
            logger.info(f"✅ Loaded {name}")
        
        logger.info(f"📁 {len(self.known_embeddings)} known faces loaded")

    def enroll_current_face(self, camera_id=0):
        cap = self.caps.get(camera_id)
        if not cap or not cap.isOpened():
            logger.error(f"❌ Camera {camera_id} not available")
            return
        
        ret, frame = cap.read()
        if not ret:
            logger.error("❌ No frame")
            return
        
        faces = self.model.get(frame)
        if not faces:
            logger.error("❌ No face detected")
            return
        
        embedding = faces[0].embedding / np.linalg.norm(faces[0].embedding)
        name = input("👤 Enter name: ").strip()
        np.save(self.embeddings_dir / f"{name}.npy", embedding)
        self.load_known_faces()
        logger.info(f"✅ ENROLLED: {name}")

    def cosine_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2)

    def crop_face(self, frame, bbox):
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = y2-y1, x2-x1
        pad = int(min(h, w) * 0.2)
        return frame[max(0,y1-pad):min(frame.shape[0],y2+pad), 
                     max(0,x1-pad):min(frame.shape[1],x2+pad)]

    async def send_telegram_alert(self, full_frame=None, face_crop=None, name=None, 
                                status=None, sim=None, unauthorized_count=0, camera_id=0):
        if not self.bot or not self.telegram_tested:
            return
        
        now = time.time()
        key = f"cam{camera_id}_{'multi_unauth' if unauthorized_count > 1 else f'{name}_{status}'}"
        
        if key in self.last_alerts and now - self.last_alerts[key] < self.alert_cooldown:
            return
        self.last_alerts[key] = now
        
        try:
            if full_frame is not None and unauthorized_count > 1:
                rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG', quality=90)
                buffer.seek(0)
                
                caption = f"🚨 Cam{camera_id}: {unauthorized_count} UNAUTHORIZED!\n🔴 Threshold: {self.threshold:.2f}"
                await self.bot.send_photo(chat_id=self.telegram_chat_id, photo=buffer, caption=caption)
            
            elif face_crop is not None:
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG', quality=90)
                buffer.seek(0)
                
                emoji = "✅" if status == "authorized" else "❌"
                caption = f"{emoji} Cam{camera_id}\n👤 {name}\n🎯 {sim:.3f}"
                await self.bot.send_photo(chat_id=self.telegram_chat_id, photo=buffer, caption=caption)
                
        except Exception as e:
            logger.error(f"❌ Telegram: {e}")

    def recognize_faces(self, frame):
        faces = self.model.get(frame)
        results = []
        
        for face in faces:
            embedding = face.embedding / np.linalg.norm(face.embedding)
            best_name = "Unknown"
            best_sim = 0.0
            
            for name, known_emb in self.known_embeddings.items():
                sim = self.cosine_similarity(embedding, known_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            
            results.append((best_name, face.bbox, best_sim))
        
        return results

    def draw_results(self, frame, results, camera_id):
        for name, bbox, sim in results:
            x1, y1, x2, y2 = bbox.astype(int)
            is_auth = sim >= self.threshold
            color = (0, 255, 0) if is_auth else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), color, -1)
            
            status = "AUTHORIZED" if is_auth else "UNAUTHORIZED"
            label = f"Cam{camera_id} {name}: {sim:.2f}"
            cv2.putText(frame, label, (x1+5, y2-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        return frame

    def process_camera(self, camera_id, cap, loop):
        """Process single camera in separate thread"""
        logger.info(f"🎥 Cam {camera_id} started")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            results = self.recognize_faces(frame)
            
            unauthorized_faces = [(name, bbox, sim) for name, bbox, sim in results 
                                if sim < self.threshold]
            
            display_frame = frame.copy()
            
            if len(unauthorized_faces) > 1:
                # Multiple unauthorized - mark & alert
                alert_frame = frame.copy()
                for name, bbox, sim in unauthorized_faces:
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.rectangle(alert_frame, (x1, y2-45), (x2, y2), (0, 0, 255), -1)
                    cv2.putText(alert_frame, f"🚨 {name}: {sim:.2f}", (x1+10, y2-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                cv2.rectangle(alert_frame, (10, 10), (alert_frame.shape[1]-10, 90), (0, 0, 255), -1)
                cv2.putText(alert_frame, f"🚨 Cam{camera_id}: {len(unauthorized_faces)} UNAUTHORIZED!", 
                           (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                try:
                    loop.run_until_complete(self.send_telegram_alert(
                        full_frame=alert_frame, unauthorized_count=len(unauthorized_faces), camera_id=camera_id
                    ))
                except:
                    pass
                display_frame = alert_frame
                
            else:
                display_frame = self.draw_results(display_frame, results, camera_id)
                
                # Single unauthorized alert
                for name, bbox, sim in results:
                    if sim < self.threshold:
                        crop = self.crop_face(frame, bbox)
                        if crop.size > 0:
                            try:
                                loop.run_until_complete(self.send_telegram_alert(
                                    face_crop=crop, name=name, status="unauthorized", sim=sim, camera_id=camera_id
                                ))
                            except:
                                pass
            
            # HUD
            status = "✅" if self.telegram_tested else "❌"
            hud_text = f"Cam{camera_id} | Known:{len(self.known_embeddings)} | Thresh:{self.threshold:.2f} | TG:{status}"
            cv2.putText(display_frame, hud_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(display_frame, "q=QUIT  e=ENROLL  +/-=THRESH  t=TEST", (10, display_frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            cv2.imshow(f'Camera {camera_id} - Face Recognition', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run_realtime(self):
        if self.bot:
            self.test_telegram()
        
        # Connect cameras
        for i, url in enumerate(self.rtsp_urls):
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                self.caps[i] = cap
                logger.info(f"✅ Cam {i}: {url}")
            else:
                logger.error(f"❌ Cam {i} failed: {url}")
        
        if not self.caps:
            logger.error("❌ No cameras connected!")
            return
        
        logger.info(f"🚀 LIVE: {len(self.caps)} cameras | Threshold={self.threshold}")
        
        # Start camera threads
        threads = []
        loops = {}
        for camera_id, cap in self.caps.items():
            loop = asyncio.new_event_loop()
            loops[camera_id] = loop
            asyncio.set_event_loop(loop)
            thread = Thread(target=self.process_camera, args=(camera_id, cap, loop), daemon=True)
            thread.start()
            threads.append(thread)
        
        # Wait for threads
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        
        # Cleanup
        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()
        logger.info("🛑 System stopped")

def main():
    parser = argparse.ArgumentParser(description="🚀 Multi-Camera Face Recognition")
    parser.add_argument('--threshold', type=float, default=0.22, help="Similarity threshold")
    parser.add_argument('--rtsp-urls', nargs='+', default=[
        0
    ], help="RTSP camera URLs")
    
    args = parser.parse_args()
    
    system = MultiCameraFaceRecognitionSystem(
        threshold=args.threshold,
        rtsp_urls=args.rtsp_urls,
        telegram_token="8204810755:AAFy1qt1ap44Nq6CjibuSuXMJXYBUtkkxeA",
        telegram_chat_id="-5160513799"
    )
    
    system.run_realtime()

if __name__ == "__main__":
    main()
# End of git.py
## To run the system use python git.py --threshold 0.22