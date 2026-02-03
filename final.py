import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import requests
from datetime import datetime
import time
import threading
from collections import defaultdict

# Config
RTSP_URL = [
    {"name": "Camera 1", "url": "rtsp://admin:Jezt@884@192.168.74.114/cam/realmonitor?channel=1&subtype=0"},
    {"name": "Camera 2", "url": "rtsp://admin:Jezt@884@192.168.74.112/cam/realmonitor?channel=1&subtype=0"},
]
TELEGRAM_BOT_TOKEN = "8204810755:AAFy1qt1ap44Nq6CjibuSuXMJXYBUtkkxeA"
TELEGRAM_CHAT_ID = "-5160513799"
SAFE_ZONE = (100, 100, 500, 400)  # x1, y1, x2, y2
ALERT_COOLDOWN = 30

model = YOLO("yolov8n.pt")

def send_telegram_alert(message, photo_path=None, camera_name="Unknown"):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        cam_msg = f"📹 <b>{camera_name}</b>\n"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": cam_msg + message, 
            "parse_mode": "HTML"
        })
        
        if photo_path:
            photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(photo_path, 'rb') as photo:
                requests.post(photo_url, files={'photo': photo}, data={
                    "chat_id": TELEGRAM_CHAT_ID, 
                    "caption": f"🚨 INTRUDER ALERT - {camera_name}"
                })
    except Exception as e:
        print(f"Telegram error: {e}")

AUTHORIZED_PERSON_IDS = [1, 2, 3]

def is_person_authorized(track_id):
    return track_id in AUTHORIZED_PERSON_IDS

def is_person_in_safe_zone(bbox, safe_zone):
    """Check if bbox is within safe_zone (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = safe_zone
    return (x1 > zx1 and y1 > zy1 and x2 < zx2 and y2 < zy2)

def process_camera(camera_info, camera_id):
    """Process single camera stream."""
    cap = cv2.VideoCapture(camera_info["url"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    tracker = sv.ByteTrack()
    last_alert = 0
    
    print(f"Started {camera_info['name']}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        frame = cv2.resize(frame, (640, 480))
        
        # YOLO + Tracking
        results = model(frame, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)
        
        # Draw safe zone
        cv2.rectangle(frame, (SAFE_ZONE[0], SAFE_ZONE[1]), 
                     (SAFE_ZONE[2], SAFE_ZONE[3]), (0, 255, 0), 3)
        cv2.putText(frame, "SAFE ZONE", (SAFE_ZONE[0], SAFE_ZONE[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        unauthorized_detected = False
        
        # Filter person detections only (class_id 0)
        person_mask = detections.class_id == 0
        person_detections = detections[person_mask]
        
        for track_id, box in zip(person_detections.tracker_id, person_detections.xyxy):
            x1, y1, x2, y2 = box.astype(int)
            
            if not is_person_in_safe_zone(box, SAFE_ZONE) or not is_person_authorized(track_id):
                color = (0, 0, 255)
                label = "🚨 INTRUDER"
                unauthorized_detected = True
            else:
                color = (0, 255, 0)
                label = f"Staff #{track_id}"
            
            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1, y1-30), (x2, y1), color, -1)
            cv2.putText(frame, label, (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Alert logic
        current_time = time.time()
        if unauthorized_detected and (current_time - last_alert) > ALERT_COOLDOWN:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = f"alert_{camera_info['name'].replace(' ', '_')}_{int(current_time)}.jpg"
            cv2.imwrite(filename, frame)
            
            message = f"""
🚨 <b>INTRUDER DETECTED!</b>

⏰ {timestamp}
📍 Outside SAFE ZONE
👥 Unknown person detected
📸 Screenshot saved: {filename}
            """
            send_telegram_alert(message, filename, camera_info['name'])
            last_alert = current_time
        
        # Show camera feed (separate window per camera)
        cv2.imshow(f'YOLO Surveillance - {camera_info["name"]}', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Stopped {camera_info['name']}")

def main():
    # Send startup message
    send_telegram_alert("🚀 <b>MULTI-CAMERA YOLO Surveillance Started</b>\n📹 2 cameras active")
    
    # Start each camera in separate thread
    threads = []
    for i, camera_info in enumerate(RTSP_URL):
        thread = threading.Thread(target=process_camera, args=(camera_info, i))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    try:
        # Keep main thread alive
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        send_telegram_alert("🔴 Multi-camera surveillance stopped")

if __name__ == "__main__":
    main()