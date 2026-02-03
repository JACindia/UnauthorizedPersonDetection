import cv2
import numpy as np
import insightface
import os
from sklearn.preprocessing import normalize
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import time

class FaceRegister:
    def __init__(self):
        # Initialize face analysis model
        self.model = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.model.prepare(ctx_id=0, det_size=(320, 320))
        
        # Registration parameters (unchanged from your original)
        self.min_face_size = 100  # Minimum face area in pixels
        self.samples_per_person = 5  # Number of samples to collect
        self.min_sample_interval = 0.5  # Seconds between samples
        self.quality_threshold = 50  # Minimum sharpness score
        
        # Create directories if they don't exist
        os.makedirs("embeddings", exist_ok=True)
        os.makedirs("registration_samples", exist_ok=True)
        
    def check_face_quality(self, face_img):
        """Face quality check (unchanged from your original)"""
        if face_img.size == 0:
            return False, 0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        
        # Quality checks
        if sharpness < self.quality_threshold:
            return False, sharpness
        if brightness < 30 or brightness > 220:
            return False, sharpness
        if face_img.shape[0] < self.min_face_size or face_img.shape[1] < self.min_face_size:
            return False, sharpness
            
        return True, sharpness
    
    def get_person_name(self):
        """Get name for registration through GUI (unchanged)"""
        root = tk.Tk()
        root.withdraw()
        
        name = simpledialog.askstring("Registration", "Enter person's name:")
        if not name:
            return None
            
        # Clean name for filename use
        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        return clean_name
    
    def select_video_file(self):
        """Open file dialog to select video"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        try:
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
            )
            return file_path
        except Exception as e:
            print(f"File selection error: {str(e)}")
            return None
        finally:
            try:
                root.destroy()
            except:
                pass
    
    def capture_from_source(self, source, name):
        """Capture samples from either camera or video file"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video source: {source}")
            return False
        
        samples = []
        last_capture_time = 0
        feedback_window = f"Registration - {name} - Press ESC to cancel"
        frame_skip = 3  # Process every 3rd frame for video files
        
        try:
            frame_count = 0
            while len(samples) < self.samples_per_person:
                ret, frame = cap.read()
                if not ret:
                    if source == 0:  # Camera
                        continue
                    else:  # Video file ended
                        break
                
                frame_count += 1
                if source != 0 and frame_count % frame_skip != 0:
                    continue
                
                # Detect faces
                faces = self.model.get(frame)
                if not faces:
                    cv2.putText(frame, "No face detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(feedback_window, frame)
                    if cv2.waitKey(10) == 27:
                        break
                    continue
                
                # Process largest face
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                bbox = face.bbox.astype(int)
                face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
                # Check quality (unchanged from your original)
                is_ok, sharpness = self.check_face_quality(face_img)
                current_time = time.time()
                
                # Visual feedback (unchanged from your original)
                color = (0, 255, 0) if is_ok else (0, 0, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                status_text = f"Samples: {len(samples)}/{self.samples_per_person}"
                if is_ok and current_time - last_capture_time > self.min_sample_interval:
                    status_text += " (Ready)"
                    cv2.putText(frame, "Press SPACE to capture", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    status_text += f" (Wait: {max(0, self.min_sample_interval - (current_time - last_capture_time)):.1f}s)"
                
                cv2.putText(frame, status_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Sharpness: {sharpness:.1f}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(feedback_window, frame)
                
                key = cv2.waitKey(10)
                if key == 27:  # ESC
                    break
                elif key == 32 and is_ok and current_time - last_capture_time > self.min_sample_interval:  # SPACE
                    # Save sample with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    sample_path = f"registration_samples/{name}_{timestamp}.jpg"
                    cv2.imwrite(sample_path, face_img)
                    
                    # Store embedding
                    samples.append(face.embedding)
                    last_capture_time = current_time
                    
                    # Visual confirmation
                    cv2.putText(frame, "CAPTURED!", (bbox[0], bbox[1]-20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow(feedback_window, frame)
                    cv2.waitKey(300)
            
            return len(samples) == self.samples_per_person
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def register_from_video(self, name):
        """Register faces from a video file"""
        video_path = self.select_video_file()
        if not video_path:
            return False
        
        return self.capture_from_source(video_path, name)
    
    def register_from_camera(self, name):
        """Register faces from live camera"""
        return self.capture_from_source(0, name)  # 0 for default camera
    
    def save_embeddings(self, name, samples):
        """Save all collected embeddings for a person (unchanged)"""
        if not samples:
            return False
        
        # Save mean embedding
        mean_embedding = np.mean(samples, axis=0)
        np.save(f"embeddings/{name}_mean.npy", mean_embedding)
        
        # Save individual embeddings
        for i, emb in enumerate(samples):
            np.save(f"embeddings/{name}_{i}.npy", emb)
        
        return True
    
    def register_person(self, source_type):
        """Complete registration workflow"""
        name = self.get_person_name()
        if not name:
            return False
        
        # Capture samples from selected source
        if source_type == "camera":
            success = self.register_from_camera(name)
        elif source_type == "video":
            success = self.register_from_video(name)
        else:
            return False
        
        if not success:
            messagebox.showwarning("Registration", "Registration cancelled or incomplete")
            return False
        
        # Load all samples for this person
        embeddings = []
        sample_files = [f for f in os.listdir("registration_samples") if f.startswith(name)]
        
        for file in sample_files:
            try:
                img = cv2.imread(f"registration_samples/{file}")
                faces = self.model.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        
        if not embeddings:
            messagebox.showerror("Error", "No valid faces found in captured samples")
            return False
        
        # Save embeddings
        self.save_embeddings(name, embeddings)
        
        messagebox.showinfo("Success", f"Successfully registered {name} with {len(embeddings)} samples")
        return True
    
    def run_registration(self):
        """Main registration interface"""
        print("=== Face Registration System ===")
        print("1. Register from live camera")
        print("2. Register from video file")
        print("3. Exit")
        
        while True:
            choice = input("\nSelect option (1/2/3): ").strip()
            
            if choice == '1':
                self.register_person("camera")
            elif choice == '2':
                self.register_person("video")
            elif choice == '3':
                break
            else:
                print("Invalid choice, please enter 1, 2, or 3")

if __name__ == "__main__":
    register = FaceRegister()
    register.run_registration()