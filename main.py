import cv2
import mediapipe as mp
import time
import random
import numpy as np

face_data = []
distance_threshold = 50
detect_interval = 1
last_faces = []

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Face Tracker Nguwawor', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Face Tracker Nguwawor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

random_ahh = [
    "Neja Beras", "Ohim Lutung", "Sigit Rendang", "Dimas Jordan", "Kapal Emas", "Royan Sybau"
]

def generate_random_info():
    return random.choice(random_ahh)

def statik_ingfo():
    return f"Suka Nasi Padang: {random.randint(10, 100)}%"

def how_much():
    return f"Tingkat IQ: {random.randint(80, 150)}"

start_time = time.time()
frame_count = 0
fps = 0

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            faces.append((x, y, w, h))

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        matched = False

        for face in face_data:
            prev_x, prev_y = face['pos']
            dist = np.hypot(center[0] - prev_x, center[1] - prev_y)
            if dist < distance_threshold:
                face['pos'] = center
                info = face['info']
                info_static = face['static']
                info_how = face['how']
                matched = True
                break

        if not matched:
            info = generate_random_info()
            info_static = statik_ingfo()
            info_how  = how_much()
            face_data.append({
                'pos': center,
                'info': info,
                'static': info_static,
                'how': info_how
            })

        label = f"ID Wajah: {face_data.index(next(f for f in face_data if f['info'] == info)) + 1}"

        # Draw rectangle and info
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.putText(frame, info, (x + w + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, info, (x + w + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, info_static, (x + w + 10, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, info_static, (x + w + 10, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, info_how, (x + w + 10, y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, info_how, (x + w + 10, y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # FPS display
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    # Show output
    frame_show = cv2.resize(frame, (1920, 1080))
    cv2.imshow('Face Tracker Nguwawor', frame_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
