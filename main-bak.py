import cv2
import random
import time
import numpy as np

face_data = []
distance_threshold = 50
detect_interval = 5
last_faces = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("kneaf pula (eror haar apalah)")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('Face Tracker Nguwawor', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Face Tracker Nguwawor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

random_ahh = [
    "Neja Beras", "Ohim Lutung", "Sigit Rendang", "Dimas Jordan", "Kapal Emas", "Royan Sybau"
]

def generate_random_info():
    return random.choice(random_ahh)

def statik_ingfo():
    return f"Suka Nasi Padang: {random.randint(10, 100)}%"

start_time = time.time()
frame_count = 0
fps = 0

scale = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    if frame_count % detect_interval == 0:
        last_faces = face_cascade.detectMultiScale(
            gray_small,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40)
        )

    faces = last_faces

    for (x, y, w, h) in faces:
        x_big, y_big, w_big, h_big = [int(v / scale) for v in (x, y, w, h)]
        center = (x_big + w_big // 2, y_big + h_big // 2)
        matched = False

        for face in face_data:
            prev_x, prev_y = face['pos']
            dist = np.hypot(center[0] - prev_x, center[1] - prev_y)
            if dist < distance_threshold:
                face['pos'] = center
                info = face['info']
                info_static = face['static']
                matched = True
                break

        if not matched:
            info = generate_random_info()
            info_static = statik_ingfo()
            face_data.append({
                'pos': center,
                'info': info,
                'static': info_static
            })

        label = f"ID Wajah: {face_data.index(next(f for f in face_data if f['info'] == info)) + 1}"

        cv2.rectangle(frame, (x_big, y_big), (x_big + w_big, y_big + h_big), (245, 245, 245), 1)
        cv2.putText(frame, label, (x_big, y_big - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
        cv2.putText(frame, info, (x_big + w_big + 10, y_big + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, info_static, (x_big + w_big + 10, y_big + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    frame_show = cv2.resize(frame, (1280, 720))
    cv2.imshow('Face Tracker Nguwawor', frame_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
