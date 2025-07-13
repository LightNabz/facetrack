import cv2
import random
import time

distance_threshold_sq = 50 * 50
detect_interval = 5
face_data = []
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
    "Neja Beras", "Ohim Lutung", "Sigit Rendang",
    "Dimas Jordan", "Kapal Emas", "Royan Sybau"
]

def generate_random_info():
    return random.choice(random_ahh)

def statik_ingfo():
    return f"Suka Nasi Padang: {random.randint(10, 100)}%"

start_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_count % detect_interval == 0:
        last_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60) 
        )
    faces = last_faces

    now = time.time()
    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        matched = False

        for face in face_data:
            dx = cx - face['pos'][0]
            dy = cy - face['pos'][1]
            dist_sq = dx * dx + dy * dy

            if dist_sq < distance_threshold_sq:
                face['pos'] = (cx, cy)
                face['last_seen'] = now
                info = face['info']
                info_static = face['static']
                matched = True
                break

        if not matched:
            info = generate_random_info()
            info_static = statik_ingfo()
            face_data.append({
                'pos': (cx, cy),
                'info': info,
                'static': info_static,
                'last_seen': now
            })

        label = f"ID Wajah: {face_data.index(next(f for f in face_data if f['info'] == info)) + 1}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 245, 245), 1)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
        cv2.putText(frame, info, (x + w + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, info_static, (x + w + 10, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    face_data = [f for f in face_data if now - f['last_seen'] < 3]

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
