"""Live facial recognition on a camera feed.

Keys:  e = enroll a new user (type the name in the terminal)
       q / Esc = quit

Run:   python main.py
"""
import cv2

import face_lib as fl

ENROLL_SAMPLES = 10

detector = fl.load_session(fl.DETECTOR_PATH)
embedder = fl.load_session(fl.EMBEDDER_PATH)
db = fl.load_db()
print(f"Loaded {len(db)} enrolled user(s): {list(db)}")

cap = cv2.VideoCapture(0)
enrolling, samples, new_name = False, [], ""

while True:
    ok, frame = cap.read()
    if not ok:
        break

    boxes = fl.detect_faces(detector, frame)
    if enrolling:
        boxes = boxes[:1]  # enroll from a single face only

    for (x1, y1, x2, y2) in boxes:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        emb = fl.embed(embedder, crop)

        if enrolling:
            samples.append(emb)
            label, color = f"Enrolling {new_name}: {len(samples)}/{ENROLL_SAMPLES}", (0, 165, 255)
            if len(samples) == ENROLL_SAMPLES:
                fl.enroll(db, new_name, samples)
                enrolling = False
                print(f"Enrolled '{new_name}'.")
        else:
            name, dist = fl.identify(db, emb)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({dist:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition (e=enroll, q=quit)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("e") and not enrolling:
        new_name = input("Name of new user: ").strip()
        if new_name:
            samples, enrolling = [], True
            print("Look at the camera...")

cap.release()
cv2.destroyAllWindows()
