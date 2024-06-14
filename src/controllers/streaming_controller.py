import numpy as np
import cv2
import json
from data.db.students import students

from .classification_controller import test 

def detect_face(frame):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_classifier.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

def annotate_frame(frame, x, y, w, h, y_pred):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    cv2.putText(frame, f"{y_pred}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if y_pred in students.keys() and not students[y_pred]["present"]:
        students[y_pred]["present"] = True
        with open('src/data/db/students.py', 'w') as file:
            file.write("students=" + json.dumps(students, indent=4))

def stream(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = detect_face(gray_image)

            for (x, y, w, h) in face:
                resized  = cv2.resize(gray_image[y:y + h, x:x + w], (180, 180))
                reshaped = np.array(resized).reshape(1, -1)
                y_pred = test(reshaped)
                annotate_frame(frame, x, y, w, h, y_pred)

            ret, buffer = cv2.imencode('.jpg', frame)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

