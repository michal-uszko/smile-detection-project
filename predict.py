from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from time import sleep
import cv2
import numpy as np
import imutils


print("[INFO] Loading model...")
model = load_model('model/classifier.h5')

face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
sleep(2.0)

while True:
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(224, 224),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in face_rects:
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

        roi = cv2.resize(frame[fY: fY + fH, fX: fX + fW], (224, 224))
        roi = img_to_array(roi) / 255.0
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]

        if preds >= 0.5:
            is_smiling = "Smiling"
        else:
            is_smiling = "Not smiling"

        cv2.putText(frame, is_smiling, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

