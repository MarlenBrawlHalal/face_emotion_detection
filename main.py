import tensorflow as tf
import cv2
import pathlib
import numpy as np
import os

model = tf.keras.models.load_model('facial_expression_model.h5')

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

class_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

cam = cv2.VideoCapture(0)

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / 'data/haarcascade_frontalface_default.xml'

while True:
    _, img = cam.read()
    img_bcp = img.copy()

    face_classifier = cv2.CascadeClassifier(str(cascade_path))
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_coordinates = face_classifier.detectMultiScale(gray_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped_face = img_bcp[y:y+h,x:x+w]
    final_img = cv2.resize(cropped_face, (224, 224))
    final_img = np.expand_dims(final_img, axis=0)
    final_img = final_img / 255.0

    predictions = model.predict(final_img)

    cv2.putText(img, class_dict[np.argmax(predictions)], (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.imshow('Emotion recognition', img)

    if cv2.waitKey(1) == ord('s'):
        break

cam.release()
cv2.destroyAllWindows()