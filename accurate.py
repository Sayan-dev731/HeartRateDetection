import tensorflow as tf
import cv2
import numpy as np

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    return faces

def get_heart_rate(frame, faces):
    if len(faces) == 0:
        return 0

    face = faces[0]
    x, y, w, h = face
    roi = frame[y:y+h, x:x+w]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale image to 11x11
    gray = cv2.resize(gray, (11, 11))

    # Get the heart rate from the model
    model = tf.keras.models.load_model('my_model.h5')
    heart_rate = model.predict(gray)

    return heart_rate


def main():
    cap = cv2.VideoCapture(0)

    # Load the model
    model = tf.keras.models.load_model('my_model.h5')

    while True:
        ret, frame = cap.read()

        faces = detect_face(frame)
        heart_rate = get_heart_rate(frame, faces)

        # Add a green frame to the face
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, str(heart_rate), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
