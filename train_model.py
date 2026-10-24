
import os
import cv2
import numpy as np
import tensorflow as tf

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emodetect.keras")

# Emotion labels must match the training folder order
EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

IMG_SIZE = 48

# Load the trained model
print("Loading EmoDetect model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded! Starting webcam...")

# Load OpenCV face detector
cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    print("Check camera permissions or try changing VideoCapture(0) to VideoCapture(1).")
    raise SystemExit(1)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Mirror the webcam like a selfie camera
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        # Extract detected face
        face = gray[y:y + h, x:x + w]

        # Resize and normalize
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0

        # Model input shape: (1, 48, 48, 1)
        face = np.expand_dims(face, axis=(0, -1))

        # Predict emotion
        predictions = model.predict(face, verbose=0)[0]
        emotion_index = int(np.argmax(predictions))
        emotion = EMOTIONS[emotion_index]
        confidence = float(predictions[emotion_index]) * 100

        # Draw face rectangle
        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

        # Display predicted expression
        label = f"{emotion}: {confidence:.1f}%"
        label_y = max(y - 10, 25)

        cv2.putText(
            frame, label, (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0), 2
        )

    # Display instructions
    cv2.putText(
        frame, "EmoDetect - Live Emotion Recognition",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2
    )
    cv2.putText(
        frame, "Press Q to quit",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2
    )

    cv2.imshow("EmoDetect - Webcam", frame)

    # Press Q to close the webcam window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam resources
camera.release()
cv2.destroyAllWindows()

print("EmoDetect webcam closed.")
