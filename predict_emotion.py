
import os
import numpy as np
import tensorflow as tf
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emodetect.keras")

IMG_SIZE = 48

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Ask for an image path
image_path = input("Enter the full path of your face image: ").strip().strip('"')

# Read the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the image path.")
    raise SystemExit(1)

# Convert color image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize to the model's expected input size
face = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

# Normalize pixel values
face = face.astype("float32") / 255.0

# Add batch and channel dimensions
face = np.expand_dims(face, axis=(0, -1))

# Predict emotion
predictions = model.predict(face, verbose=0)[0]

predicted_index = int(np.argmax(predictions))
predicted_emotion = EMOTIONS[predicted_index]
confidence = float(predictions[predicted_index]) * 100

print("\n--- EmoDetect Prediction ---")
print("Predicted expression:", predicted_emotion.capitalize())
print(f"Model confidence score: {confidence:.2f}%")
