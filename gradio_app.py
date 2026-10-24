
import os
import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

# Load the trained EmoDetect model
MODEL_PATH = os.path.join("models", "emodetect.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# FER-2013 class order used during training
EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Load face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)


def predict_emotion(image):
    if image is None:
        return None, "Please upload or capture a face image."

    # Gradio provides the image in RGB format
    image = image.astype(np.uint8)
    annotated = image.copy()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return annotated, "No face detected. Please try another image."

    # Select the largest detected face
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))

    # Predict facial expression
    predictions = model.predict(face, verbose=0)[0]
    emotion_index = int(np.argmax(predictions))
    emotion = EMOTIONS[emotion_index]
    score = float(predictions[emotion_index]) * 100

    # Draw prediction on the image
    cv2.rectangle(
        annotated, (x, y), (x + w, y + h), (0, 255, 0), 2
    )
    cv2.putText(
        annotated,
        f"{emotion}: {score:.1f}%",
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    result = (
        f"Predicted expression: {emotion}\n"
        f"Model score: {score:.1f}%\n\n"
        "Note: This is a model prediction, not a reliable measurement "
        "of a person's actual feelings."
    )

    return annotated, result


with gr.Blocks(title="EmoDetect") as demo:
    gr.Markdown("# EmoDetect")
    gr.Markdown("### AI-Powered Facial Expression Recognition")

    with gr.Tab("Upload Photo"):
        photo_input = gr.Image(
            label="Upload a face image",
            type="numpy"
        )
        photo_button = gr.Button("Detect Expression")
        photo_output = gr.Image(label="Detected Face")
        photo_result = gr.Textbox(label="Prediction")

        photo_button.click(
            fn=predict_emotion,
            inputs=photo_input,
            outputs=[photo_output, photo_result]
        )

    with gr.Tab("Webcam"):
        webcam_input = gr.Image(
            label="Capture a face image",
            sources=["webcam"],
            type="numpy"
        )
        webcam_button = gr.Button("Detect Expression")
        webcam_output = gr.Image(label="Detected Face")
        webcam_result = gr.Textbox(label="Prediction")

        webcam_button.click(
            fn=predict_emotion,
            inputs=webcam_input,
            outputs=[webcam_output, webcam_result]
        )

if __name__ == "__main__":
    demo.launch()
