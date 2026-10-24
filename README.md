# EmoDetect — AI-Powered Facial Emotion Recognition

EmoDetect is a deep learning project that recognizes facial expressions from uploaded images and webcam captures. It uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify facial expressions into seven categories.

## ✨ Features

- 📸 Upload a face image and predict its facial expression.
- 📷 Capture an image using your webcam for prediction.
- 🧠 CNN-based facial expression classification.
- 😊 Recognizes seven expression categories.
- 🖼️ Detects faces using OpenCV's Haar Cascade classifier.
- 🌐 Interactive user interface built with Gradio.

## 🎭 Emotion Categories

EmoDetect predicts one of these seven categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## 🛠️ Technologies Used

- **Python** — Core programming language
- **TensorFlow / Keras** — Deep learning model development
- **OpenCV** — Face detection and image preprocessing
- **NumPy** — Numerical operations
- **Gradio** — Interactive web interface
- **FER-2013** — Facial expression recognition dataset

## 🏗️ Project Structure

```text
EmoDetect/
├── models/
│   └── emodetect.keras
├── train_model.py
├── predict_emotion.py
├── webcam_emotion.py
├── gradio_app.py
├── README.md
├── .gitignore
└── venv/
```

> **Note:** The `venv/` folder is your local Python environment and should not be uploaded to GitHub. The trained model and dataset may also be excluded from Git using `.gitignore`.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/linzars2006-oss/EmoDedect.git
cd EmoDedect
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install tensorflow opencv-python==4.10.0.84 numpy gradio
```

### 4. Prepare the trained model

Place the trained model at:

```text
models/emodetect.keras
```

If the model is not included in the repository, train it locally using the FER-2013 dataset and `train_model.py`. Make sure the dataset folder structure matches what the training script expects.

### 5. Run EmoDetect

Launch the Gradio application:

```bash
python gradio_app.py
```

Open the local URL displayed in the terminal to use the application.

## 🧠 Model Training

EmoDetect uses a CNN trained on the FER-2013 facial expression dataset. The model processes grayscale face images resized to **48 × 48 pixels** and predicts one of seven expression classes.

The training script is:

```bash
python train_model.py
```

## 📊 Model Performance

The model achieved approximately **52.58% test accuracy** in the current experiment.

Performance can vary depending on image quality, lighting, face position, and the expression shown. The prediction represents a visual expression classification and does not reliably determine a person's actual emotional state.

## 🚀 Future Improvements

- Improve model accuracy through data augmentation and hyperparameter tuning.
- Explore transfer learning with suitable pretrained models.
- Improve face detection under different lighting conditions.
- Support multiple face detection.
- Add confidence visualizations and model evaluation metrics.

## 🎓 Learning Outcomes

Through EmoDetect, I explored:

- CNN-based image classification.
- Image preprocessing and normalization.
- Face detection using OpenCV.
- Training and evaluating a deep learning model.
- Integrating an ML model into a Gradio interface.

## 👩‍💻 Author

**Linza R.S.**

AI & Machine Learning Student

---

*EmoDetect is an educational AI/ML project. Facial expression predictions may be inaccurate and should not be treated as definitive assessments of a person's feelings.*
