# 🌿 Plant Disease Detection using Deep Learning

An AI-powered web application that detects plant leaf diseases from uploaded images using **TensorFlow** and **MobileNetV2**. The application is deployed using **Streamlit Community Cloud**, allowing users to upload or capture a leaf image and receive instant disease predictions.

## 🚀 Live Demo

🔗 https://plantdiseaseprediction-mxhnu7gp69xsuuwzmb46pm.streamlit.app/

---

## 📖 Overview

Plant diseases significantly impact crop production and food security. This project leverages **Transfer Learning** with **MobileNetV2** to classify plant leaf diseases from images. The model is integrated into a clean and interactive Streamlit web application for real-time predictions.

---

## ✨ Features

- 🌿 AI-powered plant disease detection
- 📷 Upload images or capture using your camera
- ⚡ Instant disease prediction
- 🎯 Confidence score with progress bar
- 📱 Simple and responsive Streamlit interface
- ☁️ Deployed on Streamlit Community Cloud

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- MobileNetV2 (Transfer Learning)
- NumPy
- Pillow (PIL)
- Streamlit

---

## 📂 Project Structure

```
Plant_Disease_Prediction/
│
├── Plant_App.py                 # Streamlit application
├── Plant.ipynb                  # Model training notebook
├── plant_disease_model.keras    # Trained model
├── class_names.npy              # Class labels
├── requirements.txt
├── README.md
└── assets/ (optional)
```

---

## ⚙️ How It Works

1. Upload or capture a leaf image.
2. The image is resized to **224 × 224** pixels.
3. Pixel values are normalized.
4. The processed image is passed to the trained MobileNetV2 model.
5. The model predicts the disease class.
6. The predicted disease and confidence score are displayed.

---

## 🧠 Model

- Base Model: **MobileNetV2**
- Framework: TensorFlow/Keras
- Input Size: **224 × 224**
- Output: Plant disease class with confidence score

---

## 📊 Dataset

The model is trained using the **PlantVillage Dataset**, which contains thousands of labeled images of healthy and diseased plant leaves across multiple crop species. This dataset is widely used for plant disease classification research. :contentReference[oaicite:0]{index=0}

---

## 📸 Application Preview

Add screenshots here.

Example:

```
assets/home.png

assets/result.png
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Plant_Disease_Prediction.git
```

Go to project folder

```bash
cd Plant_Disease_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run Plant_App.py
```

---

## 🌐 Deployment

This application is deployed on **Streamlit Community Cloud**.

Live Application:

https://plantdiseaseprediction-mxhnu7gp69xsuuwzmb46pm.streamlit.app/

---

## 🔮 Future Improvements

- Disease treatment recommendations
- Fertilizer suggestions
- Grad-CAM visualization
- More plant species support
- Mobile application
- Multi-language support

---

## 👨‍💻 Developer

**Prateek Verma**

B.Tech Information Technology Student

Interested in Artificial Intelligence, Machine Learning, Deep Learning, and Computer Vision.

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.
