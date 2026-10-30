# 🌿 PlantCare AI – Intelligent Plant Disease Detection

An AI-powered plant health analysis application that uses deep learning to identify possible plant diseases from leaf images.

PlantCare AI allows users to upload a plant leaf image or capture one using their camera and receive an AI-generated disease prediction with confidence scores.

## 🚀 Live Demo

<https://plantdiseaseprediction-prateek.streamlit.app/>

Try PlantCare AI live without installing anything!

---

## ✨ Features

- **AI-Based Disease Detection:** Predicts plant disease classes using a trained deep learning model.
- **Image Upload:** Upload leaf images in JPG, JPEG, and PNG formats.
- **Live Camera Capture:** Capture plant images directly through your device camera.
- **Prediction Confidence:** Displays the model's confidence score for its prediction.
- **Top Predictions:** Shows the top three predicted disease classes.
- **Interactive Dashboard:** Clean, modern, and responsive Streamlit interface.
- **Fast Predictions:** Uses TensorFlow for image classification.

---

## 🧠 How It Works

PlantCare AI uses a trained **MobileNetV2** deep learning model to classify plant leaf images.

1. Upload or capture a plant leaf image.
2. Convert the image into RGB format.
3. Resize the image to 224 × 224 pixels.
4. Process the image using the trained TensorFlow model.
5. Generate prediction scores.
6. Display the predicted disease, confidence score, and top three predictions.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning |
| MobileNetV2 | Image classification |
| Streamlit | Web application |
| NumPy | Numerical operations |
| Pillow | Image processing |
| PlantVillage | Dataset |

---

## 📂 Project Structure

```text
PlantCare-AI/
│
├── app.py
├── plant_disease_model.keras
├── class_names.npy
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/PlantCare-AI.git
cd PlantCare-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📦 Requirements

Create a `requirements.txt` file:

```text
streamlit
tensorflow
numpy
Pillow
```

---

## 📊 Dataset

This project uses the **PlantVillage Dataset**, a collection of plant leaf images containing healthy and diseased plant samples.

---

## ⚠️ Disclaimer

PlantCare AI provides AI-generated predictions for informational purposes only. Predictions are not guaranteed to be accurate and should not replace professional agricultural diagnosis.

---

## 🚀 Future Enhancements

- Improve model accuracy through fine-tuning.
- Add more plant species and disease categories.
- Provide disease descriptions and symptoms.
- Include prevention and treatment recommendations.
- Add prediction history.
- Improve performance on real-world field images.

---

## 👨‍💻 Developer

**Prateek Verma**

Developed with ❤️ using Python, TensorFlow, and Streamlit.

---

<p align="center">
  🌿 <b>PlantCare AI</b><br>
  Making plant health analysis smarter.
</p>
