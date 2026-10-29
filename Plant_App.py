# Streamlit se web app banane ke liye
import streamlit as st

# Trained Deep Learning model load karne ke liye
import tensorflow as tf

# Arrays aur mathematical operations ke liye
import numpy as np

# Uploaded image ko read aur process karne ke liye
from PIL import Image

# ----------------------------------------
# 🌿 Web App Header
# ----------------------------------------

st.title("🌿 Plant Disease Detection")

st.caption(
    "AI-powered leaf disease prediction using TensorFlow & MobileNetV2"
)

st.markdown("---")

# ----------------------------------------
# 🌿 Sidebar
# ----------------------------------------

with st.sidebar:

    st.header("🌿 About")

    st.write("""
This application predicts plant leaf diseases using a
Deep Learning model trained on the PlantVillage dataset.
""")

    st.markdown("---")

    st.write("### 🤖 Model")
    st.write("MobileNetV2")

    st.write("### 📊 Dataset")
    st.write("PlantVillage")

    st.write("### 👨‍💻 Developer")
    st.write("Pooja Bisht")

# ----------------------------------------
# Model Load
# ----------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# Class Names
class_names = np.load("class_names.npy", allow_pickle=True)

# ----------------------------------------
# Image Upload
# ----------------------------------------

uploaded_file = st.file_uploader(
    "📁 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input(
    "📷 Capture Leaf Image"
)

# ----------------------------------------
# Prediction Function
# ----------------------------------------

def predict(image):

   

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image = np.array(image, dtype=np.float32)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image, verbose=0)

    # Get predicted class
    predicted_index = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][predicted_index])

    print("Number of model outputs:", len(prediction[0]))
    print("Number of class names:", len(class_names))
    print("Predicted index:", predicted_index)
    print("Predicted class:", class_names[predicted_index])
    print("Confidence:", confidence)

    return class_names[predicted_index], confidence

# ----------------------------------------
# Image Selection
# ----------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

elif camera_image is not None:

    image = Image.open(camera_image).convert("RGB")

else:

    image = None

# ----------------------------------------
# Prediction Section
# ----------------------------------------

if image is not None:

    # Selected Image Show Karega
    st.image(
        image,
        caption="📷 Selected Leaf Image",
        use_container_width=True
    )

    st.markdown("")

    # Predict Button
    if st.button(
        "🔍 Predict Disease",
        use_container_width=True,
        type="primary"
    ):

        with st.spinner("🔍 Analyzing Leaf..."):

            disease, confidence = predict(image)

        # Folder Name Ko Readable Banayenge
        disease = disease.replace("___"," → ")
        disease = disease.replace("_"," ")

        st.markdown("---")

        st.subheader("🌿 Prediction Result")

        # Disease
        st.success(f"Predicted Disease : {disease}")

        # Confidence
        st.metric(
            label="🎯 Confidence",
            value=f"{confidence*100:.2f}%"
        )

        # Progress Bar
        st.progress(int(confidence*100))

        # Confidence Message
        if confidence >= 0.80:

            st.success("✅ High Confidence Prediction")

        elif confidence >= 0.60:

            st.warning("⚠ Moderate Confidence Prediction")

        else:

            st.error("❗Low Confidence. Try another leaf image.")

# ----------------------------------------
# Footer
# ----------------------------------------

st.markdown("---")

st.caption(
    "🌿 Developed by Pooja Bisht | Powered by TensorFlow • MobileNetV2 • Streamlit"
)