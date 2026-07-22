# Streamlit se web app banane ke liye
import streamlit as st

# Trained Deep Learning model load karne ke liye
import tensorflow as tf

# Arrays aur mathematical operations ke liye
import numpy as np

# Uploaded image ko read aur process karne ke liye
from PIL import Image

# Web app ki basic settings

st.set_page_config(

    page_title="Plant Disease Detection",   # Browser tab ka title

    page_icon="🌿",                         # Browser tab icon

    layout="centered"                      # Page center me dikhega
)

# Main Heading
st.title("🌿 Plant Disease Detection")

# Short Description
st.write("Upload a leaf image to predict the disease.")

# Model ko sirf ek hi baar load karega.
# Har refresh par dobara load nahi hoga.
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

# Function call karke model load kar rahe hain.
model = load_model()

# Saved class names load kar rahe hain.

# Prediction ke baad index ko
# disease name me convert karenge.
class_names = np.load("class_names.npy",allow_pickle=True)

# User image upload karega.
uploaded_file = st.file_uploader("Choose a Leaf Image",type=["jpg","jpeg","png"])

# Uploaded image ko model ke according
# preprocess aur predict karega.

def predict(image):

    # Image ko 224x224 me resize kar rahe hain.
    image = image.resize((224,224))

    # PIL Image ko NumPy Array me convert kar rahe hain.
    image = np.array(image)

    # Pixel values ko 0-1 range me convert kar rahe hain.
    image = image.astype("float32") / 255.0

    # Batch Dimension add kar rahe hain.
    image = np.expand_dims(image, axis=0)

    # Prediction karwa rahe hain.
    prediction = model.predict(image, verbose=0)

    # Highest probability wala index.
    predicted_index = np.argmax(prediction)

    # Highest confidence value.
    confidence = np.max(prediction)

    # Disease Name aur Confidence return karega.
    return class_names[predicted_index], confidence

# User directly camera se photo click bhi kar sakta hai.
camera_image = st.camera_input("📷 Capture Leaf Image")

# Agar image upload hui hai.
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

# Agar camera se image click hui hai.
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

# Agar dono me se koi image mili.
else:
    image = None

# Agar image available hai.
if image is not None:

    # Uploaded/Captured image show karega.
    st.image(image, caption="Selected Image", use_container_width=True)

    # Prediction tabhi hoga jab button click hoga.
    if st.button("🔍 Predict Disease", use_container_width=True):

        with st.spinner("🔍 Analyzing Leaf..."):

            # Prediction Function call karenge.
            disease, confidence = predict(image)

        # Folder name ko readable format me convert karenge.
        disease = disease.replace("___", " → ")
        disease = disease.replace("_", " ")

        # Prediction Result
        st.success(f"🌿 Predicted Disease : {disease}")

        # Confidence Percentage
        st.metric(
            label="🎯 Confidence",
            value=f"{confidence*100:.2f}%"
        )

        # Confidence Progress Bar
        st.progress(int(confidence * 100))

        st.markdown("---")

        st.subheader("Prediction Result")

        st.write("### 🌿 Disease")
        st.success(disease)

        st.write("### 🎯 Confidence")
        st.write(f"{confidence*100:.2f}%")

st.markdown("---")
st.caption("Made with ❤️ using TensorFlow, MobileNetV2 & Streamlit")