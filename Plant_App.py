import html
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PlantCare AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "plant_disease_model.keras"
CLASS_NAMES_PATH = Path(__file__).resolve().parent / "class_names.npy"
IMAGE_SIZE = (224, 224)


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background: radial-gradient(circle at 8% 4%, rgba(181,226,191,.22), transparent 26%),
                    linear-gradient(180deg,#f7fbf7 0%,#f1f7f1 100%);
    }
    [data-testid="stHeader"] { background: rgba(247,251,247,.85); }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
        background: linear-gradient(120deg,#103f2b 0%,#176443 58%,#26815a 100%);
        padding: 2.2rem 2.3rem; border-radius: 25px; color: white; margin-bottom: 1.4rem;
        box-shadow: 0 16px 40px rgba(20,78,49,.15);
    }
    .hero h1 {
        font-family: 'Manrope',sans-serif; font-size: clamp(2rem,4vw,3rem); font-weight: 800;
        letter-spacing: -1.4px; line-height: 1.12; margin: .35rem 0 .7rem; color: white;
    }
    .hero p { color: #e0f2e5; font-size: 1.03rem; max-width: 650px; margin-bottom: 0; }
    .eyebrow { text-transform: uppercase; letter-spacing: 2px; font-size: .72rem; font-weight: 700; color: #b9e9c8; }
    .hero-pill { display: inline-block; margin-top: 1.2rem; padding: .42rem .8rem; border: 1px solid rgba(255,255,255,.25); background: rgba(255,255,255,.10); border-radius: 999px; font-size: .8rem; color: #f0fff3; }
    .section-heading { font-family: 'Manrope',sans-serif; color: #153c2b; font-weight: 800; font-size: 1.25rem; margin: .4rem 0 .2rem; }
    .muted { color: #64766b; font-size: .92rem; }
    .panel { background: rgba(255,255,255,.9); border: 1px solid #e0ebe1; border-radius: 20px; padding: 1.2rem 1.3rem; box-shadow: 0 8px 24px rgba(28,67,42,.045); }
    .result-label { color: #5c7564; font-size: .78rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; }
    .result-name { font-family: 'Manrope',sans-serif; color: #17452f; font-weight: 800; font-size: clamp(1.3rem,2.3vw,1.8rem); line-height: 1.25; margin: .45rem 0 .8rem; overflow-wrap: anywhere; }
    .small-note { color: #66776b; font-size: .82rem; line-height: 1.5; }
    div.stButton > button {
        background: linear-gradient(100deg,#176443,#27865b); color: white; border: 0;
        border-radius: 12px; min-height: 3rem; font-weight: 700; transition: all .2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(100deg,#125235,#1e7149); color: white; border: 0;
        transform: translateY(-1px); box-shadow: 0 7px 18px rgba(23,100,67,.2);
    }
    [data-testid="stFileUploader"] { background: #fbfefb; border: 1px dashed #a9cbb3; border-radius: 15px; padding: .65rem; }
    [data-testid="stSidebar"] { background: #f0f7f0; border-right: 1px solid #e0ebe1; }
    .sidebar-brand { background: linear-gradient(135deg,#164a32,#287c52); color: white; padding: 1.15rem; border-radius: 17px; margin-bottom: 1rem; }
    .sidebar-brand h3 { color: white; margin: .2rem 0; font-family: 'Manrope',sans-serif; }
    .sidebar-brand p { color: #d9f0df; font-size: .85rem; margin-bottom: 0; }
    .footer { text-align: center; color: #718276; font-size: .82rem; padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div style="font-size:1.8rem;">🌿</div>
            <h3>PlantCare AI</h3>
            <p>Smart leaf health assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### About this project")
    st.write(
        "Upload a plant leaf photo or capture one with your camera "
        "to get an AI-generated disease-class prediction."
    )
    st.divider()
    st.markdown("**🧠 Model**")
    st.write("MobileNetV2")
    st.markdown("**🗂️ Dataset**")
    st.write("PlantVillage")
    st.markdown("**🧪 Framework**")
    st.write("TensorFlow · Streamlit")
    st.divider()
    st.caption("AI predictions are estimates, not confirmed diagnoses.")
    st.caption("Developer: Pooja Bisht")


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">PLANT HEALTH • AI ASSISTANT</div>
        <h1>A healthier leaf<br>starts with a closer look.</h1>
        <p>Upload a plant leaf photo to view the disease class predicted by your trained
        deep-learning model in a clean, easy-to-read dashboard.</p>
        <div class="hero-pill">🌱 MobileNetV2 &nbsp; · &nbsp; Image-based prediction</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL AND CLASS NAMES
# These definitions are intentionally at module level.
# =========================================================
@st.cache_resource(show_spinner="Loading plant disease model...")
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_class_names(class_names_path: str):
    names = np.load(class_names_path, allow_pickle=True)
    return [str(name) for name in names.tolist()]


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH.name}")
    st.info("Place plant_disease_model.keras in the same folder as app.py.")
    st.stop()

if not CLASS_NAMES_PATH.is_file():
    st.error(f"Class names file not found: {CLASS_NAMES_PATH.name}")
    st.info("Place class_names.npy in the same folder as app.py.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    class_names = load_class_names(str(CLASS_NAMES_PATH))
except Exception as exc:
    st.error("The model or class names could not be loaded.")
    st.exception(exc)
    st.stop()

# Validate model output and labels before user submits an image.
try:
    output_shape = model.output_shape
    output_count = int(output_shape[-1])
    if output_count != len(class_names):
        st.error(
            f"Model/class mismatch: model predicts {output_count} classes, "
            f"but class_names.npy contains {len(class_names)} labels. "
            "Use the class_names.npy generated for this exact trained model."
        )
        st.stop()
except (AttributeError, TypeError, ValueError, IndexError):
    # Some custom models do not expose a simple output_shape.
    pass


# =========================================================
# IMAGE INPUT
# =========================================================
st.markdown(
    '<div class="section-heading">🔎 Check your plant leaf</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="muted">Choose a clear, well-lit photo where the leaf is easy to see.</div>',
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1.02, 0.98], gap="large")

with left:
    input_method = st.radio(
        "Choose how to add a photo",
        ["📁 Upload image", "📷 Use camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image = None

    if input_method == "📁 Upload image":
        uploaded_file = st.file_uploader(
            "Drop a leaf photo here",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except (UnidentifiedImageError, OSError):
                st.error("This image could not be opened. Please try another JPG or PNG.")
    else:
        camera_image = st.camera_input("Take a leaf photo")
        if camera_image is not None:
            try:
                image = Image.open(camera_image).convert("RGB")
            except (UnidentifiedImageError, OSError):
                st.error("This camera image could not be opened. Please try again.")

    st.markdown(
        '<div class="small-note">Tip: avoid blurry photos, strong shadows, and backgrounds '
        'that hide most of the leaf.</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div class="section-heading">🖼️ Image preview</div>',
        unsafe_allow_html=True,
    )
    if image is not None:
        st.image(image, caption="Selected leaf image", use_container_width=True)
    else:
        st.markdown(
            """
            <div class="panel" style="min-height:260px;display:flex;align-items:center;
            justify-content:center;text-align:center;">
                <div>
                    <div style="font-size:3rem;">🍃</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#214b34;">
                        Your leaf photo will appear here
                    </div>
                    <div class="small-note" style="margin-top:.5rem;">
                        Upload an image or use your camera to get started.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PREDICTION
# =========================================================
st.write("")
if image is not None and st.button(
    "🌿 Analyze leaf", type="primary", use_container_width=True
):
    try:
        with st.spinner("Examining the leaf image..."):
            resized = image.resize(IMAGE_SIZE)
            image_array = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(image_array, axis=0)

            predictions = model.predict(batch, verbose=0)
            scores = np.asarray(predictions[0], dtype=np.float32).reshape(-1)

            if len(scores) != len(class_names):
                st.error(
                    f"Model returned {len(scores)} scores, but there are "
                    f"{len(class_names)} class names. Please check that the model "
                    "and class_names.npy belong together."
                )
                st.stop()

            predicted_index = int(np.argmax(scores))
            confidence = float(scores[predicted_index])
            disease_raw = class_names[predicted_index]
            disease_display = disease_raw.replace("___", " → ").replace("_", " ")

        st.markdown("---")
        st.markdown(
            '<div class="section-heading">📋 Prediction report</div>',
            unsafe_allow_html=True,
        )

        result_col, confidence_col = st.columns([1.35, 0.65], gap="large")

        with result_col:
            safe_disease_display = html.escape(disease_display)
            st.markdown(
                f"""
                <div class="panel">
                    <div class="result-label">Predicted class</div>
                    <div class="result-name">{safe_disease_display}</div>
                    <div class="small-note">
                        This is the class selected by the model for the supplied image.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with confidence_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.metric("🎯 Model confidence", f"{confidence * 100:.1f}%")
            st.progress(max(0, min(100, int(confidence * 100))))
            if confidence >= 0.80:
                st.success("High model confidence")
            elif confidence >= 0.60:
                st.warning("Moderate model confidence")
            else:
                st.error("Low model confidence")
            st.markdown("</div>", unsafe_allow_html=True)

        st.info(
            "Confidence is the model's score for its selected class; it does not guarantee "
            "that the prediction is correct. Use a clear photo and seek expert advice "
            "before making plant-treatment decisions."
        )

    except Exception as exc:
        st.error("Prediction could not be completed. Check the model and image, then try again.")
        st.exception(exc)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        🌿 PlantCare AI &nbsp;·&nbsp; Powered by TensorFlow, MobileNetV2 & Streamlit<br>
        <span style="font-size:.75rem;">A mini-project for plant leaf disease prediction</span>
    </div>
    """,
    unsafe_allow_html=True,
)
