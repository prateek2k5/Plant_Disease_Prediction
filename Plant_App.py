import html
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.npy"
IMAGE_SIZE = (224, 224)


# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #07130e;
    --surface: #102219;
    --surface-light: #162d20;
    --green: #39d98a;
    --green-light: #a4f4c5;
    --muted: #91a99a;
    --border: rgba(139, 210, 162, 0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(34,197,94,.13), transparent 35%),
        radial-gradient(ellipse at 90% 40%, rgba(16,185,129,.07), transparent 30%),
        linear-gradient(145deg, #07130e, #0a1911 55%, #07120d);
    color: #e7f5eb;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label, .stMarkdown {
    color: #e7f5eb;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 3rem;
    border-radius: 28px;
    background:
        radial-gradient(circle at 90% 20%, rgba(74,222,128,.22), transparent 30%),
        linear-gradient(120deg, #102b1c, #145232 65%, #17633b);
    border: 1px solid rgba(167,243,208,.2);
    box-shadow: 0 20px 65px rgba(0,0,0,.28);
    margin-bottom: 2rem;
}

.hero:after {
    content: "🌿";
    position: absolute;
    right: 6%;
    top: 15%;
    font-size: 8rem;
    opacity: .13;
    transform: rotate(-20deg);
}

.eyebrow {
    color: #a7f3c0;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: .73rem;
    font-weight: 700;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -2px;
    margin: 1rem 0;
    color: white;
}

.hero p {
    color: #c7e8d2;
    font-size: 1.05rem;
    max-width: 580px;
    line-height: 1.8;
}

.hero-tag {
    display: inline-block;
    margin-top: 1.3rem;
    padding: .55rem 1rem;
    border-radius: 30px;
    border: 1px solid rgba(255,255,255,.2);
    background: rgba(0,0,0,.15);
    color: #d8ffe5;
    font-size: .83rem;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.35rem;
    color: #b5f4cb;
    margin-bottom: .35rem;
}

.section-subtitle {
    color: var(--muted);
    font-size: .9rem;
    margin-bottom: 1.3rem;
}

.panel {
    background: linear-gradient(145deg, #142a1d, #0e2016);
    border: 1px solid var(--border);
    border-radius: 21px;
    padding: 1.5rem;
    box-shadow: 0 12px 35px rgba(0,0,0,.18);
}

.result-label {
    color: #8eaf99;
    font-size: .75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 700;
}

.result-name {
    color: #a7f3c0;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.3rem, 2.5vw, 2rem);
    font-weight: 800;
    line-height: 1.35;
    margin-top: .65rem;
    overflow-wrap: anywhere;
}

.small-note {
    color: #a1b8a8;
    font-size: .82rem;
    line-height: 1.7;
}

.sidebar-brand {
    background: linear-gradient(135deg, #153d27, #17643e);
    border: 1px solid rgba(134,239,172,.2);
    padding: 1.3rem;
    border-radius: 19px;
    margin-bottom: 1.5rem;
}

.sidebar-brand h2 {
    color: white;
    font-family: 'Manrope', sans-serif;
    font-size: 1.45rem;
    margin: .5rem 0 .2rem;
}

.sidebar-brand p {
    color: #c8ead4;
    font-size: .85rem;
    margin: 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #09170f, #0d1d13);
    border-right: 1px solid #203b2a;
}

[data-testid="stSidebar"] * {
    color: #e0f1e5;
}

div.stButton > button {
    min-height: 3.2rem;
    border-radius: 13px;
    border: 1px solid #35b86e;
    background: linear-gradient(110deg, #16803e, #27b66c);
    color: white;
    font-weight: 700;
    transition: all .2s ease;
}

div.stButton > button:hover {
    background: linear-gradient(110deg, #126331, #159957);
    border-color: #86efac;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(34,197,94,.2);
}

[data-testid="stFileUploader"] {
    background: #11251a;
    border: 1px dashed #44875a;
    border-radius: 17px;
    padding: .8rem;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-baseweb="select"] > div,
[data-testid="stTextInput"] input {
    background: #14251a;
    color: white;
    border-color: #355541;
}

[data-testid="stMetric"] {
    background: #14291d;
    border: 1px solid #31513b;
    padding: 1rem;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #b8cdbd !important;
}

[data-testid="stMetricValue"] {
    color: #a7f3c0 !important;
}

[data-testid="stProgressBar"] > div {
    background: #234632;
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

.footer {
    text-align: center;
    color: #829d8a;
    font-size: .82rem;
    padding-top: 1.5rem;
}

.footer strong {
    color: #a7f3c0;
}

hr {
    border-color: #294333;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:2rem;">🌿</div>
        <h2>PlantCare AI</h2>
        <p>Intelligent plant health analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### About PlantCare")
    st.write(
        "An AI-powered plant health assistant that analyzes "
        "leaf images and identifies possible disease classes."
    )

    st.divider()

    st.markdown("**🧠 AI Technology**")
    st.caption("MobileNetV2")

    st.markdown("**🌱 Dataset**")
    st.caption("PlantVillage")

    st.markdown("**⚡ Powered by**")
    st.caption("TensorFlow")

    st.divider()

    st.caption("Developed by")
    st.markdown("**Prateek Verma**")

    st.caption("Predictions are AI-generated estimates.")


# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ SMART PLANT HEALTH ANALYSIS</div>
    <h1>Grow healthier.<br>Live greener.</h1>
    <p>
        Discover what's happening with your plants.
        Upload a leaf image and let artificial intelligence
        analyze its health in seconds.
    </p>
    <div class="hero-tag">
        🌿 AI-powered &nbsp; · &nbsp; Fast &nbsp; · &nbsp; Simple
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_class_names(path):
    names = np.load(path, allow_pickle=True)
    return [str(name) for name in names.tolist()]


if not MODEL_PATH.is_file():
    st.error("Model file not found: plant_disease_model.keras")
    st.stop()

if not CLASS_NAMES_PATH.is_file():
    st.error("Class names file not found: class_names.npy")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    class_names = load_class_names(str(CLASS_NAMES_PATH))
except Exception as exc:
    st.error("Unable to load the AI model or class names.")
    st.exception(exc)
    st.stop()

try:
    output_shape = model.output_shape
    if int(output_shape[-1]) != len(class_names):
        st.error(
            "Model output and class names do not match. "
            "Please use the correct class_names.npy."
        )
        st.stop()
except (AttributeError, TypeError, ValueError, IndexError):
    pass


# =========================================================
# IMAGE UPLOAD SECTION
# =========================================================
st.markdown(
    '<div class="section-title">Explore your plant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload a clear leaf image to begin the analysis.'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")

image = None

with left:
    with st.container(border=True):
        st.markdown("### 📸 Add a leaf image")

        input_method = st.radio(
            "Select image source",
            ["📁 Upload image", "📷 Camera"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if input_method == "📁 Upload image":
            uploaded_file = st.file_uploader(
                "Choose your plant image",
                type=["jpg", "jpeg", "png"],
                help="Upload a JPG or PNG image.",
            )

            if uploaded_file:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                except (UnidentifiedImageError, OSError):
                    st.error("Unable to read this image.")

        else:
            camera_image = st.camera_input("Capture your leaf")

            if camera_image:
                try:
                    image = Image.open(camera_image).convert("RGB")
                except (UnidentifiedImageError, OSError):
                    st.error("Unable to read the camera image.")

        st.caption(
            "Tip: Use a well-lit photo with the leaf clearly visible."
        )

with right:
    st.markdown("### 🖼️ Image preview")

    if image is not None:
        st.image(
            image,
            caption="Selected plant image",
            use_container_width=True,
        )
    else:
        st.markdown("""
        <div class="panel" style="
            min-height:290px;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
        ">
            <div>
                <div style="font-size:4rem;">🍃</div>
                <div style="font-size:1.1rem;font-weight:700;color:#a7f3c0;">
                    Your plant image goes here
                </div>
                <div class="small-note">
                    Upload or capture an image to get started.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================
st.write("")

if image is not None:
    if st.button(
        "🌿  Analyze Plant Health",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("AI is analyzing your plant..."):

                resized = image.resize(IMAGE_SIZE)

                image_array = np.asarray(
                    resized,
                    dtype=np.float32,
                )

                batch = np.expand_dims(image_array, axis=0)

                predictions = model.predict(batch, verbose=0)

                scores = np.asarray(
                    predictions[0],
                    dtype=np.float32,
                ).reshape(-1)

                if len(scores) != len(class_names):
                    st.error("Prediction output does not match class names.")
                    st.stop()

                predicted_index = int(np.argmax(scores))

                confidence = float(scores[predicted_index])

                if not np.isfinite(confidence):
                    st.error("The model returned an invalid prediction.")
                    st.stop()

                disease_raw = class_names[predicted_index]

                disease_display = (
                    disease_raw
                    .replace("___", " → ")
                    .replace("_", " ")
                )

                safe_disease = html.escape(disease_display)

                top_indices = np.argsort(scores)[-3:][::-1]

            st.divider()

            st.markdown(
                '<div class="section-title">✨ Analysis Results</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-subtitle">'
                'Here is what our AI model found.'
                '</div>',
                unsafe_allow_html=True,
            )

            result_col, confidence_col = st.columns(
                [1.4, 0.8],
                gap="large",
            )

            with result_col:
                st.markdown(f"""
                <div class="panel">
                    <div class="result-label">
                        DETECTED PLANT CONDITION
                    </div>
                    <div class="result-name">
                        {safe_disease}
                    </div>
                    <div class="small-note">
                        This is the disease class predicted by the AI model
                        for your uploaded leaf image.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with confidence_col:
                with st.container(border=True):
                    st.metric(
                        "🎯 Prediction Confidence",
                        f"{confidence * 100:.2f}%",
                    )

                    st.progress(
                        float(np.clip(confidence, 0, 1))
                    )

                    if confidence >= 0.80:
                        st.success("High confidence")
                    elif confidence >= 0.60:
                        st.warning("Moderate confidence")
                    else:
                        st.error("Low confidence")

            st.write("")

            # TOP PREDICTIONS
            st.markdown("### 📊 Top predictions")

            for index in top_indices:
                label = html.escape(
                    class_names[int(index)]
                    .replace("___", " → ")
                    .replace("_", " ")
                )

                score = float(scores[index])

                st.markdown(
                    f"**{label}** — {score * 100:.2f}%"
                )

                st.progress(float(np.clip(score, 0, 1)))

            st.info(
                "AI predictions are estimates and do not guarantee "
                "a confirmed diagnosis. Consult an agricultural "
                "expert before applying treatments."
            )

        except Exception as exc:
            st.error(
                "Something went wrong during prediction. "
                "Please check your model and try again."
            )
            st.exception(exc)


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown("""
<div class="footer">
    <div style="font-size:1.5rem;">🌿</div>
    <strong>PlantCare AI</strong>
    <br>
    Intelligent plant health analysis
    <br><br>
    Designed & Developed by <strong>Prateek Verma</strong>
</div>
""", unsafe_allow_html=True)
