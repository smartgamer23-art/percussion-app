# Import necessary libraries for the Streamlit web application
import streamlit as st  # Web framework for creating interactive ML applications
from fastai.vision.all import load_learner, PILImage  # fastai utilities for model loading and image handling
import tempfile  # Create temporary files for audio and spectrogram processing
import matplotlib.pyplot as plt  # Plotting library for visualizations
import librosa.display  # Audio visualization utilities
import pathlib  # For pathlib compatibility fix

# Import custom audio processing functions
from pneumonia_audio_utils import audio_to_melspec, save_temp_spectrogram

# -----------------------------
# App configuration
# -----------------------------
# Configure Streamlit page settings for optimal user experience
st.set_page_config(
    page_title="Medical Percussion Test AI",  # Browser tab title
    layout="centered"  # Center the content for focused presentation
)

# Display main application title and subtitle
st.title("🫁 AI-Assisted Percussion Test")
st.subheader("Acoustic Classification of Chest Percussion Sounds")

# Display application description and classification categories
# Explains the medical percussion test and what the AI model detects
st.markdown("""
This tool analyzes percussion sounds and classifies them as:

- **Resonant** → Air-filled lung (normal)
- **Dull** → Fluid-filled lung (pneumonia-like)

*For educational and competition use only.*
""")

# -----------------------------
# Load model
# -----------------------------
# Cache the model loading to improve performance and avoid reloading on each interaction
# The @st.cache_resource decorator ensures the model is loaded only once
@st.cache_resource
def load_model():
    """
    Load the pre-trained percussion classifier model.

    Returns:
        learner: Trained fastai vision learner for percussion sound
classification
    """
    # Temporary fix for pathlib compatibility issues when loading models saved on different OS
    original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.PurePosixPath
    
    try:
        learner = load_learner("percussion_classifier.pkl")
    finally:
        pathlib.PosixPath = original_posix_path
    
    return learner

# Initialize the model once at application startup
learn = load_model()

# -----------------------------
# Shared processing function
# -----------------------------
def process_and_display(audio_path):
    """Run the full pipeline (spectrogram → prediction → display) for
a saved WAV file."""
    mel, y, sr = audio_to_melspec(audio_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        save_temp_spectrogram(mel, tmp_img.name)
        spec_img = tmp_img.name

    pred_class, pred_idx, probs = learn.predict(PILImage.create(spec_img))
    confidence = probs[pred_idx].item() * 100

    if pred_class == "air":
        diagnosis = "🟢 Resonant Percussion"
        explanation = "Findings consistent with air-filled lung tissue."
    else:
        diagnosis = "🔴 Dull Percussion"
        explanation = "Findings suggest fluid-filled lung (pneumonia-like)."

    st.markdown("## 🧠 AI Interpretation")
    st.markdown(f"### **{diagnosis}**")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    st.markdown(explanation)

    st.markdown("## 🔊 Audio Signal")
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    st.markdown("## 📊 Spectrogram Analysis")
    fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel",
cmap="magma", ax=ax)
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)

    st.markdown("---")
    st.caption("AI model trained on simulated percussion sounds (balloon model).")


# -----------------------------
# Audio input — two tabs
# -----------------------------
tab_record, tab_upload = st.tabs(["🎙️ Record", "📂 Upload WAV"])

# --- Record tab ---
with tab_record:
    if "record_key" not in st.session_state:
        st.session_state.record_key = 0

    audio_bytes = st.audio_input(
        "Tap to start · Tap again to stop",
        key=f"recorder_{st.session_state.record_key}"
    )
    if audio_bytes:
        if st.button("🔄 Record Again", type="primary",
use_container_width=True):
            st.session_state.record_key += 1
            st.rerun()

    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes.read())
            audio_path = tmp.name
        process_and_display(audio_path)

# --- Upload tab ---
with tab_upload:
    uploaded_file = st.file_uploader("Upload a percussion WAV file",
type=["wav"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name
        process_and_display(audio_path)