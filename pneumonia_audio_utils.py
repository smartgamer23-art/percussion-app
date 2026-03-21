# Utility functions for audio processing and spectrogram generation
# Used by the Streamlit application for percussion sound classification

import librosa  # Audio analysis and feature extraction
import numpy as np  # Numerical operations
import librosa.display  # Audio visualization utilities
import matplotlib.pyplot as plt  # Plotting library
from pathlib import Path  # Object-oriented filesystem paths

def audio_to_melspec(
    audio_path,
    sr=22050,      # Sample rate (22.05 kHz standard for speech/audio)
    duration=2.0,  # Duration to process (seconds)
    n_mels=128,    # Number of mel frequency bands
    fmin=50,       # Minimum frequency (Hz) - filters out very low frequencies
    fmax=2000      # Maximum frequency (Hz) - focuses on percussion sound range
):
    """
    Convert audio file to mel-spectrogram for percussion sound analysis.
    
    This function loads an audio file, normalizes its length, and converts
    it to a mel-spectrogram representation suitable for the classification model.
    Unlike the notebook version, this returns the waveform and sample rate
    for additional visualization in the web app.
    
    Args:
        audio_path: Path to the audio file (.wav format)
        sr: Target sample rate for audio resampling
        duration: Length of audio clip to process in seconds
        n_mels: Number of mel filterbanks (frequency bins)
        fmin: Minimum frequency to include in the mel-spectrogram
        fmax: Maximum frequency to include in the mel-spectrogram
    
    Returns:
        mel_db: Mel-spectrogram in decibels (dB scale)
        y: Audio waveform (time-series data)
        sr: Sample rate of the audio
    """
    # Load audio file, convert to mono, and limit duration
    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=duration)

    # Ensure consistent audio length by padding or trimming
    target_len = int(sr * duration)
    if len(y) < target_len:
        # Pad with zeros if audio is shorter than target
        y = np.pad(y, (0, target_len - len(y)))
    else:
        # Trim if audio is longer than target
        y = y[:target_len]

    # Convert audio waveform to mel-spectrogram
    # Mel scale better represents human perception of frequency
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,  # Filter out low-frequency noise
        fmax=fmax   # Focus on relevant percussion frequency range
    )

    # Convert power spectrogram to decibel (logarithmic) scale
    # This better represents human perception of loudness
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Return mel-spectrogram, waveform, and sample rate for visualization
    return mel_db, y, sr


def save_temp_spectrogram(mel, out_path):
    """
    Save mel-spectrogram as a temporary image file for model prediction.
    
    Creates a clean image representation of the mel-spectrogram without
    axes, labels, or padding. This format matches the training data format
    and is optimized for the image classification model.
    
    Args:
        mel: Mel-spectrogram array (2D numpy array in dB scale)
        out_path: Output file path for saving the image (PNG format)
    """
    # Create a 3x3 inch figure for consistent image dimensions
    plt.figure(figsize=(3,3))
    
    # Display the spectrogram using 'magma' colormap
    # Same colormap used during model training for consistency
    librosa.display.specshow(mel, cmap="magma")
    
    # Remove axes for a clean image suitable for neural network input
    plt.axis("off")
    
    # Save with tight bounding box to eliminate white space
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    
    # Close figure to free memory (important in web applications)
    plt.close()
