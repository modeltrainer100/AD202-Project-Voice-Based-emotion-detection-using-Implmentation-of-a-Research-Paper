"""
Speech Emotion Recognition Streamlit App (FIXED)
Place this file in the same directory as your model files:
  - emotion_lstm_model.h5
  - best_model_label_encoder.joblib
  - best_model_feature_scaler.joblib
"""

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .emotion-label {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .confidence-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants (must match training)
SAMPLE_RATE = 22050
DURATION = 3.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

# Emotion colors
EMOTION_COLORS = {
    'angry': '#ff4444',
    'disgust': '#9c27b0',
    'fearful': '#ff9800',
    'happy': '#4caf50',
    'neutral': '#607d8b',
    'sad': '#2196f3',
    'surprised': '#ffeb3b',
    'calm': '#00bcd4'
}

# Emotion emojis
EMOTION_EMOJIS = {
    'angry': '😠',
    'disgust': '🤢',
    'fearful': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprised': '😲',
    'calm': '😌'
}

@st.cache_resource
def load_models():
    """Load pre-trained model, scaler, and label encoder"""
    try:
        model_paths = ['emotion_lstm_model.h5']
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                model = load_model(path)
                st.sidebar.success(f"✅ Loaded model: {path}")
                break
        
        if model is None:
            raise FileNotFoundError("No model file found")
        
        # Load label encoder
        le_paths = ['best_model_label_encoder.joblib', 'label_encoder.joblib']
        label_encoder = None
        for path in le_paths:
            if os.path.exists(path):
                label_encoder = joblib.load(path)
                break
        
        if label_encoder is None:
            raise FileNotFoundError("No label encoder file found")
        
        # Load scaler
        scaler_paths = ['best_model_feature_scaler.joblib', 'feature_scaler.joblib']
        scaler = None
        for path in scaler_paths:
            if os.path.exists(path):
                scaler = joblib.load(path)
                break
        
        if scaler is None:
            raise FileNotFoundError("No scaler file found")
        
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("""
        **Please ensure these files exist in the same directory:**
        - emotion_lstm_model.h5
        - best_model_label_encoder.joblib
        - best_model_feature_scaler.joblib
        """)
        st.stop()

def load_audio_file(audio_file, sr=SAMPLE_RATE, duration=DURATION):
    """Load and preprocess audio"""
    try:
        file_extension = audio_file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        audio, _ = librosa.load(tmp_path, sr=sr, mono=True, duration=duration)
        os.unlink(tmp_path)
        
        # Pad or truncate
        if len(audio) > SAMPLES_PER_TRACK:
            audio = audio[:SAMPLES_PER_TRACK]
        else:
            pad_len = SAMPLES_PER_TRACK - len(audio)
            audio = np.pad(audio, (0, pad_len), mode='constant')
        
        return audio
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None

def extract_mfcc_features(signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
                         n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Extract ONLY MFCC features (40 features) - FIXED VERSION"""
    try:
        # Extract MFCC coefficients
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, 
                                    n_fft=n_fft, hop_length=hop_length)
        
        # Transpose to (time_steps, n_mfcc)
        mfcc = mfcc.T
        
        return mfcc
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def predict_emotion(audio_signal, model, scaler, label_encoder):
    """Predict emotion from audio signal"""
    try:
        # Extract MFCC features (40 features only)
        features = extract_mfcc_features(audio_signal)
        if features is None:
            return None, None
        
        st.write(f"DEBUG: Features shape before scaling: {features.shape}")
        
        # Pad to model's expected timesteps
        model_timesteps = model.input_shape[1]
        t = features.shape[0]
        
        if t < model_timesteps:
            pad_len = model_timesteps - t
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
        else:
            features = features[:model_timesteps, :]
        
        st.write(f"DEBUG: Features shape after padding: {features.shape}")
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        st.write(f"DEBUG: Features shape after scaling: {features_scaled.shape}")
        
        # Reshape for LSTM: (batch_size, timesteps, features)
        features_scaled = features_scaled.reshape(1, model_timesteps, -1)
        
        st.write(f"DEBUG: Final input shape for model: {features_scaled.shape}")
        
        # Predict
        predictions = model.predict(features_scaled, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_emotion = label_encoder.inverse_transform([predicted_idx])[0]
        
        return predicted_emotion, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error(f"Exception details: {str(e)}")
        return None, None

def create_probability_chart(emotions, probabilities):
    """Create interactive bar chart of emotion probabilities"""
    colors = [EMOTION_COLORS.get(em.lower(), '#999999') for em in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probabilities * 100,
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Emotion Probability Distribution',
        xaxis_title='Emotion',
        yaxis_title='Confidence (%)',
        yaxis_range=[0, 105],
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_waveform_plot(audio_signal):
    """Create waveform visualization"""
    time = np.linspace(0, len(audio_signal) / SAMPLE_RATE, len(audio_signal))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=audio_signal,
        mode='lines',
        line=dict(color='#667eea', width=1),
        name='Amplitude'
    ))
    
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=300,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_spectrogram(audio_signal):
    """Create mel spectrogram visualization"""
    mel_spec = librosa.feature.melspectrogram(y=audio_signal, sr=SAMPLE_RATE)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig = px.imshow(
        mel_spec_db,
        aspect='auto',
        origin='lower',
        color_continuous_scale='Viridis',
        labels={'x': 'Time', 'y': 'Mel Frequency', 'color': 'dB'}
    )
    
    fig.update_layout(
        title='Mel Spectrogram',
        height=300,
        template='plotly_white'
    )
    
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">🎤 Speech Emotion Recognition</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered emotion detection from voice recordings</p>', 
                unsafe_allow_html=True)
    
    with st.spinner('Loading AI models...'):
        model, scaler, label_encoder = load_models()
    
    with st.sidebar:
        st.header("📊 About")
        st.info(
            "This app uses a deep learning LSTM model to recognize emotions "
            "from speech audio. Upload an audio file to get started!"
        )
        
        st.header("🎯 Supported Emotions")
        emotions = sorted(label_encoder.classes_)
        for emotion in emotions:
            emoji = EMOTION_EMOJIS.get(emotion.lower(), '🔹')
            st.write(f"{emoji} {emotion.capitalize()}")
        
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. Upload an audio file
        2. The AI analyzes the speech
        3. View the detected emotion and confidence scores
        4. Explore visualizations
        """)
        
        st.header("⚙️ Model Info")
        st.write(f"**Model Type:** LSTM")
        st.write(f"**Input Shape:** {model.input_shape}")
        st.write(f"**Feature Type:** MFCC (40 coefficients)")
        st.write(f"**Classes:** {len(emotions)}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎵 Upload Audio File")
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'wave', 'mp3', 'ogg', 'flac', 'm4a'],
            help="Upload a speech audio file"
        )
    
    if audio_file is not None:
        audio_format = audio_file.name.split('.')[-1].lower()
        if audio_format == 'mp3':
            st.audio(audio_file, format='audio/mp3')
        elif audio_format in ['ogg']:
            st.audio(audio_file, format='audio/ogg')
        elif audio_format in ['m4a']:
            st.audio(audio_file, format='audio/mp4')
        else:
            st.audio(audio_file, format='audio/wav')
        
        if st.button("🔍 Analyze Emotion", type="primary"):
            with st.spinner('Analyzing audio...'):
                audio_signal = load_audio_file(audio_file)
                
                if audio_signal is not None:
                    emotion, probabilities = predict_emotion(
                        audio_signal, model, scaler, label_encoder
                    )
                    
                    if emotion is not None:
                        st.success("✅ Analysis Complete!")
                        
                        emoji = EMOTION_EMOJIS.get(emotion.lower(), '🔹')
                        confidence = probabilities[np.argmax(probabilities)] * 100
                        
                        st.markdown(f"""
                        <div class="emotion-box">
                            <div class="emotion-label">{emoji} {emotion.upper()}</div>
                            <div class="confidence-label">Confidence: {confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        tab1, tab2, tab3 = st.tabs(
                            ["📊 Probabilities", "🌊 Waveform", "🎨 Spectrogram"]
                        )
                        
                        with tab1:
                            emotions_list = label_encoder.classes_
                            fig = create_probability_chart(emotions_list, probabilities)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Detailed Confidence Scores")
                            prob_data = sorted(
                                zip(emotions_list, probabilities),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            
                            for em, prob in prob_data:
                                emoji_em = EMOTION_EMOJIS.get(em.lower(), '🔹')
                                st.progress(float(prob), 
                                          text=f"{emoji_em} {em.capitalize()}: {prob*100:.2f}%")
                        
                        with tab2:
                            fig = create_waveform_plot(audio_signal)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Duration", f"{DURATION} sec")
                            col2.metric("Sample Rate", f"{SAMPLE_RATE} Hz")
                            col3.metric("Samples", f"{len(audio_signal):,}")
                        
                        with tab3:
                            fig = create_spectrogram(audio_signal)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info(
                                "The spectrogram shows the frequency content of the "
                                "audio over time. Brighter colors indicate stronger frequencies."
                            )
    
    else:
        st.info("👆 Please upload an audio file to begin emotion analysis")
        
        with st.expander("📋 Audio Requirements"):
            st.markdown("""
            **Supported Formats:** WAV, MP3, OGG, FLAC, M4A
            
            **Quality Tips:**
            - Clear speech recording
            - Minimal background noise
            - 3 seconds or longer
            - Single speaker preferred
            """)
    
    st.markdown("---")
    st.markdown("### 👥 Project Team")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Team Members:**
        - Aditya Upendra Gupta (AD24B1003)
        - Kartavya Gupta (AD24B1028)
        - Anshika Agarwal (AD24B107)
        """)
    
    with col2:
        st.markdown("""
        **Under the Guidance of:**  
        Dr. Gyanswar
        
        **Institution:**  
        IIIT Raichur
        """)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using TensorFlow and Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()