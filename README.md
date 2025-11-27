# 🎭 Voice-Based Emotion Detection Using Dual-Layer LSTM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**An advanced Speech Emotion Recognition system achieving 83.26% accuracy**

[Research Paper](#-research-foundation) • [Architecture](#-architecture-deep-dive) • [Results](#-performance-metrics) • [Demo](#-live-demo) • [Getting Started](#-quick-start)

</div>

---

## 🌟 Project Highlights

<table>
<tr>
<td width="50%">

### 🎯 **Achievement Unlocked**
- **83.26%** Test Accuracy
- **<100ms** Inference Time
- **11,680** Training Samples
- **8** Emotion Classes

</td>
<td width="50%">

### 🚀 **Key Features**
- Dual-layer LSTM architecture
- Real-time emotion prediction
- Interactive web interface
- Comprehensive visualization

</td>
</tr>
</table>

---

## 📊 Performance Journey

```
Initial Model (REVDAS only)
├─ Dataset Size: 1,440 samples
├─ Accuracy Range: 57% - 73%
└─ Best Accuracy: 73.88%

Final Model (Expanded Dataset)
├─ Dataset Size: 11,680 samples
├─ Accuracy Range: 76% - 83%
└─ Best Accuracy: 83.26% ✨
```

### 📈 Improvement Breakdown

| Metric | Initial Model | Final Model | Improvement |
|--------|--------------|-------------|-------------|
| **Test Accuracy** | 73.88% | 83.26% | **+9.38%** |
| **Test Loss** | 0.71 | 0.52 | **-26.8%** |
| **Dataset Size** | 1,440 | 11,680 | **+710%** |

---

## 🎓 Research Foundation

### Paper Implementation
This project implements and extends the methodology from:

> **"Improvement and Implementation of a Speech Emotion Recognition Model Based on Dual-Layer LSTM"**  
> *by Xiaoran Yang, Shuhan Yu, and Wenxi Xu*

### 🔬 Our Contributions

1. **📦 Dataset Expansion** - Integrated multiple datasets for robust training
2. **🎛️ Hyperparameter Optimization** - 500+ configurations tested
3. **🖥️ Production Deployment** - Streamlit-based interactive application
4. **📊 Comprehensive Analysis** - In-depth performance evaluation

---

## 🗂️ Datasets Used

<div align="center">

| Dataset | Samples | Speakers | Description |
|---------|---------|----------|-------------|
| **REVDAS** | 1,440 | - | Initial training baseline |
| **RAVDESS** | ✓ | 24 actors | Professional emotional speech |
| **CREMA-D** | ✓ | - | Diverse speaker demographics |
| **SAVEE** | ✓ | - | British English speakers |
| **TESS** | ✓ | - | Toronto emotional speech |
| **MELD** | ✓ | - | Multi-party conversations |

**Total Training Samples:** 11,680 files

</div>

### 🎯 Emotion Classes

```python
emotions = [
    '😠 Angry',
    '😌 Calm', 
    '🤢 Disgust',
    '😨 Fearful',
    '😊 Happy',
    '😐 Neutral',
    '😢 Sad',
    '😲 Surprised'
]
```

---

## 🏗️ Architecture Deep Dive

### Model Structure

```
┌─────────────────────────────────────────────────┐
│            Input Audio (3 seconds)              │
│               22,050 Hz, Mono                   │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│          Feature Extraction Pipeline            │
│  • 40 MFCCs + Deltas                           │
│  • Chroma Features                              │
│  • Mel Spectrogram                              │
│  • Spectral Contrast                            │
│  • Tonnetz                                      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│       LSTM Layer 1 (256 units)                  │
│       return_sequences=True                     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│    Batch Normalization + Dropout (0.4)          │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│       LSTM Layer 2 (128 units)                  │
│       return_sequences=False                    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│    Batch Normalization + Dropout (0.5)          │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│       Dense Layer (256 units, ReLU)             │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│     Output Layer (8 units, Softmax)             │
│        Emotion Probabilities                    │
└─────────────────────────────────────────────────┘
```

### 🎼 Feature Extraction Details

| Feature Type | Dimensions | Purpose |
|-------------|-----------|---------|
| **MFCCs** | 40 coefficients | Capture timbre and spectral envelope |
| **Delta MFCCs** | 40 coefficients | Track temporal changes |
| **Chroma** | 12 bins | Represent harmonic content |
| **Mel Spectrogram** | Variable | Frequency representation |
| **Spectral Contrast** | 7 bands | Texture information |
| **Tonnetz** | 6 features | Harmonic relationships |

---

## 🔧 Optimal Configuration

After 500+ hyperparameter searches, the winning configuration:

```python
{
    "architecture": {
        "lstm_1_units": 256,
        "lstm_2_units": 128,
        "dense_units": 256,
        "dropout_lstm": 0.4,
        "dropout_dense": 0.5
    },
    "training": {
        "batch_size": 128,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy"
    },
    "preprocessing": {
        "sample_rate": 22050,
        "duration": 3.0,
        "n_mfcc": 40
    }
}
```

---

## 📊 Performance Metrics

### 🎯 Confusion Matrix Insights

**High Performance Emotions:**
- ✅ **Angry** - Distinctive high arousal patterns
- ✅ **Fearful** - Clear acoustic signatures
- ✅ **Disgust** - Strong spectral characteristics
- ✅ **Calm** - Low arousal, stable features

**Challenging Emotions:**
- ⚠️ **Happy vs Sad** - Similar energy profiles
- ⚠️ **Neutral** - Overlaps with low-arousal states

### ⚡ Inference Performance

```
┌──────────────────────────────────────┐
│   End-to-End Processing Pipeline     │
├──────────────────────────────────────┤
│  Audio Loading:         ~20ms        │
│  Feature Extraction:    ~50ms        │
│  Model Inference:       ~15ms        │
│  Visualization:         ~10ms        │
├──────────────────────────────────────┤
│  Total Time:           <100ms ✨      │
└──────────────────────────────────────┘
```

---

## 🖥️ Live Demo

### Streamlit Application Features

<table>
<tr>
<td width="33%" align="center">

#### 📤 Upload
Drag & drop or browse audio files

</td>
<td width="33%" align="center">

#### 🔮 Predict
Real-time emotion detection

</td>
<td width="33%" align="center">

#### 📊 Visualize
Interactive charts & plots

</td>
</tr>
</table>

### Sample Output

```
🎵 Audio Analysis Complete!

Detected Emotion: 😊 Happy (87.3% confidence)

Emotion Probabilities:
  Happy:     ████████████████████████████ 87.3%
  Surprised: ████████ 8.2%
  Neutral:   ███ 2.1%
  Calm:      ██ 1.4%
  Others:    █ 1.0%
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda
```

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/emotion-recognition-lstm.git
cd emotion-recognition-lstm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 💻 Usage

#### Train Model
```bash
python src/train.py \
  --dataset expanded \
  --epochs 100 \
  --batch_size 128 \
  --lstm_units 256 128
```

#### Run Inference
```python
from src.model import EmotionRecognizer

# Initialize model
model = EmotionRecognizer.load('models/best_model.h5')

# Predict emotion
emotion, confidence = model.predict('audio_sample.wav')
print(f"Emotion: {emotion} ({confidence:.2%})")
```

#### Launch Web App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
emotion-recognition-lstm/
│
├── 📂 data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed audio
│   └── features/         # Extracted features
│
├── 📂 models/
│   ├── checkpoints/      # Training checkpoints
│   └── final/            # Best model (83.26%)
│
├── 📂 src/
│   ├── preprocessing.py  # Audio preprocessing
│   ├── features.py       # Feature extraction
│   ├── model.py          # LSTM architecture
│   ├── train.py          # Training pipeline
│   └── evaluate.py       # Performance metrics
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Features.ipynb
│   ├── 03_Training.ipynb
│   └── 04_Analysis.ipynb
│
├── 📂 app/
│   └── streamlit_app.py  # Web interface
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 report.pdf
```

---

## 🔬 Key Findings

### 1️⃣ Dataset Size Matters Most

Expanding from 1,440 to 11,680 samples provided the **largest performance boost** (+9.38% accuracy), far exceeding architectural improvements alone.

### 2️⃣ Dual-Layer LSTM Effectiveness

The hierarchical structure proves essential:
- **Layer 1:** Captures short-term acoustic details
- **Layer 2:** Learns abstract emotional patterns

### 3️⃣ Regularization is Critical

Dropout rates of 0.4-0.5 prevented overfitting while maintaining strong generalization.

### 4️⃣ Feature Diversity Helps

Combining MFCCs, Chroma, Mel-spectrograms, and Spectral Contrast created a robust representation resistant to speaker variability.

---

## 🎯 Limitations & Future Work

### Current Limitations

- 🎬 **Acted Speech:** Limited to professional recordings
- 🌍 **English Only:** No multilingual support
- 🔇 **Clean Audio:** Minimal noise robustness testing
- 📊 **Discrete Classes:** Continuous emotion space not modeled

### 🚀 Future Improvements

- [ ] Real-world spontaneous speech testing
- [ ] Multi-language emotion recognition
- [ ] Noise robustness enhancement
- [ ] Continuous emotion dimension modeling
- [ ] Attention mechanism integration
- [ ] Transfer learning from pre-trained models

---

## 👥 Team

**Indian Institute of Information Technology (IIIT) Raichur**

- **Aditya Upendra Gupta** (AD24B1003)
- **Anshika Agarwal** (AD24B1007)
- **Kartavya Gupta** (AD24B1028)

**Supervisor:** Dr. Dubacharla Gyaneshwar

---

## 📚 References

<details>
<summary>Click to expand full reference list</summary>

1. Yang, X., Yu, S., & Xu, W. "Improvement and Implementation of a Speech Emotion Recognition Model Based on Dual-Layer LSTM"

2. Livingstone, S. R., & Russo, F. A. (2018). "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)", *PLOS ONE*

3. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory", *Neural Computation*

4. El Ayadi, M., Kamel, M. S., & Karray, F. (2011). "Survey on Speech Emotion Recognition", *Pattern Recognition*

5. McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python"

</details>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- RAVDESS Dataset creators
- TensorFlow/Keras development team
- Streamlit for deployment framework
- Research paper authors for methodology

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star!

**Made with ❤️ by IIIT Raichur Students**

[Report Issues](https://github.com/your-username/emotion-recognition-lstm/issues) • [Request Features](https://github.com/your-username/emotion-recognition-lstm/issues)

</div>
