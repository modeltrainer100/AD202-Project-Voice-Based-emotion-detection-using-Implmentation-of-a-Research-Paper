# 🎭 Speech Emotion Recognition Using Dual-Layer LSTM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow.svg" alt="Status">
</p>

## 📄 About the Paper

This project implements the research paper **"Improvement and Implementation of a Speech Emotion Recognition Model Based on Dual-Layer LSTM"** by Xiaoran Yang, Shuhan Yu, and Wenxi Xu.

<p align="center">
  <img src="path/to/your/paper_screenshot.png" alt="Paper Header" width="700">
</p>

### 📚 Paper Overview

The paper presents an enhanced speech emotion recognition (SER) system that builds upon existing models by introducing an additional LSTM layer to improve accuracy and computational efficiency. 

**Key Innovations:**
- 🧠 **Dual-Layer LSTM Architecture**: Captures long-term dependencies in audio sequences
- 📈 **2% Accuracy Improvement**: Outperforms single-layer LSTM models
- ⚡ **Reduced Latency**: Enhanced real-time performance
- 🎯 **Complex Pattern Recognition**: Better extraction of emotional features from noisy audio

The dual-layer architecture addresses limitations of single-layer LSTM structures in extracting emotional features from audio data, especially when dealing with noisy or complex emotional shifts in speech.

---

## 🎯 Project Objective

We are implementing this paper **from scratch** as a comprehensive machine learning project. This is a complete ground-up implementation where we:

✅ Design and code the dual-layer LSTM architecture from scratch  
✅ Implement custom feature extraction pipelines for audio data  
✅ Train the model on emotion-labeled speech datasets  
✅ Evaluate performance metrics against baseline models  
✅ Reproduce and validate the research findings  

---

## 📊 Dataset: RAVDESS

We are using the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset, which was also validated in the original paper.

### Dataset Characteristics

| Feature | Description |
|---------|-------------|
| **Actors** | 24 professional actors (12 male, 12 female) |
| **Emotions** | 7 categories: neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| **Modality** | Audio recordings of emotional speech |
| **Quality** | Professionally recorded with controlled acoustic conditions |
| **Balance** | Equal representation across emotions and genders |

### Why RAVDESS?

- ✨ Industry-standard benchmark for SER research
- 🎯 Clean, labeled data ideal for supervised learning
- 📊 Sufficient size for training deep learning models
- 🔬 Enables direct comparison with paper's reported results

**Dataset Link:** [RAVDESS on Kaggle](https://www.kaggle.com/uwrfkaggle/ravdess-emotional-speech-audio)

---

## 🏗️ Architecture

### Dual-Layer LSTM Model
```
Input (Audio Features)
        ↓
   LSTM Layer 1 (128 units)
        ↓
     Dropout (0.3)
        ↓
   LSTM Layer 2 (64 units)
        ↓
     Dropout (0.3)
        ↓
   Dense Layer (64 units, ReLU)
        ↓
   Output Layer (7 units, Softmax)
```

### Feature Extraction Pipeline

- **MFCC** (Mel-Frequency Cepstral Coefficients)
- **Chroma Features**
- **Mel Spectrogram**
- **Zero Crossing Rate**
- **Spectral Centroid**

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras** - Deep Learning Framework
- **Librosa** - Audio Processing
- **NumPy** - Numerical Computing
- **Pandas** - Data Manipulation
- **Matplotlib / Seaborn** - Visualization
- **Scikit-learn** - ML Utilities

---

## 📁 Project Structure
```
speech-emotion-recognition/
│
├── data/
│   ├── raw/                    # Raw RAVDESS dataset
│   ├── processed/              # Preprocessed features
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Audio preprocessing functions
│   ├── feature_extraction.py   # Feature extraction utilities
│   ├── model.py                # Dual-Layer LSTM model
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
│
├── models/
│   └── saved_models/           # Trained model checkpoints
│
├── results/
│   ├── plots/                  # Visualization outputs
│   └── metrics/                # Performance metrics
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download RAVDESS dataset**
```bash
# Download from Kaggle or official source
# Place in data/raw/ directory
```

---

## 💻 Usage

### 1. Preprocess Data
```bash
python src/data_preprocessing.py --data_path data/raw/ --output_path data/processed/
```

### 2. Extract Features
```bash
python src/feature_extraction.py --input_path data/processed/ --output_path data/features/
```

### 3. Train Model
```bash
python src/train.py --epochs 100 --batch_size 32 --learning_rate 0.001
```

### 4. Evaluate Model
```bash
python src/evaluate.py --model_path models/saved_models/best_model.h5
```

### 5. Run Inference
```python
from src.model import DualLayerLSTM
from src.feature_extraction import extract_features

# Load model
model = DualLayerLSTM.load('models/saved_models/best_model.h5')

# Predict emotion
audio_file = 'path/to/audio.wav'
features = extract_features(audio_file)
emotion = model.predict(features)
print(f"Predicted Emotion: {emotion}")
```

---

## 📈 Results

### Expected Performance (Based on Paper)

| Metric | Single-Layer LSTM | Dual-Layer LSTM (Our Implementation) |
|--------|------------------|--------------------------------------|
| Accuracy | ~XX% | ~XX% (+2%) |
| Precision | ~XX% | ~XX% |
| Recall | ~XX% | ~XX% |
| F1-Score | ~XX% | ~XX% |

> **Note:** Results will be updated as we progress with the implementation.

---

## 🗺️ Roadmap

- [x] Project setup and planning
- [ ] Data collection and preprocessing
- [ ] Feature extraction implementation
- [ ] Dual-Layer LSTM model architecture
- [ ] Model training and validation
- [ ] Performance evaluation
- [ ] Hyperparameter tuning
- [ ] Real-time emotion recognition interface
- [ ] Documentation and final report

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

- **Your Name** - [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)
- **Teammate 2** - [GitHub](https://github.com/teammate2)

---

## 🙏 Acknowledgments

- Original Paper Authors: Xiaoran Yang, Shuhan Yu, and Wenxi Xu
- RAVDESS Dataset Creators
- Open-source community for tools and libraries

---

## 📧 Contact

For questions or feedback, please reach out:
- Email: your.email@example.com
- Project Link: [https://github.com/yourusername/speech-emotion-recognition](https://github.com/yourusername/speech-emotion-recognition)

---

<p align="center">Made with ❤️ and 🎵</p>
