# QuantumNeuro Diagnostics - Huntington's Disease Prediction

A production-ready web application for Huntington's Disease detection using quantum-inspired neural networks.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
Navigate to: http://localhost:5000

## 📁 Project Structure

```
huntington_diagnosis_app/
├── app.py                    # Flask backend with quantum model
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Main UI template
├── static/
│   ├── css/
│   │   └── style.css        # Custom styles
│   └── js/
│       └── main.js          # Frontend logic
└── trainedmodel_file/
    └── quantum_huntington_train2.pth  # Pre-trained model
```

## 🧠 Features

- **Quantum-Inspired AI**: Superposition, Entanglement, Interference, Measurement layers
- **95.5% Accuracy**: Trained on brain MRI dataset
- **Real-time Analysis**: Upload and get instant results
- **Modern UI**: Glassmorphism, animations, responsive design
- **REST API**: `/api/predict`, `/api/status`, `/api/stats`

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/predict` | POST | Upload image for prediction |
| `/api/status` | GET | Server and model status |
| `/api/stats` | GET | Prediction statistics |

## ⚠️ Medical Disclaimer

This AI system is for research and screening purposes only. It is NOT a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider.

## 📋 Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- 4GB+ RAM
