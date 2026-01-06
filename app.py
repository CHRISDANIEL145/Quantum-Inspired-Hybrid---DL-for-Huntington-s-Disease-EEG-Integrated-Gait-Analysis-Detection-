"""
Huntington's Disease Prediction Web Application
================================================
Flask backend for quantum neural network-based MRI brain scan analysis.

Features:
- Load pre-trained quantum model
- Image preprocessing for MRI scans
- Prediction API endpoint
- RESTful JSON responses
"""

import os
import io
import time
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms, models
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'trainedmodel_file/quantum_huntington_train2.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
CLASS_NAMES = ['Huntington', 'Normal']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================================
# QUANTUM LAYERS (Mirror from train2.py)
# ============================================================================

class QuantumSuperposition(nn.Module):
    def __init__(self, in_dim, out_dim, n_states=4):
        super().__init__()
        self.n_states = n_states
        state_dim = out_dim // n_states
        
        self.states = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, state_dim, bias=False),
                nn.LayerNorm(state_dim),
                nn.GELU()
            ) for _ in range(n_states)
        ])
        
        self.amplitudes = nn.Parameter(torch.ones(n_states) / np.sqrt(n_states))
        self.phases = nn.Parameter(torch.zeros(n_states))
        
    def forward(self, x):
        state_outputs = []
        for i, state_fn in enumerate(self.states):
            amplitude = torch.abs(self.amplitudes[i])
            phase = self.phases[i]
            state_out = state_fn(x)
            state_out = amplitude * state_out * torch.cos(phase) + \
                        amplitude * state_out * torch.sin(phase)
            state_outputs.append(state_out)
        
        return torch.cat(state_outputs, dim=-1)


class QuantumEntanglement(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.entanglement_strength = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        b = x.size(0)
        q = q.view(b, self.n_heads, self.head_dim)
        k = k.view(b, self.n_heads, self.head_dim)
        v = v.view(b, self.n_heads, self.head_dim)
        
        correlation = torch.einsum('bnd,bmd->bnm', q, k) / np.sqrt(self.head_dim)
        entanglement = F.softmax(correlation, dim=-1)
        entanglement = self.dropout(entanglement)
        
        entangled = torch.einsum('bnm,bmd->bnd', entanglement, v)
        entangled = entangled.reshape(b, self.dim)
        
        out = self.out_proj(entangled)
        out = self.entanglement_strength * out + (1 - self.entanglement_strength) * residual
        
        return out


class QuantumInterference(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.wave1 = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim)
        )
        self.wave2 = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim)
        )
        
        self.phase_diff = nn.Parameter(torch.zeros(dim))
        self.interference_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        psi1 = self.wave1(x)
        psi2 = self.wave2(x)
        
        psi2_phased = psi2 * torch.cos(self.phase_diff) + psi2 * torch.sin(self.phase_diff)
        
        constructive = psi1 + psi2_phased
        destructive = psi1 - psi2_phased
        
        interference_weight = self.interference_gate(torch.cat([constructive, destructive], dim=-1))
        result = interference_weight * constructive + (1 - interference_weight) * destructive
        
        return F.gelu(result) + x


class QuantumMeasurement(nn.Module):
    def __init__(self, dim, n_basis=8):
        super().__init__()
        self.dim = dim
        self.n_basis = n_basis
        
        self.basis_vectors = nn.Parameter(torch.randn(n_basis, dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(1))
        self.projection = nn.Sequential(
            nn.Linear(n_basis, dim, bias=False),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        basis_norm = F.normalize(self.basis_vectors, dim=-1)
        x_norm = F.normalize(x, dim=-1)
        
        overlaps = torch.matmul(x_norm, basis_norm.T)
        probabilities = overlaps ** 2
        collapse_weights = F.softmax(probabilities / self.temperature.clamp(min=0.1), dim=-1)
        measured = self.projection(collapse_weights)
        
        return x + 0.1 * measured


class QuantumFeatureBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        
        self.superposition = QuantumSuperposition(dim, dim, n_states=4)
        self.entanglement = QuantumEntanglement(dim, n_heads=4)
        self.interference = QuantumInterference(dim)
        self.measurement = QuantumMeasurement(dim, n_basis=8)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.superposition(x)
        x = self.dropout(x)
        x = self.entanglement(x)
        x = self.dropout(x)
        x = self.interference(x)
        x = self.dropout(x)
        x = self.measurement(x)
        x = self.norm(x)
        return x


class QuantumHuntingtonModel(nn.Module):
    def __init__(self, n_classes=2, drop=0.3):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(weights=None)
        feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.initial_proj = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        
        self.quantum_block = QuantumFeatureBlock(256, dropout=drop)
        
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(drop / 2),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.initial_proj(feat)
        feat = self.quantum_block(feat)
        return self.classifier(feat)


# ============================================================================
# MODEL LOADING
# ============================================================================

model = None
model_loaded = False
model_load_error = None


def load_model():
    global model, model_loaded, model_load_error
    try:
        model = QuantumHuntingtonModel(n_classes=2, drop=0.3)
        
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            model.to(DEVICE)
            model.eval()
            model_loaded = True
            print(f"✅ Model loaded successfully on {DEVICE}")
        else:
            model_load_error = f"Model file not found: {MODEL_PATH}"
            print(f"❌ {model_load_error}")
    except Exception as e:
        model_load_error = str(e)
        print(f"❌ Error loading model: {e}")


# Load model on startup
load_model()


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def preprocess_image(image_bytes):
    """Preprocess uploaded image for model inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor, image


# ============================================================================
# PREDICTION STATS
# ============================================================================

prediction_stats = {
    'total_predictions': 0,
    'huntington_count': 0,
    'normal_count': 0,
    'avg_processing_time': 0,
    'last_prediction_time': None
}


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render main application page."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Return API and model status."""
    return jsonify({
        'status': 'online',
        'model_loaded': model_loaded,
        'model_error': model_load_error,
        'device': str(DEVICE),
        'model_path': MODEL_PATH,
        'classes': CLASS_NAMES,
        'stats': prediction_stats
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded MRI scan."""
    global prediction_stats
    
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'details': model_load_error
        }), 500
    
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    try:
        start_time = time.time()
        
        # Read and preprocess image
        image_bytes = file.read()
        tensor, original_image = preprocess_image(image_bytes)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        processing_time = time.time() - start_time
        
        # Get prediction details
        confidence_value = confidence.item()
        class_idx = predicted_class.item()
        class_name = CLASS_NAMES[class_idx]
        
        # Calculate risk level
        if class_name == 'Huntington':
            if confidence_value >= 0.9:
                risk_level = 'High Risk'
                risk_color = '#EF4444'
            elif confidence_value >= 0.7:
                risk_level = 'Moderate Risk'
                risk_color = '#F59E0B'
            else:
                risk_level = 'Low Risk'
                risk_color = '#10B981'
        else:
            risk_level = 'Normal'
            risk_color = '#10B981'
        
        # Update stats
        prediction_stats['total_predictions'] += 1
        if class_name == 'Huntington':
            prediction_stats['huntington_count'] += 1
        else:
            prediction_stats['normal_count'] += 1
        
        # Update average processing time
        n = prediction_stats['total_predictions']
        old_avg = prediction_stats['avg_processing_time']
        prediction_stats['avg_processing_time'] = (old_avg * (n-1) + processing_time) / n
        prediction_stats['last_prediction_time'] = datetime.now().isoformat()
        
        # Get all class probabilities
        all_probs = {CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(len(CLASS_NAMES))}
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': class_name,
                'class_index': class_idx,
                'confidence': round(confidence_value * 100, 2),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'probabilities': all_probs
            },
            'processing_time': round(processing_time * 1000, 2),  # milliseconds
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': 'Quantum-Inspired Neural Network',
                'accuracy': '95.5%'
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats')
def get_stats():
    """Get prediction statistics."""
    return jsonify(prediction_stats)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🧠 Huntington's Disease Prediction System")
    print("   Quantum-Inspired Neural Network")
    print("=" * 60)
    print(f"   Device: {DEVICE}")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Model Loaded: {model_loaded}")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
