/**
 * QuantumNeuro Diagnostics - Main JavaScript
 * Handles file upload, API calls, and UI interactions
 */

// Global state
let selectedFile = null;
let isAnalyzing = false;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initAOS();
    initMobileMenu();
    initDragDrop();
    initNavScroll();
    loadStats();
});

// Initialize Particles.js
function initParticles() {
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: { value: 60, density: { enable: true, value_area: 1000 } },
                color: { value: ['#F472B6', '#6366F1', '#FBBF24', '#A855F7'] },
                shape: { type: 'circle' },
                opacity: { value: 0.3, random: true },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#F472B6',
                    opacity: 0.12,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 1,
                    direction: 'none',
                    random: true,
                    out_mode: 'out'
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: { enable: true, mode: 'grab' },
                    resize: true
                },
                modes: {
                    grab: { distance: 140, line_linked: { opacity: 0.3 } }
                }
            }
        });
    }
}

// Initialize AOS animations
function initAOS() {
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 800,
            easing: 'ease-out-cubic',
            once: true,
            offset: 50
        });
    }
}

// Mobile menu toggle
function initMobileMenu() {
    const btn = document.getElementById('mobileMenuBtn');
    const menu = document.getElementById('mobileMenu');

    if (btn && menu) {
        btn.addEventListener('click', () => {
            menu.classList.toggle('hidden');
        });
    }
}

// Navigation scroll effect
function initNavScroll() {
    const nav = document.getElementById('navbar');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            nav.classList.add('shadow-lg');
            nav.style.background = 'rgba(15, 23, 42, 0.95)';
        } else {
            nav.classList.remove('shadow-lg');
            nav.style.background = 'rgba(15, 23, 42, 0.8)';
        }
    });
}

// Drag and drop functionality
function initDragDrop() {
    const zone = document.getElementById('uploadZone');

    if (zone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            zone.addEventListener(event, preventDefaults);
        });

        ['dragenter', 'dragover'].forEach(event => {
            zone.addEventListener(event, () => zone.classList.add('dragover'));
        });

        ['dragleave', 'drop'].forEach(event => {
            zone.addEventListener(event, () => zone.classList.remove('dragover'));
        });

        zone.addEventListener('drop', handleDrop);
    }
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length) {
        handleFileSelect({ target: { files: files } });
    }
}

// File selection handler
function handleFileSelect(event) {
    const file = event.target.files[0];

    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file (JPG, PNG)', 'error');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showToast('File size must be less than 10MB', 'error');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('uploadPlaceholder').classList.add('hidden');
        document.getElementById('imagePreview').classList.remove('hidden');
        document.getElementById('previewImg').src = e.target.result;
        document.getElementById('fileName').textContent = file.name;
    };
    reader.readAsDataURL(file);

    // Enable analyze button
    document.getElementById('analyzeBtn').disabled = false;

    showToast('Image loaded successfully', 'success');
}

// Scroll to upload section
function scrollToUpload() {
    document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
}

// Analyze image
async function analyzeImage() {
    if (!selectedFile || isAnalyzing) return;

    isAnalyzing = true;

    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    const loadingSection = document.getElementById('loadingSection');
    const loadingBar = document.getElementById('loadingBar');
    const loadingPercent = document.getElementById('loadingPercent');

    // Update UI
    analyzeBtn.disabled = true;
    analyzeBtnText.textContent = 'Analyzing...';
    loadingSection.classList.remove('hidden');

    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        if (progress < 90) {
            progress += Math.random() * 15;
            progress = Math.min(progress, 90);
            loadingBar.style.width = progress + '%';
            loadingPercent.textContent = Math.round(progress) + '%';
        }
    }, 200);

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Complete progress
        clearInterval(progressInterval);
        loadingBar.style.width = '100%';
        loadingPercent.textContent = '100%';

        setTimeout(() => {
            if (data.success) {
                displayResults(data);
                showToast('Analysis complete!', 'success');
            } else {
                showToast('Error: ' + data.error, 'error');
            }

            // Reset loading
            loadingSection.classList.add('hidden');
            loadingBar.style.width = '0%';
            analyzeBtnText.textContent = 'Analyze with Quantum AI';
            analyzeBtn.disabled = false;
            isAnalyzing = false;
        }, 500);

    } catch (error) {
        clearInterval(progressInterval);
        showToast('Connection error. Please try again.', 'error');
        loadingSection.classList.add('hidden');
        analyzeBtnText.textContent = 'Analyze with Quantum AI';
        analyzeBtn.disabled = false;
        isAnalyzing = false;
    }
}

// Display results
function displayResults(data) {
    const pred = data.prediction;

    // Show results section
    document.getElementById('resultsPlaceholder').classList.add('hidden');
    document.getElementById('resultsContent').classList.remove('hidden');

    // Animate confidence circle
    const confidence = pred.confidence;
    const circumference = 339.292;
    const offset = circumference - (confidence / 100) * circumference;

    const progressCircle = document.getElementById('progressCircle');
    progressCircle.style.stroke = pred.risk_color;
    progressCircle.style.strokeDashoffset = offset;

    // Animate confidence value
    animateValue('confidenceValue', 0, confidence, 1000, '%');

    // Update prediction info
    document.getElementById('predictionClass').textContent = pred.class;
    document.getElementById('riskLevel').textContent = pred.risk_level;

    // Update risk badge
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = pred.risk_level;
    riskBadge.style.backgroundColor = pred.risk_color;
    riskBadge.style.color = '#fff';

    // Update probability bars
    const huntProb = pred.probabilities['Huntington'] * 100;
    const normalProb = pred.probabilities['Normal'] * 100;

    document.getElementById('probHunt').textContent = huntProb.toFixed(1) + '%';
    document.getElementById('probNormal').textContent = normalProb.toFixed(1) + '%';
    document.getElementById('probHuntBar').style.width = huntProb + '%';
    document.getElementById('probNormalBar').style.width = normalProb + '%';

    // Processing time
    document.getElementById('processingTime').textContent = data.processing_time + 'ms';

    // Update stats
    loadStats();
}

// Animate number value
function animateValue(id, start, end, duration, suffix = '') {
    const element = document.getElementById(id);
    const range = end - start;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeProgress = 1 - Math.pow(1 - progress, 3); // Ease out cubic
        const current = start + range * easeProgress;

        element.textContent = current.toFixed(1) + suffix;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Load stats from API
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();

        document.getElementById('stat-predictions').textContent = stats.total_predictions;

        if (stats.avg_processing_time > 0) {
            document.getElementById('stat-speed').textContent =
                '<' + Math.ceil(stats.avg_processing_time * 1000 / 100) / 10 + 's';
        }
    } catch (error) {
        console.log('Could not load stats');
    }
}

// Toggle FAQ
function toggleFaq(button) {
    const content = button.nextElementSibling;
    const icon = button.querySelector('i');

    content.classList.toggle('hidden');
    content.classList.toggle('show');
    icon.style.transform = content.classList.contains('show') ? 'rotate(180deg)' : 'rotate(0)';
}

// Toast notification
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');

    const toast = document.createElement('div');
    toast.className = `toast toast-${type} flex items-center gap-2 text-white text-sm`;

    const icons = {
        success: 'check_circle',
        error: 'error',
        info: 'info'
    };

    toast.innerHTML = `
        <i class="material-icons text-lg">${icons[type]}</i>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Check API status on load
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();

        if (!status.model_loaded) {
            showToast('Warning: Model not loaded. ' + status.model_error, 'error');
        }
    } catch (error) {
        showToast('Cannot connect to server', 'error');
    }
}

// Run status check
setTimeout(checkStatus, 1000);
