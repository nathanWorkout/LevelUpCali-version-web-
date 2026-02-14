// Configuration
const API_URL = "https://levelupcali-version-web.onrender.com";
//const API_URL = "http://localhost:5000";

// State
let state = {
    currentFile: null,
    isProcessing: false
};

// Elements
const elements = {
    imageInput: document.getElementById('image-input'),
    imageUploadZone: document.getElementById('image-upload-zone'),
    imagePreview: document.getElementById('image-preview'),
    imagePreviewImg: document.getElementById('image-preview-img'),
    removeImageBtn: document.getElementById('remove-image'),
    analyzeImageBtn: document.getElementById('analyze-image-btn'),
    resultsSection: document.getElementById('results-section'),
    resultImage: document.getElementById('result-image'),
    figureBadge: document.getElementById('figure-badge'),
    causeText: document.getElementById('cause-text'),
    compensationText: document.getElementById('compensation-text'),
    correctionText: document.getElementById('correction-text'),
    deviationsSection: document.getElementById('deviations-section'),
    deviationsGrid: document.getElementById('deviations-grid'),
    newAnalysisBtn: document.getElementById('new-analysis-btn')
};

// ============================================================================
// INIT
// ============================================================================
function init() {
    elements.imageUploadZone.addEventListener('click', () => elements.imageInput.click());
    elements.imageUploadZone.addEventListener('dragover', handleDragOver);
    elements.imageUploadZone.addEventListener('dragleave', handleDragLeave);
    elements.imageUploadZone.addEventListener('drop', handleDrop);
    elements.imageInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));
    elements.removeImageBtn.addEventListener('click', resetUploadState);
    elements.analyzeImageBtn.addEventListener('click', analyzeImage);
    elements.newAnalysisBtn.addEventListener('click', startNewAnalysis);
    
    console.log('App initialisée - Envoi fichier RAW');
}

// ============================================================================
// FILE HANDLING
// ============================================================================
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
}

function handleFileSelect(file) {
    if (!file) return;
    
    // Validation
    if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
        showError('Format invalide. Utilise JPG ou PNG.');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showError('Fichier trop volumineux (max 10MB).');
        return;
    }
    
    state.currentFile = file;
    
    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.imagePreviewImg.src = e.target.result;
        elements.imageUploadZone.style.display = 'none';
        elements.imagePreview.style.display = 'block';
        elements.analyzeImageBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function resetUploadState() {
    state.currentFile = null;
    elements.imageInput.value = '';
    elements.imagePreviewImg.src = '';
    elements.imageUploadZone.style.display = 'block';
    elements.imagePreview.style.display = 'none';
    elements.analyzeImageBtn.disabled = true;
}

// ============================================================================
// ANALYSIS - ENVOI FICHIER RAW
// ============================================================================
async function analyzeImage() {
    if (!state.currentFile || state.isProcessing) return;
    
    console.log('Analyse image (fichier brut)...');
    
    setLoading(elements.analyzeImageBtn, true);
    state.isProcessing = true;
    
    try {
        // FormData avec fichier brut
        const formData = new FormData();
        formData.append('image', state.currentFile);
        
        const response = await fetch(`${API_URL}/analyze_static`, {
            method: 'POST',
            body: formData  // Pas de Content-Type header, laisse le navigateur gérer
        });
        
        const data = await response.json();
        
        if (data.status === 'ok') {
            displayResults(data);
        } else {
            showError(data.message || 'Erreur lors de l\'analyse');
        }
    } catch (error) {
        console.error('Erreur:', error);
        showError('Erreur lors de l\'analyse. Vérifie que le serveur est accessible.');
    } finally {
        setLoading(elements.analyzeImageBtn, false);
        state.isProcessing = false;
    }
}

// ============================================================================
// RESULTS
// ============================================================================
function displayResults(data) {
    elements.resultsSection.style.display = 'block';
    
    // Image annotée
    if (data.image_base64) {
        elements.resultImage.src = `data:image/jpeg;base64,${data.image_base64}`;
        elements.resultImage.style.display = 'block';
    }
    
    // Figure
    const figureNames = {
        'handstand': 'Handstand',
        'planche': 'Planche',
        'front_lever': 'Front Lever'
    };
    elements.figureBadge.textContent = figureNames[data.detected_figure] || data.detected_figure;
    
    // Analysis
    if (data.analysis) {
        elements.causeText.textContent = data.analysis.cause || 'N/A';
        elements.compensationText.textContent = data.analysis.compensation || 'N/A';
        elements.correctionText.textContent = data.analysis.correction || 'N/A';
    }
    
    // Deviations
    if (data.deviations && Object.keys(data.deviations).length > 0) {
        displayDeviations(data.deviations);
    } else {
        elements.deviationsSection.style.display = 'none';
    }
    
    setTimeout(() => {
        elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function displayDeviations(deviations) {
    elements.deviationsSection.style.display = 'block';
    elements.deviationsGrid.innerHTML = '';
    
    const labels = {
        'hanches_flechies': 'Hanches fléchies',
        'coudes_flechis': 'Coudes fléchis',
        'genoux_flechis': 'Genoux fléchis',
        'hanches_basses': 'Hanches basses',
        'position_epaules': 'Position épaules'
    };
    
    for (const [key, value] of Object.entries(deviations)) {
        const item = document.createElement('div');
        item.className = 'deviation-item';
        item.innerHTML = `
            <span class="deviation-label">${labels[key] || key}</span>
            <span class="deviation-value ${value === 'Oui' ? 'deviation-error' : ''}">${value}</span>
        `;
        elements.deviationsGrid.appendChild(item);
    }
}

function startNewAnalysis() {
    elements.resultsSection.style.display = 'none';
    resetUploadState();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// UI
// ============================================================================
function setLoading(button, isLoading) {
    const btnText = button.querySelector('.btn-text');
    const btnLoader = button.querySelector('.btn-loader');
    
    if (isLoading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'flex';
        button.disabled = true;
    } else {
        btnText.style.display = 'block';
        btnLoader.style.display = 'none';
        button.disabled = false;
    }
}

function showError(message) {
    alert(message);
}

// ============================================================================
// START
// ============================================================================
document.addEventListener('DOMContentLoaded', init);
console.log('LevelUpCali v13.0 - Optimisé Web');
console.log('API:', API_URL);
console.log('Flux: Navigateur → Fichier RAW → Flask → NumPy → MediaPipe');