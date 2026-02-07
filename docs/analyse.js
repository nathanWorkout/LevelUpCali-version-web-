// Configuration
const API_URL = "https://levelupcali-version-web.onrender.com";

// Exemple d'appel
fetch(`${API_URL}/analyze_static`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    image_base64: imageData
  })
})

// State
let state = {
    currentFile: null,
    isProcessing: false
};

// Elements (cache DOM)
const elements = {
    // Image
    imageInput: document.getElementById('image-input'),
    imageUploadZone: document.getElementById('image-upload-zone'),
    imagePreview: document.getElementById('image-preview'),
    imagePreviewImg: document.getElementById('image-preview-img'),
    removeImageBtn: document.getElementById('remove-image'),
    analyzeImageBtn: document.getElementById('analyze-image-btn'),
    
    // Results
    resultsSection: document.getElementById('results-section'),
    resultImage: document.getElementById('result-image'),
    figureBadge: document.getElementById('figure-badge'),
    statsChips: document.getElementById('stats-chips'),
    causeText: document.getElementById('cause-text'),
    compensationText: document.getElementById('compensation-text'),
    correctionText: document.getElementById('correction-text'),
    deviationsSection: document.getElementById('deviations-section'),
    deviationsGrid: document.getElementById('deviations-grid'),
    newAnalysisBtn: document.getElementById('new-analysis-btn')
};

// ============================================================================
// INITIALIZATION
// ============================================================================
function init() {
    // Image handlers
    elements.imageUploadZone.addEventListener('click', () => elements.imageInput.click());
    elements.imageUploadZone.addEventListener('dragover', handleDragOver);
    elements.imageUploadZone.addEventListener('dragleave', handleDragLeave);
    elements.imageUploadZone.addEventListener('drop', handleDrop);
    elements.imageInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));
    elements.removeImageBtn.addEventListener('click', resetUploadState);
    elements.analyzeImageBtn.addEventListener('click', analyzeImage);
    
    // New analysis
    elements.newAnalysisBtn.addEventListener('click', startNewAnalysis);
    
    console.log('App initialisée - Analyse image uniquement');
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
    
    console.log(`Fichier: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
    
    // Validation
    const validation = validateFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }
    
    state.currentFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.imagePreviewImg.src = e.target.result;
        elements.imageUploadZone.style.display = 'none';
        elements.imagePreview.style.display = 'block';
        elements.analyzeImageBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function validateFile(file) {
    const validImageTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    
    if (!validImageTypes.includes(file.type)) {
        return { valid: false, error: 'Format invalide. Utilise JPG ou PNG.' };
    }
    if (file.size > 10 * 1024 * 1024) {
        return { valid: false, error: 'Fichier trop volumineux (max 10MB).' };
    }
    
    return { valid: true };
}

function resetUploadState() {
    state.currentFile = null;
    
    // Image
    elements.imageInput.value = '';
    elements.imagePreviewImg.src = '';
    elements.imageUploadZone.style.display = 'block';
    elements.imagePreview.style.display = 'none';
    elements.analyzeImageBtn.disabled = true;
}

// ============================================================================
// ANALYSIS
// ============================================================================
async function analyzeImage() {
    if (!state.currentFile || state.isProcessing) return;
    
    console.log('🔍 Analyse image...');
    
    setLoading(elements.analyzeImageBtn, true);
    state.isProcessing = true;
    
    try {
        const base64 = await fileToBase64(state.currentFile);
        const base64Data = base64.split(',')[1];
        
        const response = await fetch(`${API_URL}/analyze_static`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_base64: base64Data })
        });
        
        const data = await response.json();
        
        console.log('Réponse serveur:', data);
        
        if (data.status === 'ok') {
            displayResults(data);
        } else {
            showError(data.message || 'Erreur lors de l\'analyse');
        }
    } catch (error) {
        console.error('Erreur:', error);
        showError('Erreur lors de l\'analyse. Vérifie que le serveur est lancé sur le port 5000.');
    } finally {
        setLoading(elements.analyzeImageBtn, false);
        state.isProcessing = false;
    }
}

// ============================================================================
// RESULTS DISPLAY
// ============================================================================
function displayResults(data) {
    console.log('Affichage résultats:', data.detected_figure);
    
    // Show section
    elements.resultsSection.style.display = 'block';
    
    // Media - Image avec landmarks
    if (data.image_base64) {
        elements.resultImage.src = `data:image/jpeg;base64,${data.image_base64}`;
        elements.resultImage.style.display = 'block';
        console.log('Image avec landmarks affichée');
    }
    
    // Figure badge avec traduction
    const figureNames = {
        'handstand': 'Handstand',
        'planche': 'Planche',
        'front_lever': 'Front Lever',
        'push_up': 'Push-up',
        'pull_up': 'Pull-up',
        'dips': 'Dips'
    };
    
    elements.figureBadge.textContent = figureNames[data.detected_figure] || data.detected_figure;
    
    // Stats (pas de stats pour image statique, mais on garde pour compatibilité)
    elements.statsChips.innerHTML = '';
    
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
    
    // Scroll
    setTimeout(() => {
        elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function displayDeviations(deviations) {
    elements.deviationsSection.style.display = 'block';
    elements.deviationsGrid.innerHTML = '';
    
    const labels = {
        'hanches_flechies': 'Hanches fléchies',
        'coudes_flechis': ' Coudes fléchis',
        'genoux_flechis': ' Genoux fléchis',
        'hanches_basses': ' Hanches basses',
        'position_epaules': ' Position épaules',
        'amplitude_insuffisante': ' Amplitude insuffisante',
        'verrouillage_incomplet': ' Verrouillage incomplet',
        'execution_rapide': ' Exécution rapide',
        'extension_incomplete': ' Extension incomplète',
        'profondeur_insuffisante': ' Profondeur insuffisante'
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
// UI HELPERS
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
    alert(`${message}`);
}

// ============================================================================
// UTILITIES
// ============================================================================
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// ============================================================================
// START
// ============================================================================
document.addEventListener('DOMContentLoaded', init);

console.log('LevelUpCali - Analyse Biomécanique v12.0');
console.log('API:', API_URL);
console.log('Mode: Analyse image statique avec landmarks colorés');