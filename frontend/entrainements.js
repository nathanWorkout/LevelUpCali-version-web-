// ================== VARIABLES GLOBALES ==================
let trainings = [];
let currentEditId = null;
let exerciseCounter = 0;

// ================== INITIALISATION ==================
document.addEventListener('DOMContentLoaded', () => {
    loadTrainings();
    renderTrainings();
    
    // Event listeners
    document.getElementById('btnCreateTraining').addEventListener('click', openCreateModal);
    document.getElementById('btnCloseModal').addEventListener('click', closeModal);
    document.getElementById('btnCancelForm').addEventListener('click', closeModal);
    document.getElementById('btnCloseDetails').addEventListener('click', closeDetailsModal);
    document.getElementById('btnAddExercise').addEventListener('click', addExerciseField);
    document.getElementById('trainingForm').addEventListener('submit', handleFormSubmit);
    document.getElementById('btnDeleteTraining').addEventListener('click', deleteTraining);
    document.getElementById('btnEditTraining').addEventListener('click', editCurrentTraining);
    document.getElementById('btnCompleteTraining').addEventListener('click', completeTraining);
    
    // Filtres
    document.getElementById('filterType').addEventListener('change', applyFilters);
    document.getElementById('filterDifficulty').addEventListener('change', applyFilters);
    
    // Fermer modals en cliquant à l'extérieur
    document.getElementById('trainingModal').addEventListener('click', (e) => {
        if (e.target.id === 'trainingModal') closeModal();
    });
    
    document.getElementById('detailsModal').addEventListener('click', (e) => {
        if (e.target.id === 'detailsModal') closeDetailsModal();
    });
});

// ================== GESTION DES DONNÉES ==================
function loadTrainings() {
    const stored = localStorage.getItem('levelUpCaliTrainings');
    if (stored) {
        trainings = JSON.parse(stored);
    } else {
        // Données d'exemple
        trainings = [
            {
                id: 1,
                name: "Pull Day Intense",
                type: "pull",
                difficulty: "intermediate",
                durationHours: 1,
                durationMinutes: 0,
                duration: 60,
                rest: 90,
                restUnit: "sec",
                description: "Entraînement complet pour le dos et les biceps avec focus sur les tractions",
                exercises: [
                    { name: "Tractions pronation", sets: 4, reps: "8-12", isStatic: false, notes: "Pleine amplitude" },
                    { name: "Tractions supination", sets: 3, reps: "10-15", isStatic: false, notes: "" },
                    { name: "Planche", sets: 4, holdTime: 30, isStatic: true, notes: "Maintenir la position" }
                ]
            }
        ];
        saveTrainings();
    }
}

function saveTrainings() {
    localStorage.setItem('levelUpCaliTrainings', JSON.stringify(trainings));
}

// ================== AFFICHAGE ==================
function renderTrainings() {
    const container = document.getElementById('trainingList');
    
    if (trainings.length === 0) {
        container.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: #b0b0b0;">
                <p style="font-size: 18px; margin-bottom: 10px;">Aucun entraînement créé</p>
                <p style="font-size: 14px;">Cliquez sur "Créer un entraînement" pour commencer</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    trainings.forEach((training, index) => {
        const card = createTrainingCard(training, index);
        container.appendChild(card);
    });
}

function createTrainingCard(training, index) {
    const card = document.createElement('div');
    card.className = `training-card ${training.type}`;
    card.style.animationDelay = `${index * 0.1}s`;
    card.addEventListener('click', () => showTrainingDetails(training));
    
    card.innerHTML = `
        <div class="card-header">
            <div class="card-title">${training.name}</div>
            <div class="card-badges">
                <span class="badge badge-type">${getTypeLabel(training.type)}</span>
                <span class="badge badge-difficulty ${training.difficulty}">${getDifficultyLabel(training.difficulty)}</span>
            </div>
        </div>
        ${training.description ? `<div class="card-description">${training.description}</div>` : ''}
        <div class="card-stats">
            <div class="stat-item">
                <span class="stat-value">${training.exercises.length} exercices</span>
            </div>
            ${training.duration ? `
            <div class="stat-item">
                <span class="stat-value">${formatDuration(training)}</span>
            </div>
            ` : ''}
        </div>
    `;
    
    return card;
}

// ================== MODAL CRÉATION/ÉDITION ==================
function openCreateModal() {
    currentEditId = null;
    exerciseCounter = 0;
    document.getElementById('modalTitle').textContent = 'Créer un entraînement';
    document.getElementById('trainingForm').reset();
    document.getElementById('exercisesList').innerHTML = '';
    addExerciseField(); // Ajouter un exercice par défaut
    document.getElementById('trainingModal').classList.add('active');
}

function openEditModal(training) {
    currentEditId = training.id;
    exerciseCounter = 0;
    document.getElementById('modalTitle').textContent = 'Modifier l\'entraînement';
    
    // Remplir le formulaire
    document.getElementById('trainingName').value = training.name;
    document.getElementById('trainingType').value = training.type;
    document.getElementById('trainingDifficulty').value = training.difficulty;
    document.getElementById('trainingHours').value = training.durationHours || 0;
    document.getElementById('trainingMinutes').value = training.durationMinutes || 0;
    document.getElementById('trainingRest').value = training.rest || '';
    document.getElementById('trainingRestUnit').value = training.restUnit || 'sec';
    document.getElementById('trainingDescription').value = training.description || '';
    
    // Ajouter les exercices
    document.getElementById('exercisesList').innerHTML = '';
    training.exercises.forEach(exercise => {
        addExerciseField(exercise);
    });
    
    document.getElementById('trainingModal').classList.add('active');
}

function closeModal() {
    document.getElementById('trainingModal').classList.remove('active');
    currentEditId = null;
}

// ================== GESTION DES EXERCICES ==================
function addExerciseField(exerciseData = null) {
    exerciseCounter++;
    const exercisesList = document.getElementById('exercisesList');
    
    const exerciseItem = document.createElement('div');
    exerciseItem.className = 'exercise-item';
    exerciseItem.dataset.exerciseId = exerciseCounter;
    
    const isStatic = exerciseData ? exerciseData.isStatic : false;
    const holdMinutes = exerciseData && exerciseData.holdTime ? Math.floor(exerciseData.holdTime / 60) : 0;
    const holdSeconds = exerciseData && exerciseData.holdTime ? exerciseData.holdTime % 60 : 0;
    
    exerciseItem.innerHTML = `
        <div class="exercise-header">
            <span class="exercise-number">Exercice ${exerciseCounter}</span>
            <button type="button" class="btn-remove-exercise" onclick="removeExercise(${exerciseCounter})">
                Supprimer
            </button>
        </div>
        <div class="exercise-fields">
            <div class="field-row">
                <div class="field-group">
                    <label>Nom de l'exercice *</label>
                    <input type="text" class="exercise-name" required 
                           placeholder="Ex: Tractions" 
                           value="${exerciseData ? exerciseData.name : ''}">
                </div>
                <div class="field-group">
                    <label>Type *</label>
                    <select class="exercise-type" required>
                        <option value="reps" ${!isStatic ? 'selected' : ''}>Répétitions</option>
                        <option value="hold" ${isStatic ? 'selected' : ''}>Maintien</option>
                    </select>
                </div>
                <div class="field-group">
                    <label>Séries *</label>
                    <input type="number" class="exercise-sets" required min="1" max="20" 
                           placeholder="4" 
                           value="${exerciseData ? exerciseData.sets : ''}">
                </div>
            </div>
            <div class="field-row reps-fields" style="display: ${isStatic ? 'none' : 'grid'}">
                <div class="field-group">
                    <label>Répétitions *</label>
                    <input type="text" class="exercise-reps" 
                           placeholder="8-12" 
                           value="${exerciseData && !isStatic ? exerciseData.reps : ''}">
                </div>
            </div>
            <div class="field-row hold-fields" style="display: ${isStatic ? 'grid' : 'none'}">
                <div class="field-group">
                    <label>Temps de maintien *</label>
                    <div class="duration-inputs">
                        <div class="input-with-unit">
                            <input type="number" class="hold-minutes" min="0" max="60" placeholder="0" value="${holdMinutes || ''}">
                            <span class="unit-label">min</span>
                        </div>
                        <div class="input-with-unit">
                            <input type="number" class="hold-seconds" min="0" max="59" placeholder="30" value="${holdSeconds || ''}">
                            <span class="unit-label">sec</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="field-group">
                <label>Notes (optionnel)</label>
                <input type="text" class="exercise-notes" 
                       placeholder="Conseils techniques, variations..." 
                       value="${exerciseData ? exerciseData.notes : ''}">
            </div>
        </div>
    `;
    
    exercisesList.appendChild(exerciseItem);
    
    // Ajouter l'event listener pour le changement de type
    const typeSelect = exerciseItem.querySelector('.exercise-type');
    const repsFields = exerciseItem.querySelector('.reps-fields');
    const holdFields = exerciseItem.querySelector('.hold-fields');
    const repsInput = exerciseItem.querySelector('.exercise-reps');
    
    typeSelect.addEventListener('change', (e) => {
        const isHold = e.target.value === 'hold';
        repsFields.style.display = isHold ? 'none' : 'grid';
        holdFields.style.display = isHold ? 'grid' : 'none';
        repsInput.required = !isHold;
    });
}

function removeExercise(exerciseId) {
    const exerciseItem = document.querySelector(`[data-exercise-id="${exerciseId}"]`);
    if (exerciseItem) {
        exerciseItem.style.animation = 'fadeOut 0.3s ease-out';
        setTimeout(() => {
            exerciseItem.remove();
            updateExerciseNumbers();
        }, 300);
    }
}

function updateExerciseNumbers() {
    const exercises = document.querySelectorAll('.exercise-item');
    exercises.forEach((exercise, index) => {
        exercise.querySelector('.exercise-number').textContent = `Exercice ${index + 1}`;
    });
}

function getExercisesFromForm() {
    const exercises = [];
    const exerciseItems = document.querySelectorAll('.exercise-item');
    
    exerciseItems.forEach(item => {
        const type = item.querySelector('.exercise-type').value;
        const isStatic = type === 'hold';
        
        const exercise = {
            name: item.querySelector('.exercise-name').value,
            sets: parseInt(item.querySelector('.exercise-sets').value),
            isStatic: isStatic,
            notes: item.querySelector('.exercise-notes').value
        };
        
        if (isStatic) {
            const mins = parseInt(item.querySelector('.hold-minutes').value) || 0;
            const secs = parseInt(item.querySelector('.hold-seconds').value) || 0;
            exercise.holdTime = (mins * 60) + secs;
            exercise.reps = null; // Pas de reps pour les statiques
        } else {
            exercise.reps = item.querySelector('.exercise-reps').value;
            exercise.holdTime = null; // Pas de holdTime pour les dynamiques
        }
        
        exercises.push(exercise);
    });
    
    return exercises;
}

// ================== SOUMISSION DU FORMULAIRE ==================
function handleFormSubmit(e) {
    e.preventDefault();
    
    const exercises = getExercisesFromForm();
    
    if (exercises.length === 0) {
        alert('Veuillez ajouter au moins un exercice');
        return;
    }
    
    // Calculer la durée totale en minutes
    const hours = parseInt(document.getElementById('trainingHours').value) || 0;
    const minutes = parseInt(document.getElementById('trainingMinutes').value) || 0;
    const totalMinutes = (hours * 60) + minutes;
    
    const trainingData = {
        name: document.getElementById('trainingName').value,
        type: document.getElementById('trainingType').value,
        difficulty: document.getElementById('trainingDifficulty').value,
        durationHours: hours,
        durationMinutes: minutes,
        duration: totalMinutes || null,
        rest: parseInt(document.getElementById('trainingRest').value) || null,
        restUnit: document.getElementById('trainingRestUnit').value,
        description: document.getElementById('trainingDescription').value,
        exercises: exercises
    };
    
    if (currentEditId) {
        // Modification
        const index = trainings.findIndex(t => t.id === currentEditId);
        if (index !== -1) {
            trainings[index] = { ...trainingData, id: currentEditId };
            
            // Mettre à jour aussi les entraînements complétés avec ce training
            const completed = localStorage.getItem('levelUpCaliCompletedWorkouts');
            if (completed) {
                const completedWorkouts = JSON.parse(completed);
                const updatedCompleted = completedWorkouts.map(workout => {
                    if (workout.trainingId === currentEditId) {
                        return {
                            ...workout,
                            name: trainingData.name,
                            type: trainingData.type,
                            exercises: trainingData.exercises
                        };
                    }
                    return workout;
                });
                localStorage.setItem('levelUpCaliCompletedWorkouts', JSON.stringify(updatedCompleted));
            }
        }
    } else {
        // Création
        trainingData.id = Date.now();
        trainings.push(trainingData);
    }
    
    saveTrainings();
    renderTrainings();
    closeModal();
}

// ================== MODAL DÉTAILS ==================
function formatHoldTime(seconds) {
    if (seconds >= 60) {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return secs > 0 ? `${mins}min ${secs}s` : `${mins}min`;
    }
    return `${seconds}s`;
}

function showTrainingDetails(training) {
    document.getElementById('detailsName').textContent = training.name;
    document.getElementById('detailsType').textContent = getTypeLabel(training.type);
    document.getElementById('detailsDifficulty').textContent = getDifficultyLabel(training.difficulty);
    
    // Afficher la durée formatée
    let durationText = 'Non définie';
    if (training.duration) {
        const hours = training.durationHours || 0;
        const minutes = training.durationMinutes || 0;
        
        if (hours > 0 && minutes > 0) {
            durationText = `${hours}h${minutes}`;
        } else if (hours > 0) {
            durationText = `${hours}h`;
        } else if (minutes > 0) {
            durationText = `${minutes} min`;
        }
    }
    document.getElementById('detailsDuration').textContent = durationText;
    
    // Afficher le repos avec l'unité appropriée
    let restText = 'Non défini';
    if (training.rest) {
        const unit = training.restUnit === 'min' ? 'min' : 'sec';
        restText = `${training.rest} ${unit}`;
    }
    document.getElementById('detailsRest').textContent = restText;
    
    // Description
    const descContainer = document.getElementById('detailsDescriptionContainer');
    if (training.description) {
        descContainer.style.display = 'block';
        document.getElementById('detailsDescription').textContent = training.description;
    } else {
        descContainer.style.display = 'none';
    }
    
    // Exercices
    const exercisesContainer = document.getElementById('detailsExercises');
    exercisesContainer.innerHTML = '';
    
    training.exercises.forEach(exercise => {
        const exerciseDiv = document.createElement('div');
        exerciseDiv.className = 'exercise-display-item';
        
        let valueDisplay;
        if (exercise.isStatic) {
            valueDisplay = `
                <div class="set-info">
                    <span class="set-value">${exercise.sets}</span>
                    <span class="set-label">Séries</span>
                </div>
                <div class="set-info">
                    <span class="set-value">${formatHoldTime(exercise.holdTime)}</span>
                    <span class="set-label">Maintien</span>
                </div>
            `;
        } else {
            valueDisplay = `
                <div class="set-info">
                    <span class="set-value">${exercise.sets}</span>
                    <span class="set-label">Séries</span>
                </div>
                <div class="set-info">
                    <span class="set-value">${exercise.reps}</span>
                    <span class="set-label">Reps</span>
                </div>
            `;
        }
        
        exerciseDiv.innerHTML = `
            <div class="exercise-display-info">
                <div class="exercise-display-name">${exercise.name}</div>
                ${exercise.notes ? `<div class="exercise-display-notes">${exercise.notes}</div>` : ''}
            </div>
            <div class="exercise-display-sets">
                ${valueDisplay}
            </div>
        `;
        
        exercisesContainer.appendChild(exerciseDiv);
    });
    
    // Stocker l'ID pour suppression/édition/complétion
    document.getElementById('btnDeleteTraining').dataset.trainingId = training.id;
    document.getElementById('btnEditTraining').dataset.trainingId = training.id;
    document.getElementById('btnCompleteTraining').dataset.trainingId = training.id;
    
    document.getElementById('detailsModal').classList.add('active');
}

function closeDetailsModal() {
    document.getElementById('detailsModal').classList.remove('active');
}

// ================== FONCTION AMÉLIORÉE DE SUPPRESSION ==================
function deleteTraining() {
    const trainingId = parseInt(document.getElementById('btnDeleteTraining').dataset.trainingId);
    
    if (confirm('Êtes-vous sûr de vouloir supprimer cet entraînement ? Toutes les performances associées seront également supprimées.')) {
        // Supprimer le training
        trainings = trainings.filter(t => t.id !== trainingId);
        saveTrainings();
        
        // Supprimer tous les entraînements complétés liés à ce training
        const completed = localStorage.getItem('levelUpCaliCompletedWorkouts');
        if (completed) {
            const completedWorkouts = JSON.parse(completed);
            const updatedCompleted = completedWorkouts.filter(w => w.trainingId !== trainingId);
            localStorage.setItem('levelUpCaliCompletedWorkouts', JSON.stringify(updatedCompleted));
            
            console.log(`Training ${trainingId} supprimé. ${completedWorkouts.length - updatedCompleted.length} séances complétées supprimées.`);
        }
        
        renderTrainings();
        closeDetailsModal();
        
        // Afficher un message de confirmation
        showNotification('Entraînement supprimé avec succès', 'success');
    }
}

// Fonction pour afficher des notifications (optionnel mais utile)
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? 'linear-gradient(135deg, #4CAF50, #66BB6A)' : 'linear-gradient(135deg, #2196F3, #42A5F5)'};
        color: white;
        padding: 15px 25px;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        animation: slideInRight 0.3s ease-out;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function editCurrentTraining() {
    const trainingId = parseInt(document.getElementById('btnEditTraining').dataset.trainingId);
    const training = trainings.find(t => t.id === trainingId);
    
    if (training) {
        closeDetailsModal();
        setTimeout(() => openEditModal(training), 300);
    }
}

function completeTraining() {
    const trainingId = parseInt(document.getElementById('btnCompleteTraining').dataset.trainingId);
    const training = trainings.find(t => t.id === trainingId);
    
    if (!training) return;
    
    // Fermer le modal de détails et ouvrir le modal de complétion
    closeDetailsModal();
    setTimeout(() => {
        openCompletionModal(training);
    }, 300);
}

// Nouvelle fonction pour ouvrir le modal de complétion
function openCompletionModal(training) {
    // Créer le modal s'il n'existe pas
    let modal = document.getElementById('completionModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'completionModal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content modal-large">
                <div class="modal-header">
                    <h2>Enregistrer la séance</h2>
                    <button class="btn-close" id="btnCloseCompletion">&times;</button>
                </div>
                <div class="completion-content">
                    <div class="completion-info">
                        <h3 id="completionTrainingName"></h3>
                        <p style="color: #b0b0b0; font-size: 14px;">Entrez vos performances réelles pour cette séance</p>
                    </div>
                    <div class="completion-exercises" id="completionExercises"></div>
                    <div class="form-actions">
                        <button class="btn-cancel" id="btnCancelCompletion">Annuler</button>
                        <button class="btn-submit" id="btnSaveCompletion">Enregistrer la séance</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Event listeners
        modal.querySelector('#btnCloseCompletion').addEventListener('click', closeCompletionModal);
        modal.querySelector('#btnCancelCompletion').addEventListener('click', closeCompletionModal);
        modal.querySelector('#btnSaveCompletion').addEventListener('click', saveCompletion);
        modal.addEventListener('click', (e) => {
            if (e.target.id === 'completionModal') closeCompletionModal();
        });
    }
    
    // Remplir le modal
    document.getElementById('completionTrainingName').textContent = training.name;
    const exercisesContainer = document.getElementById('completionExercises');
    exercisesContainer.innerHTML = '';
    
    training.exercises.forEach((exercise, index) => {
        const exerciseDiv = document.createElement('div');
        exerciseDiv.className = 'completion-exercise-item';
        exerciseDiv.dataset.exerciseIndex = index;
        exerciseDiv.dataset.isStatic = exercise.isStatic || false;
        
        if (exercise.isStatic) {
            // Exercice statique (hold)
            const suggestedMins = Math.floor((exercise.holdTime || 0) / 60);
            const suggestedSecs = (exercise.holdTime || 0) % 60;
            
            exerciseDiv.innerHTML = `
                <div class="completion-exercise-header">
                    <div class="completion-exercise-name">${exercise.name}</div>
                    <div class="completion-exercise-suggested">Objectif : ${exercise.sets} × ${formatHoldTime(exercise.holdTime || 0)}</div>
                </div>
                <div class="completion-exercise-fields">
                    <div class="field-group">
                        <label>Temps de maintien réalisé</label>
                        <div class="duration-inputs">
                            <div class="input-with-unit">
                                <input type="number" class="completion-hold-minutes" min="0" max="60" placeholder="0" value="${suggestedMins}">
                                <span class="unit-label">min</span>
                            </div>
                            <div class="input-with-unit">
                                <input type="number" class="completion-hold-seconds" min="0" max="59" placeholder="30" value="${suggestedSecs}">
                                <span class="unit-label">sec</span>
                            </div>
                        </div>
                    </div>
                    <div class="field-group" style="max-width: 150px;">
                        <label>Séries réalisées</label>
                        <input type="number" class="completion-sets" min="1" max="20" value="${exercise.sets}">
                    </div>
                </div>
                <div class="field-group">
                    <label>Notes (optionnel)</label>
                    <input type="text" class="completion-notes" placeholder="Ressenti, difficulté...">
                </div>
            `;
        } else {
            // Exercice dynamique (reps)
            const suggestedReps = typeof exercise.reps === 'string' && exercise.reps.includes('-') 
                ? exercise.reps.split('-')[1] 
                : exercise.reps;
            
            exerciseDiv.innerHTML = `
                <div class="completion-exercise-header">
                    <div class="completion-exercise-name">${exercise.name}</div>
                    <div class="completion-exercise-suggested">Objectif : ${exercise.sets} × ${exercise.reps} reps</div>
                </div>
                <div class="completion-exercise-fields">
                    <div class="field-group">
                        <label>Répétitions réalisées</label>
                        <input type="number" class="completion-reps" min="1" placeholder="12" value="${suggestedReps}">
                    </div>
                    <div class="field-group">
                        <label>Séries réalisées</label>
                        <input type="number" class="completion-sets" min="1" max="20" value="${exercise.sets}">
                    </div>
                </div>
                <div class="field-group">
                    <label>Notes (optionnel)</label>
                    <input type="text" class="completion-notes" placeholder="Ressenti, difficulté...">
                </div>
            `;
        }
        
        exercisesContainer.appendChild(exerciseDiv);
    });
    
    // Stocker l'ID du training pour la sauvegarde
    modal.dataset.trainingId = training.id;
    modal.dataset.trainingName = training.name;
    modal.dataset.trainingType = training.type;
    
    modal.classList.add('active');
}

function closeCompletionModal() {
    const modal = document.getElementById('completionModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

function saveCompletion() {
    const modal = document.getElementById('completionModal');
    const trainingId = parseInt(modal.dataset.trainingId);
    const trainingName = modal.dataset.trainingName;
    const trainingType = modal.dataset.trainingType;
    
    const exerciseItems = document.querySelectorAll('.completion-exercise-item');
    const completedExercises = [];
    
    exerciseItems.forEach(item => {
        const isStatic = item.dataset.isStatic === 'true';
        const name = item.querySelector('.completion-exercise-name').textContent;
        const sets = parseInt(item.querySelector('.completion-sets').value);
        const notes = item.querySelector('.completion-notes').value;
        
        const exercise = {
            name: name,
            sets: sets,
            isStatic: isStatic,
            notes: notes
        };
        
        if (isStatic) {
            const mins = parseInt(item.querySelector('.completion-hold-minutes').value) || 0;
            const secs = parseInt(item.querySelector('.completion-hold-seconds').value) || 0;
            exercise.holdTime = (mins * 60) + secs;
            exercise.reps = null;
        } else {
            exercise.reps = parseInt(item.querySelector('.completion-reps').value);
            exercise.holdTime = null;
        }
        
        completedExercises.push(exercise);
    });
    
    const today = new Date().toISOString().split('T')[0];
    
    // Créer l'enregistrement de l'entraînement complété
    const completedWorkout = {
        id: Date.now(),
        trainingId: trainingId,
        name: trainingName,
        type: trainingType,
        exercises: completedExercises,
        completedDate: today
    };
    
    console.log('Entraînement complété avec performances réelles:', completedWorkout);
    
    // Sauvegarder dans localStorage
    const stored = localStorage.getItem('levelUpCaliCompletedWorkouts');
    const completed = stored ? JSON.parse(stored) : [];
    completed.push(completedWorkout);
    localStorage.setItem('levelUpCaliCompletedWorkouts', JSON.stringify(completed));
    
    // Feedback visuel
    const btn = document.getElementById('btnSaveCompletion');
    const originalText = btn.textContent;
    btn.textContent = '✓ Enregistré !';
    btn.style.background = 'linear-gradient(135deg, #66BB6A 0%, #4CAF50 100%)';
    
    setTimeout(() => {
        closeCompletionModal();
        showNotification('Séance enregistrée avec succès !', 'success');
        setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '';
        }, 500);
    }, 1500);
}

// ================== FILTRES ==================
function applyFilters() {
    const typeFilter = document.getElementById('filterType').value;
    const difficultyFilter = document.getElementById('filterDifficulty').value;
    
    const cards = document.querySelectorAll('.training-card');
    
    cards.forEach(card => {
        const training = trainings[Array.from(cards).indexOf(card)];
        
        const matchesType = typeFilter === 'all' || training.type === typeFilter;
        const matchesDifficulty = difficultyFilter === 'all' || training.difficulty === difficultyFilter;
        
        if (matchesType && matchesDifficulty) {
            card.style.display = 'block';
            card.style.animation = 'fadeIn 0.3s ease-out';
        } else {
            card.style.display = 'none';
        }
    });
}

// ================== UTILITAIRES ==================
function getTypeLabel(type) {
    const labels = {
        pull: 'Pull',
        push: 'Push',
        legs: 'Legs',
        full: 'Full Body',
        skills: 'Skills'
    };
    return labels[type] || type;
}

function getDifficultyLabel(difficulty) {
    const labels = {
        beginner: 'Débutant',
        intermediate: 'Intermédiaire',
        advanced: 'Avancé'
    };
    return labels[difficulty] || difficulty;
}

function formatDuration(training) {
    const hours = training.durationHours || 0;
    const minutes = training.durationMinutes || 0;
    
    if (hours > 0 && minutes > 0) {
        return `${hours}h${minutes}`;
    } else if (hours > 0) {
        return `${hours}h`;
    } else if (minutes > 0) {
        return `${minutes} min`;
    }
    return '';
}