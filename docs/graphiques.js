// ================== VARIABLES GLOBALES ==================
let performanceRecords = [];
let completedWorkouts = [];
let currentExercise = null;
let currentChart = null;
let currentFilter = 'all';

// ================== INITIALISATION ==================
document.addEventListener('DOMContentLoaded', () => {
    loadCompletedWorkouts();
    cleanOrphanedWorkouts();
    generatePerformanceRecords();
    updateStats();
    renderExerciseList();
    
    // Event listeners
    document.getElementById('timeRange').addEventListener('change', updateChart);
    
    // Filtres
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentFilter = e.target.dataset.filter;
            renderExerciseList();
        });
    });
    
    // Recharger les données quand la page devient visible
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            refreshData();
        }
    });
    
    // Recharger les données au focus de la fenêtre
    window.addEventListener('focus', () => {
        refreshData();
    });
});

// ================== NETTOYAGE DES DONNÉES ==================
function cleanOrphanedWorkouts() {
    const trainingsStored = localStorage.getItem('levelUpCaliTrainings');
    const trainings = trainingsStored ? JSON.parse(trainingsStored) : [];
    const validTrainingIds = new Set(trainings.map(t => t.id));
    
    console.log('Trainings valides:', Array.from(validTrainingIds));
    
    if (completedWorkouts.length > 0) {
        const originalCount = completedWorkouts.length;
        
        completedWorkouts = completedWorkouts.filter(workout => {
            if (!workout.trainingId) return true;
            return validTrainingIds.has(workout.trainingId);
        });
        
        const removedCount = originalCount - completedWorkouts.length;
        
        if (removedCount > 0) {
            console.log(`Nettoyage: ${removedCount} séance(s) orpheline(s) supprimée(s)`);
            localStorage.setItem('levelUpCaliCompletedWorkouts', JSON.stringify(completedWorkouts));
        }
    }
}

// ================== RAFRAÎCHISSEMENT ==================
function refreshData() {
    loadCompletedWorkouts();
    cleanOrphanedWorkouts();
    generatePerformanceRecords();
    updateStats();
    renderExerciseList();
    
    if (currentExercise) {
        const exerciseStillExists = performanceRecords.some(r => r.exercise === currentExercise);
        if (exerciseStillExists) {
            showExerciseGraph(currentExercise);
        } else {
            document.getElementById('graphSection').style.display = 'none';
            currentExercise = null;
        }
    }
}

// ================== GESTION DES DONNÉES ==================
function loadCompletedWorkouts() {
    const stored = localStorage.getItem('levelUpCaliCompletedWorkouts');
    if (stored) {
        completedWorkouts = JSON.parse(stored);
    } else {
        completedWorkouts = [];
    }
}

function generatePerformanceRecords() {
    performanceRecords = [];
    
    completedWorkouts.forEach(workout => {
        if (workout.exercises && Array.isArray(workout.exercises)) {
            workout.exercises.forEach(exercise => {
                const record = {
                    id: Date.now() + Math.random(),
                    exercise: exercise.name,
                    category: workout.type,
                    reps: exercise.reps || null,
                    sets: exercise.sets,
                    holdTime: exercise.holdTime || null,
                    isStatic: exercise.isStatic || false,
                    date: workout.completedDate,
                    notes: exercise.notes || '',
                    workoutName: workout.name,
                    workoutId: workout.id,
                    trainingId: workout.trainingId
                };
                performanceRecords.push(record);
            });
        }
    });
    
    console.log('Total performances chargées:', performanceRecords.length);
}

// ================== STATISTIQUES ==================
function updateStats() {
    const totalRecords = performanceRecords.length;
    const uniqueExercises = [...new Set(performanceRecords.map(r => r.exercise))].length;
    
    let totalProgress = 0;
    let progressCount = 0;
    
    const exerciseGroups = groupByExercise(performanceRecords);
    Object.values(exerciseGroups).forEach(records => {
        if (records.length >= 2) {
            const sorted = records.sort((a, b) => new Date(a.date) - new Date(b.date));
            const first = sorted[0];
            const last = sorted[sorted.length - 1];
            
            let firstValue, lastValue;
            
            if (first.isStatic) {
                firstValue = first.holdTime || 0;
                lastValue = last.holdTime || 0;
            } else {
                firstValue = parseReps(first.reps);
                lastValue = parseReps(last.reps);
            }
            
            if (firstValue > 0) {
                const progress = ((lastValue - firstValue) / firstValue) * 100;
                totalProgress += progress;
                progressCount++;
            }
        }
    });
    
    const avgProgress = progressCount > 0 ? (totalProgress / progressCount).toFixed(1) : 0;
    
    document.getElementById('totalRecords').textContent = totalRecords;
    document.getElementById('totalExercises').textContent = uniqueExercises;
    document.getElementById('progressRate').textContent = avgProgress >= 0 ? `+${avgProgress}%` : `${avgProgress}%`;
}

// ================== LISTE DES EXERCICES ==================
function groupByExercise(records) {
    return records.reduce((acc, record) => {
        if (!acc[record.exercise]) {
            acc[record.exercise] = [];
        }
        acc[record.exercise].push(record);
        return acc;
    }, {});
}

function renderExerciseList() {
    const container = document.getElementById('exerciseList');
    const exerciseGroups = groupByExercise(performanceRecords);
    
    let filteredGroups = exerciseGroups;
    if (currentFilter !== 'all') {
        filteredGroups = {};
        Object.entries(exerciseGroups).forEach(([exercise, records]) => {
            const matchingRecords = records.filter(r => r.category === currentFilter);
            if (matchingRecords.length > 0) {
                filteredGroups[exercise] = matchingRecords;
            }
        });
    }
    
    if (Object.keys(filteredGroups).length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div class="empty-state-title">Aucune performance enregistrée</div>
                <div class="empty-state-text">
                    Pour visualiser vos graphiques de progression, vous devez d'abord compléter des entraînements.
                </div>
                <div class="empty-state-steps">
                    <ol>
                        <li>Allez dans l'onglet <strong>Planning</strong></li>
                        <li>Créez ou sélectionnez un programme d'entraînement</li>
                        <li>Dans l'onglet <strong>Entraînements</strong>, complétez une séance</li>
                        <li>Cliquez sur <strong>Marquer comme effectué</strong></li>
                        <li>Revenez ici pour voir vos graphiques de progression !</li>
                    </ol>
                </div>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    Object.entries(filteredGroups).forEach(([exercise, records]) => {
        const sortedRecords = records.sort((a, b) => new Date(a.date) - new Date(b.date));
        const latestRecord = sortedRecords[sortedRecords.length - 1];
        const isStatic = latestRecord.isStatic;
        
        let bestValue, bestLabel;
        if (isStatic) {
            bestValue = Math.max(...records.map(r => r.holdTime || 0));
            bestLabel = `Best: ${formatHoldTime(bestValue)}`;
        } else {
            bestValue = Math.max(...records.map(r => parseReps(r.reps)));
            bestLabel = `Best: ${bestValue} reps`;
        }
        
        const item = document.createElement('div');
        item.className = `exercise-item ${latestRecord.category}`;
        item.addEventListener('click', () => showExerciseGraph(exercise));
        
        item.innerHTML = `
            <div class="exercise-item-name">${exercise}</div>
            <div class="exercise-item-stats">
                <span>${records.length} enregistrements</span>
                <span class="exercise-item-best">${bestLabel}</span>
            </div>
        `;
        
        container.appendChild(item);
    });
}

function formatHoldTime(seconds) {
    if (!seconds || seconds === 0) return '0s';
    if (seconds >= 60) {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return secs > 0 ? `${mins}min ${secs}s` : `${mins}min`;
    }
    return `${seconds}s`;
}

function parseReps(reps) {
    if (!reps) return 0;
    if (typeof reps === 'number') return reps;
    const repsStr = String(reps).trim();
    if (repsStr.includes('-')) {
        const parts = repsStr.split('-');
        return parseInt(parts[1]) || parseInt(parts[0]) || 0;
    }
    return parseInt(repsStr) || 0;
}

// ================== GRAPHIQUE ==================
function showExerciseGraph(exercise) {
    currentExercise = exercise;
    const records = performanceRecords
        .filter(r => r.exercise === exercise)
        .sort((a, b) => new Date(a.date) - new Date(b.date));
    
    document.getElementById('graphTitle').textContent = `Progression - ${exercise}`;
    document.getElementById('graphSection').style.display = 'block';
    
    document.getElementById('graphSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    updateChart();
    displayPerformanceDetails(records);
}

function updateChart() {
    if (!currentExercise) return;
    
    const timeRange = document.getElementById('timeRange').value;
    let records = performanceRecords
        .filter(r => r.exercise === currentExercise)
        .sort((a, b) => new Date(a.date) - new Date(b.date));
    
    if (timeRange !== 'all') {
        const daysAgo = parseInt(timeRange);
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - daysAgo);
        records = records.filter(r => new Date(r.date) >= cutoffDate);
    }
    
    if (records.length === 0) return;
    
    const isStatic = records[0].isStatic;
    const labels = records.map(r => formatDate(new Date(r.date)));
    const data = isStatic 
        ? records.map(r => r.holdTime || 0) 
        : records.map(r => parseReps(r.reps));
    
    if (currentChart) {
        currentChart.destroy();
    }
    
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: isStatic ? 'Temps de maintien (secondes)' : 'Répétitions',
                data: data,
                borderColor: isStatic ? '#FF9800' : '#4CAF50',
                backgroundColor: isStatic ? 'rgba(255, 152, 0, 0.1)' : 'rgba(76, 175, 80, 0.1)',
                borderWidth: 3,
                pointRadius: 6,
                pointBackgroundColor: isStatic ? '#FF9800' : '#4CAF50',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointHoverRadius: 8,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: isStatic ? '#FF9800' : '#4CAF50',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            if (isStatic) {
                                return formatHoldTime(context.parsed.y);
                            }
                            return `${context.parsed.y} répétitions`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#b0b0b0',
                        font: {
                            size: 12,
                            weight: 600
                        },
                        stepSize: 1,
                        callback: function(value) {
                            if (isStatic) {
                                return formatHoldTime(value);
                            }
                            return value;
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    }
                },
                x: {
                    ticks: {
                        color: '#b0b0b0',
                        font: {
                            size: 12,
                            weight: 600
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    }
                }
            }
        }
    });
}

function displayPerformanceDetails(records) {
    const container = document.getElementById('performanceDetails');
    if (records.length === 0) return;
    
    const isStatic = records[0].isStatic;
    container.innerHTML = '<h3 style="color: #ffffff; font-size: 18px; margin-bottom: 15px;">Historique détaillé</h3>';
    
    records.reverse().forEach((record, index) => {
        const recordDiv = document.createElement('div');
        recordDiv.className = 'performance-record';
        
        let progressHtml = '';
        if (index < records.length - 1) {
            const prevRecord = records[index + 1];
            
            if (isStatic) {
                const currentTime = record.holdTime || 0;
                const prevTime = prevRecord.holdTime || 0;
                const diff = currentTime - prevTime;
                if (diff !== 0) {
                    const arrow = diff > 0 ? '↑' : '↓';
                    const className = diff > 0 ? '' : 'negative';
                    progressHtml = `
                        <div class="record-progress ${className}">
                            <span class="arrow">${arrow}</span>
                            <span>${formatHoldTime(Math.abs(diff))}</span>
                        </div>
                    `;
                }
            } else {
                const currentReps = parseReps(record.reps);
                const prevReps = parseReps(prevRecord.reps);
                const diff = currentReps - prevReps;
                if (diff !== 0) {
                    const arrow = diff > 0 ? '↑' : '↓';
                    const className = diff > 0 ? '' : 'negative';
                    progressHtml = `
                        <div class="record-progress ${className}">
                            <span class="arrow">${arrow}</span>
                            <span>${Math.abs(diff)} reps</span>
                        </div>
                    `;
                }
            }
        }
        
        let valueDisplay;
        if (isStatic) {
            valueDisplay = `${formatHoldTime(record.holdTime || 0)}${record.sets ? ` × ${record.sets}` : ''}`;
        } else {
            valueDisplay = `${parseReps(record.reps)} <span class="unit">reps</span>${record.sets ? ` × ${record.sets}` : ''}`;
        }
        
        recordDiv.innerHTML = `
            <div>
                <div class="record-date">${formatDate(new Date(record.date))}</div>
                ${record.workoutName ? `<div class="record-notes">Entraînement: ${record.workoutName}</div>` : ''}
                ${record.notes ? `<div class="record-notes">${record.notes}</div>` : ''}
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div class="record-value">
                    ${valueDisplay}
                </div>
                ${progressHtml}
            </div>
        `;
        
        container.appendChild(recordDiv);
    });
}

// ================== UTILITAIRES ==================
function formatDate(date) {
    return date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' });
}