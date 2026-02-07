// ================== VARIABLES GLOBALES ==================
let currentWeekStart = new Date();
let events = [];
let selectedEventId = null;

// ================== INITIALISATION ==================
document.addEventListener('DOMContentLoaded', () => {
    // Charger les événements depuis localStorage
    loadEvents();
    
    // Définir la semaine courante au début de la semaine
    setToStartOfWeek(currentWeekStart);
    
    // Afficher le calendrier
    renderCalendar();
    
    // Event listeners
    document.getElementById('btnAddEvent').addEventListener('click', openModal);
    document.getElementById('btnCloseModal').addEventListener('click', closeModal);
    document.getElementById('btnCancel').addEventListener('click', closeModal);
    document.getElementById('btnPrevWeek').addEventListener('click', previousWeek);
    document.getElementById('btnNextWeek').addEventListener('click', nextWeek);
    document.getElementById('eventForm').addEventListener('submit', handleFormSubmit);
    document.getElementById('btnCloseDetails').addEventListener('click', closeDetailsModal);
    document.getElementById('btnDeleteEvent').addEventListener('click', deleteEvent);
    
    // Fermer le modal en cliquant à l'extérieur
    document.getElementById('eventModal').addEventListener('click', (e) => {
        if (e.target.id === 'eventModal') closeModal();
    });
    
    document.getElementById('eventDetailsModal').addEventListener('click', (e) => {
        if (e.target.id === 'eventDetailsModal') closeDetailsModal();
    });
});

// ================== GESTION DES DATES ==================
function setToStartOfWeek(date) {
    const day = date.getDay();
    const diff = day === 0 ? -6 : 1 - day; // Lundi = début de semaine
    date.setDate(date.getDate() + diff);
    date.setHours(0, 0, 0, 0);
}

function getWeekDates(startDate) {
    const dates = [];
    for (let i = 0; i < 7; i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i);
        dates.push(date);
    }
    return dates;
}

function formatWeekRange(startDate) {
    const endDate = new Date(startDate);
    endDate.setDate(endDate.getDate() + 6);
    
    const options = { day: 'numeric', month: 'long', year: 'numeric' };
    const start = startDate.toLocaleDateString('fr-FR', { day: 'numeric', month: 'long' });
    const end = endDate.toLocaleDateString('fr-FR', options);
    
    return `${start} - ${end}`;
}

function formatDate(date) {
    return date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'long', year: 'numeric' });
}

function isToday(date) {
    const today = new Date();
    return date.getDate() === today.getDate() &&
           date.getMonth() === today.getMonth() &&
           date.getFullYear() === today.getFullYear();
}

function getDayName(date) {
    const days = ['Dimanche', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi'];
    return days[date.getDay()];
}

// ================== NAVIGATION ==================
function previousWeek() {
    animateWeekChange(() => {
        currentWeekStart.setDate(currentWeekStart.getDate() - 7);
        renderCalendar();
    });
}

function nextWeek() {
    animateWeekChange(() => {
        currentWeekStart.setDate(currentWeekStart.getDate() + 7);
        renderCalendar();
    });
}

function animateWeekChange(callback) {
    const grid = document.getElementById('calendarGrid');
    
    // Ajouter l'animation de sortie
    grid.classList.add('fade-out');
    
    // Attendre la fin de l'animation avant de changer le contenu
    setTimeout(() => {
        grid.classList.remove('fade-out');
        callback();
        grid.classList.add('fade-in');
        
        // Retirer la classe d'entrée après l'animation
        setTimeout(() => {
            grid.classList.remove('fade-in');
        }, 400);
    }, 300);
}

// ================== RENDU DU CALENDRIER ==================
function renderCalendar() {
    const weekDates = getWeekDates(currentWeekStart);
    document.getElementById('currentWeek').textContent = formatWeekRange(currentWeekStart);
    
    const calendarGrid = document.getElementById('calendarGrid');
    calendarGrid.innerHTML = '';
    
    weekDates.forEach(date => {
        const dayColumn = createDayColumn(date);
        calendarGrid.appendChild(dayColumn);
    });
}

function createDayColumn(date) {
    const column = document.createElement('div');
    column.className = 'day-column';
    
    const header = document.createElement('div');
    header.className = 'day-header';
    
    const dayName = document.createElement('div');
    dayName.className = 'day-name';
    dayName.textContent = getDayName(date);
    
    const dayDate = document.createElement('div');
    dayDate.className = 'day-date' + (isToday(date) ? ' today' : '');
    dayDate.textContent = date.getDate();
    
    header.appendChild(dayName);
    header.appendChild(dayDate);
    column.appendChild(header);
    
    const eventsContainer = document.createElement('div');
    eventsContainer.className = 'events-container';
    
    // Filtrer les événements pour ce jour
    const dayEvents = getEventsForDate(date);
    dayEvents.forEach(event => {
        const eventCard = createEventCard(event);
        eventsContainer.appendChild(eventCard);
    });
    
    column.appendChild(eventsContainer);
    return column;
}

function createEventCard(event) {
    const card = document.createElement('div');
    card.className = `event-card ${event.type}`;
    card.addEventListener('click', () => showEventDetails(event));
    
    const time = document.createElement('div');
    time.className = 'event-time';
    time.textContent = `${event.startTime} - ${event.endTime}`;
    
    const title = document.createElement('div');
    title.className = 'event-title';
    title.textContent = event.title;
    
    const type = document.createElement('div');
    type.className = 'event-type';
    type.textContent = getTypeLabel(event.type);
    
    card.appendChild(time);
    card.appendChild(title);
    card.appendChild(type);
    
    return card;
}

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

// ================== GESTION DES ÉVÉNEMENTS ==================
function getEventsForDate(date) {
    return events.filter(event => {
        const eventDate = new Date(event.date);
        return eventDate.getDate() === date.getDate() &&
               eventDate.getMonth() === date.getMonth() &&
               eventDate.getFullYear() === date.getFullYear();
    }).sort((a, b) => a.startTime.localeCompare(b.startTime));
}

function loadEvents() {
    const stored = localStorage.getItem('levelUpCaliEvents');
    if (stored) {
        events = JSON.parse(stored);
    }
}

function saveEvents() {
    localStorage.setItem('levelUpCaliEvents', JSON.stringify(events));
}

// ================== MODAL ==================
function openModal() {
    document.getElementById('eventModal').classList.add('active');
    document.getElementById('eventForm').reset();
    // Définir la date d'aujourd'hui par défaut
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('eventDate').value = today;
}

function closeModal() {
    document.getElementById('eventModal').classList.remove('active');
}

function handleFormSubmit(e) {
    e.preventDefault();
    
    const newEvent = {
        id: Date.now(),
        title: document.getElementById('eventTitle').value,
        date: document.getElementById('eventDate').value,
        startTime: document.getElementById('eventStartTime').value,
        endTime: document.getElementById('eventEndTime').value,
        type: document.getElementById('eventType').value,
        notes: document.getElementById('eventNotes').value
    };
    
    events.push(newEvent);
    saveEvents();
    renderCalendar();
    closeModal();
}

// ================== DÉTAILS DE L'ÉVÉNEMENT ==================
function showEventDetails(event) {
    selectedEventId = event.id;
    
    document.getElementById('detailsTitle').textContent = event.title;
    document.getElementById('detailsDate').textContent = formatDate(new Date(event.date));
    document.getElementById('detailsTime').textContent = `${event.startTime} - ${event.endTime}`;
    document.getElementById('detailsType').textContent = getTypeLabel(event.type);
    
    const notesContainer = document.getElementById('detailsNotesContainer');
    if (event.notes) {
        notesContainer.style.display = 'block';
        document.getElementById('detailsNotes').textContent = event.notes;
    } else {
        notesContainer.style.display = 'none';
    }
    
    document.getElementById('eventDetailsModal').classList.add('active');
}

function closeDetailsModal() {
    document.getElementById('eventDetailsModal').classList.remove('active');
    selectedEventId = null;
}

function deleteEvent() {
    if (selectedEventId && confirm('Êtes-vous sûr de vouloir supprimer cet entraînement ?')) {
        events = events.filter(event => event.id !== selectedEventId);
        saveEvents();
        renderCalendar();
        closeDetailsModal();
    }
}