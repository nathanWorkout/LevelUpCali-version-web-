# 🏋️ LevelUpCali

> Moteur d'analyse biomécanique pour le street workout — version web

---

## Problématique

Dans le street workout, l'analyse technique repose presque exclusivement sur l'observation visuelle.
Même pour des pratiquants avancés ou des coachs expérimentés, certaines compensations biomécaniques sont :

- difficiles à percevoir,
- parfois invisibles à l'œil nu,
- souvent détectées trop tard (stagnation, mauvaise progression, risque de blessure).

L'œil humain juge le rendu visuel, mais le corps, lui, s'adapte mécaniquement.

---

## Solution

LevelUpCali est un moteur d'analyse biomécanique spécialisé dans le street workout.

Il s'appuie sur :

- la **vision par ordinateur**,
- des **règles expertes biomécaniques** personnalisées,
- une **modélisation logique du corps humain** pour analyser des images et détecter les compensations invisibles visuellement.

Le système ne se contente pas de dire « figure correcte ou non » : il identifie les **causes biomécaniques sous-jacentes** et leurs conséquences sur le reste du corps.

Chaque décision est explicable, traçable et liée à une logique biomécanique réelle :

```
Problème → Compensation → Correction
```

---

## Fonctionnement

1. Détection des landmarks corporels via **MediaPipe**
2. Calcul des **angles articulaires** et des lignes corporelles
3. Application de **règles expertes** spécifiques au street workout
4. Identification des défauts techniques et des **compensations associées**
5. Génération de feedbacks techniques clairs, hiérarchisés et exploitables

> Les règles expertes garantissent des décisions fiables et explicables, adaptées aux exigences du street workout, contrairement aux modèles purement statistiques.

---

## Fonctionnalités

### 🔬 Analyse biomécanique

- Détection automatique des articulations
- Calcul précis des angles articulaires
- Analyse des lignes corporelles (alignement, stabilité)
- Détection de compensations mécaniques
- Annotation visuelle de l'image avec mise en évidence des erreurs

**Figures actuellement analysées :**

| Figure | Erreurs détectées |
|---|---|
| Handstand | Hanches fléchies, coudes fléchis, genoux fléchis, épaules insuffisamment ouvertes |
| Planche | Hanches basses, coudes fléchis, position des épaules |
| Front Lever | Hanches basses, coudes fléchis, position des épaules |

**Exemple — Front Lever :**
> Hanches trop basses → compensation au niveau des bras / épaules → feedback : rétroversion du bassin + dépression scapulaire

### 📅 Planning

- Calendrier hebdomadaire pour organiser ses séances
- Création d'événements avec type, horaires et notes

### 💪 Entraînements

- Création et gestion de routines personnalisées
- Support des exercices dynamiques (répétitions) et statiques (maintien)
- Filtres par type (Pull / Push / Legs / Full Body / Skills) et niveau
- Enregistrement des performances réelles après chaque séance

### 📊 Graphiques

- Visualisation de la progression par exercice
- Filtres temporels (7 jours, 30 jours, 3 mois, 1 an)
- Statistiques globales : performances enregistrées, exercices suivis, taux de progression

---

## Architecture technique

| Couche | Technologie |
|---|---|
| Front-end | HTML / CSS / JavaScript (Vanilla) + Chart.js |
| Back-end | Python / Flask — déployé sur **Render** |
| Vision par ordinateur | MediaPipe |
| Analyse biomécanique | Règles expertes personnalisées |

---

## API — Analyse biomécanique

### `POST /analyze_static`

Analyse une image statique et retourne le diagnostic biomécanique.

**Requête :** `multipart/form-data` — champ `image` (JPG ou PNG, max 10 Mo)

**Réponse :**

```json
{
  "status": "ok",
  "detected_figure": "front_lever",
  "image_base64": "...",
  "analysis": {
    "cause": "Hanches trop basses, le corps n'est pas aligné horizontalement",
    "compensation": "Les bras se plient pour compenser le manque de gainage",
    "correction": "Contracte abdos/fessiers en rétroversion + tire plus fort avec les épaules"
  },
  "deviations": {
    "hanches_basses": "Oui"
  }
}
```

---

## Structure du projet

```
levelupcali/
├── index.html            # Page d'accueil
├── planning.html         # Calendrier
├── planning.js
├── entrainements.html    # Gestion des routines
├── entrainements.js
├── graphiques.html       # Suivi des performances
├── graphiques.js
├── analyse.html          # Analyse biomécanique
├── analyse.js
├── app.py                # API Flask (backend)
└── *.css                 # Feuilles de style par page
```

---

## Conseils pour l'analyse

Pour obtenir les meilleurs résultats :

- **Vue de profil** — se placer perpendiculairement à la caméra (90°)
- **Corps complet** — tout le corps doit être visible dans le cadre
- **Bon éclairage** — éviter les contre-jours et zones sombres
- **Distance** — reculer la caméra à 2-3 mètres minimum

---

## Améliorations futures

- Analyse biomécanique complète sur toutes les figures (pull-ups, dips, pompes...)
- Analyse vidéo en temps réel
- Générateur de programmes entièrement personnalisés
- Détection avancée des schémas de stagnation
- Enrichissement progressif des règles expertes

---

## Statut du projet

Projet en développement actif.
Conçu, développé et maintenu par un pratiquant de street workout.
Approche biomécanique réelle, orientée performance et sécurité.
