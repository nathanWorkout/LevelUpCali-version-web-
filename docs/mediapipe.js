import {
  FilesetResolver,  // Charge le moteur de traitement de vision
  PoseLandmarker,   // Charge le modèle qui détecte les 33 points du corps
  DrawingUtils      // Pour dessiner les points sur le cnavas
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

// ============================================================================
// INITIALISATION DES MODÈLES
// ============================================================================
const STATIC_SKILLS = {
  handstand: {
    elbow:    { min: 160 },
    shoulder: { min: 160 },
    hip:      { min: 158 },
    knee:     { min: 165 }
  },

  planche: {
    elbow:    { min: 160 },
    shoulder: { min: 25, max: 65 },
    hip:      { min: 148 }
  },

  front_lever: {
    elbow:            { min: 160 },
    shoulder:         { min: 25, max: 65 },
    hip:              { min: 160 },
    tolerance_biceps: 3
  }
};


let poseLandmarker; // Modèle de détection
let runningMode = "IMAGE"; // Mode image
async function initializePoseLandmarker() {

    // Charge le webAssembly nécessaire pour les tâches de vision (WASM = c++ pour le web)
    // await = attende que le chargement soit terminé avant de continuer
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {

        baseOptions: {
            modelAssetPath: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/pose_landmarker_full.task"
        },

        runningMode: runningMode, // On passe en mode image
        numPoses: 1 // Max une personne détectée
    });
}

initializePoseLandmarker();