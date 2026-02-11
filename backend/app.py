from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os
import mediapipe as mp
import sys
import logging
from datetime import datetime
from scipy.signal import find_peaks, savgol_filter


# CONFIGURATION & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    try:
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        sys.stdout = open(sys.stdout.fileno(), 'w', encoding='utf-8', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), 'w', encoding='utf-8', buffering=1)
    except Exception as e:
        logger.warning(f"Impossible de configurer le mode Windows: {e}")


# INITIALISATION FLASK & MEDIAPIPE
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = False
app.config['PROPAGATE_EXCEPTIONS'] = True


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Modèles pour mouvements dynamiques
dynamic_skills = {
    "push_up": {
        "elbow": {"min": 80, "max": 175},
        "hip_drift_max": 0.08,
        "asymmetry_max": 12
    },
    "pull_up": {
        "elbow": {"min": 45, "max": 165},
        "shoulder": {"min": 30, "max": 160},
        "kipping_max": 0.10
    },
    "dips": {
        "elbow": {"min": 60, "max": 175},
        "shoulder": {"min": 45, "max": 150},
        "lean_max": 0.10
    }
}

# Modèles pour figures statiques
static_skills = {
    "handstand": {
        "elbow": {"min": 165, "max": 180},
        "shoulder": {"min": 165, "max": 180},
        "hip": {"min": 165, "max": 180}
    },
    "planche": {
        "elbow": {"min": 165, "max": 180},
        "shoulder": {"min": 30, "max": 60},
        "hip": {"min": 165, "max": 180}
    },
    "front_lever": {
        "elbow": {"min": 162, "max": 180},  # 165 - 3° tolérance biceps
        "shoulder": {"min": 30, "max": 60},
        "hip": {"min": 167, "max": 180},     # 170 - 3° tolérance
        "tolerance_biceps": 3  # Tolérance pour développement musculaire des biceps
    }
}

# ============================================================================
# Fonctions de prétraitement d'image
# ============================================================================
def adjust_brightness_auto(image):
    """
    Ajuste automatiquement la luminosité de l'image pour améliorer la détection MediaPipe.
    Version EXTRÊME - Réduit massivement la luminosité.
    """
    try:
        # Validation de l'image d'entrée
        if image is None or image.size == 0:
            logger.error("Image invalide reçue dans adjust_brightness_auto")
            return image
            
        # Conversion en LAB pour analyser la luminosité
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Analyse de la luminosité moyenne
        mean_brightness = np.mean(l_channel)
        logger.info(f" Luminosité détectée: {mean_brightness:.1f}/255")
        
        # CIBLE EXTRÊMEMENT BASSE : 60-70 (au lieu de 100)
        target_brightness = 65
        
        # RÉDUCTION MASSIVE pour TOUTES les images > 70
        if mean_brightness > 70:
            logger.info(f" Réduction EXTRÊME de la luminosité de {mean_brightness:.1f} vers {target_brightness}...")
            
            # Calculer le facteur de réduction
            brightness_factor = target_brightness / mean_brightness
            
            # Appliquer une réduction encore plus agressive
            # Multiplier par un facteur supplémentaire de 0.75 pour vraiment assombrir
            l_channel = np.clip(l_channel * brightness_factor * 0.75, 0, 255).astype(np.uint8)
            
            corrected_lab = cv2.merge([l_channel, a, b])
            corrected_image = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
            new_brightness = np.mean(cv2.split(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2LAB))[0])
            logger.info(f" Luminosité EXTRÊMEMENT réduite: {mean_brightness:.1f} → {new_brightness:.1f}")
            return corrected_image
        
        # Si déjà sombre, laisser tel quel
        else:
            logger.info(" Image déjà très sombre, parfait pour MediaPipe")
            return image
            
    except Exception as e:
        logger.error(f" Erreur ajustement luminosité: {e}")
        return image

# Détection environnement Render
IS_RENDER = os.environ.get("RENDER") is not None

def auto_rotate_image(image):
    """
    Détecte et corrige l'orientation de l'image si le corps est horizontal.
    """
    try:
        h, w = image.shape[:2]
        
        # Si l'image est clairement en portrait (hauteur > largeur)
        # et que le mouvement détecté est horizontal, on peut la faire pivoter
        if h > w * 1.2:  # Image en mode portrait
            logger.info(f"📱 Image portrait détectée ({w}x{h}), test de rotation...")
            
            # Essayer une rotation de 90° dans les deux sens
            rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Tester quelle rotation détecte le mieux le corps
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3) as pose:
                # Test image originale
                rgb_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result_orig = pose.process(rgb_orig)
                conf_orig = result_orig.pose_landmarks.landmark[0].visibility if result_orig.pose_landmarks else 0
                
                # Test rotation 90°
                rgb_90 = cv2.cvtColor(rotated_90, cv2.COLOR_BGR2RGB)
                result_90 = pose.process(rgb_90)
                conf_90 = result_90.pose_landmarks.landmark[0].visibility if result_90.pose_landmarks else 0
                
                # Test rotation 270°
                rgb_270 = cv2.cvtColor(rotated_270, cv2.COLOR_BGR2RGB)
                result_270 = pose.process(rgb_270)
                conf_270 = result_270.pose_landmarks.landmark[0].visibility if result_270.pose_landmarks else 0
                
                logger.info(f"Confiances: orig={conf_orig:.2f}, 90°={conf_90:.2f}, 270°={conf_270:.2f}")
                
                # Choisir la meilleure orientation
                if conf_90 > conf_orig and conf_90 > conf_270:
                    logger.info("Rotation 90° appliquée")
                    return rotated_90
                elif conf_270 > conf_orig and conf_270 > conf_90:
                    logger.info("Rotation 270° appliquée")
                    return rotated_270
                    
        return image
        
    except Exception as e:
        logger.warning(f"Auto-rotation échouée: {e}")
        return image


def enhance_image_for_detection(image):
    """
    Applique plusieurs améliorations pour optimiser la détection MediaPipe.
    Version EXTRÊME - Assombrissement massif.
    """
    original_image = image.copy()
    
    try:
        # 0. Rotation automatique si nécessaire
        logger.info(" Vérification orientation...")
        image = auto_rotate_image(image)
        
        # 1. Ajustement de luminosité EXTRÊME (TOUJOURS actif)
        logger.info("Ajustement EXTRÊME de la luminosité...")
        image = adjust_brightness_auto(image)
        
        if image is None:
            logger.warning("adjust_brightness_auto a retourné None, utilisation image originale")
            image = original_image
        
        # 2. Amélioration du contraste MAXIMALE (TOUJOURS actif)
        try:
            logger.info("Amélioration MAXIMALE du contraste...")
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Analyser le contraste de l'image
            contrast = np.std(l)
            logger.info(f"Contraste actuel: {contrast:.1f}")
            
            # Renforcement maximal du contraste pour toutes les images
            if contrast < 30:
                logger.info("Contraste très faible - CLAHE ultra-agressif...")
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
            else:
                logger.info("Contraste normal - CLAHE agressif...")
                clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            
            l = clahe.apply(l)
            
            # Augmenter davantage la saturation des couleurs
            a = cv2.convertScaleAbs(a, alpha=1.2, beta=0)
            b = cv2.convertScaleAbs(b, alpha=1.2, beta=0)
            
            enhanced_lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            logger.info("Contraste MAXIMISÉ avec succès")
        except Exception as e:
            logger.warning(f"Amélioration contraste échouée: {e}")
        
        # 3. Réduction du bruit (uniquement en local)
        if not IS_RENDER:
            try:
                logger.info("Réduction du bruit (local uniquement)...")
                image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
                logger.info("Réduction du bruit appliquée")
            except Exception as e:
                logger.warning(f"Réduction bruit échouée: {e}")
        
        # Vérification finale
        if image is None or image.size == 0:
            logger.error("Image finale invalide, retour à l'originale")
            return original_image
            
        # Log de la luminosité finale
        try:
            final_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            final_brightness = np.mean(cv2.split(final_lab)[0])
            logger.info(f"Luminosité FINALE pour MediaPipe: {final_brightness:.1f}/255")
        except:
            pass
            
        logger.info(f"Image finale: shape={image.shape}, dtype={image.dtype}")
        return image
        
    except Exception as e:
        logger.error(f"Erreur critique lors de l'amélioration de l'image: {e}")
        return original_image

# ============================================================================
# Fonctions pour les calculs
# ============================================================================
def calculate_angle(a, b, c):
    """Calcule l'angle entre trois points."""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    except Exception as e:
        logger.error(f"Erreur calcul angle: {e}")
        return 0.0

def get_landmark_points(landmarks, name):
    """Récupère les coordonnées x, y d'un point anatomique."""
    try:
        lm = landmarks.landmark[mp_pose.PoseLandmark[name.upper()]]
        return [lm.x, lm.y]
    except (KeyError, AttributeError) as e:
        logger.error(f"Erreur récupération landmark {name}: {e}")
        return [0.0, 0.0]

def calculate_user_angles(landmarks, image_shape):
    """Calcule tous les angles articulaires de l'utilisateur."""
    joints = {
        "left_elbow": ["left_shoulder", "left_elbow", "left_wrist"],
        "right_elbow": ["right_shoulder", "right_elbow", "right_wrist"],
        "left_shoulder": ["left_elbow", "left_shoulder", "left_hip"],
        "right_shoulder": ["right_elbow", "right_shoulder", "right_hip"],
        "left_hip": ["left_shoulder", "left_hip", "left_knee"],
        "right_hip": ["right_shoulder", "right_hip", "right_knee"],
        "left_knee": ["left_hip", "left_knee", "left_ankle"],
        "right_knee": ["right_hip", "right_knee", "right_ankle"],
        "left_ankle": ["left_knee", "left_ankle", "left_foot_index"],
        "right_ankle": ["right_knee", "right_ankle", "right_foot_index"]
    }

    user_angles = {}
    for joint, points in joints.items():
        try:
            a = get_landmark_points(landmarks, points[0])
            b = get_landmark_points(landmarks, points[1])
            c = get_landmark_points(landmarks, points[2])
            angle = calculate_angle(a, b, c)
            user_angles[joint] = float(angle)
        except Exception as e:
            logger.warning(f"Impossible de calculer {joint}: {e}")
            user_angles[joint] = 0.0

    return user_angles

def calculate_scapular_position(landmarks):
    """Calcule la position des scapulas (protraction/rétraction)."""
    try:
        lm = landmarks.landmark
        left_shoulder_z = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z
        right_shoulder_z = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
        left_elbow_z = lm[mp_pose.PoseLandmark.LEFT_ELBOW].z
        right_elbow_z = lm[mp_pose.PoseLandmark.RIGHT_ELBOW].z

        avg_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2
        avg_elbow_z = (left_elbow_z + right_elbow_z) / 2
        z_diff = avg_shoulder_z - avg_elbow_z

        is_protracted = z_diff > 0.05
        is_retracted = z_diff < -0.05

        return {
            "is_protracted": bool(is_protracted),
            "is_retracted": bool(is_retracted),
            "is_neutral": bool(not is_protracted and not is_retracted)
        }
    except Exception as e:
        logger.error(f"Erreur calcul scapulaire: {e}")
        return {"is_protracted": False, "is_retracted": False, "is_neutral": True}

def get_error_landmarks(figure, deviations):
    """Retourne les landmarks à colorier en rouge selon les erreurs détectées."""
    error_landmarks = []
    
    if figure == "handstand":
        if deviations.get("hanches_flechies") == "Oui":
            error_landmarks.extend(['LEFT_HIP', 'RIGHT_HIP'])
        if deviations.get("coudes_flechis") == "Oui":
            error_landmarks.extend(['LEFT_ELBOW', 'RIGHT_ELBOW'])
        if deviations.get("genoux_flechis") == "Oui":
            error_landmarks.extend(['LEFT_KNEE', 'RIGHT_KNEE'])
    
    elif figure == "planche":
        if deviations.get("hanches_basses") == "Oui":
            error_landmarks.extend(['LEFT_HIP', 'RIGHT_HIP'])
        if deviations.get("coudes_flechis") == "Oui":
            error_landmarks.extend(['LEFT_ELBOW', 'RIGHT_ELBOW'])
        if deviations.get("position_epaules") == "Oui":
            error_landmarks.extend(['LEFT_SHOULDER', 'RIGHT_SHOULDER'])
    
    elif figure == "front_lever":
        if deviations.get("hanches_basses") == "Oui":
            error_landmarks.extend(['LEFT_HIP', 'RIGHT_HIP'])
        if deviations.get("coudes_flechis") == "Oui":
            error_landmarks.extend(['LEFT_ELBOW', 'RIGHT_ELBOW'])
        if deviations.get("position_epaules") == "Oui":
            error_landmarks.extend(['LEFT_SHOULDER', 'RIGHT_SHOULDER'])
    
    return list(set(error_landmarks))


def draw_landmarks_with_errors(image, landmarks, error_landmarks=None):
    """Dessine les landmarks avec les erreurs en rouge."""
    if error_landmarks is None:
        error_landmarks = []
    
    # Dessiner les connexions en blanc
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=0),
        mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=0)
    )
    
    # Dessiner les points individuels
    h, w, _ = image.shape
    for idx, landmark in enumerate(landmarks.landmark):
        landmark_name = mp_pose.PoseLandmark(idx).name
        
        # Coordonnées du point
        cx = int(landmark.x * w)
        cy = int(landmark.y * h)
        
        # Couleur selon si c'est une erreur ou non
        if landmark_name in error_landmarks:
            color = (0, 0, 255)  # Rouge pour les erreurs
            radius = 8
        else:
            color = (255, 255, 255)  # Blanc pour les points normaux
            radius = 4
        
        # Dessiner le point avec contour noir
        cv2.circle(image, (cx, cy), radius, color, -1)
        cv2.circle(image, (cx, cy), radius, (0, 0, 0), 1)

# ============================================================================
# DÉTECTION DE FIGURE DYNAMIQUE vs STATIQUE
# ============================================================================
def classify_movement_type(landmarks_sequence):
    """Détermine si c'est un mouvement DYNAMIQUE ou STATIQUE."""
    if len(landmarks_sequence) < 10:
        return "unknown", None

    try:
        sample_size = min(30, len(landmarks_sequence))

        nose_y_values = []
        hip_y_values = []
        elbow_y_values = []

        for landmarks in landmarks_sequence[:sample_size]:
            nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

            nose_y_values.append(nose.y)
            hip_y = (left_hip.y + right_hip.y) / 2
            hip_y_values.append(hip_y)
            elbow_y = (left_elbow.y + right_elbow.y) / 2
            elbow_y_values.append(elbow_y)

        # Variations
        nose_var = np.std(nose_y_values)
        hip_var = np.std(hip_y_values)
        elbow_var = np.std(elbow_y_values)

        total_variation = nose_var + hip_var + elbow_var

        logger.info(f"Classification: nose_var={nose_var:.3f}, hip_var={hip_var:.3f}, elbow_var={elbow_var:.3f}, total={total_variation:.3f}")

        # Si forte variation → DYNAMIQUE
        if total_variation > 0.15 or elbow_var > 0.05:
            logger.info("→ Mouvement DYNAMIQUE détecté")
            return "dynamic", detect_dynamic_exercise(landmarks_sequence)

        # Sinon → STATIQUE
        else:
            logger.info("→ Figure STATIQUE détectée")
            return "static", detect_static_figure(landmarks_sequence)

    except Exception as e:
        logger.error(f"Erreur classification: {e}")
        return "unknown", None

def detect_dynamic_exercise(landmarks_sequence):
    """Détecte le type d'exercice DYNAMIQUE (push-up, pull-up, dips)."""
    try:
        sample_size = min(20, len(landmarks_sequence))

        nose_y_values = []
        hip_y_values = []
        elbow_y_values = []
        shoulder_y_values = []

        for landmarks in landmarks_sequence[:sample_size]:
            nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

            nose_y_values.append(nose.y)
            hip_y = (left_hip.y + right_hip.y) / 2
            hip_y_values.append(hip_y)
            elbow_y = (left_elbow.y + right_elbow.y) / 2
            elbow_y_values.append(elbow_y)
            shoulder_y_values.append(left_shoulder.y)

        nose_var = np.std(nose_y_values)
        hip_var = np.std(hip_y_values)
        elbow_var = np.std(elbow_y_values)

        avg_nose_y = np.mean(nose_y_values)
        avg_hip_y = np.mean(hip_y_values)
        avg_shoulder_y = np.mean(shoulder_y_values)

        logger.info(f"Détection dynamique: nose_var={nose_var:.3f}, elbow_var={elbow_var:.3f}")

        # Pull-up: forte variation du nez (monte/descend), les coudes varient beaucoup
        if nose_var > 0.08 and elbow_var > 0.08 and avg_nose_y < avg_shoulder_y:
            logger.info("→ PULL-UP détecté")
            return "pull_up"

        # Dips: variation moyenne du corps, les coudes varient, le corps descend
        if elbow_var > 0.05 and hip_var > 0.03 and avg_nose_y < avg_hip_y:
            logger.info("→ DIPS détecté")
            return "dips"

        # Push-up: faible variation verticale, les coudes varient
        if elbow_var > 0.03:
            logger.info("→ PUSH-UP détecté")
            return "push_up"

        return "unknown"

    except Exception as e:
        logger.error(f"Erreur détection exercice dynamique: {e}")
        return "unknown"

def detect_static_figure(landmarks_sequence):
    """Détecte le type de figure STATIQUE avec logique multi-critères robuste."""
    try:
        sample_size = min(10, len(landmarks_sequence))

        nose_y_values = []
        hip_y_values = []
        wrist_y_values = []
        shoulder_y_values = []
        ankle_y_values = []
        elbow_y_values = []

        # Positions Z (profondeur)
        nose_z_values = []
        wrist_z_values = []
        shoulder_z_values = []
        hip_z_values = []
        ankle_z_values = []

        # Positions X (horizontal)
        wrist_x_values = []
        shoulder_x_values = []
        hip_x_values = []

        for landmarks in landmarks_sequence[:sample_size]:
            lm = landmarks.landmark

            nose = lm[mp_pose.PoseLandmark.NOSE]
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Y (vertical)
            nose_y_values.append(nose.y)
            hip_y = (left_hip.y + right_hip.y) / 2
            hip_y_values.append(hip_y)
            wrist_y = (left_wrist.y + right_wrist.y) / 2
            wrist_y_values.append(wrist_y)
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_y_values.append(shoulder_y)
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            ankle_y_values.append(ankle_y)
            elbow_y = (left_elbow.y + right_elbow.y) / 2
            elbow_y_values.append(elbow_y)

            # Z (profondeur)
            nose_z_values.append(nose.z)
            wrist_z = (left_wrist.z + right_wrist.z) / 2
            wrist_z_values.append(wrist_z)
            shoulder_z = (left_shoulder.z + right_shoulder.z) / 2
            shoulder_z_values.append(shoulder_z)
            hip_z = (left_hip.z + right_hip.z) / 2
            hip_z_values.append(hip_z)
            ankle_z = (left_ankle.z + right_ankle.z) / 2
            ankle_z_values.append(ankle_z)

            # X (horizontal)
            wrist_x = (left_wrist.x + right_wrist.x) / 2
            wrist_x_values.append(wrist_x)
            shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_x_values.append(shoulder_x)
            hip_x = (left_hip.x + right_hip.x) / 2
            hip_x_values.append(hip_x)

        # Moyennes
        avg_nose_y = np.mean(nose_y_values)
        avg_hip_y = np.mean(hip_y_values)
        avg_wrist_y = np.mean(wrist_y_values)
        avg_shoulder_y = np.mean(shoulder_y_values)
        avg_ankle_y = np.mean(ankle_y_values)
        avg_elbow_y = np.mean(elbow_y_values)

        avg_nose_z = np.mean(nose_z_values)
        avg_wrist_z = np.mean(wrist_z_values)
        avg_shoulder_z = np.mean(shoulder_z_values)
        avg_hip_z = np.mean(hip_z_values)
        avg_ankle_z = np.mean(ankle_z_values)

        avg_wrist_x = np.mean(wrist_x_values)
        avg_shoulder_x = np.mean(shoulder_x_values)
        avg_hip_x = np.mean(hip_x_values)

        logger.info(f"Détection Y: nose={avg_nose_y:.3f}, hip={avg_hip_y:.3f}, wrist={avg_wrist_y:.3f}, shoulder={avg_shoulder_y:.3f}, elbow={avg_elbow_y:.3f}, ankle={avg_ankle_y:.3f}")
        logger.info(f"Détection Z: nose={avg_nose_z:.3f}, wrist={avg_wrist_z:.3f}, shoulder={avg_shoulder_z:.3f}, hip={avg_hip_z:.3f}, ankle={avg_ankle_z:.3f}")
        logger.info(f"Détection X: wrist={avg_wrist_x:.3f}, shoulder={avg_shoulder_x:.3f}, hip={avg_hip_x:.3f}")

        # ====================================================================
        # HANDSTAND
        # ====================================================================
        ankle_nose_diff = avg_nose_y - avg_ankle_y
        logger.info(f"Différence verticale handstand (nose_y - ankle_y): {ankle_nose_diff:.3f}")
        
        wrists_at_bottom = avg_wrist_y > avg_nose_y
        
        if ankle_nose_diff > 0.15 and avg_nose_y > avg_hip_y and wrists_at_bottom:
            logger.info("→ HANDSTAND détecté (chevilles en haut, tête en bas, mains au sol)")
            return "handstand"

        # ====================================================================
        # CORPS HORIZONTAL : Différencier FRONT LEVER vs PLANCHE
        # ====================================================================
        y_difference = avg_hip_y - avg_nose_y

        logger.info(f"Différence verticale (hip_y - nose_y): {y_difference:.3f}")

        if y_difference > 0.10:
            logger.info("→ FRONT LEVER détecté (nez plus haut que hanches)")
            return "front_lever"

        if abs(y_difference) < 0.25:

            wrists_above_shoulders = avg_wrist_y < avg_shoulder_y + 0.05
            wrists_behind_shoulders_z = avg_wrist_z > avg_shoulder_z + 0.02
            nose_behind_hips_z = avg_nose_z > avg_hip_z + 0.05
            ankles_closer_than_hips = avg_ankle_z < avg_hip_z
            wrist_shoulder_x_distance = abs(avg_wrist_x - avg_shoulder_x)
            wrists_far_from_shoulders_x = wrist_shoulder_x_distance > 0.08
            elbows_above_shoulders = avg_elbow_y <= avg_shoulder_y + 0.05

            logger.info(f"=== Analyse Front Lever vs Planche ===")
            logger.info(f"  1. Poignets au-dessus épaules (Y): {wrists_above_shoulders}")
            logger.info(f"  2. Poignets derrière épaules (Z): {wrists_behind_shoulders_z}")
            logger.info(f"  3. Nez derrière hanches (Z): {nose_behind_hips_z}")
            logger.info(f"  4. Chevilles plus proches caméra: {ankles_closer_than_hips}")
            logger.info(f"  5. Poignets éloignés épaules (X): {wrists_far_from_shoulders_x} (dist={wrist_shoulder_x_distance:.3f})")
            logger.info(f"  6. Coudes au-dessus épaules (Y): {elbows_above_shoulders}")

            front_lever_score = sum([
                wrists_above_shoulders,
                wrists_behind_shoulders_z,
                nose_behind_hips_z,
                ankles_closer_than_hips,
                wrists_far_from_shoulders_x,
                elbows_above_shoulders
            ])

            logger.info(f"  → Score Front Lever: {front_lever_score}/6")

            if front_lever_score >= 3:
                logger.info("→ FRONT LEVER détecté")
                return "front_lever"
            else:
                logger.info("→ PLANCHE détectée")
                return "planche"

        return "unknown"

    except Exception as e:
        logger.error(f"Erreur détection figure statique: {e}")
        return "unknown"

# ============================================================================
# EXTRACTION DES CYCLES POUR MOUVEMENTS DYNAMIQUES
# ============================================================================
def extract_cycles_generic(signal, timestamps, model, signal_inverted=False):
    """Extrait les cycles d'un mouvement à partir d'un signal."""
    try:
        if len(signal) < 10:
            return []

        # Lissage
        window_length = min(11, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
        if window_length < 5:
            smoothed = signal
        else:
            smoothed = savgol_filter(signal, window_length, 3)

        # Détection des pics
        if signal_inverted:
            peaks, _ = find_peaks(smoothed, distance=10, prominence=5)
            valleys, _ = find_peaks(-np.array(smoothed), distance=10, prominence=5)
        else:
            valleys, _ = find_peaks(-np.array(smoothed), distance=10, prominence=5)
            peaks, _ = find_peaks(smoothed, distance=10, prominence=5)

        cycles = []
        for i in range(min(len(peaks), len(valleys))):
            if signal_inverted:
                start_idx = valleys[i] if i < len(valleys) else peaks[i]
                end_idx = peaks[i]
            else:
                start_idx = peaks[i]
                end_idx = valleys[i] if i < len(valleys) else peaks[i]

            if start_idx < end_idx < len(signal):
                min_val = min(smoothed[start_idx:end_idx+1])
                max_val = max(smoothed[start_idx:end_idx+1])
                amplitude = max_val - min_val
                duration = timestamps[end_idx] - timestamps[start_idx]

                cycles.append({
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "min_value": float(min_val),
                    "max_value": float(max_val),
                    "amplitude": float(amplitude),
                    "duration": float(duration)
                })

        return cycles

    except Exception as e:
        logger.error(f"Erreur extraction cycles: {e}")
        return []

# ============================================================================
# ANALYSES - FIGURES STATIQUES
# ============================================================================
def analyze_static_hold_unified(figure, angles_data, model):
    """
    Analyse unifiée des figures statiques (image ou vidéo).
    Retourne les mêmes conseils peu importe le type de média.
    """
    if not angles_data:
        return {
            "cause": "Pas de données disponibles",
            "compensation": "N/A",
            "correction": "N/A",
            "deviations": {}
        }

    # Normalisation : convertir en liste si c'est un dict unique (image)
    if isinstance(angles_data, dict):
        angles_sequence = [angles_data]
    else:
        angles_sequence = angles_data

    # Calcul des moyennes sur toute la séquence
    avg_left_elbow = np.mean([a.get("left_elbow", 180) for a in angles_sequence])
    avg_right_elbow = np.mean([a.get("right_elbow", 180) for a in angles_sequence])
    avg_left_hip = np.mean([a.get("left_hip", 180) for a in angles_sequence])
    avg_right_hip = np.mean([a.get("right_hip", 180) for a in angles_sequence])
    avg_left_shoulder = np.mean([a.get("left_shoulder", 180) for a in angles_sequence])
    avg_right_shoulder = np.mean([a.get("right_shoulder", 180) for a in angles_sequence])
    avg_left_knee = np.mean([a.get("left_knee", 180) for a in angles_sequence])
    avg_right_knee = np.mean([a.get("right_knee", 180) for a in angles_sequence])

    deviations = {}
    primary_issue = None

    logger.info(f"=== ANALYSE {figure.upper()} ===")
    logger.info(f"Angles hanches: left={avg_left_hip:.1f}°, right={avg_right_hip:.1f}°")
    logger.info(f"Angles coudes: left={avg_left_elbow:.1f}°, right={avg_right_elbow:.1f}°")
    logger.info(f"Seuils hanches: min={model['hip']['min']}°")

    # ========================================================================
    # HANDSTAND
    # ========================================================================
    if figure == "handstand":
        # Erreur 1 : Hanches fléchies
        if avg_left_hip < model["hip"]["min"] or avg_right_hip < model["hip"]["min"]:
            deviations["hanches_flechies"] = "Oui"
            primary_issue = {
                "cause": "Hanches fléchies, alignement perdu",
                "compensation": "Cambrure lombaire excessive et coudes fléchis pour compenser",
                "correction": "Contracte abdos et fessiers pour aligner le corps verticalement"
            }

        # Erreur 2 : Coudes fléchis
        if (avg_left_elbow < model["elbow"]["min"] or avg_right_elbow < model["elbow"]["min"]) and not primary_issue:
            deviations["coudes_flechis"] = "Oui"
            primary_issue = {
                "cause": "Coudes fléchis pendant le maintien",
                "compensation": "Le dos se cambre pour maintenir l'équilibre",
                "correction": "Verrouille complètement les coudes et engage les triceps"
            }

        # Erreur 3 : Genoux fléchis
        if (avg_left_knee < 170 or avg_right_knee < 170) and not primary_issue:
            deviations["genoux_flechis"] = "Oui"
            primary_issue = {
                "cause": "Genoux fléchis, jambes non tendues",
                "compensation": "Perte d'alignement et instabilité",
                "correction": "Verrouille complètement les genoux, engage les quadriceps"
            }

    # ========================================================================
    # PLANCHE
    # ========================================================================
    elif figure == "planche":
        # Erreur 1 : Hanches basses (bassin abaissé)
        if avg_left_hip < model["hip"]["min"] or avg_right_hip < model["hip"]["min"]:
            deviations["hanches_basses"] = "Oui"
            primary_issue = {
                "cause": "Hanches trop basses, gainage insuffisant",
                "compensation": "Les bras compensent en se pliant pour soutenir le poids",
                "correction": "Renforce le gainage : serre abdos et fessiers, rétroversion du bassin pour monter les hanches"
            }

        # Erreur 2 : Coudes fléchis
        if (avg_left_elbow < model["elbow"]["min"] or avg_right_elbow < model["elbow"]["min"]) and not primary_issue:
            deviations["coudes_flechis"] = "Oui"
            primary_issue = {
                "cause": "Bras fléchis pendant la planche",
                "compensation": "Perte de protraction scapulaire",
                "correction": "Pousse activement dans le sol, verrouille les coudes, engage les scapulas"
            }

        # Erreur 3 : Épaules trop hautes ou trop basses
        if (avg_left_shoulder < model["shoulder"]["min"] or avg_left_shoulder > model["shoulder"]["max"]) and not primary_issue:
            deviations["position_epaules"] = "Oui"
            primary_issue = {
                "cause": "Position des épaules incorrecte",
                "compensation": "Instabilité et perte de protraction scapulaire",
                "correction": "Pousse dans le sol pour protracter les épaules vers l'avant"
            }

    # ========================================================================
    # FRONT LEVER - AVEC PRIORISATION STRICTE ET TOLÉRANCE BICEPS
    # ========================================================================
    elif figure == "front_lever":
        # Appliquer tolérance spécifique pour les biceps développés
        biceps_tolerance = model.get("tolerance_biceps", 0)
        logger.info(f"Tolérance biceps appliquée: {biceps_tolerance}°")
        
        # Détection de toutes les erreurs possibles avec tolérance
        hanches_basses = avg_left_hip < model["hip"]["min"] or avg_right_hip < model["hip"]["min"]
        coudes_flechis = (avg_left_elbow < model["elbow"]["min"] - biceps_tolerance or 
                          avg_right_elbow < model["elbow"]["min"] - biceps_tolerance)
        epaules_incorrectes = ((avg_left_shoulder < model["shoulder"]["min"] or avg_left_shoulder > model["shoulder"]["max"]) or
                              (avg_right_shoulder < model["shoulder"]["min"] or avg_right_shoulder > model["shoulder"]["max"]))
        
        logger.info(f"Détection erreurs: hanches_basses={hanches_basses}, coudes_flechis={coudes_flechis}, epaules={epaules_incorrectes}")
        logger.info(f"Angles coudes après tolérance: left={avg_left_elbow:.1f}° (seuil={model['elbow']['min'] - biceps_tolerance:.1f}°), right={avg_right_elbow:.1f}° (seuil={model['elbow']['min'] - biceps_tolerance:.1f}°)")
        
        # Marquer toutes les erreurs détectées dans deviations
        if hanches_basses:
            deviations["hanches_basses"] = "Oui"
            logger.info("Hanches basses détectées")
        if coudes_flechis:
            deviations["coudes_flechis"] = "Oui"
            logger.info("Coudes fléchis détectés")
        if epaules_incorrectes:
            deviations["position_epaules"] = "Oui"
            logger.info("Épaules incorrectes détectées")
        
        # PRIORISATION : Hanches > Coudes > Épaules
        if hanches_basses:
            logger.info("→ PRIORITÉ: Hanches basses (cause principale)")
            primary_issue = {
                "cause": "Hanches trop basses, le corps n'est pas aligné horizontalement",
                "compensation": "Les bras se plient pour compenser le manque de gainage et de force de traction",
                "correction": "Renforce ton gainage : contracte abdos/fessiers en rétroversion + tire plus fort avec les épaules pour monter les hanches"
            }
        
        elif coudes_flechis:
            logger.info("→ PRIORITÉ: Coudes fléchis")
            primary_issue = {
                "cause": "Bras fléchis pendant le front lever",
                "compensation": "Perte de rétraction scapulaire et tension constante",
                "correction": "Tu as la force d'effectuer le front lever, il te manque la technique. Pour cela, il faut POUSSER la barre vers tes pieds en contractant les épaules et en gardant les bras tendus"
            }
        
        elif epaules_incorrectes:
            logger.info("→ PRIORITÉ: Épaules incorrectes")
            primary_issue = {
                "cause": "Position des épaules incorrecte, pas assez de dépression scapulaire",
                "compensation": "Les bras compensent en se pliant",
                "correction": "Tire les épaules vers le bas et vers l'arrière (rétraction + dépression scapulaire)"
            }

    # ========================================================================
    # SI AUCUNE ERREUR DÉTECTÉE
    # ========================================================================
    if not primary_issue:
        primary_issue = {
            "cause": "Maintien correct de la figure",
            "compensation": "Aucune",
            "correction": "Excellente tenue ! Continue comme ça "
        }

    logger.info(f"Déviations finales: {deviations}")
    logger.info(f"Cause: {primary_issue['cause']}")
    
    return {**primary_issue, "deviations": deviations}

# ============================================================================
# ANALYSES SPÉCIFIQUES - MOUVEMENTS DYNAMIQUES
# ============================================================================
def analyze_push_up(cycles, model):
    """Analyse des pompes."""
    if not cycles:
        return {
            "cause": "Aucun cycle détecté",
            "compensation": "N/A",
            "correction": "N/A",
            "deviations": {}
        }

    avg_min = np.mean([c["min_value"] for c in cycles])
    avg_max = np.mean([c["max_value"] for c in cycles])
    avg_amplitude = np.mean([c["amplitude"] for c in cycles])
    avg_duration = np.mean([c["duration"] for c in cycles])

    deviations = {}
    primary_issue = None

    if avg_min > model["elbow"]["min"] + 10:
        deviations["amplitude_insuffisante"] = "Oui"
        primary_issue = {
            "cause": "Tu ne descends pas assez, tes coudes restent trop hauts",
            "compensation": "Le dos se cambre pour compenser le manque de flexion",
            "correction": "Descends jusqu'à 90° de flexion des coudes"
        }

    if avg_max < model["elbow"]["max"] - 10 and not primary_issue:
        deviations["verrouillage_incomplet"] = "Oui"
        primary_issue = {
            "cause": "Tu n'étends pas complètement les coudes en haut",
            "compensation": "Les épaules compensent le manque d'extension",
            "correction": "Verrouille complètement les coudes en position haute"
        }

    if avg_duration < 0.8 and not primary_issue:
        deviations["execution_rapide"] = "Oui"
        primary_issue = {
            "cause": "Exécution trop rapide, mouvement non contrôlé",
            "compensation": "Le corps réduit l'amplitude pour suivre le rythme",
            "correction": "Ralentir: 1-2 secondes par phase"
        }

    if not primary_issue:
        primary_issue = {
            "cause": "Exécution correcte",
            "compensation": "Aucune",
            "correction": "Continuer avec cette technique"
        }

    return {**primary_issue, "deviations": deviations}

def analyze_pull_up(cycles, model):
    """Analyse des tractions."""
    if not cycles:
        return {
            "cause": "Aucun cycle détecté",
            "compensation": "N/A",
            "correction": "N/A",
            "deviations": {}
        }

    avg_min = np.mean([c["min_value"] for c in cycles])
    avg_max = np.mean([c["max_value"] for c in cycles])
    avg_amplitude = np.mean([c["amplitude"] for c in cycles])

    deviations = {}
    primary_issue = None

    if avg_min > model["elbow"]["min"] + 15:
        deviations["amplitude_insuffisante"] = "Oui"
        primary_issue = {
            "cause": "Tu ne montes pas assez haut : le menton ne dépasse pas la barre",
            "compensation": "Le mouvement est raccourci pour rester fluide et enchaîner les répétitions",
            "correction": "Tire un peu plus haut à chaque répétition, quitte à faire moins de reps mais complètes"
        }

    if avg_max < model["elbow"]["max"] - 15 and not primary_issue:
        deviations["extension_incomplete"] = "Oui"
        primary_issue = {
            "cause": "Tu ne descends pas complètement bras tendus entre les répétitions",
            "compensation": "Tu gardes une tension constante pour éviter la partie la plus difficile du mouvement",
            "correction": "Marque une vraie position basse avec les bras tendus avant de repartir (scapula relâché)"
        }

    if not primary_issue:
        primary_issue = {
            "cause": "Amplitude et contrôle corrects sur l'ensemble des répétitions",
            "compensation": "Aucune",
            "correction": "Excellente technique, continue !"
        }

    return {**primary_issue, "deviations": deviations}

def analyze_dips(cycles, model):
    """Analyse des dips."""
    if not cycles:
        return {
            "cause": "Aucun cycle détecté",
            "compensation": "N/A",
            "correction": "N/A",
            "deviations": {}
        }

    avg_min = np.mean([c["min_value"] for c in cycles])
    avg_max = np.mean([c["max_value"] for c in cycles])
    avg_amplitude = np.mean([c["amplitude"] for c in cycles])

    deviations = {}
    primary_issue = None

    if avg_min > model["elbow"]["min"] + 10:
        deviations["profondeur_insuffisante"] = "Oui"
        primary_issue = {
            "cause": "Tu ne descends pas assez bas, les coudes restent trop ouverts",
            "compensation": "Le mouvement est volontairement raccourci pour rester confortable et stable",
            "correction": "Descends progressivement plus bas jusqu'à atteindre au moins 90° de flexion des coudes"
        }

    if avg_max < model["elbow"]["max"] - 10 and not primary_issue:
        deviations["verrouillage_incomplet"] = "Oui"
        primary_issue = {
            "cause": "Tu ne verrouilles pas complètement les bras en position haute",
            "compensation": "Tu évites la fin du mouvement pour garder du rythme ou réduire l'effort",
            "correction": "Marque une vraie position haute avec les bras tendus avant de redescendre"
        }

    if not primary_issue:
        primary_issue = {
            "cause": "Amplitude et contrôle corrects sur l'ensemble des répétitions",
            "compensation": "Aucune",
            "correction": "Technique solide, maintiens-la !"
        }

    return {**primary_issue, "deviations": deviations}

# ============================================================================
# ROUTES FLASK
# ============================================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "API d'analyse de mouvement complète",
        "version": "10.6 - Assombrissement EXTRÊME (cible 65 avec facteur 0.75)",
        "endpoints": {
            "static": "/analyze_static",
            "video_dynamic": "/analyze_video_dynamic",
            "video_static": "/analyze_video_static"
        },
        "supported_exercises": {
            "static": ["handstand", "planche", "front_lever"],
            "dynamic": ["push_up", "pull_up", "dips"]
        },
        "features": {
            "dual_image_processing": "Image prétraitée pour détection, image originale pour affichage",
            "extreme_darkening": "Assombrissement EXTRÊME - Cible 65 (au lieu de 100) avec facteur 0.75",
            "simple_strategy": "Si luminosité > 70 → réduction massive vers 65",
            "auto_rotation": "Rotation automatique portrait → paysage si meilleure détection",
            "max_contrast": "Contraste MAXIMISÉ (CLAHE 5.0 pour faible contraste, 3.5 sinon)",
            "boosted_saturation": "Saturation augmentée à 1.2",
            "noise_reduction": "Réduction du bruit - local uniquement",
            "detailed_logging": "Logs détaillés du décodage et traitement d'image",
            "robust_fallback": "Gestion d'erreurs avec fallback sur image originale",
            "biceps_tolerance": "Tolérance de 3° pour front lever (biceps développés)"
        },
        "brightness_strategy": {
            "target": "65/255 (TRÈS SOMBRE)",
            "action": "Si > 70 → multiplication par (65/luminosité) × 0.75",
            "example": "Image à 200 → (65/200) × 0.75 = 0.24 → luminosité finale ~48"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route("/analyze_static", methods=["POST"])
def analyze_static():
    logger.info("=== ANALYSE STATIQUE (IMAGE) ===")

    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Requête non JSON"}), 400

        data = request.get_json()
        image_base64 = data.get("image_base64")

        if not image_base64:
            logger.error("Aucune image dans la requête")
            return jsonify({"status": "error", "message": "Aucune image reçue"}), 400

        logger.info(f"Image base64 reçue - Longueur: {len(image_base64)} caractères")
        logger.info(f" Premiers caractères: {image_base64[:100]}...")

        # Décodage
        try:
            # Nettoyer le base64 (enlever les éventuels préfixes data:image)
            if ',' in image_base64:
                logger.info("🔧 Détection préfixe data:image, extraction du base64 pur...")
                image_base64 = image_base64.split(',')[1]
                logger.info(f"Base64 nettoyé - Longueur: {len(image_base64)}")
            
            # Ajouter du padding si nécessaire
            missing_padding = len(image_base64) % 4
            if missing_padding:
                logger.info(f"🔧 Ajout de {4 - missing_padding} caractères de padding")
                image_base64 += '=' * (4 - missing_padding)
            
            logger.info("🔧 Décodage base64...")
            image_bytes = base64.b64decode(image_base64)
            logger.info(f"Base64 décodé - {len(image_bytes)} bytes")
            
            logger.info("🔧 Conversion en array numpy...")
            nparr = np.frombuffer(image_bytes, np.uint8)
            logger.info(f"Array numpy créé - shape: {nparr.shape}, dtype: {nparr.dtype}")
            
            logger.info("🔧 Décodage de l'image avec OpenCV...")
            original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if original_image is None:
                logger.error("cv2.imdecode a retourné None")
                logger.error(f"   - Taille du buffer: {len(image_bytes)} bytes")
                logger.error(f"   - Premiers bytes: {image_bytes[:20]}")
                return jsonify({
                    "status": "error", 
                    "message": "Image invalide - Impossible de décoder l'image. Formats supportés: JPG, PNG",
                    "debug": {
                        "buffer_size": len(image_bytes),
                        "base64_length": len(image_base64)
                    }
                }), 400
            
            logger.info(f"Image décodée - shape: {original_image.shape}, dtype: {original_image.dtype}")
            
        except base64.binascii.Error as e:
            logger.error(f"Erreur décodage base64: {e}")
            return jsonify({
                "status": "error", 
                "message": f"Erreur décodage base64: {str(e)}",
                "debug": "Le format base64 est invalide"
            }), 400
        except Exception as e:
            logger.error(f"Erreur décodage: {e}", exc_info=True)
            return jsonify({
                "status": "error", 
                "message": f"Erreur décodage: {str(e)}",
                "debug": f"Type d'erreur: {type(e).__name__}"
            }), 400

        # ========================================================================
        # PRÉTRAITEMENT AUTOMATIQUE DE L'IMAGE (pour MediaPipe uniquement)
        # ========================================================================
        logger.info("📸 Prétraitement de l'image pour MediaPipe...")
        preprocessed_image = enhance_image_for_detection(original_image.copy())

        # Analyse Mediapipe sur l'image PRÉTRAITÉE
        logger.info("Lancement de MediaPipe Pose...")
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            preprocessed_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
            preprocessed_rgb.flags.writeable = False
            results = pose.process(preprocessed_rgb)
            preprocessed_rgb.flags.writeable = True

            if not results.pose_landmarks:
                logger.warning("Aucun corps détecté après prétraitement")
                return jsonify({
                    "status": "error",
                    "message": "Aucun corps détecté. Assure-toi d'être bien visible et que tout ton corps est dans le cadre.",
                    "detected_figure": "none"
                }), 400

            logger.info("✓ Corps détecté avec succès")

            user_angles = calculate_user_angles(results.pose_landmarks, preprocessed_rgb.shape)

            # Détection figure statique
            figure = detect_static_figure([results.pose_landmarks])
            logger.info(f"Figure détectée: {figure}")

            # Analyse unifiée
            if figure in static_skills:
                model = static_skills[figure]
                analysis = analyze_static_hold_unified(figure, user_angles, model)
            else:
                analysis = {
                    "cause": "Figure non reconnue",
                    "compensation": "N/A",
                    "correction": "N/A",
                    "deviations": {}
                }

            # Dessiner les landmarks avec erreurs en ROUGE sur l'IMAGE ORIGINALE
            error_landmarks = get_error_landmarks(figure, analysis.get("deviations", {}))
            logger.info(f"Landmarks en erreur: {error_landmarks}")
            
            # Utiliser l'image ORIGINALE pour l'annotation et l'affichage
            logger.info("Annotation de l'image originale (luminosité non modifiée)...")
            original_image_copy = original_image.copy()
            draw_landmarks_with_errors(original_image_copy, results.pose_landmarks, error_landmarks)

            # Encoder l'image ORIGINALE annotée
            logger.info("Encodage de l'image annotée en JPEG...")
            _, buffer = cv2.imencode('.jpg', original_image_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_base64_out = base64.b64encode(buffer).decode('utf-8')
            logger.info(f"Image encodée - {len(image_base64_out)} caractères")

            return jsonify({
                "status": "ok",
                "analysis_type": "static_image",
                "message": "Analyse statique terminée",
                "image_base64": image_base64_out,
                "detected_figure": figure,
                "user_angles": user_angles,
                "analysis": {
                    "cause": analysis["cause"],
                    "compensation": analysis["compensation"],
                    "correction": analysis["correction"]
                },
                "deviations": analysis.get("deviations", {}),
                "error_landmarks": error_landmarks
            })

    except Exception as e:
        logger.error(f"Erreur dans analyze_static: {e}", exc_info=True)
        return jsonify({
            "status": "error", 
            "message": f"Erreur serveur: {str(e)}",
            "error_type": type(e).__name__
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Route non trouvée"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur 500: {error}", exc_info=True)
    return jsonify({"status": "error", "message": "Erreur serveur"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)