from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os
import mediapipe as mp
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)
CORS(app)

# ============================================================================
# MODÈLES DE RÉFÉRENCE
# ============================================================================
STATIC_SKILLS = {
    "handstand": {
        "elbow":    {"min": 160},
        "shoulder": {"min": 160},
        "hip":      {"min": 158},
        "knee":     {"min": 165}
    },
    "planche": {
        "elbow":    {"min": 160},
        "shoulder": {"min": 25, "max": 65},
        "hip":      {"min": 148}
    },
    "front_lever": {
        "elbow":            {"min": 160},
        "shoulder":         {"min": 25, "max": 65},
        "hip":              {"min": 160},
        "tolerance_biceps": 3
    }
}

# ============================================================================
# PRÉTRAITEMENT IMAGE
# ============================================================================
def preprocess_image(image):
    gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    logger.info(f"Luminosité: {brightness:.1f}/255")
    return image

# ============================================================================
# CALCULS GÉOMÉTRIQUES
# ============================================================================
def calculate_angle(a, b, c):
    """Calcule l'angle ABC entre 3 points 2D."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle   = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def get_point(landmarks, name):
    lm = landmarks.landmark[mp_pose.PoseLandmark[name.upper()]]
    return [lm.x, lm.y]

def calculate_angles(landmarks):
    """Calcule tous les angles articulaires + métadonnées de fiabilité."""
    joints = {
        "left_elbow":     ["left_shoulder",  "left_elbow",    "left_wrist"],
        "right_elbow":    ["right_shoulder", "right_elbow",   "right_wrist"],
        "left_shoulder":  ["left_elbow",     "left_shoulder", "left_hip"],
        "right_shoulder": ["right_elbow",    "right_shoulder","right_hip"],
        "left_hip":       ["left_shoulder",  "left_hip",      "left_knee"],
        "right_hip":      ["right_shoulder", "right_hip",     "right_knee"],
        "left_knee":      ["left_hip",       "left_knee",     "left_ankle"],
        "right_knee":     ["right_hip",      "right_knee",    "right_ankle"]
    }

    angles = {}
    for joint, points in joints.items():
        try:
            angles[joint] = float(calculate_angle(
                get_point(landmarks, points[0]),
                get_point(landmarks, points[1]),
                get_point(landmarks, points[2])
            ))
        except:
            angles[joint] = 0.0

    # X des épaules → détecter vue de face vs profil
    try:
        angles["_ls_x"] = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
        angles["_rs_x"] = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
    except:
        angles["_ls_x"] = 0.5
        angles["_rs_x"] = 0.5

    # Visibilité des coudes ET des hanches (fiabilité du calcul 2D)
    try:
        angles["_le_vis"] = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].visibility
        angles["_re_vis"] = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility
        angles["_lh_vis"] = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].visibility
        angles["_rh_vis"] = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].visibility
    except:
        angles["_le_vis"] = 1.0
        angles["_re_vis"] = 1.0
        angles["_lh_vis"] = 1.0
        angles["_rh_vis"] = 1.0

    logger.info(
        f"Angles → L_elbow:{angles['left_elbow']:.1f}° R_elbow:{angles['right_elbow']:.1f}° "
        f"L_shoulder:{angles['left_shoulder']:.1f}° R_shoulder:{angles['right_shoulder']:.1f}° "
        f"L_hip:{angles['left_hip']:.1f}° R_hip:{angles['right_hip']:.1f}° "
        f"L_knee:{angles['left_knee']:.1f}° R_knee:{angles['right_knee']:.1f}°"
    )
    logger.info(
        f"Visibilité → coude_G:{angles['_le_vis']:.2f} coude_D:{angles['_re_vis']:.2f} "
        f"hanche_G:{angles['_lh_vis']:.2f} hanche_D:{angles['_rh_vis']:.2f}"
    )

    return angles

# ============================================================================
# DÉTECTION DE FIGURE
# ============================================================================
def detect_figure(landmarks):
    """Détecte la figure : handstand, planche ou front_lever."""
    try:
        lm = landmarks.landmark

        # Positions verticales (Y) — coords image : 0=haut, 1=bas
        nose_y     = lm[mp_pose.PoseLandmark.NOSE].y
        hip_y      = (lm[mp_pose.PoseLandmark.LEFT_HIP].y      + lm[mp_pose.PoseLandmark.RIGHT_HIP].y)      / 2
        wrist_y    = (lm[mp_pose.PoseLandmark.LEFT_WRIST].y     + lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)     / 2
        ankle_y    = (lm[mp_pose.PoseLandmark.LEFT_ANKLE].y     + lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)     / 2
        shoulder_y = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y  + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)  / 2

        logger.info(
            f"Y → nose:{nose_y:.3f} shoulder:{shoulder_y:.3f} "
            f"hip:{hip_y:.3f} ankle:{ankle_y:.3f} wrist:{wrist_y:.3f}"
        )

        # ── HANDSTAND ────────────────────────────────────────────────────────
        # Critère 1 : les chevilles sont NETTEMENT au-dessus des épaules
        #             marge > 0.05 → évite les faux positifs planche (tout au même Y)
        # Critère 2 : le nez est en dessous des hanches → personne inversée
        inversion_margin = shoulder_y - ankle_y   # positif = chevilles plus hautes
        is_inverted      = inversion_margin > 0.05
        nose_below_hip   = nose_y > hip_y

        logger.info(
            f"Handstand check → marge_inversion:{inversion_margin:.3f} "
            f"is_inverted:{is_inverted}  nose_below_hip:{nose_below_hip}"
        )

        if is_inverted and nose_below_hip:
            logger.info("→ HANDSTAND détecté")
            return "handstand"

        # ── FRONT LEVER vs PLANCHE ────────────────────────────────────────────
        y_diff = hip_y - nose_y
        logger.info(f"Diff verticale (hip-nose): {y_diff:.3f}")

        if y_diff > 0.10:
            logger.info("→ FRONT LEVER (nez plus haut)")
            return "front_lever"

        if abs(y_diff) < 0.25:
            wrist_z    = (lm[mp_pose.PoseLandmark.LEFT_WRIST].z    + lm[mp_pose.PoseLandmark.RIGHT_WRIST].z)    / 2
            shoulder_z = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z) / 2
            nose_z     = lm[mp_pose.PoseLandmark.NOSE].z
            hip_z      = (lm[mp_pose.PoseLandmark.LEFT_HIP].z      + lm[mp_pose.PoseLandmark.RIGHT_HIP].z)      / 2

            front_lever_score = sum([
                wrist_y < nose_y + 0.05,
                wrist_z > shoulder_z + 0.02,
                nose_z  > hip_z + 0.05
            ])

            logger.info(f"Score Front Lever: {front_lever_score}/3")

            if front_lever_score >= 2:
                logger.info("→ FRONT LEVER")
                return "front_lever"
            else:
                logger.info("→ PLANCHE")
                return "planche"

        logger.info("→ Figure inconnue")
        return "unknown"

    except Exception as e:
        logger.error(f"Erreur détection: {e}")
        return "unknown"

# ============================================================================
# ANALYSE BIOMÉCANIQUE
# ============================================================================
def analyze_figure(figure, angles, model):
    """Analyse biomécanique avec priorisation des erreurs."""

    le = angles.get("left_elbow",     180)
    re = angles.get("right_elbow",    180)
    lh = angles.get("left_hip",       180)
    rh = angles.get("right_hip",      180)
    ls = angles.get("left_shoulder",  180)
    rs = angles.get("right_shoulder", 180)
    lk = angles.get("left_knee",      180)
    rk = angles.get("right_knee",     180)

    # Fiabilité par articulation (visibilité MediaPipe)
    # Coudes : 0.35 → accepte profil normal, rejette vue de face franche (calcul 2D faux)
    # Hanches : 0.6  → plus strict, généralement bien visibles
    coudes_fiables  = angles.get("_le_vis", 1.0) > 0.35 and angles.get("_re_vis", 1.0) > 0.35
    hanches_fiables = angles.get("_lh_vis", 1.0) > 0.6  and angles.get("_rh_vis", 1.0) > 0.6

    deviations = {}
    issue      = None

    # ========================================================================
    # HANDSTAND
    # ========================================================================
    if figure == "handstand":
        knee_min = model.get("knee", {}).get("min", 160)

        # Erreur 1 — Hanches fléchies (seulement si bien visibles)
        if hanches_fiables and (lh < model["hip"]["min"] or rh < model["hip"]["min"]):
            deviations["hanches_flechies"] = "Oui"
            issue = {
                "cause":        "Hanches fléchies, alignement perdu",
                "compensation": "Cambrure lombaire excessive et coudes fléchis pour compenser",
                "correction":   "Contracte abdos et fessiers pour aligner le corps verticalement"
            }

        # Erreur 2 — Coudes fléchis (seulement si bien visibles)
        elif coudes_fiables and (le < model["elbow"]["min"] or re < model["elbow"]["min"]):
            deviations["coudes_flechis"] = "Oui"
            issue = {
                "cause":        "Coudes fléchis pendant le maintien",
                "compensation": "Le dos se cambre pour maintenir l'équilibre",
                "correction":   "Verrouille complètement les coudes et engage les triceps"
            }

        # Erreur 3 — Genoux fléchis
        elif lk < knee_min or rk < knee_min:
            deviations["genoux_flechis"] = "Oui"
            issue = {
                "cause":        "Genoux fléchis, jambes non tendues",
                "compensation": "Perte d'alignement et instabilité",
                "correction":   "Verrouille complètement les genoux, engage les quadriceps"
            }

        # Erreur 4 — Épaules insuffisamment ouvertes
        elif ls < model["shoulder"]["min"] or rs < model["shoulder"]["min"]:
            deviations["position_epaules"] = "Oui"
            issue = {
                "cause":        "Épaules insuffisamment ouvertes, manque d'élévation scapulaire",
                "compensation": "Perte d'équilibre et instabilité en haut du handstand",
                "correction":   "Pousse activement dans le sol, ouvre les épaules au maximum et élève les scapulas pour verrouiller la position"
            }

    # ========================================================================
    # PLANCHE
    # ========================================================================
    elif figure == "planche":
        ls_x        = angles.get("_ls_x", 0.5)
        rs_x        = angles.get("_rs_x", 0.5)
        vue_de_face = abs(ls_x - rs_x) < 0.15

        logger.info(
            f"Planche → vue_de_face:{vue_de_face}  "
            f"coudes_fiables:{coudes_fiables}  hanches_fiables:{hanches_fiables}"
        )

        # Erreur 1 — Hanches basses (ignorée si vue de face ou hanches peu visibles)
        if not vue_de_face and hanches_fiables and (lh < model["hip"]["min"] or rh < model["hip"]["min"]):
            deviations["hanches_basses"] = "Oui"
            issue = {
                "cause":        "Hanches trop basses, gainage insuffisant",
                "compensation": "Les bras compensent en se pliant pour soutenir le poids",
                "correction":   "Renforce le gainage : serre abdos et fessiers, rétroversion du bassin"
            }

        # Erreur 2 — Coudes fléchis (ignorée si coudes peu visibles)
        elif coudes_fiables and (le < model["elbow"]["min"] or re < model["elbow"]["min"]):
            deviations["coudes_flechis"] = "Oui"
            issue = {
                "cause":        "Bras fléchis pendant la planche",
                "compensation": "Perte de protraction scapulaire",
                "correction":   "Pousse activement dans le sol, verrouille les coudes"
            }

        # Erreur 3 — Position des épaules
        elif ls < model["shoulder"]["min"] or ls > model["shoulder"]["max"]:
            deviations["position_epaules"] = "Oui"
            issue = {
                "cause":        "Position des épaules incorrecte",
                "compensation": "Instabilité et perte de protraction scapulaire",
                "correction":   "Pousse dans le sol pour protracter les épaules vers l'avant ET vers le bas (protraction + dépression scapulaire)"
            }

    # ========================================================================
    # FRONT LEVER
    # ========================================================================
    elif figure == "front_lever":
        tolerance = model.get("tolerance_biceps", 0)

        hanches_basses      = hanches_fiables and (lh < model["hip"]["min"] or rh < model["hip"]["min"])
        coudes_flechis      = coudes_fiables and (
                                  le < model["elbow"]["min"] - tolerance or
                                  re < model["elbow"]["min"] - tolerance
                              )
        epaules_incorrectes = (ls < model["shoulder"]["min"] or ls > model["shoulder"]["max"] or
                               rs < model["shoulder"]["min"] or rs > model["shoulder"]["max"])

        if hanches_basses:
            deviations["hanches_basses"] = "Oui"
        if coudes_flechis:
            deviations["coudes_flechis"] = "Oui"
        if epaules_incorrectes:
            deviations["position_epaules"] = "Oui"

        # Priorisation : Hanches > Coudes > Épaules
        if hanches_basses:
            issue = {
                "cause":        "Hanches trop basses, le corps n'est pas aligné horizontalement",
                "compensation": "Les bras se plient pour compenser le manque de gainage et de force de traction",
                "correction":   "Renforce ton gainage : contracte abdos/fessiers en rétroversion + tire plus fort avec les épaules pour monter les hanches"
            }
        elif coudes_flechis:
            issue = {
                "cause":        "Bras fléchis pendant le front lever",
                "compensation": "Perte de rétraction scapulaire et tension constante",
                "correction":   "Tu as la force d'effectuer le front lever, il te manque la technique. Il faut POUSSER la barre vers tes pieds en contractant les épaules et en gardant les bras tendus"
            }
        elif epaules_incorrectes:
            issue = {
                "cause":        "Position des épaules incorrecte, pas assez de dépression scapulaire",
                "compensation": "Les bras compensent en se pliant",
                "correction":   "Tire les épaules vers le bas et vers l'arrière (rétraction + dépression scapulaire)"
            }

    # Aucune erreur détectée
    if not issue:
        issue = {
            "cause":        "Maintien correct de la figure",
            "compensation": "Aucune",
            "correction":   "Excellente tenue ! Continue comme ça 💪"
        }

    logger.info(f"Analyse {figure}: {issue['cause']}")
    return {**issue, "deviations": deviations}

# ============================================================================
# ANNOTATION VISUELLE
# ============================================================================
def get_error_landmarks(figure, deviations):
    """Retourne la liste des landmarks à colorier en rouge."""
    errors = []

    if deviations.get("hanches_flechies") or deviations.get("hanches_basses"):
        errors.extend(['LEFT_HIP', 'RIGHT_HIP'])

    if deviations.get("coudes_flechis"):
        errors.extend(['LEFT_ELBOW', 'RIGHT_ELBOW'])

    if deviations.get("genoux_flechis"):
        errors.extend(['LEFT_KNEE', 'RIGHT_KNEE'])

    if deviations.get("position_epaules"):
        errors.extend(['LEFT_SHOULDER', 'RIGHT_SHOULDER'])

    return list(set(errors))

def draw_landmarks(image, landmarks, error_landmarks):
    """Dessine les landmarks avec les erreurs en rouge."""
    h, w = image.shape[:2]

    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=0),
        mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=0)
    )

    for idx, lm in enumerate(landmarks.landmark):
        name = mp_pose.PoseLandmark(idx).name
        cx   = int(lm.x * w)
        cy   = int(lm.y * h)

        if name in error_landmarks:
            color, radius = (0, 0, 255), 8
        else:
            color, radius = (255, 255, 255), 4

        cv2.circle(image, (cx, cy), radius, color, -1)
        cv2.circle(image, (cx, cy), radius, (0, 0, 0), 1)

# ============================================================================
# ROUTES
# ============================================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":      "ok",
        "version":     "15.0 - Seuil inversion + fiabilité hanches/coudes",
        "description": "API d'analyse biomécanique Calisthenics",
        "features": {
            "handstand_detection": "marge inversion > 0.05 (anti faux positif planche)",
            "fiabilite":           "coudes ET hanches ignorés si visibilité MediaPipe < 0.6"
        }
    })

@app.route("/analyze_static", methods=["POST"])
def analyze_static():
    """
    Analyse d'une image statique.
    INPUT  : FormData avec fichier image (clé: 'image')
    OUTPUT : JSON avec analyse + image annotée en base64
    """
    logger.info("=== NOUVELLE ANALYSE ===")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"Files: {list(request.files.keys())}")

    try:
        # 1. Réception
        if 'image' not in request.files:
            return jsonify({
                "status":  "error",
                "message": "Aucune image dans la requête. Clé attendue: 'image'"
            }), 400

        file = request.files['image']
        logger.info(f"Fichier reçu: {file.filename}")

        # 2. Décodage
        file_bytes     = file.read()
        nparr          = np.frombuffer(file_bytes, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if original_image is None:
            return jsonify({
                "status":  "error",
                "message": "Image invalide. Formats supportés: JPG, PNG"
            }), 400

        logger.info(f"Image décodée: {original_image.shape}")

        # 3. Prétraitement
        preprocessed = preprocess_image(original_image.copy())

        # 4. Détection MediaPipe
        with mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.3,
            model_complexity=1
        ) as pose:

            rgb     = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                logger.warning("Aucun corps détecté")
                return jsonify({
                    "status":          "error",
                    "message":         "Aucun corps détecté. Assure-toi d'être bien visible et que tout ton corps est dans le cadre.",
                    "detected_figure": "none"
                }), 400

            logger.info("Corps détecté")

            # 5. Calculs
            angles = calculate_angles(results.pose_landmarks)
            figure = detect_figure(results.pose_landmarks)

            # 6. Analyse biomécanique
            if figure in STATIC_SKILLS:
                analysis = analyze_figure(figure, angles, STATIC_SKILLS[figure])
            else:
                analysis = {
                    "cause":        "Figure non reconnue",
                    "compensation": "N/A",
                    "correction":   "Essaie handstand, planche ou front lever",
                    "deviations":   {}
                }

            # 7. Annotation visuelle
            annotated_image = original_image.copy()
            error_landmarks = get_error_landmarks(figure, analysis.get("deviations", {}))
            draw_landmarks(annotated_image, results.pose_landmarks, error_landmarks)

            # 8. Encodage base64
            _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            logger.info(f"Analyse terminée: {figure}")

            # 9. Réponse
            return jsonify({
                "status":          "ok",
                "image_base64":    image_b64,
                "detected_figure": figure,
                "user_angles":     angles,
                "analysis": {
                    "cause":        analysis["cause"],
                    "compensation": analysis["compensation"],
                    "correction":   analysis["correction"]
                },
                "deviations": analysis.get("deviations", {})
            })

    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        return jsonify({
            "status":  "error",
            "message": f"Erreur serveur: {str(e)}"
        }), 500

# ============================================================================
# GESTION D'ERREURS
# ============================================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Route non trouvée"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur 500: {error}", exc_info=True)
    return jsonify({"status": "error", "message": "Erreur serveur interne"}), 500

# ============================================================================
# DÉMARRAGE
# ============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Démarrage serveur sur port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)