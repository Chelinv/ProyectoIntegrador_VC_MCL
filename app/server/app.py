import os
import base64
import numpy as np
import cv2
import pandas as pd
import mahotas
import mediapipe as mp
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectPercentile, f_classif
import tensorflow as tf

app = Flask(__name__, static_folder="../web")
CORS(app)

# ==========================================
# 1. Configuración de Rutas y Variables
# ==========================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
clases_oficiales = ['Afro-ecuadorians', 'European descendants', 'Indigenous', 'Mestizos']

models = {'svm': {}, 'rf': {}, 'kmeans': {}}
scalers = {'svm': {}, 'kmeans': {}}
selectors = {'svm': {}, 'rf': {}, 'kmeans': {}}
cnn_model = None

hog_descriptor = None
brisk_descriptor = None
MAX_KEYPOINTS = 30

import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Descargar modelo FaceLandmarker si no existe
task_path = os.path.join(project_root, 'app', 'server', 'face_landmarker.task')
if not os.path.exists(task_path):
    print("Descargando modelo FaceLandmarker de MediaPipe...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        task_path
    )

base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Grupos de landmarks para cada facción
EYES_IDX = [33, 133, 159, 145, 153, 154, 155, 133, 362, 263, 386, 374, 380, 381, 382, 384, 385, 387, 388, 390, 398]
EYEBROWS_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
NOSE_IDX = [1, 2, 98, 327, 279, 49, 114, 128, 29, 290, 277, 437]
MOUTH_IDX = [0, 13, 14, 17, 61, 291, 37, 267, 314, 17, 84, 181]

FACCIONES_MP = {
    "eye": EYES_IDX,
    "eyebrow": EYEBROWS_IDX,
    "nose": NOSE_IDX,
    "mouth": MOUTH_IDX
}

# ==========================================
# 2. Inicialización de Modelos
# ==========================================
def init_models():
    global cnn_model, hog_descriptor, brisk_descriptor
    print("Iniciando carga de modelos...")
    
    import joblib
    model_dir = os.path.join(project_root, "outputs", "models")
    
    for ext in ['zernike', 'hog', 'brisk']:
        print(f"Cargando modelos para extractor: {ext.upper()}...")
        try:
            models['svm'][ext] = joblib.load(os.path.join(model_dir, f"svm_model_{ext}.pkl"))
            scalers['svm'][ext] = joblib.load(os.path.join(model_dir, f"svm_scaler_{ext}.pkl"))
            selectors['svm'][ext] = joblib.load(os.path.join(model_dir, f"svm_selector_{ext}.pkl"))
            
            models['rf'][ext] = joblib.load(os.path.join(model_dir, f"rf_model_{ext}.pkl"))
            selectors['rf'][ext] = joblib.load(os.path.join(model_dir, f"rf_selector_{ext}.pkl"))
            
            models['kmeans'][ext] = joblib.load(os.path.join(model_dir, f"kmeans_model_{ext}.pkl"))
            scalers['kmeans'][ext] = joblib.load(os.path.join(model_dir, f"kmeans_scaler_{ext}.pkl"))
            selectors['kmeans'][ext] = joblib.load(os.path.join(model_dir, f"kmeans_selector_{ext}.pkl"))
        except Exception as e:
            print(f"  -> Advertencia: No se pudieron cargar los modelos para {ext}: {e}")

    ruta_cnn = os.path.join(project_root, "outputs", "models", "modelo_cnn.keras")
    if os.path.exists(ruta_cnn):
        cnn_model = tf.keras.models.load_model(ruta_cnn)
        print("CNN cargada exitosamente.")

    # Inicializar Extractores
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    try:
        hog_descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    except AttributeError:
        pass

    try:
        brisk_descriptor = cv2.ORB_create()
    except AttributeError:
        pass

# ==========================================
# 3. Pipeline de Preprocesamiento (MediaPipe)
# ==========================================
def extract_biometric_mosaic(img_color):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        raise ValueError("No face detected by MediaPipe")
        
    landmarks = detection_result.face_landmarks[0]
    h, w, _ = img_color.shape
    gray_full = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    crops = {}
    
    for faccion_name, indices in FACCIONES_MP.items():
        xs = [int(landmarks[i].x * w) for i in indices]
        ys = [int(landmarks[i].y * h) for i in indices]
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        pad_x = int((xmax - xmin) * 0.05)
        pad_y = int((ymax - ymin) * 0.05)
        
        x1 = max(0, xmin - pad_x)
        y1 = max(0, ymin - pad_y)
        x2 = min(w, xmax + pad_x)
        y2 = min(h, ymax + pad_y)
        
        recorte = gray_full[y1:y2, x1:x2]
        if recorte.size == 0:
            raise ValueError(f"Invalid crop size for {faccion_name}")
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(recorte)
        img_blur = cv2.bilateralFilter(img_clahe, 9, 75, 75)
        img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        crops[faccion_name] = cv2.resize(img_thresh, (64, 64))
        
    # Mosaico 2x2: Arriba(eye, eyebrow), Abajo(nose, mouth)
    top_row = np.hstack((crops["eye"], crops["eyebrow"]))
    bottom_row = np.hstack((crops["nose"], crops["mouth"]))
    mosaic = np.vstack((top_row, bottom_row))
    
    return mosaic, crops, gray_full

init_models()

# ==========================================
# 4. Endpoints de la API Flask
# ==========================================
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/preprocess_demo', methods=['POST'])
def preprocess_demo():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        base64_img = data['image'].split(',')[1]
        image_bytes = base64.b64decode(base64_img)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Usar parámetros matemáticos estrictos (Fase 2)
        gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(gray)
        
        img_bilateral = cv2.bilateralFilter(img_clahe, 9, 75, 75)
        
        img_thresh = cv2.adaptiveThreshold(img_bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        def mat_to_base64(mat):
            _, buffer = cv2.imencode('.jpg', mat)
            return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
            
        return jsonify({
            'gray': mat_to_base64(gray),
            'clahe': mat_to_base64(img_clahe), 
            'bilateral': mat_to_base64(img_bilateral),
            'threshold': mat_to_base64(img_thresh)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract_demo', methods=['POST'])
def extract_demo():
    try:
        from skimage.feature import hog as skimage_hog
        from skimage import exposure
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        base64_img = data['image'].split(',')[1]
        image_bytes = base64.b64decode(base64_img)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # 1. Preprocesamiento MediaPipe
        mosaic, crops, _ = extract_biometric_mosaic(img_original)
        
        # Redimensionamos a 256x256 para visualización en la UI
        img_resized = cv2.resize(mosaic, (256, 256))
        img_color_base = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        base_vis = img_color_base.copy()

        # Zernike Visualization (4 Discos)
        img_zernike = img_color_base.copy()
        centros = [(64, 64), (192, 64), (64, 192), (192, 192)]
        for cx, cy in centros:
            cv2.circle(img_zernike, (cx, cy), 60, (0, 255, 255), 2)
            cv2.circle(img_zernike, (cx, cy), 4, (0, 0, 255), -1)

        # HOG en Mosaico
        _, hog_image = skimage_hog(img_resized, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True, channel_axis=None)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_vis = (hog_image_rescaled * 255).astype(np.uint8)
        img_hog = cv2.applyColorMap(hog_vis, cv2.COLORMAP_JET)

        # BRISK en Mosaico (Visualización con ORB para evitar crash)
        orb = cv2.ORB_create(nfeatures=500)
        kp, _ = orb.detectAndCompute(img_resized, None)
        img_brisk = cv2.drawKeypoints(img_resized, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # --- FULL IMAGE ---
        img_full_resized = cv2.resize(img_original, (256, 256))
        
        # Zernike on Full
        full_zernike = img_full_resized.copy()
        cv2.circle(full_zernike, (128, 128), 120, (0, 255, 255), 2)
        cv2.circle(full_zernike, (128, 128), 4, (0, 0, 255), -1)

        # HOG on Full
        gray_full_resized = cv2.cvtColor(img_full_resized, cv2.COLOR_BGR2GRAY)
        _, hog_full_img = skimage_hog(gray_full_resized, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True, channel_axis=None)
        hog_full_rescaled = exposure.rescale_intensity(hog_full_img, in_range=(0, 10))
        hog_full_vis = (hog_full_rescaled * 255).astype(np.uint8)
        full_hog = cv2.applyColorMap(hog_full_vis, cv2.COLORMAP_JET)

        # BRISK on Full
        kp_full, _ = orb.detectAndCompute(img_full_resized, None)
        full_brisk = cv2.drawKeypoints(img_full_resized, kp_full, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        def mat_to_base64(mat):
            _, buffer = cv2.imencode('.jpg', mat)
            return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'full_base': mat_to_base64(img_full_resized),
            'full_zernike': mat_to_base64(full_zernike),
            'full_hog': mat_to_base64(full_hog),
            'full_brisk': mat_to_base64(full_brisk),
            'base': mat_to_base64(base_vis),
            'zernike': mat_to_base64(img_zernike),
            'hog': mat_to_base64(img_hog),
            'brisk': mat_to_base64(img_brisk)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        req_model = data.get('model', 'cnn')
        req_extractor = data.get('extractor', 'hog')
        
        base64_img = data['image'].split(',')[1]
        image_bytes = base64.b64decode(base64_img)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        mosaic, crops, _ = extract_biometric_mosaic(img_original)
        
        features = None
        
        # 1. Extracción de características iterando sobre las 4 facciones
        if req_model != 'cnn':
            z_acumulado = []
            h_acumulado = []
            b_acumulado = []
            
            # EL ORDEN DEBE SER IDENTICO A EXTRACCION_CARACTERISTICAS.PY
            FACCIONES_ORDEN = ["eye", "eyebrow", "nose", "mouth"]
            
            for faccion in FACCIONES_ORDEN:
                crop_64 = crops[faccion]
                
                if req_extractor == 'zernike':
                    z_feats = mahotas.features.zernike_moments(crop_64, radius=32)
                    z_acumulado.extend(z_feats)
                    
                elif req_extractor == 'hog':
                    h_feats = hog_descriptor.compute(crop_64).flatten()
                    h_acumulado.extend(h_feats)
                    
                elif req_extractor == 'brisk':
                    keypoints, descriptores = brisk_descriptor.detectAndCompute(crop_64, None)
                    v_brisk = np.zeros(MAX_KEYPOINTS * 64, dtype=np.uint8)
                    if descriptores is not None:
                        desc_aplanados = descriptores.flatten()
                        longitud_real = len(desc_aplanados)
                        if longitud_real > len(v_brisk):
                            v_brisk = desc_aplanados[:len(v_brisk)]
                        else:
                            v_brisk[:longitud_real] = desc_aplanados
                    b_acumulado.extend(v_brisk)

            if req_extractor == 'zernike':
                features = np.array(z_acumulado).reshape(1, -1)
            elif req_extractor == 'hog':
                features = np.array(h_acumulado).reshape(1, -1)
            elif req_extractor == 'brisk':
                features = np.array(b_acumulado).reshape(1, -1)

            features = features.astype(np.float64)

        # 2. Predicción
        if req_model == 'cnn':
            if cnn_model:
                # CNN ahora recibe el mosaico 128x128 normalizado a [0, 1]
                X_cnn = (mosaic / 255.0).reshape(1, 128, 128, 1)
                pred_cnn_raw = cnn_model.predict(X_cnn, verbose=0)[0]
                class_idx = np.argmax(pred_cnn_raw)
                prob_cnn = float(np.max(pred_cnn_raw))
                distribucion = {clases_oficiales[i]: float(pred_cnn_raw[i]) for i in range(len(clases_oficiales))}
                return jsonify({"clase": clases_oficiales[class_idx], "prob": prob_cnn, "distribucion": distribucion})
            else:
                return jsonify({'error': 'CNN model not loaded'}), 500
                
        elif req_model == 'svm':
            if 'svm' in models and req_extractor in models['svm']:
                m = models['svm'][req_extractor]
                s = scalers['svm'][req_extractor]
                sel = selectors['svm'].get(req_extractor)
                feat_sel = sel.transform(features) if sel else features
                feat_scaled = s.transform(feat_sel)
                
                probs_raw = m.predict_proba(feat_scaled)[0]
                max_idx = np.argmax(probs_raw)
                pred = m.classes_[max_idx]
                prob = probs_raw[max_idx]
                
                distribucion = {m.classes_[i]: float(probs_raw[i]) for i in range(len(m.classes_))}
                return jsonify({"clase": pred, "prob": float(prob), "distribucion": distribucion})
            else:
                return jsonify({'error': 'SVM model not loaded for this extractor'}), 500
                
        elif req_model == 'rf':
            if 'rf' in models and req_extractor in models['rf']:
                m = models['rf'][req_extractor]
                sel = selectors['rf'].get(req_extractor)
                feat_sel = sel.transform(features) if sel else features
                
                probs_raw = m.predict_proba(feat_sel)[0]
                max_idx = np.argmax(probs_raw)
                pred = m.classes_[max_idx]
                prob = probs_raw[max_idx]
                
                distribucion = {m.classes_[i]: float(probs_raw[i]) for i in range(len(m.classes_))}
                return jsonify({"clase": pred, "prob": float(prob), "distribucion": distribucion})
            else:
                return jsonify({'error': 'RF model not loaded for this extractor'}), 500
                
        elif req_model == 'kmeans':
            if 'kmeans' in models and req_extractor in models['kmeans']:
                m = models['kmeans'][req_extractor]
                s = scalers['kmeans'][req_extractor]
                sel = selectors['kmeans'].get(req_extractor)
                feat_sel = sel.transform(features) if sel else features
                feat_scaled = s.transform(feat_sel)
                cluster_idx = m.predict(feat_scaled)[0]
                return jsonify({"clase": f"Grupo {cluster_idx}", "prob": 1.0})
            else:
                return jsonify({'error': 'K-Means model not loaded for this extractor'}), 500
                
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Iniciando Servidor Flask en http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
