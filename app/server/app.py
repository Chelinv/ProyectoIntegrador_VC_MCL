import os
import base64
import numpy as np
import cv2
import pandas as pd
import mahotas
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

# Modelos en memoria
models = {
    'svm': {},
    'rf': {},
    'kmeans': {}
}
scalers = {
    'svm': {},
    'kmeans': {}
}
selectors = {
    'svm': {},
    'rf': {},
    'kmeans': {}
}
cnn_model = None

# Extractores
hog_descriptor = None
brisk_descriptor = None
MAX_KEYPOINTS = 30

# ==========================================
# 2. Inicialización de Modelos
# ==========================================
def init_models():
    global cnn_model, hog_descriptor, brisk_descriptor
    print("Iniciando carga de modelos...")
    
    import joblib
    model_dir = os.path.join(project_root, "outputs", "models")
    
    extractors = ['zernike', 'hog', 'brisk']
    
    for ext in extractors:
        print(f"Cargando modelos para extractor: {ext.upper()}...")
        # SVM
        try:
            models['svm'][ext] = joblib.load(os.path.join(model_dir, f"svm_model_{ext}.pkl"))
            selectors['svm'][ext] = joblib.load(os.path.join(model_dir, f"svm_selector_{ext}.pkl"))
            scalers['svm'][ext] = joblib.load(os.path.join(model_dir, f"svm_scaler_{ext}.pkl"))
        except Exception as e:
            print(f"  -> Advertencia: No se pudo cargar SVM para {ext}")
            
        # RF
        try:
            models['rf'][ext] = joblib.load(os.path.join(model_dir, f"rf_model_{ext}.pkl"))
            selectors['rf'][ext] = joblib.load(os.path.join(model_dir, f"rf_selector_{ext}.pkl"))
        except Exception as e:
            print(f"  -> Advertencia: No se pudo cargar RF para {ext}")
            
        # KMeans
        try:
            models['kmeans'][ext] = joblib.load(os.path.join(model_dir, f"kmeans_model_{ext}.pkl"))
            selectors['kmeans'][ext] = joblib.load(os.path.join(model_dir, f"kmeans_selector_{ext}.pkl"))
            scalers['kmeans'][ext] = joblib.load(os.path.join(model_dir, f"kmeans_scaler_{ext}.pkl"))
        except Exception as e:
            print(f"  -> Advertencia: No se pudo cargar KMeans para {ext}")

    # Cargar CNN
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
    hog_descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    brisk_descriptor = cv2.BRISK_create()

    brisk_descriptor = cv2.BRISK_create()

# Detector de Rostros basado en Segmentación de Color (HSV) y Contornos
def extract_face_roi(img_color):
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Filtros Morfológicos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No face detected")
        
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 1500:
        raise ValueError("No face detected")
        
    x, y, w, h = cv2.boundingRect(c)
    gray_full = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return gray_full[y:y+h, x:x+w], gray_full

init_models()

# ==========================================
# 3. Pipeline de Preprocesamiento (Vision.py)
# ==========================================
def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    face_roi, _ = extract_face_roi(img)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(face_roi)
    blur = cv2.bilateralFilter(clahe_img, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img_resized = cv2.resize(thresh, (64, 64))
    return img_resized

# ==========================================
# 4. Endpoints de la API Flask
# ==========================================
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/8266730/<path:filename>')
def serve_data(filename):
    return send_from_directory(os.path.join(project_root, 'data', '01_raw'), filename)

@app.route('/Dataset_Preprocesado/<path:filename>')
def serve_preprocesado(filename):
    return send_from_directory(os.path.join(project_root, 'data', '02_processed'), filename)

@app.route('/Resultados_Clustering/<path:filename>')
def serve_clustering(filename):
    return send_from_directory(os.path.join(project_root, 'outputs', 'plots'), filename)

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
        
        # A. Detección de Rostro por HSV
        face_roi, gray = extract_face_roi(img_original)
        
        # B. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(face_roi)
        
        # C. Filtro Bilateral
        img_blur = cv2.bilateralFilter(img_clahe, 9, 75, 75)
        
        # D. Umbralización Adaptativa
        img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Convert to Base64
        def mat_to_base64(mat):
            _, buffer = cv2.imencode('.jpg', mat)
            return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
            
        return jsonify({
            'gray': mat_to_base64(gray),
            'clahe': mat_to_base64(img_clahe),
            'bilateral': mat_to_base64(img_blur),
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
        
        # 1. Preprocesamiento Completo con Detección de Rostro HSV
        face_roi, gray = extract_face_roi(img_original)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(face_roi)
        img_blur = cv2.bilateralFilter(img_clahe, 9, 75, 75)
        img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Redimensionamos a 256x256 para visualización
        img_resized = cv2.resize(img_thresh, (256, 256))
        img_color_base = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        # 2. Base Binarizada
        base_vis = img_color_base.copy()

        # 2.5 Zernike Visualization (Unit Disk)
        img_zernike = img_color_base.copy()
        center = (128, 128)
        radius = 120
        cv2.circle(img_zernike, center, radius, (0, 255, 255), 2) # Yellow circle
        cv2.circle(img_zernike, center, 4, (0, 0, 255), -1) # Red center dot
        cv2.putText(img_zernike, "Zernike Disk", (80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 3. HOG
        _, hog_image = skimage_hog(img_resized, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True, channel_axis=None)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_vis = (hog_image_rescaled * 255).astype(np.uint8)
        img_hog = cv2.applyColorMap(hog_vis, cv2.COLORMAP_JET)

        # 5. BRISK
        brisk = cv2.BRISK_create()
        kp, _ = brisk.detectAndCompute(img_resized, None)
        img_brisk = cv2.drawKeypoints(img_resized, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        def mat_to_base64(mat):
            _, buffer = cv2.imencode('.jpg', mat)
            return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

        return jsonify({
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
        img_preprocesada = preprocess_image(image_bytes)
        
        features = None
        
        # 1. Extracción de características
        if req_model != 'cnn':
            if req_extractor == 'zernike':
                # El preprocesamiento base la deja en 64x64
                zernike_features = mahotas.features.zernike_moments(img_preprocesada, radius=32)
                features = np.array(zernike_features).reshape(1, -1)
                
            elif req_extractor == 'hog':
                caracteristicas_hog = hog_descriptor.compute(img_preprocesada).flatten()
                features = caracteristicas_hog.reshape(1, -1)
                
            elif req_extractor == 'brisk':
                keypoints, descriptores = brisk_descriptor.detectAndCompute(img_preprocesada, None)
                vector_brisk = np.zeros(MAX_KEYPOINTS * 64, dtype=np.uint8)
                if descriptores is not None:
                    desc_aplanados = descriptores.flatten()
                    longitud_real = len(desc_aplanados)
                    if longitud_real > len(vector_brisk):
                        vector_brisk = desc_aplanados[:len(vector_brisk)]
                    else:
                        vector_brisk[:longitud_real] = desc_aplanados
                features = vector_brisk.reshape(1, -1)

            features = features.astype(np.float64)

        # 2. Predicción
        if req_model == 'cnn':
            if cnn_model:
                X_cnn = img_preprocesada.reshape(1, 64, 64, 1)
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
                sel = selectors['svm'][req_extractor]
                s = scalers['svm'][req_extractor]
                feat_sel = sel.transform(features)
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
                sel = selectors['rf'][req_extractor]
                feat_sel = sel.transform(features)
                
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
                sel = selectors['kmeans'][req_extractor]
                s = scalers['kmeans'][req_extractor]
                feat_sel = sel.transform(features)
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
