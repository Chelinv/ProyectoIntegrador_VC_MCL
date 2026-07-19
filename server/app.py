import os
import base64
import numpy as np
import cv2
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

app = Flask(__name__, static_folder="../FrontEnd")
CORS(app)

# ==========================================
# 1. Configuración de Rutas y Variables
# ==========================================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
clases_oficiales = ['Afro-ecuadorians', 'European descendants', 'Indigenous', 'Mestizos']

# Modelos en memoria
models = {
    'svm': {},
    'rf': {},
    'kmeans': {}
}
scalers = {}
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
    
    rutas_csv = {
        'hu': os.path.join(base_dir, "Datasets_Caracteristicas", "dataset_hu.csv"),
        'hog': os.path.join(base_dir, "Datasets_Caracteristicas", "dataset_hog.csv"),
        'brisk': os.path.join(base_dir, "Datasets_Caracteristicas", "dataset_brisk.csv")
    }
    
    for extractor_name, ruta in rutas_csv.items():
        if os.path.exists(ruta):
            print(f"Entrenando modelos para extractor: {extractor_name.upper()}...")
            df = pd.read_csv(ruta)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            # Escalar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scalers[extractor_name] = scaler
            
            # SVM
            svm_m = SVC(kernel='linear', probability=True, random_state=42)
            svm_m.fit(X_scaled, y)
            models['svm'][extractor_name] = svm_m
            
            # RF
            rf_m = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_m.fit(X, y)
            models['rf'][extractor_name] = rf_m
            
            # K-Means
            km_m = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
            km_m.fit(X_scaled)
            models['kmeans'][extractor_name] = km_m
        else:
            print(f"ERROR: No se encontró {ruta}")

    # Cargar CNN
    ruta_cnn = os.path.join(base_dir, "Resultados_DeepLearning", "modelo_cnn.keras")
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

init_models()

# ==========================================
# 3. Pipeline de Preprocesamiento (Vision.py)
# ==========================================
def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
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

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(os.path.join(base_dir, '..', 'data'), filename)

@app.route('/Dataset_Preprocesado/<path:filename>')
def serve_preprocesado(filename):
    return send_from_directory(os.path.join(base_dir, 'Dataset_Preprocesado'), filename)

@app.route('/Resultados_Clustering/<path:filename>')
def serve_clustering(filename):
    return send_from_directory(os.path.join(base_dir, 'Resultados_Clustering'), filename)

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
            if req_extractor == 'hu':
                momentos = cv2.moments(img_preprocesada)
                hu = cv2.HuMoments(momentos).flatten()
                for i in range(0, 7):
                    if hu[i] != 0:
                        hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]))
                features = np.array(hu).reshape(1, -1)
                
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
                return jsonify({"clase": clases_oficiales[class_idx], "prob": prob_cnn})
            else:
                return jsonify({'error': 'CNN model not loaded'}), 500
                
        elif req_model == 'svm':
            if 'svm' in models and req_extractor in models['svm']:
                m = models['svm'][req_extractor]
                s = scalers[req_extractor]
                feat_scaled = s.transform(features)
                pred = m.predict(feat_scaled)[0]
                prob = np.max(m.predict_proba(feat_scaled))
                return jsonify({"clase": pred, "prob": float(prob)})
            else:
                return jsonify({'error': 'SVM model not loaded for this extractor'}), 500
                
        elif req_model == 'rf':
            if 'rf' in models and req_extractor in models['rf']:
                m = models['rf'][req_extractor]
                pred = m.predict(features)[0]
                prob = np.max(m.predict_proba(features))
                return jsonify({"clase": pred, "prob": float(prob)})
            else:
                return jsonify({'error': 'RF model not loaded for this extractor'}), 500
                
        elif req_model == 'kmeans':
            if 'kmeans' in models and req_extractor in models['kmeans']:
                m = models['kmeans'][req_extractor]
                s = scalers[req_extractor]
                feat_scaled = s.transform(features)
                cluster_id = m.predict(feat_scaled)[0]
                return jsonify({"clase": f"Grupo {cluster_id}", "prob": 1.0})
            else:
                return jsonify({'error': 'K-Means model not loaded for this extractor'}), 500
                
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Iniciando Servidor Flask en http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
