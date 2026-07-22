import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time

# 1. Definir rutas y parámetros
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_DIR = os.path.join(project_root, "data", "01_raw")
IMG_SIZE_PER_CROP = (64, 64)
IMG_SIZE = (128, 128) # Mosaico 2x2 de 64x64
BATCH_SIZE = 32

print("Iniciando Fase de Deep Learning con TensorFlow/Keras...")
print("=" * 50)

# Verificar si el dataset existe
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"No se encontró el directorio {DATASET_DIR}.")

from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Inicializar MediaPipe (mismo pipeline que el servidor)
task_path = os.path.join(project_root, 'app', 'server', 'face_landmarker.task')
base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

EYES_IDX = [33,133,159,145,153,154,155,133,362,263,386,374,380,381,382,384,385,387,388,390,398]
EYEBROWS_IDX = [70,63,105,66,107,55,65,52,53,46,336,296,334,293,300,276,283,282,295,285]
NOSE_IDX = [1,2,98,327,279,49,114,128,29,290,277,437]
MOUTH_IDX = [0,13,14,17,61,291,37,267,314,17,84,181]
FACCIONES_MP = {"eye":EYES_IDX,"eyebrow":EYEBROWS_IDX,"nose":NOSE_IDX,"mouth":MOUTH_IDX}
FACCIONES = ["eye", "eyebrow", "nose", "mouth"]

def extract_crops_mediapipe(img_bgr):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    crops = {}
    for faccion, indices in FACCIONES_MP.items():
        xs = [int(landmarks[i].x * w) for i in indices]
        ys = [int(landmarks[i].y * h) for i in indices]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad_x = int((xmax - xmin) * 0.05)
        pad_y = int((ymax - ymin) * 0.05)
        x1 = max(0, xmin - pad_x); y1 = max(0, ymin - pad_y)
        x2 = min(w, xmax + pad_x); y2 = min(h, ymax + pad_y)
        recorte = gray[y1:y2, x1:x2]
        if recorte.size == 0:
            return None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_c = clahe.apply(recorte)
        img_b = cv2.bilateralFilter(img_c, 9, 75, 75)
        img_t = cv2.adaptiveThreshold(img_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        crops[faccion] = cv2.resize(img_t, IMG_SIZE_PER_CROP)
    return crops

print(f"Cargando imágenes desde: {DATASET_DIR} con MediaPipe (creando mosaicos biométricos)...")

X = []
y = []
class_names = ["Afro-ecuadorians", "European descendants", "Indigenous", "Mestizos"]
num_classes = len(class_names)
print(f"Clases: {class_names}")

for i, class_name in enumerate(class_names):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    imagenes = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    ok = 0
    for img_file in imagenes:
        img_bgr = cv2.imread(os.path.join(class_dir, img_file))
        if img_bgr is None:
            continue
        crops = extract_crops_mediapipe(img_bgr)
        if crops is None:
            continue
        # Crear mosaico 2x2: (Arriba: ojo, ceja), (Abajo: nariz, boca)
        top_row = np.hstack((crops["eye"], crops["eyebrow"]))
        bottom_row = np.hstack((crops["nose"], crops["mouth"]))
        mosaic = np.vstack((top_row, bottom_row))
        X.append(mosaic)
        y.append(i)
        ok += 1
    print(f"  {class_name}: {ok} mosaicos creados")

X = np.array(X)
y = np.array(y)
X = X.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)

print(f"Forma de X original: {X.shape}")

# 1. IMPORTANTE: Dividir PRIMERO en entrenamiento y prueba para evitar fugas de datos (Data Leakage)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. (Balanceo Eliminado a petición del usuario)
# Se entrenará directamente con los datos originales desbalanceados
print(f"Forma de X_train (sin balanceo): {X_train.shape}")
print(f"Forma de X_val intacto (imágenes reales): {X_val.shape}")

# 3. Definir la Arquitectura de la CNN (Custom Architecture Simple)
model = models.Sequential([
    # Capa de reescalado (Normalización de 0-255 a 0-1)
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    
    # Bloque Convolucional 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Bloque Convolucional 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Bloque Convolucional 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Aplanamiento y Capas Densas
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Regularización
    layers.Dense(num_classes, activation='softmax') # Clasificación final (4 clases)
])

# Compilar el modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# 4. Entrenar la CNN
print("\nComenzando el entrenamiento de la CNN...")
EPOCHS = 25
inicio_tiempo = time.time()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)
tiempo_entrenamiento = time.time() - inicio_tiempo

# 5. Evaluación Detallada (Similar a SVM)
print(f"\n{'='*40}")
print("Resultados para CNN con TensorFlow/Keras")
print(f"{'='*40}")

# Evaluar y predecir para obtener métricas completas
print("Evaluando el modelo...")
y_true = y_val
preds = model.predict(X_val, verbose=0)
y_pred = np.argmax(preds, axis=1)

# Calcular métricas
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"1. Accuracy (Exactitud) : {acc:.4f}")
print(f"2. Precision (Precisión): {prec:.4f}")
print(f"3. Recall (Sensibilidad): {rec:.4f}")
print(f"4. F1-Score             : {f1:.4f}")
print(f"5. Tiempo Entrenamiento : {tiempo_entrenamiento:.2f} segundos")

# 6. Graficar Pérdida y Exactitud
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Exactitud (Entrenamiento)')
plt.plot(history.history['val_accuracy'], label='Exactitud (Validación)')
plt.title('Exactitud durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida (Entrenamiento)')
plt.plot(history.history['val_loss'], label='Pérdida (Validación)')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')

os.makedirs(os.path.join(project_root, "outputs", "models"), exist_ok=True)
os.makedirs(os.path.join(project_root, "outputs", "plots"), exist_ok=True)
plt.savefig(os.path.join(project_root, "outputs", "plots", "CNN_Training_History.png"))
print("\nGráficas de entrenamiento guardadas en la carpeta 'outputs/plots'.")

# Guardar el modelo en formato .keras
ruta_modelo = os.path.join(project_root, "outputs", "models", "modelo_cnn.keras")
model.save(ruta_modelo)
print(f"Modelo guardado exitosamente en: {ruta_modelo}")
