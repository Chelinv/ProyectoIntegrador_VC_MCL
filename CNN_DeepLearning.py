import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time

# 1. Definir rutas y parámetros
base_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(base_dir, "Dataset_Preprocesado")
IMG_SIZE = (64, 64) # Mismo tamaño que se usó en extracción de características
BATCH_SIZE = 32

print("Iniciando Fase de Deep Learning con TensorFlow/Keras...")
print("=" * 50)

# Verificar si el dataset preprocesado existe
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"No se encontró el directorio {DATASET_DIR}. Ejecuta Vision.py primero.")

# 2. Cargar el dataset (80% entrenamiento, 20% prueba)
print(f"Cargando imágenes desde: {DATASET_DIR}")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale" # Las imágenes procesadas están en blanco y negro
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

class_names = train_dataset.class_names
print(f"Clases detectadas: {class_names}")
num_classes = len(class_names)

# Optimizar para rendimiento
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Definir la Arquitectura de la CNN
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
    layers.Dropout(0.5), # Regularización para evitar sobreajuste
    layers.Dense(num_classes, activation='softmax') # Clasificación final (4 clases)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# 4. Entrenar la CNN
print("\nComenzando el entrenamiento de la CNN...")
EPOCHS = 20
inicio_tiempo = time.time()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)
tiempo_entrenamiento = time.time() - inicio_tiempo

# 5. Evaluación Detallada (Similar a SVM)
print(f"\n{'='*40}")
print("Resultados para CNN con TensorFlow/Keras")
print(f"{'='*40}")

# Obtener las predicciones reales sobre el dataset de validación
y_true = []
y_pred_probs = []

for img_batch, label_batch in val_dataset:
    y_true.extend(label_batch.numpy())
    preds = model.predict(img_batch, verbose=0)
    y_pred_probs.extend(preds)

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)

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

os.makedirs(os.path.join(base_dir, "Resultados_DeepLearning"), exist_ok=True)
plt.savefig(os.path.join(base_dir, "Resultados_DeepLearning", "CNN_Training_History.png"))
print("\nGráficas de entrenamiento guardadas en la carpeta 'Resultados_DeepLearning'.")

# Guardar el modelo entrenado para predicciones futuras
ruta_modelo = os.path.join(base_dir, "Resultados_DeepLearning", "modelo_cnn.keras")
model.save(ruta_modelo)
print(f"Modelo guardado exitosamente en: {ruta_modelo}")
