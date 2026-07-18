import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Rutas
base_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(base_dir, "Dataset_Preprocesado")
MODEL_PATH = os.path.join(base_dir, "Resultados_DeepLearning", "modelo_cnn.keras")
CLASES = ["Afro-ecuadorians", "European descendants", "Indigenous", "Mestizos"]

if not os.path.exists(MODEL_PATH):
    print("Error: No se encontró el modelo. Primero debes ejecutar CNN_DeepLearning.py para entrenarlo y guardarlo.")
    exit()

print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Elegir una imagen aleatoria para probar
clase_aleatoria = random.choice(CLASES)
ruta_clase = os.path.join(DATASET_DIR, clase_aleatoria)
imagenes_disponibles = os.listdir(ruta_clase)
imagen_aleatoria = random.choice(imagenes_disponibles)
ruta_imagen = os.path.join(ruta_clase, imagen_aleatoria)

print(f"\nProbando con imagen: {imagen_aleatoria}")
print(f"Clase real: {clase_aleatoria}")

# 3. Cargar y preparar la imagen para la red neuronal
img = tf.keras.utils.load_img(ruta_imagen, target_size=(64, 64), color_mode="grayscale")
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Crear un batch de tamaño 1

# 4. Hacer la predicción
predicciones = model.predict(img_array)
indice_prediccion = np.argmax(predicciones[0])
clase_predicha = CLASES[indice_prediccion]
confianza = predicciones[0][indice_prediccion] * 100

print(f"\n--- RESULTADO DE LA PREDICCIÓN ---")
print(f"El modelo predice que es: {clase_predicha} (Seguridad: {confianza:.2f}%)")

if clase_predicha == clase_aleatoria:
    print("¡PREDICCIÓN CORRECTA!")
else:
    print("PREDICCIÓN INCORRECTA.")

# 5. Mostrar la imagen con el resultado
plt.imshow(tf.keras.utils.load_img(ruta_imagen, color_mode="grayscale"), cmap='gray')
plt.title(f"Real: {clase_aleatoria}\nPrediccion: {clase_predicha}")
plt.axis('off')
plt.show()
