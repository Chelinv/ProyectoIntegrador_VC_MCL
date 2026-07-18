import os
# pyrefly: ignore [missing-import]
import cv2
import time
import numpy as np
import pandas as pd

# 1. Definir rutas de los datos
base_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(base_dir, "Dataset_Preprocesado")
OUTPUT_DIR = os.path.join(base_dir, "Datasets_Caracteristicas")

CLASES = ["Afro-ecuadorians", "European descendants", "Indigenous", "Mestizos"]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 2. Configurar los extractores
winSize = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

brisk = cv2.BRISK_create()
MAX_KEYPOINTS = 30

datos_hu = []
datos_hog = []
datos_brisk = []

# Cronómetros individuales inicializados en cero
tiempo_hu = 0.0
tiempo_hog = 0.0
tiempo_brisk = 0.0

print("Iniciando extracción y cronometraje individual...")
print("-" * 50)

# 3. Recorrer las imágenes
for clase in CLASES:
    ruta_clase = os.path.join(INPUT_DIR, clase)
    if not os.path.isdir(ruta_clase):
        continue

    imagenes = [f for f in os.listdir(ruta_clase) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for f in imagenes:
        ruta_img = os.path.join(ruta_clase, f)
        img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)

        if img is None: continue
        img_resized = cv2.resize(img, (64, 64))

        # --- TÉCNICA 1: Momentos de Hu ---
        t_inicio_hu = time.time()  # Iniciar cronómetro Hu
        momentos = cv2.moments(img_resized)
        hu = cv2.HuMoments(momentos).flatten()
        for i in range(0, 7):
            if hu[i] != 0:
                hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]))
        fila_hu = list(hu) + [clase]
        datos_hu.append(fila_hu)
        tiempo_hu += (time.time() - t_inicio_hu)  # Pausar y sumar tiempo Hu

        # --- TÉCNICA 2: HOG ---
        t_inicio_hog = time.time()  # Iniciar cronómetro HOG
        hog_features = hog.compute(img_resized).flatten()
        fila_hog = list(hog_features) + [clase]
        datos_hog.append(fila_hog)
        tiempo_hog += (time.time() - t_inicio_hog)  # Pausar y sumar tiempo HOG

        # --- TÉCNICA 3: BRISK ---
        t_inicio_brisk = time.time()  # Iniciar cronómetro BRISK
        keypoints, descriptores = brisk.detectAndCompute(img_resized, None)
        vector_brisk = np.zeros(MAX_KEYPOINTS * 64, dtype=np.uint8)

        if descriptores is not None:
            desc_aplanados = descriptores.flatten()
            longitud_real = len(desc_aplanados)
            if longitud_real > len(vector_brisk):
                vector_brisk = desc_aplanados[:len(vector_brisk)]
            else:
                vector_brisk[:longitud_real] = desc_aplanados

        fila_brisk = list(vector_brisk) + [clase]
        datos_brisk.append(fila_brisk)
        tiempo_brisk += (time.time() - t_inicio_brisk)  # Pausar y sumar tiempo BRISK

    print(f"Extracción completada para la clase: {clase}")

# 4. Guardar los 3 datasets
df_hu = pd.DataFrame(datos_hu)
df_hog = pd.DataFrame(datos_hog)
df_brisk = pd.DataFrame(datos_brisk)

df_hu.to_csv(os.path.join(OUTPUT_DIR, "dataset_hu.csv"), index=False)
df_hog.to_csv(os.path.join(OUTPUT_DIR, "dataset_hog.csv"), index=False)
df_brisk.to_csv(os.path.join(OUTPUT_DIR, "dataset_brisk.csv"), index=False)

# 5. Imprimir la comparativa individual
print("\n" + "=" * 50)
print("TABLA COMPARATIVA DE EXTRACCIÓN (Para informe LaTeX)")
print("=" * 50)
print(f"{'Algoritmo':<15} | {'N. Características':<20} | {'Formato':<10} | {'Tiempo Total (430 img)'}")
print("-" * 75)
print(f"{'Momentos (HU)':<15} | {df_hu.shape[1] - 1:<20} | {'Float64':<10} | {tiempo_hu:.4f} segundos")
print(f"{'Avanzado (HOG)':<15} | {df_hog.shape[1] - 1:<20} | {'Float32':<10} | {tiempo_hog:.4f} segundos")
print(f"{'Investigado (BRISK)':<15} | {df_brisk.shape[1] - 1:<20} | {'Uint8':<10} | {tiempo_brisk:.4f} segundos")
print("-" * 75)