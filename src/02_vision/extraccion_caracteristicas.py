import os
# pyrefly: ignore [missing-import]
import cv2
import time
import numpy as np
import pandas as pd
import mahotas
import json

# 1. Definir rutas de los datos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(project_root, "data", "02_processed")
OUTPUT_DIR = os.path.join(project_root, "data", "03_features")

CLASES = ["Afro-ecuadorians", "European descendants", "Indigenous", "Mestizos"]
FACCIONES = ["eye", "eyebrow", "nose", "mouth"]

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
MAX_KEYPOINTS = 30 # Por cada facción

datos_zernike = []
datos_hog = []
datos_brisk = []

tiempo_zernike = 0.0
tiempo_hog = 0.0
tiempo_brisk = 0.0

total_personas = 0

print("Iniciando extracción combinada (4 facciones por persona) y cronometraje...")
print("-" * 50)

# 3. Recorrer las imágenes
for clase in CLASES:
    ruta_clase = os.path.join(INPUT_DIR, clase)
    if not os.path.isdir(ruta_clase):
        continue

    # Buscar todos los archivos "eye" para obtener los IDs únicos de las personas
    archivos_eye = [f for f in os.listdir(ruta_clase) if f.lower().endswith('_eye.jpg')]

    for f_eye in archivos_eye:
        # Extraer el ID base (e.g. 001_1)
        base_name = f_eye[:f_eye.lower().rfind('_eye.jpg')]
        
        # Vectores para acumular las 4 facciones
        z_acumulado = []
        h_acumulado = []
        b_acumulado = []
        
        todas_facciones_leidas = True
        
        for faccion in FACCIONES:
            ruta_img = os.path.join(ruta_clase, f"{base_name}_{faccion}.jpg")
            img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                todas_facciones_leidas = False
                break
                
            img_resized = cv2.resize(img, (64, 64))
            
            # --- ZERNIKE ---
            t_inicio = time.time()
            z_feats = mahotas.features.zernike_moments(img_resized, radius=32)
            z_acumulado.extend(z_feats)
            tiempo_zernike += (time.time() - t_inicio)
            
            # --- HOG ---
            t_inicio = time.time()
            h_feats = hog.compute(img_resized).flatten()
            h_acumulado.extend(h_feats)
            tiempo_hog += (time.time() - t_inicio)
            
            # --- BRISK ---
            t_inicio = time.time()
            keypoints, descriptores = brisk.detectAndCompute(img_resized, None)
            v_brisk = np.zeros(MAX_KEYPOINTS * 64, dtype=np.uint8)
            if descriptores is not None:
                desc_aplanados = descriptores.flatten()
                longitud_real = len(desc_aplanados)
                if longitud_real > len(v_brisk):
                    v_brisk = desc_aplanados[:len(v_brisk)]
                else:
                    v_brisk[:longitud_real] = desc_aplanados
            b_acumulado.extend(v_brisk)
            tiempo_brisk += (time.time() - t_inicio)

        if todas_facciones_leidas:
            datos_zernike.append(z_acumulado + [clase])
            datos_hog.append(h_acumulado + [clase])
            datos_brisk.append(b_acumulado + [clase])
            total_personas += 1

    print(f"Extracción completada para la clase: {clase}")

# 4. Guardar los 3 datasets
df_zernike = pd.DataFrame(datos_zernike)
df_hog = pd.DataFrame(datos_hog)
df_brisk = pd.DataFrame(datos_brisk)

df_zernike.to_csv(os.path.join(OUTPUT_DIR, "dataset_zernike.csv"), index=False)
df_hog.to_csv(os.path.join(OUTPUT_DIR, "dataset_hog.csv"), index=False)
df_brisk.to_csv(os.path.join(OUTPUT_DIR, "dataset_brisk.csv"), index=False)

# 5. Guardar JSON para la web (como pidió el usuario)
num_feat_zernike = int(df_zernike.shape[1] - 1)
num_feat_hog = int(df_hog.shape[1] - 1)
num_feat_brisk = int(df_brisk.shape[1] - 1)

resultados_json = {
    "total_personas": total_personas,
    "algoritmos": [
        {
            "nombre": "Zernike",
            "caracteristicas": num_feat_zernike,
            "formato": "Float64",
            "tiempo_segundos": round(tiempo_zernike, 4),
            "costo_computacional": "Bajo",
            "representacion": "Momentos ortogonales (Forma global)"
        },
        {
            "nombre": "HOG",
            "caracteristicas": num_feat_hog,
            "formato": "Float32",
            "tiempo_segundos": round(tiempo_hog, 4),
            "costo_computacional": "Medio",
            "representacion": "Histogramas de gradientes (Textura y bordes)"
        },
        {
            "nombre": "BRISK",
            "caracteristicas": num_feat_brisk,
            "formato": "Uint8",
            "tiempo_segundos": round(tiempo_brisk, 4),
            "costo_computacional": "Alto",
            "representacion": "Puntos clave locales (Patrones binarios)"
        }
    ]
}

json_path = os.path.join(project_root, "resultados_extraccion.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(resultados_json, f, indent=4)

print("\n" + "=" * 50)
print(f"Resultados de extracción guardados en: {json_path}")
print("TABLA COMPARATIVA DE EXTRACCIÓN (4 Facciones concatenadas)")
print("=" * 50)
print(f"{'Algoritmo':<15} | {'N. Características':<20} | {'Formato':<10} | {'Tiempo Total'}")
print("-" * 75)
print(f"{'Zernike':<15} | {num_feat_zernike:<20} | {'Float64':<10} | {tiempo_zernike:.4f} seg")
print(f"{'HOG':<15} | {num_feat_hog:<20} | {'Float32':<10} | {tiempo_hog:.4f} seg")
print(f"{'BRISK':<15} | {num_feat_brisk:<20} | {'Uint8':<10} | {tiempo_brisk:.4f} seg")
print("-" * 75)