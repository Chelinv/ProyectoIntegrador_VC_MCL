"""
extraccion_mediapipe.py
Re-extrae HOG, Zernike y BRISK usando MediaPipe (identico al servidor Flask).
Esto elimina el domain gap entre entrenamiento y prediccion.
"""
import os
import cv2
import time
import numpy as np
import pandas as pd
import mahotas
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR  = os.path.join(project_root, "data", "01_raw")
OUTPUT_DIR = os.path.join(project_root, "data", "03_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASES    = ["Afro-ecuadorians", "European descendants", "Indigenous", "Mestizos"]
FACCIONES = ["eye", "eyebrow", "nose", "mouth"]

# MediaPipe
task_path = os.path.join(project_root, 'app', 'server', 'face_landmarker.task')
if not os.path.exists(task_path):
    print("Descargando modelo FaceLandmarker...")
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

EYES_IDX     = [33,133,159,145,153,154,155,133,362,263,386,374,380,381,382,384,385,387,388,390,398]
EYEBROWS_IDX = [70,63,105,66,107,55,65,52,53,46,336,296,334,293,300,276,283,282,295,285]
NOSE_IDX     = [1,2,98,327,279,49,114,128,29,290,277,437]
MOUTH_IDX    = [0,13,14,17,61,291,37,267,314,17,84,181]
FACCIONES_MP = {"eye":EYES_IDX,"eyebrow":EYEBROWS_IDX,"nose":NOSE_IDX,"mouth":MOUTH_IDX}

# Extractores
hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
try:
    brisk_ext = cv2.BRISK_create()
except:
    brisk_ext = cv2.ORB_create()
MAX_KEYPOINTS = 30

def extract_crops(img_bgr):
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
        xmin,xmax = min(xs),max(xs)
        ymin,ymax = min(ys),max(ys)
        pad_x = int((xmax-xmin)*0.05)
        pad_y = int((ymax-ymin)*0.05)
        x1 = max(0, xmin-pad_x); y1 = max(0, ymin-pad_y)
        x2 = min(w, xmax+pad_x); y2 = min(h, ymax+pad_y)
        recorte = gray[y1:y2, x1:x2]
        if recorte.size == 0:
            return None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_c = clahe.apply(recorte)
        img_b = cv2.bilateralFilter(img_c, 9, 75, 75)
        img_t = cv2.adaptiveThreshold(img_b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        crops[faccion] = cv2.resize(img_t, (64,64))
    return crops

datos_z, datos_h, datos_b = [], [], []
t_z = t_h = t_b = 0.0
total_ok = total_skip = 0

print("Iniciando re-extraccion con MediaPipe...")
print("=" * 60)

for clase in CLASES:
    ruta_clase = os.path.join(INPUT_DIR, clase)
    if not os.path.isdir(ruta_clase):
        print(f"[WARN] No encontrado: {ruta_clase}")
        continue
    imagenes = sorted([f for f in os.listdir(ruta_clase) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    ok_clase = 0
    for img_file in imagenes:
        img_bgr = cv2.imread(os.path.join(ruta_clase, img_file))
        if img_bgr is None:
            total_skip += 1
            continue
        crops = extract_crops(img_bgr)
        if crops is None:
            print(f"  [SKIP] Sin cara: {img_file}")
            total_skip += 1
            continue
        z_acu, h_acu, b_acu = [], [], []
        for faccion in FACCIONES:
            img64 = crops[faccion]
            t0 = time.time()
            z_acu.extend(mahotas.features.zernike_moments(img64, radius=32))
            t_z += time.time()-t0
            t0 = time.time()
            h_acu.extend(hog.compute(img64).flatten())
            t_h += time.time()-t0
            t0 = time.time()
            kp, desc = brisk_ext.detectAndCompute(img64, None)
            v = np.zeros(MAX_KEYPOINTS*64, dtype=np.uint8)
            if desc is not None:
                flat = desc.flatten(); n = min(len(flat),len(v)); v[:n] = flat[:n]
            b_acu.extend(v)
            t_b += time.time()-t0
        datos_z.append(z_acu+[clase])
        datos_h.append(h_acu+[clase])
        datos_b.append(b_acu+[clase])
        total_ok += 1; ok_clase += 1
    print(f"  {clase}: {ok_clase} imagenes OK")

print(f"\nTotal: {total_ok} OK | {total_skip} saltadas")
df_z = pd.DataFrame(datos_z)
df_h = pd.DataFrame(datos_h)
df_b = pd.DataFrame(datos_b)
df_z.to_csv(os.path.join(OUTPUT_DIR,"dataset_zernike.csv"),index=False)
df_h.to_csv(os.path.join(OUTPUT_DIR,"dataset_hog.csv"),index=False)
df_b.to_csv(os.path.join(OUTPUT_DIR,"dataset_brisk.csv"),index=False)

print(f"\n{'ALGORITMO':<15} | {'CARACTERISTICAS':<18} | {'TIEMPO'}")
print("-"*55)
print(f"{'Zernike':<15} | {df_z.shape[1]-1:<18} | {t_z:.2f}s")
print(f"{'HOG':<15} | {df_h.shape[1]-1:<18} | {t_h:.2f}s")
print(f"{'BRISK/ORB':<15} | {df_b.shape[1]-1:<18} | {t_b:.2f}s")
print("\nRe-extraccion con MediaPipe completada!")
print("Ejecuta: svm_model.py, random_forest.py y clustering.py para reentrenar.")
