import os
import cv2
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import csv

DATASET_DIR = "8266730"

CLASES = ["Afro-ecuadorians", "European descendants", "Indigenous", "Mestizos"]

conteo_por_clase = {}
resoluciones_lista = []
canales = set()
formatos = set()
pesos_kb = []
total_imagenes = 0
imagenes_corruptas = 0

for clase in CLASES:
    ruta_clase = os.path.join(DATASET_DIR, clase)
    if not os.path.isdir(ruta_clase):
        print(f"No encontré la carpeta: {ruta_clase}")
        continue

    imagenes = [f for f in os.listdir(ruta_clase) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    conteo_por_clase[clase] = len(imagenes)
    total_imagenes += len(imagenes)

    for f in imagenes:
        ruta_img = os.path.join(ruta_clase, f)
        img = cv2.imread(ruta_img)
        if img is not None:
            h, w, c = img.shape
            resoluciones_lista.append((w, h))
            canales.add(c)
            formatos.add(Path(f).suffix.lower())
            pesos_kb.append(os.path.getsize(ruta_img) / 1024)
        else:
            imagenes_corruptas += 1
            print(f"No se pudo leer: {ruta_img}")

res_counter = Counter(resoluciones_lista)

print("="*50)
print("ANÁLISIS EXPLORATORIO DEL DATASET")
print("="*50)

print(f"\n1. NÚMERO DE INSTANCIAS: {total_imagenes}")
print(f"   Imágenes corruptas/no legibles: {imagenes_corruptas}")

print(f"\n2.