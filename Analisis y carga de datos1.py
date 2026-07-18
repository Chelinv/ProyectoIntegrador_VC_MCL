import os
# pyrefly: ignore [missing-import]
import cv2
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import csv

DATASET_DIR = r"c:\Users\Lalavias\Documents\Machine proyecto final\8266730"
OUT_DIR = r"c:\Users\Lalavias\Documents\Machine proyecto final\ProyectoIntegrador_VC_MCL"

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

print(f"\n2. NÚMERO DE CLASES: {len(conteo_por_clase)}")
for clase, cantidad in conteo_por_clase.items():
    porcentaje = (cantidad / total_imagenes * 100) if total_imagenes else 0
    print(f"   - {clase}: {cantidad} imágenes ({porcentaje:.1f}%)")

valores = list(conteo_por_clase.values())
if valores:
    ratio = max(valores) / min(valores) if min(valores) > 0 else float('inf')
    if ratio <= 1.5:
        print("   Dataset relativamente balanceado")
    else:
        print(f"   Dataset desbalanceado (ratio max/min = {ratio:.2f})")

print(f"\n3. CARACTERÍSTICAS GENERALES DE LAS IMÁGENES:")
print(f"   - Resoluciones únicas encontradas: {len(res_counter)}")
print(f"   - Top 3 resoluciones más comunes: {res_counter.most_common(3)}")
print(f"   - Canales de color (3 = RGB/BGR): {canales}")
print(f"   - Formatos de archivo: {formatos}")
if pesos_kb:
    print(f"   - Peso promedio: {sum(pesos_kb)/len(pesos_kb):.1f} KB")
    print(f"   - Peso mínimo: {min(pesos_kb):.1f} KB")
    print(f"   - Peso máximo: {max(pesos_kb):.1f} KB")

plt.figure(figsize=(7,5))
plt.bar(conteo_por_clase.keys(), conteo_por_clase.values(), color="#4C72B0")
plt.xticks(rotation=30, ha="right")
plt.title("Número de instancias por clase")
plt.ylabel("Cantidad de imágenes")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "instancias_por_clase.png"))
plt.show()

resumen_path = os.path.join(OUT_DIR, "resumen_exploratorio.csv")
with open(resumen_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["clase", "cantidad_imagenes"])
    for clase, cantidad in conteo_por_clase.items():
        writer.writerow([clase, cantidad])
    writer.writerow([])
    writer.writerow(["total_imagenes", total_imagenes])
    writer.writerow(["imagenes_corruptas", imagenes_corruptas])
    writer.writerow(["canales", ",".join(map(str, canales))])
    writer.writerow(["formatos", ",".join(formatos)])
    writer.writerow(["resolucion_mas_comun", res_counter.most_common(1)[0] if res_counter else "N/A"])
    if pesos_kb:
        writer.writerow(["peso_promedio_kb", f"{sum(pesos_kb)/len(pesos_kb):.1f}"])

print(f"\nResumen guardado en: {resumen_path}")
print(f"Gráfico guardado en: {os.path.join(OUT_DIR, 'instancias_por_clase.png')}")