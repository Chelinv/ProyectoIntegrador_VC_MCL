import os
# pyrefly: ignore [missing-import]
import cv2
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import csv
import xml.etree.ElementTree as ET
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_DIR = os.path.join(project_root, "data", "01_raw")
OUT_DIR = project_root

CLASES = ["Afro-ecuadorians", "European descendants", "Indigenous", "Mestizos"]

conteo_por_clase = {}
resoluciones_lista = []
canales = set()
formatos = set()
pesos_kb = []
total_imagenes = 0
imagenes_corruptas = 0

# --- NUEVAS VARIABLES PARA METADATA ---
metadata_dir = os.path.join(DATASET_DIR, "Metadata")
csv_encuestas = os.path.join(metadata_dir, "Ethnic_facial_characteristics_answers.csv")

imagenes_validas = [] # Para guardar nombres base sin extensión
xml_records = {} # Para almacenar qué facciones tiene cada imagen

print("="*50)
print("ANÁLISIS EXPLORATORIO DEL DATASET (IMÁGENES + METADATOS)")
print("="*50)

# 1. ANALISIS DE IMAGENES
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
            imagenes_validas.append(Path(f).stem) # Guarda el ID (ej: '001_1')
        else:
            imagenes_corruptas += 1
            print(f"No se pudo leer: {ruta_img}")

res_counter = Counter(resoluciones_lista)

print(f"\n1. ANÁLISIS DE IMÁGENES:")
print(f"   - Total imágenes leídas: {total_imagenes}")
print(f"   - Imágenes corruptas/no legibles: {imagenes_corruptas}")
print(f"   - Resoluciones únicas: {len(res_counter)}")
print(f"   - Formatos: {formatos}")

# 2. VALIDACIÓN DE XML (Bounding Boxes)
print(f"\n2. VALIDACIÓN DE ARCHIVOS XML (BIOMETRÍA):")
archivos_xml = [f for f in os.listdir(metadata_dir) if f.endswith('.xml')]
total_boxes = 0

for xml_file in archivos_xml:
    try:
        tree = ET.parse(os.path.join(metadata_dir, xml_file))
        root = tree.getroot()
        for row in root.findall('.//row'):
            filename = row.find('filename').text
            obj_class = row.find('class').text
            if filename and obj_class:
                img_id = Path(filename).stem
                if img_id not in xml_records:
                    xml_records[img_id] = set()
                xml_records[img_id].add(obj_class)
                total_boxes += 1
    except Exception as e:
        print(f"   - Error leyendo {xml_file}: {e}")

imagenes_con_xml = len(xml_records)
imagenes_sin_xml = total_imagenes - imagenes_con_xml
print(f"   - Total de Bounding Boxes detectadas en todos los XML: {total_boxes}")
print(f"   - Imágenes con datos biométricos (XML): {imagenes_con_xml}")
if imagenes_sin_xml > 0:
    print(f"   - ADVERTENCIA: {imagenes_sin_xml} imágenes NO tienen datos en los XML.")
else:
    print(f"   - ÉXITO: El 100% de las imágenes tiene datos biométricos en XML.")

# 3. ANÁLISIS DE CSV (Encuestas)
print(f"\n3. ANÁLISIS DE ENCUESTAS DEMOGRÁFICAS (CSV):")
try:
    df_csv = pd.read_csv(csv_encuestas, encoding='latin1')
    # Normalizar IDs si es necesario
    df_csv['ID'] = df_csv['ID'].astype(str).str.strip()
    
    ids_en_csv = set(df_csv['ID'].tolist())
    ids_en_imagenes = set(imagenes_validas)
    
    imagenes_con_csv = len(ids_en_imagenes.intersection(ids_en_csv))
    imagenes_sin_csv = total_imagenes - imagenes_con_csv
    
    print(f"   - Filas totales en el CSV: {len(df_csv)}")
    print(f"   - Imágenes que hacen 'match' con el CSV: {imagenes_con_csv}")
    if imagenes_sin_csv > 0:
         print(f"   - ADVERTENCIA: {imagenes_sin_csv} imágenes NO tienen encuesta en el CSV.")
    else:
         print(f"   - ÉXITO: El 100% de las imágenes tiene una encuesta asociada.")
         
    # Estadísticas pedidas: Género y Edad
    # Limpiamos posibles espacios en las columnas
    df_csv.columns = df_csv.columns.str.strip()
    
    if 'Gender' in df_csv.columns:
        conteo_genero = df_csv['Gender'].value_counts()
        print(f"\n   -> ESTADÍSTICAS DE GÉNERO:")
        for genero, count in conteo_genero.items():
            print(f"      * {genero}: {count}")
            
    if 'Age' in df_csv.columns:
        # Convertir a numerico por si hay textos
        edades = pd.to_numeric(df_csv['Age'], errors='coerce').dropna()
        promedio_edad = edades.mean()
        min_edad = edades.min()
        max_edad = edades.max()
        print(f"\n   -> ESTADÍSTICAS DE EDAD:")
        print(f"      * Edad promedio: {promedio_edad:.1f} años")
        print(f"      * Edad mínima: {min_edad:.0f} años")
        print(f"      * Edad máxima: {max_edad:.0f} años")

except Exception as e:
    print(f"   - Error leyendo el CSV de encuestas: {e}")

# GRAFICO (Mantenemos el de clases)
plt.figure(figsize=(7,5))
plt.bar(conteo_por_clase.keys(), conteo_por_clase.values(), color="#4C72B0")
plt.xticks(rotation=30, ha="right")
plt.title("Número de instancias por clase")
plt.ylabel("Cantidad de imágenes")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "instancias_por_clase.png"))

print("\n" + "="*50)
print(f"Gráfico guardado en: {os.path.join(OUT_DIR, 'instancias_por_clase.png')}")
print("="*50)