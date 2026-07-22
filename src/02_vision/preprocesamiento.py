import os
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_DIR = os.path.join(project_root, "data", "01_raw")
OUTPUT_DIR = os.path.join(project_root, "data", "02_processed")
METADATA_DIR = os.path.join(DATASET_DIR, "Metadata")

CLASES = ["Afro-ecuadorians", "European descendants", "Indigenous", "Mestizos"]
FACCIONES = ["eyebrow", "eye", "nose", "mouth"]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Cargando datos biométricos (XML)...")
# Diccionario para guardar cajas delimitadoras
# formato: bboxes[filename][clase_faccion] = (xmin, ymin, xmax, ymax)
bboxes = {}

archivos_xml = [f for f in os.listdir(METADATA_DIR) if f.endswith('.xml')]
for xml_file in archivos_xml:
    tree = ET.parse(os.path.join(METADATA_DIR, xml_file))
    root = tree.getroot()
    for row in root.findall('.//row'):
        filename = row.find('filename').text
        faccion = row.find('class').text
        
        # Validar si existe la faccion
        if filename and faccion and faccion in FACCIONES:
            img_id = Path(filename).stem
            xmin = int(row.find('xmin').text)
            ymin = int(row.find('ymin').text)
            xmax = int(row.find('xmax').text)
            ymax = int(row.find('ymax').text)
            
            if img_id not in bboxes:
                bboxes[img_id] = {}
            
            # Si ya existe esta facción (ej: ojo izquierdo ya fue detectado y ahora leemos ojo derecho),
            # fusionamos las cajas para abarcar ambos ojos en un solo recorte grande.
            if faccion in bboxes[img_id]:
                curr_xmin, curr_ymin, curr_xmax, curr_ymax = bboxes[img_id][faccion]
                bboxes[img_id][faccion] = (
                    min(curr_xmin, xmin),
                    min(curr_ymin, ymin),
                    max(curr_xmax, xmax),
                    max(curr_ymax, ymax)
                )
            else:
                bboxes[img_id][faccion] = (xmin, ymin, xmax, ymax)

print(f"Datos biométricos cargados para {len(bboxes)} imágenes.")
print("-" * 50)
print("Iniciando el NUEVO preprocesamiento (Recortes + CLAHE + Bilateral + Adaptativo)...")
print("-" * 50)

# Listas para guardar ejemplos de la primera imagen (4 etnias x 4 facciones = 16 subplots)
ejemplos = []

for clase in CLASES:
    ruta_clase_entrada = os.path.join(DATASET_DIR, clase)
    ruta_clase_salida = os.path.join(OUTPUT_DIR, clase)

    if not os.path.exists(ruta_clase_salida):
        os.makedirs(ruta_clase_salida)

    if not os.path.isdir(ruta_clase_entrada):
        continue

    imagenes = [f for f in os.listdir(ruta_clase_entrada) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    capturo_ejemplo = False

    for f in imagenes:
        ruta_img = os.path.join(ruta_clase_entrada, f)
        img_id = Path(f).stem
        
        if img_id not in bboxes:
            print(f"ADVERTENCIA: No se encontró XML para {f}")
            continue

        img_original = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            continue

        recortes_procesados = {}

        for faccion in FACCIONES:
            if faccion in bboxes[img_id]:
                xmin, ymin, xmax, ymax = bboxes[img_id][faccion]
                
                # Expandir un poco el margen (padding) un 5% para no cortar tan al ras
                h, w = img_original.shape
                pad_x = int((xmax - xmin) * 0.05)
                pad_y = int((ymax - ymin) * 0.05)
                
                x1 = max(0, xmin - pad_x)
                y1 = max(0, ymin - pad_y)
                x2 = min(w, xmax + pad_x)
                y2 = min(h, ymax + pad_y)

                # 1. Recortar
                recorte = img_original[y1:y2, x1:x2]
                
                # 2. Mejorar Contraste: CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_contraste = clahe.apply(recorte)

                # 3. Filtro Bilateral
                img_ruido = cv2.bilateralFilter(img_contraste, 9, 75, 75)

                # 4. Umbralización Adaptativa
                img_binarizada = cv2.adaptiveThreshold(img_ruido, 255,
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 11, 2)
                
                recortes_procesados[faccion] = img_binarizada
                
                # Guardar el recorte (ej: 001_1_nose.jpg)
                nombre_base = Path(f).stem
                ruta_salida = os.path.join(ruta_clase_salida, f"{nombre_base}_{faccion}.jpg")
                cv2.imwrite(ruta_salida, img_binarizada)

        if not capturo_ejemplo and len(recortes_procesados) == 4:
            ejemplos.append({
                "clase": clase,
                "img_nombre": f,
                "recortes": recortes_procesados
            })
            capturo_ejemplo = True

    print(f"Clase '{clase}' procesada y fragmentada.")

print("-" * 50)
print("¡Preprocesamiento Biométrico completado!")

# Generar gráfica (4 filas = 4 etnias, 4 columnas = 4 facciones)
if len(ejemplos) > 0:
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Recortes Biométricos Procesados (CLAHE + Bilateral + Adaptativo)", fontsize=16, fontweight='bold', y=0.95)
    
    for i, ej in enumerate(ejemplos):
        for j, faccion in enumerate(FACCIONES):
            if i < 4:
                axs[i, j].imshow(ej["recortes"][faccion], cmap='gray')
                axs[i, j].set_title(f"{ej['clase'][:10]}... - {faccion}")
                axs[i, j].axis('off')
                
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "recortes_biometricos.png"))
    # plt.show()