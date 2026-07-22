import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
try:
    from skimage.feature import hog
    from skimage import exposure
    has_skimage = True
except ImportError:
    has_skimage = False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
class_dir_raw = os.path.join(project_root, 'data', '01_raw', 'Afro-ecuadorians')
class_dir_processed = os.path.join(project_root, 'data', '02_processed', 'Afro-ecuadorians')
xml_path = os.path.join(project_root, 'data', '01_raw', 'Metadata', 'Afro-Ecuadorians.xml')
out_dir = os.path.join(project_root, 'app', 'web')
os.makedirs(out_dir, exist_ok=True)

base_name = '003_10'
FACCIONES = ["eye", "eyebrow", "nose", "mouth"]

# 1. Parsear XML para obtener las cajas
bboxes = {}
tree = ET.parse(xml_path)
for row in tree.getroot().findall('.//row'):
    filename = row.find('filename').text
    if filename and filename.startswith(base_name):
        faccion = row.find('class').text
        if faccion in FACCIONES:
            xmin = int(row.find('xmin').text)
            ymin = int(row.find('ymin').text)
            xmax = int(row.find('xmax').text)
            ymax = int(row.find('ymax').text)
            if faccion in bboxes:
                cx, cy, cxx, cyy = bboxes[faccion]
                bboxes[faccion] = (min(cx, xmin), min(cy, ymin), max(cxx, xmax), max(cyy, ymax))
            else:
                bboxes[faccion] = (xmin, ymin, xmax, ymax)

# 2. Cargar imagen original y dibujar cajas
img_original_path = os.path.join(class_dir_raw, f"{base_name}.JPG")
img_orig = cv2.imread(img_original_path)
if img_orig is not None:
    # Colores: BGR
    colores = {"eye": (255, 0, 0), "eyebrow": (0, 255, 0), "nose": (0, 0, 255), "mouth": (0, 255, 255)}
    for faccion, (xmin, ymin, xmax, ymax) in bboxes.items():
        cv2.rectangle(img_orig, (xmin, ymin), (xmax, ymax), colores.get(faccion, (255,255,255)), 8)
        cv2.putText(img_orig, faccion, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, colores.get(faccion, (255,255,255)), 4)
    # Redimensionar la original a 512x512 para que empate con el mosaico
    img_orig_512 = cv2.resize(img_orig, (512, 512))
else:
    img_orig_512 = np.zeros((512, 512, 3), dtype=np.uint8)

# 3. Cargar los 4 recortes procesados
crops = []
for faccion in FACCIONES:
    img_path = os.path.join(class_dir_processed, f"{base_name}_{faccion}.jpg")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256)) # 4 de 256x256 hacen un mosaico de 512x512
    crops.append(img)

# Crear mosaico 2x2 (512x512)
top_row = np.hstack((crops[0], crops[1]))
bottom_row = np.hstack((crops[2], crops[3]))
mosaic = np.vstack((top_row, bottom_row))
gray_mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)

# Dibujar cruz roja en el mosaico para separar las facciones
cv2.line(mosaic, (256, 0), (256, 512), (0,0,255), 4)
cv2.line(mosaic, (0, 256), (512, 256), (0,0,255), 4)

# 4. Unir original detectado + mosaico (Side by Side)
side_by_side = np.hstack((img_orig_512, mosaic))
cv2.imwrite(os.path.join(out_dir, 'explicacion_mosaico.jpg'), side_by_side)

# --- 1. Zernike Visualization ---
img_zernike = mosaic.copy()
centros = [(128, 128), (384, 128), (128, 384), (384, 384)]
for cx, cy in centros:
    cv2.circle(img_zernike, (cx, cy), 100, (0, 255, 255), 4)
    cv2.circle(img_zernike, (cx, cy), 8, (0, 0, 255), -1)
cv2.imwrite(os.path.join(out_dir, 'zernike_vis.jpg'), img_zernike)

# --- 2. HOG Visualization ---
if has_skimage:
    _, hog_image = hog(gray_mosaic, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True, channel_axis=None)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_vis = (hog_image_rescaled * 255).astype(np.uint8)
    hog_vis_bgr = cv2.applyColorMap(hog_vis, cv2.COLORMAP_JET)
    cv2.line(hog_vis_bgr, (256, 0), (256, 512), (255,255,255), 2)
    cv2.line(hog_vis_bgr, (0, 256), (512, 256), (255,255,255), 2)
    cv2.imwrite(os.path.join(out_dir, 'hog_vis.jpg'), hog_vis_bgr)

# --- 3. ORB Visualization ---
orb = cv2.ORB_create(nfeatures=500)
kp, _ = orb.detectAndCompute(mosaic, None)
img_kp = cv2.drawKeypoints(mosaic, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.line(img_kp, (256, 0), (256, 512), (255,255,255), 2)
cv2.line(img_kp, (0, 256), (512, 256), (255,255,255), 2)
cv2.imwrite(os.path.join(out_dir, 'brisk_vis.jpg'), img_kp)

print("¡Visualizaciones de Mosaico Biométrico + Original generadas!")
