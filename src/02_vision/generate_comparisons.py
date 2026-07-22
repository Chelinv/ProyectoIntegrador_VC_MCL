import cv2
import os
import numpy as np
try:
    from skimage.feature import hog
    from skimage import exposure
    has_skimage = True
except ImportError:
    has_skimage = False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
class_dir_raw = os.path.join(project_root, 'data', '01_raw', 'Afro-ecuadorians')
class_dir_processed = os.path.join(project_root, 'data', '02_processed', 'Afro-ecuadorians')
out_dir = os.path.join(project_root, 'app', 'web')

base_name = '003_10'
FACCIONES = ["eye", "eyebrow", "nose", "mouth"]

# 1. Full Raw Image
img_original_path = os.path.join(class_dir_raw, f"{base_name}.JPG")
img_orig = cv2.imread(img_original_path)
full_raw = cv2.resize(img_orig, (512, 512))
cv2.imwrite(os.path.join(out_dir, 'full_raw.jpg'), full_raw)

# 2. Mosaic Thresh
crops_thresh = []
for faccion in FACCIONES:
    img_path = os.path.join(class_dir_processed, f"{base_name}_{faccion}.jpg")
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (256, 256))
    else:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
    crops_thresh.append(img)

mosaic_thresh = np.vstack((
    np.hstack((crops_thresh[0], crops_thresh[1])),
    np.hstack((crops_thresh[2], crops_thresh[3]))
))
cv2.line(mosaic_thresh, (256, 0), (256, 512), (0,0,255), 4)
cv2.line(mosaic_thresh, (0, 256), (512, 256), (0,0,255), 4)
cv2.imwrite(os.path.join(out_dir, 'mosaic_thresh.jpg'), mosaic_thresh)

def generate_zernike(img, is_mosaic, out_name):
    img_vis = img.copy()
    if is_mosaic:
        centros = [(128, 128), (384, 128), (128, 384), (384, 384)]
        r = 100
    else:
        centros = [(256, 256)]
        r = 200
    for cx, cy in centros:
        cv2.circle(img_vis, (cx, cy), r, (0, 255, 255), 4)
        cv2.circle(img_vis, (cx, cy), 8, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(out_dir, out_name), img_vis)

def generate_hog(img, out_name):
    if not has_skimage: return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, hog_image = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True, channel_axis=None)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_vis = (hog_image_rescaled * 255).astype(np.uint8)
    hog_vis_bgr = cv2.applyColorMap(hog_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(out_dir, out_name), hog_vis_bgr)

def generate_brisk(img, out_name):
    orb = cv2.ORB_create(nfeatures=500)
    kp, _ = orb.detectAndCompute(img, None)
    img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(out_dir, out_name), img_kp)

# Generate for full raw
generate_zernike(full_raw, False, 'zernike_full.jpg')
generate_hog(full_raw, 'hog_full.jpg')
generate_brisk(full_raw, 'brisk_full.jpg')

# Generate for mosaic thresh
generate_zernike(mosaic_thresh, True, 'zernike_thresh.jpg')
generate_hog(mosaic_thresh, 'hog_thresh.jpg')
generate_brisk(mosaic_thresh, 'brisk_thresh.jpg')

print('Todas las imagenes comparativas generadas con exito')
