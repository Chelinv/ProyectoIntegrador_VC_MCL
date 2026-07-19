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
img_path = os.path.join(project_root, 'data', '02_processed', 'Afro-ecuadorians', '003_1.JPG')
out_dir = os.path.join(project_root, 'app', 'web')

img = cv2.imread(img_path)
if img is None:
    print("Could not load image.")
    exit(1)

img_resized = cv2.resize(img, (256, 256))
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# 1. Hu Moments Visualization (Shape contour)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_hu = img_resized.copy()
cv2.drawContours(img_hu, contours, -1, (0, 0, 255), 2) # Draw red contours
moments = cv2.moments(thresh)
if moments["m00"] != 0:
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    cv2.circle(img_hu, (cX, cY), 5, (255, 0, 0), -1) # Blue dot for center
cv2.imwrite(os.path.join(out_dir, 'hu_vis.jpg'), img_hu)

# 2. HOG Visualization
if has_skimage:
    _, hog_image = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True, channel_axis=None)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_vis = (hog_image_rescaled * 255).astype(np.uint8)
    hog_vis_bgr = cv2.applyColorMap(hog_vis, cv2.COLORMAP_JET) # Make it look cool
    cv2.imwrite(os.path.join(out_dir, 'hog_vis.jpg'), hog_vis_bgr)
else:
    print("scikit-image not installed, generating mock HOG")
    img_hog = img_resized.copy()
    for i in range(0, 256, 16):
        cv2.line(img_hog, (i, 0), (i, 256), (0,255,0), 1)
        cv2.line(img_hog, (0, i), (256, i), (0,255,0), 1)
    cv2.imwrite(os.path.join(out_dir, 'hog_vis.jpg'), img_hog)

# 3. BRISK Visualization
brisk = cv2.BRISK_create()
kp, _ = brisk.detectAndCompute(img_resized, None)
img_kp = cv2.drawKeypoints(img_resized, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(os.path.join(out_dir, 'brisk_vis.jpg'), img_kp)

# Save original
cv2.imwrite(os.path.join(out_dir, 'orig_vis.jpg'), img_resized)

print("All visualizations saved!")
