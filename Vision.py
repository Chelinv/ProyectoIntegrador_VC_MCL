import os
import cv2
import matplotlib.pyplot as plt

DATASET_DIR = "C:\\Users\\HOME\\OneDrive\\Documentos\\septimo\\vision\\8266730\\8266730"
OUTPUT_DIR = "C:\\Users\\HOME\\OneDrive\\Documentos\\septimo\\vision\\Proyecto integrador\\Dataset_Preprocesado"


CLASES = ["Afro-ecuadorians", "European descendants", "Indigenous", "Mestizos"]

# Crear el directorio base de salida si no existe
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Listas para guardar las imágenes de ejemplo para la gráfica
ejemplos_nombres = []
ejemplos_originales = []
ejemplos_procesados = []

print("Iniciando el NUEVO preprocesamiento (CLAHE + Bilateral + Adaptativo)...")
print("-" * 50)

# 2. Bucle para recorrer y procesar cada clase
for clase in CLASES:
    ruta_clase_entrada = os.path.join(DATASET_DIR, clase)
    ruta_clase_salida = os.path.join(OUTPUT_DIR, clase)

    # Crear la subcarpeta de la clase en el directorio de salida
    if not os.path.exists(ruta_clase_salida):
        os.makedirs(ruta_clase_salida)

    if not os.path.isdir(ruta_clase_entrada):
        continue

    imagenes = [f for f in os.listdir(ruta_clase_entrada) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    capturo_ejemplo = False # Bandera para tomar solo 1 foto de ejemplo por clase

    for f in imagenes:
        ruta_img = os.path.join(ruta_clase_entrada, f)
        ruta_salida = os.path.join(ruta_clase_salida, f)

        # A. Cargar imagen en Escala de Grises
        img_original = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)

        if img_original is None:
            continue

        # --- INICIO DEL NUEVO PIPELINE DE PREPROCESAMIENTO ---

        # B. Mejorar Contraste: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_contraste = clahe.apply(img_original)

        # C. Eliminación de Ruido: Filtro Bilateral (Conserva bordes)
        # Parámetros: d (diámetro del vecindario), sigmaColor, sigmaSpace
        img_ruido = cv2.bilateralFilter(img_contraste, 9, 75, 75)

        # D. Umbralización: Adaptativa (Ideal para iluminación irregular)
        img_binarizada = cv2.adaptiveThreshold(img_ruido, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)

        # --- FIN DEL PIPELINE ---

        # Guardar la imagen procesada en el nuevo directorio
        cv2.imwrite(ruta_salida, img_binarizada)

        # Guardar la primera imagen de esta clase para el gráfico de ejemplo
        if not capturo_ejemplo:
            ejemplos_nombres.append(clase)
            ejemplos_originales.append(img_original)
            ejemplos_procesados.append(img_binarizada)
            capturo_ejemplo = True

    print(f"✅ Clase '{clase}' procesada. Imágenes guardadas en Dataset_Preprocesado.")

print("-" * 50)
print("¡NUEVO Preprocesamiento completado!")

# 3. Generar la gráfica comparativa (Original vs Procesada)
fig, axs = plt.subplots(4, 2, figsize=(10, 16))
fig.suptitle("Resultados del Preprocesamiento (CLAHE + Bilateral + Adaptativo)", fontsize=16, fontweight='bold', y=0.92)

for i in range(4):
    # Columna Izquierda: Original
    axs[i, 0].imshow(ejemplos_originales[i], cmap='gray')
    axs[i, 0].set_title(f"Original: {ejemplos_nombres[i]}", fontsize=12)
    axs[i, 0].axis('off')

    # Columna Derecha: Procesada
    axs[i, 1].imshow(ejemplos_procesados[i], cmap='gray')
    axs[i, 1].set_title("CLAHE + Filtro Bilateral + Adaptativo", fontsize=12)
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()