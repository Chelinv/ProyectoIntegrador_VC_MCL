import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Rutas de los datasets generados en la Meta 2
INPUT_DIR = r"c:\Users\Lalavias\Documents\Machine proyecto final\ProyectoIntegrador_VC_MCL\Datasets_Caracteristicas"
archivos_csv = ["dataset_hu.csv", "dataset_hog.csv", "dataset_brisk.csv"]

# Diccionario maestro para guardar las divisiones de nuestros 3 datasets
datos_preparados = {}

print("Iniciando Paso 1: Preparación de Datos y Split...")
print("=" * 50)

for archivo in archivos_csv:
    ruta_csv = os.path.join(INPUT_DIR, archivo)
    
    if not os.path.exists(ruta_csv):
        print(f"⚠️ Archivo no encontrado: {ruta_csv}")
        continue
        
    # 1. Cargar el CSV
    df = pd.read_csv(ruta_csv)
    
    # 2. Separar variables (X e y)
    # df.iloc[:, :-1] selecciona todas las filas y todas las columnas EXCEPTO la última (X)
    # df.iloc[:, -1] selecciona todas las filas y SOLAMENTE la última columna (y)
    X = df.iloc[:, :-1].values 
    y_texto = df.iloc[:, -1].values
    
    # 3. Codificar etiquetas (String -> Números)
    # Convierte "Afro-ecuadorians", "Mestizos", etc. en 0, 1, 2, 3
    le = LabelEncoder()
    y_codificado = le.fit_transform(y_texto)
    
    # 4. Split (Entrenamiento/Prueba)
    # test_size=0.30 significa 70% entrenamiento y 30% prueba.
    # stratify=y_codificado garantiza que la proporción de clases se mantenga.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_codificado, test_size=0.30, random_state=42, stratify=y_codificado
    )
    
    # Guardamos todo en nuestro diccionario usando el nombre del archivo como llave
    datos_preparados[archivo] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "clases_nombres": le.classes_ # Guardamos los nombres para la matriz de confusión luego
    }
    
    print(f"✅ Dataset procesado: {archivo}")
    print(f"   - Total de instancias originales: {len(df)}")
    print(f"   - Tamaño de Entrenamiento (X_train): {X_train.shape}")
    print(f"   - Tamaño de Prueba (X_test): {X_test.shape}")
    print(f"   - Clases codificadas: {list(zip(le.classes_, range(len(le.classes_))))}")
    print("-" * 50)

print("¡Paso 1 completado exitosamente! Datos listos en memoria para entrenar.")