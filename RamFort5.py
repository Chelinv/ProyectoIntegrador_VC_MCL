import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

datasets = {
    "HU (Momentos)": r"c:\Users\Lalavias\Documents\Machine proyecto final\ProyectoIntegrador_VC_MCL\Datasets_Caracteristicas\dataset_hu.csv",
    "HOG (Avanzado)": r"c:\Users\Lalavias\Documents\Machine proyecto final\ProyectoIntegrador_VC_MCL\Datasets_Caracteristicas\dataset_hog.csv",
    "BRISK (Investigado)": r"c:\Users\Lalavias\Documents\Machine proyecto final\ProyectoIntegrador_VC_MCL\Datasets_Caracteristicas\dataset_brisk.csv"
}

def entrenar_evaluar_rf(nombre_dataset, ruta_dataset):
    print(f"\n{'='*40}")
    print(f"Resultados para Random Forest con: {nombre_dataset}")
    print(f"{'='*40}")
    
    df = pd.read_csv(ruta_dataset)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"1. Accuracy (Exactitud) : {acc:.4f}")
    print(f"2. Precision (Precisión): {prec:.4f}")
    print(f"3. Recall (Sensibilidad): {rec:.4f}")
    print(f"4. F1-Score             : {f1:.4f}")

for nombre, ruta in datasets.items():
    if os.path.exists(ruta):
        entrenar_evaluar_rf(nombre, ruta)
    else:
        print(f"No se encontró el archivo: {ruta}")
