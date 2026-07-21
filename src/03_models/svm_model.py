import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectPercentile, f_classif
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
datasets = {
    "Zernike (Momentos)": os.path.join(project_root, "data", "03_features", "dataset_zernike.csv"),
    "HOG (Avanzado)": os.path.join(project_root, "data", "03_features", "dataset_hog.csv"),
    "BRISK (Investigado)": os.path.join(project_root, "data", "03_features", "dataset_brisk.csv")
}

def entrenar_evaluar_svm(nombre_dataset, ruta_dataset):
    print(f"\n{'='*40}")
    print(f"Resultados para SVM con: {nombre_dataset}")
    print(f"{'='*40}")
    
    df = pd.read_csv(ruta_dataset)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Dividimos los datos: 80% para que el modelo estudie (entrenamiento) y 20% para el examen final (prueba).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   

    # 1. Selección del 50% de las características más correlacionadas
    selector = SelectPercentile(f_classif, percentile=50)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # 2. (Balanceo Eliminado)

    # 3. Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # SVM (requiere probability=True para el servidor web)
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    
    # --- GUARDAR MODELOS PARA EL SERVIDOR ---
    import joblib
    extractor_key = nombre_dataset.split()[0].lower()
    model_dir = os.path.join(project_root, "outputs", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(svm_model, os.path.join(model_dir, f"svm_model_{extractor_key}.pkl"))
    joblib.dump(selector, os.path.join(model_dir, f"svm_selector_{extractor_key}.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, f"svm_scaler_{extractor_key}.pkl"))
    # ----------------------------------------
    
    # Calculamos las métricas para ver qué tan bien le fue. Usamos 'weighted' por si 
    # nuestras clases están un poco desbalanceadas.
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"1. Accuracy (Exactitud) : {acc:.4f}")
    print(f"2. Precision (Precisión): {prec:.4f}")
    print(f"3. Recall (Sensibilidad): {rec:.4f}")
    print(f"4. F1-Score             : {f1:.4f}")

if __name__ == "__main__":
    for nombre, ruta in datasets.items():
        if os.path.exists(ruta):
            entrenar_evaluar_svm(nombre, ruta)
        else:
            print(f"No se encontró el archivo: {ruta}")
