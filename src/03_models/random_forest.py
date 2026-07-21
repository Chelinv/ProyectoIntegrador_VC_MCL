import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectPercentile, f_classif
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
datasets = {
    "Zernike (Momentos)": os.path.join(project_root, "data", "03_features", "dataset_zernike.csv"),
    "HOG (Avanzado)": os.path.join(project_root, "data", "03_features", "dataset_hog.csv"),
    "BRISK (Investigado)": os.path.join(project_root, "data", "03_features", "dataset_brisk.csv")
}

def entrenar_evaluar_rf(nombre_dataset, ruta_dataset):
    print(f"\n{'='*40}")
    print(f"Resultados para Gradient Boosting con: {nombre_dataset}")
    print(f"{'='*40}")
    
    df = pd.read_csv(ruta_dataset)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Selección de características
    selector = SelectPercentile(f_classif, percentile=50)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # 2. (Balanceo Eliminado)

    # 3. Entrenamiento con Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train_sel, y_train)
    y_pred = gb_model.predict(X_test_sel)
    
    # --- GUARDAR MODELOS PARA EL SERVIDOR ---
    import joblib
    extractor_key = nombre_dataset.split()[0].lower()
    model_dir = os.path.join(project_root, "outputs", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(gb_model, os.path.join(model_dir, f"rf_model_{extractor_key}.pkl"))
    joblib.dump(selector, os.path.join(model_dir, f"rf_selector_{extractor_key}.pkl"))
    # (RF no usa scaler en este script)
    # ----------------------------------------
    
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
