import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

#Carpeta base del proyecto
base_dir = os.path.dirname(os.path.abspath(__file__))

#Rutas a los archivos de caracteristicas
datasets = {
    "HU (Momentos)": os.path.join(base_dir, "Datasets_Caracteristicas", "dataset_hu.csv"),
    "HOG (Avanzado)": os.path.join(base_dir, "Datasets_Caracteristicas", "dataset_hog.csv"),
    "BRISK (Investigado)": os.path.join(base_dir, "Datasets_Caracteristicas", "dataset_brisk.csv")
}

#Crear carpeta de resultados si no existe
os.makedirs(os.path.join(base_dir, "Resultados_Clustering"), exist_ok=True)

#Funcion para calcular el indice de dunn
# (distancia minima entre clusters / diametro maximo de un cluster)
def dunn_index(X, labels):
    unique_labels = np.unique(labels)
    
    #Calcula diametros maximos de cada cluster
    diametros = []
    for label in unique_labels:
        cluster_pts = X[labels == label]
        if len(cluster_pts) > 1:
            # Distancias entre puntos del mismo cluster
            dists = np.sqrt(((cluster_pts[:, None, :] - cluster_pts[None, :, :]) ** 2).sum(axis=-1))
            diametros.append(np.max(dists))
        else:
            diametros.append(0.0)
    max_diametro = np.max(diametros)
    if max_diametro == 0.0:
        return 0.0
    
    # Calcula distancias minimas entre clusters
    min_dist_inter = np.inf
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            pts_i = X[labels == unique_labels[i]]
            pts_j = X[labels == unique_labels[j]]
            if len(pts_i) > 0 and len(pts_j) > 0:
                # Distancias entre puntos de los clusters i y j
                dists = np.sqrt(((pts_i[:, None, :] - pts_j[None, :, :]) ** 2).sum(axis=-1))
                min_dist_inter = min(min_dist_inter, np.min(dists))
                
    return min_dist_inter / max_diametro

resultados_ari = {}

# Procesar cada uno de los datasets
for nombre, ruta in datasets.items():
    if not os.path.exists(ruta):
        print(f"No se encontro el archivo: {ruta}")
        continue
        
    print(f"\n=== PROCESANDO DATASET: {nombre} ===")
    
    #Cargar el archivo csv
    df = pd.read_csv(ruta)
    X = df.iloc[:, :-1].values
    y_str = df.iloc[:, -1].values
    
    #Cambiar las etiquetas de texto a numeros
    le = LabelEncoder()
    y_true = le.fit_transform(y_str)
    
    # Escalar o normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Correlacion (solo para HU por el tamano de HOG y BRISK)
    if nombre == "HU (Momentos)":
        plt.figure(figsize=(6, 5))
        sns.heatmap(pd.DataFrame(X_scaled).corr(), annot=True, cmap="coolwarm")
        plt.title(f"Matriz de Correlacion - {nombre}")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "Resultados_Clustering", f"correlacion_{nombre.split()[0].lower()}.png"))
        plt.close()
        print("  * Grafico de correlacion guardado.")
        
    # Metodo del codo
    inercias = []
    k_valores = range(1, 11)
    for k in k_valores:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_scaled)
        inercias.append(kmeans_temp.inertia_)
        
    plt.figure(figsize=(6, 4))
    plt.plot(k_valores, inercias, marker="o", color="blue")
    plt.title(f"Metodo del Codo - {nombre}")
    plt.xlabel("Numero de clusters (k)")
    plt.ylabel("Inercia (Suma de distancias internas)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Resultados_Clustering", f"codo_{nombre.split()[0].lower()}.png"))
    plt.close()
    print("  * Grafico del metodo del codo guardado.")
    
    #Entrenar K-means con k=4 (4 etnias)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X_scaled)
    
    # Calcular las metricas de validacion
    silueta = silhouette_score(X_scaled, pred_labels)
    dunn = dunn_index(X_scaled, pred_labels)
    ari = adjusted_rand_score(y_true, pred_labels)
    ami = adjusted_mutual_info_score(y_true, pred_labels)
    nmi = normalized_mutual_info_score(y_true, pred_labels)
    
    #Mostrar resultados
    print("  * Resultados obtenidos:")
    print(f"    - Coeficiente de Silueta: {silueta:.4f}")
    print(f"    - Indice de Dunn: {dunn:.4f}")
    print(f"    - ARI: {ari:.4f}")
    print(f"    - AMI: {ami:.4f}")
    print(f"    - NMI: {nmi:.4f}")
    print("=" * 40)
    
    # Guardamos el resultado del ARI para compararlos al final
    resultados_ari[nombre] = ari

print("\n" + "=" * 50)
print("RESULTADO FINAL DEL CLUSTERING")
print("=" * 50)

# Buscar cuál tuvo el ARI más alto (el mejor ajustado a las etiquetas reales)
mejor_dataset = max(resultados_ari, key=resultados_ari.get)
mejor_puntuacion = resultados_ari[mejor_dataset]

print(f"-> Tras comparar los 3 métodos de extracción de características,")
print(f"-> el MEJOR dataset para agrupamiento (K-Means) es: {mejor_dataset}")
print(f"-> con una puntuación ARI (Adjusted Rand Score) de: {mejor_puntuacion:.4f}")
print("=" * 50)
print("\nClustering Terminado")
