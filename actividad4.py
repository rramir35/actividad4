# actividad4_transporte_no_supervisado.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime


print("🚍 ACTIVIDAD 4 - APRENDIZAJE NO SUPERVISADO")
print("📅 Ejecutado:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# 1. CREAR DATASET (mismo de la actividad 3)

datos = {
    "estacion": ["Portal Norte", "Calle 100", "Calle 72", "Calle 26", 
                 "Avenida Jiménez", "Portal Sur"],
    "distancia_al_centro_km": [12.0, 8.5, 5.0, 3.5, 2.0, 15.0],
    "tiempo_promedio_min": [15, 10, 7, 6, 5, 18],
    "flujo_usuarios": [8000, 12000, 15000, 10000, 9000, 5000]
}

df = pd.DataFrame(datos)
print("\n📊 DATASET DE ESTACIONES:")
print(df)

# Guardar dataset
df.to_csv("dataset_estaciones.csv", index=False)
print("\n✅ Dataset guardado como 'dataset_estaciones.csv'")


# 2. PREPARAR DATOS PARA CLUSTERING

# Seleccionar características numéricas
X = df[["distancia_al_centro_km", "tiempo_promedio_min", "flujo_usuarios"]]

# Normalizar datos (importante para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 3. APLICAR K-MEANS (aprendizaje no supervisado)

# Probar con 3 clusters (grupos de estaciones similares)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)


print("🤖 RESULTADOS DEL CLUSTERING (K-Means)")

print("\n📊 ESTACIONES CON SU CLUSTER ASIGNADO:")
print(df[["estacion", "cluster"]])


# 4. INTERPRETAR LOS CLUSTERS


print("📋 INTERPRETACIÓN DE LOS GRUPOS:")


for i in range(3):
    estaciones_cluster = df[df["cluster"] == i]["estacion"].tolist()
    print(f"\n🔵 Cluster {i}: {', '.join(estaciones_cluster)}")

# Análisis por cluster
cluster_0 = df[df["cluster"] == 0]
cluster_1 = df[df["cluster"] == 1]
cluster_2 = df[df["cluster"] == 2]

print("\n📊 CARACTERÍSTICAS PROMEDIO POR CLUSTER:")
print(f"Cluster 0 - Distancia: {cluster_0['distancia_al_centro_km'].mean():.1f}km, Tiempo: {cluster_0['tiempo_promedio_min'].mean():.0f}min, Flujo: {cluster_0['flujo_usuarios'].mean():.0f}")
print(f"Cluster 1 - Distancia: {cluster_1['distancia_al_centro_km'].mean():.1f}km, Tiempo: {cluster_1['tiempo_promedio_min'].mean():.0f}min, Flujo: {cluster_1['flujo_usuarios'].mean():.0f}")
print(f"Cluster 2 - Distancia: {cluster_2['distancia_al_centro_km'].mean():.1f}km, Tiempo: {cluster_2['tiempo_promedio_min'].mean():.0f}min, Flujo: {cluster_2['flujo_usuarios'].mean():.0f}")


# 5. VISUALIZACIÓN

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_data = df[df["cluster"] == i]
    plt.scatter(cluster_data["distancia_al_centro_km"], 
                cluster_data["tiempo_promedio_min"], 
                c=colors[i], label=f'Cluster {i}', s=100)

plt.xlabel("Distancia al centro (km)")
plt.ylabel("Tiempo promedio (minutos)")
plt.title("Clustering de Estaciones de Transporte Masivo")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("clustering_estaciones.png", dpi=150, bbox_inches="tight")
print("\n📊 Visualización guardada como 'clustering_estaciones.png'")


# 6. PRUEBAS REALIZADAS


print("🔍 PRUEBAS REALIZADAS:")

print("✅ Prueba 1: Se aplicó K-Means con 3 clusters")
print("✅ Prueba 2: Se normalizaron los datos con StandardScaler")
print("✅ Prueba 3: Se visualizaron los grupos en un gráfico 2D")
print("✅ Prueba 4: Se interpretaron las características de cada cluster")


# RESUMEN FINAL


print("📋 RESUMEN DEL SISTEMA NO SUPERVISADO")

print("✅ Algoritmo: K-Means (Clustering)")
print("✅ Número de clusters: 3")
print("✅ Dataset: 6 estaciones de transporte")
print("✅ Características: distancia, tiempo, flujo de usuarios")
print("✅ Archivos generados:")
print("   - dataset_estaciones.csv")
print("   - clustering_estaciones.png")

print("🏁 FIN DE LA EJECUCIÓN")