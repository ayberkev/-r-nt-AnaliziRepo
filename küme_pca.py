import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

RAW_GITHUB_URL = "https://raw.githubusercontent.com/ayberkev/-r-nt-AnaliziRepo/main/TEIAS_Elektrik_Tuketim_Verileri_2015_2025.csv"

def load_and_preprocess_data():
    data_set = pd.read_csv(RAW_GITHUB_URL)
    
    data_set_attributes = data_set[[
        'Toplam_Üretim_MWh', 
        'Toplam_Tüketim_MWh', 
        'Yenilenebilir_Oranı', 
        'Pik_Talep_MW'
    ]]
    
    data_set_attributes = data_set_attributes.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_set_attributes)
    
    feature_names = data_set_attributes.columns
    
    return X_scaled, feature_names, data_set_attributes

def apply_pca_kmeans_analysis():
    
    X_scaled, feature_names, X_original = load_and_preprocess_data()
    
    n_components = 3
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA uygulandı: Boyutlar {X_scaled.shape[1]} -> {n_components} (Bileşen)")

    K = 3
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(X_pca)
    
    cluster_labels = kmeans.labels_

    print(f"K-Means uygulandı: K={K} küme oluşturuldu.")

    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    
    print(f"\nKümeleme Metrikleri")
    print(f"Ortalama Silhouette Skoru: {silhouette_avg:.4f}")
    
    X_original['Kume_Etiketi'] = cluster_labels
    
    cluster_summary = X_original.groupby('Kume_Etiketi')[feature_names].mean()
    
    print("\nKüme Karakteristikleri (Ortalama Değerler)")
    print(cluster_summary.sort_values(by='Toplam_Tüketim_MWh', ascending=False))
    
    print("\nKümeleme Yorumu (Önerilen)")
    print("Ortalama değerleri inceleyerek küme etiketlerini (0, 1, 2) sizin tanımlarınızla eşleştirebilirsiniz:")
    print("Küme 1 (Yüksek Tüketim/Pik): 'Toplam_Tüketim_MWh' ve 'Pik_Talep_MW' sütunlarında en yüksek değere sahip olan kümedir.")
    print("Küme 3 (Düşük Tüketim/Yüksek Yenilenebilir): 'Toplam_Tüketim_MWh' ve 'Pik_Talep_MW' sütunlarında en düşük değere sahip olan, 'Yenilenebilir_Oranı'nda yüksek değere sahip olan kümedir.")
    print("Küme 2 (Orta Düzey): Diğer iki küme arasında kalan değerlere sahip olan kümedir.")
    
    return X_original

if __name__ == '__main__':
    final_clustered_data = apply_pca_kmeans_analysis()