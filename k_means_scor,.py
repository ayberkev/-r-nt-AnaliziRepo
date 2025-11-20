import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

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
    return X_scaled

def calculate_silhouette_scores(min_k, max_k):
    X = load_and_preprocess_data()
    
    k_range = range(min_k, max_k + 1)
    silhouette_scores = {}
    
    for k in k_range:
        if k == 1:
            continue
            
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores[k] = score
        
    results_df = pd.DataFrame(list(silhouette_scores.items()), columns=['Kume_Sayisi_K', 'Ortalama_Silhouette_Skoru'])
    return results_df

def plot_silhouette_results(results_df):
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['Kume_Sayisi_K'], results_df['Ortalama_Silhouette_Skoru'], marker='o', linestyle='--')
    plt.title('K-Means Kümeleme: Silhouette Skoru Yöntemi')
    plt.xlabel('Küme Sayısı (K)')
    plt.ylabel('Ortalama Silhouette Skoru')
    plt.grid(True)
    plt.xticks(results_df['Kume_Sayisi_K'])

    best_k = results_df.loc[results_df['Ortalama_Silhouette_Skoru'].idxmax()]
    plt.scatter(best_k['Kume_Sayisi_K'], best_k['Ortalama_Silhouette_Skoru'], color='red', s=100)
    plt.text(best_k['Kume_Sayisi_K'], best_k['Ortalama_Silhouette_Skoru'] + 0.005, f"En İyi K={int(best_k['Kume_Sayisi_K'])}", color='red')
    plt.show()

if __name__ == '__main__':
    min_k_value = 2
    max_k_value = 7
    
    silhouette_results = calculate_silhouette_scores(min_k_value, max_k_value)
    
    print(" Silhouette Skoru Analiz Sonuçları (K=2 ile K=7 Arası)")
    print(silhouette_results)
    
    plot_silhouette_results(silhouette_results)