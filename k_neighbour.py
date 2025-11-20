import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

RAW_GITHUB_URL = "https://raw.githubusercontent.com/ayberkev/-r-nt-AnaliziRepo/main/TEIAS_Elektrik_Tuketim_Verileri_2015_2025.csv"
DOSYA_ADI = "TEIAS_Elektrik_Tuketim_Verileri_2015_2025.csv"

def load_data():
    data_set = pd.read_csv(RAW_GITHUB_URL)
    data_set_classes = data_set['Önerilen_Enerji_Türü'].astype('category').cat.codes
    data_set_attributes = data_set[[
        'Toplam_Üretim_MWh', 
        'Toplam_Tüketim_MWh', 
        'Yenilenebilir_Oranı', 
        'Pik_Talep_MW'
    ]]
    n_classes = len(data_set['Önerilen_Enerji_Türü'].unique())
    return data_set_attributes, data_set_classes, n_classes

def preprocess_data(data_set_attributes):
    data_set_attributes = data_set_attributes.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    data_set_attributes_scaled = scaler.fit_transform(data_set_attributes)
    return data_set_attributes_scaled

def apply_pca(data_set_attributes_scaled, n_components=0.95, use_pca=False):
    if use_pca:
        pca = PCA(n_components=n_components)
        data_set_attributes_scaled_pca = pca.fit_transform(data_set_attributes_scaled) 
        return data_set_attributes_scaled_pca
    else:
        return data_set_attributes_scaled
    
def apply_lda(data_set_attributes_scaled, data_set_classes, n_classes, use_lda=False):
    n_components = min(data_set_attributes_scaled.shape[1], n_classes - 1)
    
    if use_lda:
        lda = LDA(n_components=n_components)
        data_set_attributes_scaled_lda = lda.fit_transform(data_set_attributes_scaled, data_set_classes) 
        return data_set_attributes_scaled_lda
    else:
        return data_set_attributes_scaled


def main_knn_csv(use_pca=False, use_lda=False):
    
    data_set_attributes, data_set_classes, n_classes = load_data()
    if data_set_attributes is None:
        return

    data_set_attributes_scaled = preprocess_data(data_set_attributes)
    
    if use_pca:
        X = apply_pca(data_set_attributes_scaled, n_components=0.95, use_pca=True)
    elif use_lda and n_classes > 1:
        X = apply_lda(data_set_attributes_scaled, data_set_classes, n_classes, use_lda=True)
    else:
        X = data_set_attributes_scaled

    y = data_set_classes.values
    
    original_df = pd.read_csv(RAW_GITHUB_URL)
    original_classes = original_df['Önerilen_Enerji_Türü'].astype('category').cat.categories.tolist()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    
    all_y_true_labels = []
    all_y_pred_labels = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_test = y[test_idx]
        
        knn_model.fit(X_train, y) # Tüm veri setini kullanmak yerine sadece train_idx'deki veriler kullanılmalıdır.
        y_pred = knn_model.predict(X_test)
        
        # Sayısal kodları orijinal metinsel etiketlere dönüştürme
        y_test_labels = [original_classes[i] for i in y_test]
        y_pred_labels = [original_classes[i] for i in y_pred]
        
        all_y_true_labels.extend(y_test_labels)
        all_y_pred_labels.extend(y_pred_labels)
        
    sonuc_df = pd.DataFrame({
        'Gercek_Etiket': all_y_true_labels,
        'KNN_Tahmini_Etiket': all_y_pred_labels
    })

    cikti_dosya_adi = 'knn_tahmin_sonuclari.csv'
    sonuc_df.to_csv(cikti_dosya_adi, index=False, encoding='utf-8')
    
    print(f"Tahmin sonuçları başarıyla '{cikti_dosya_adi}' dosyasına kaydedildi.")

if __name__ == '__main__':
    main_knn_csv(use_pca=False, use_lda=False)