import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    
    return X_scaled

if __name__ == '__main__':
    X_preprocessed = load_and_preprocess_data()
    
    print("Ön İşlenmiş Veri Matrisi (X_scaled) Başarıyla Oluşturuldu:")
    print("Şekil:", X_preprocessed.shape)
    print("İlk 5 Satır:")
    print(X_preprocessed[:5])