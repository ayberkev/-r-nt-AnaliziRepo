import pandas as pd
import os


dosya_adi = 'https://raw.githubusercontent.com/ayberkev/-r-nt-AnaliziRepo/main/TEIAS_Elektrik_Tuketim_Verileri_2015_2025.csv'


kayit_klasoru = r'C:\Users\UserPC\Desktop\py kod'


cikti_dosya_adi = os.path.join(kayit_klasoru, 'bolgesel_ortalama_analiz.csv')

try:
    
    df = pd.read_csv(dosya_adi)

    print("Veri seti başarıyla okundu.")

    
   
    gruplama_sutunu = 'Bölge'
    
   
    ortalama_sutunlar = [
        'Toplam_Üretim_MWh', 
        'Toplam_Tüketim_MWh', 
        'Yenilenebilir_Oranı', 
        'Pik_Talep_MW'
    ]
    
 
    mode_sutunu = 'Önerilen_Enerji_Türü'

  
    ortalama_sonuclar = df.groupby(gruplama_sutunu)[ortalama_sutunlar].mean()
    
   
    mode_sonuclari = df.groupby(gruplama_sutunu)[mode_sutunu].apply(lambda x: x.mode()[0]).rename('En_Sik_Onerilen_Enerji_Turu')

 
    analiz_sonucu = pd.merge(ortalama_sonuclar, mode_sonuclari, on=gruplama_sutunu)
    
 
    print("Bölgesel Ortalama Analiz Sonucu ")

    analiz_sonucu['Yenilenebilir_Oranı'] = (analiz_sonucu['Yenilenebilir_Oranı'] * 100).round(2).astype(str) + '%'
    print(analiz_sonucu)
  

  
    analiz_sonucu.to_csv(cikti_dosya_adi, encoding='utf-8')

    print(f"Kayıt Tamamlandı ###")
    print(f"Analiz sonuçları başarıyla **'{cikti_dosya_adi}'** dosyasına kaydedildi.")


except FileNotFoundError:
    print(f"HATA: '{dosya_adi}' dosyası bulunamadı. Lütfen dosyanın yüklü olduğundan veya adının doğru olduğundan emin olun.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")