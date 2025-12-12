"""
Yapay Zeka Destekli EEG Teşhis Sistemi (Demo)
DEHB (Dikkat Eksikliği ve Hiperaktivite Bozukluğu) Teşhisi

⚠️ UYARI: Bu sistem sadece eğitim ve demo amaçlıdır.
Gerçek tıbbi teşhis için kullanılamaz!
"""

import mne
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

class EEGDiagnosticAI:
    """EEG Teşhis Yapay Zekası"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def extract_features(self, raw):
        """EEG verisinden özellikler çıkar"""
        print("  → Özellikler çıkarılıyor...")
        
        features = {}
        
        # 1. Frekans bantları güç özellikleri
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        # Güç spektral yoğunluğu hesapla
        spectrum = raw.compute_psd(fmin=0.5, fmax=40, method='welch', 
                                   n_fft=2048, n_overlap=512)
        
        # Her frekans bandı için ortalama güç
        for band_name, (fmin, fmax) in bands.items():
            band_power = spectrum.get_data(fmin=fmin, fmax=fmax).mean()
            features[f'power_{band_name}'] = band_power
        
        # 2. Theta/Beta oranı (DEHB için önemli)
        theta_power = features['power_theta']
        beta_power = features['power_beta']
        features['theta_beta_ratio'] = theta_power / beta_power if beta_power > 0 else 0
        
        # 3. Alpha peak frekansı
        alpha_spectrum = spectrum.get_data(fmin=8, fmax=13)
        alpha_freqs = spectrum.freqs[(spectrum.freqs >= 8) & (spectrum.freqs <= 13)]
        if len(alpha_freqs) > 0:
            alpha_peak_idx = np.argmax(alpha_spectrum.mean(axis=0))
            features['alpha_peak_freq'] = alpha_freqs[alpha_peak_idx]
        else:
            features['alpha_peak_freq'] = 10.5  # Varsayılan
        
        # 4. Toplam güç
        features['total_power'] = spectrum.get_data(fmin=0.5, fmax=40).mean()
        
        # 5. Kanal bazlı özellikler (ön kanallar)
        eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
        
        # Varsayılan değerler
        features['frontal_alpha'] = 0.0
        features['central_beta'] = 0.0
        features['parietal_alpha'] = 0.0
        
        if len(eeg_channels) >= 3:
            # Frontal, Central, Parietal kanalları seç
            frontal_chs = [ch for ch in eeg_channels if any(x in ch for x in ['Fp', 'Fz', 'F3', 'F4'])]
            central_chs = [ch for ch in eeg_channels if any(x in ch for x in ['Cz', 'C3', 'C4'])]
            parietal_chs = [ch for ch in eeg_channels if any(x in ch for x in ['Pz', 'P3', 'P4'])]
            
            try:
                if frontal_chs:
                    frontal_spectrum = spectrum.copy().pick(frontal_chs[:3])
                    features['frontal_alpha'] = frontal_spectrum.get_data(fmin=8, fmax=13).mean()
            except:
                features['frontal_alpha'] = features['power_alpha'] * 0.8  # Fallback
            
            try:
                if central_chs:
                    central_spectrum = spectrum.copy().pick(central_chs[:3])
                    features['central_beta'] = central_spectrum.get_data(fmin=13, fmax=30).mean()
            except:
                features['central_beta'] = features['power_beta'] * 0.8  # Fallback
            
            try:
                if parietal_chs:
                    parietal_spectrum = spectrum.copy().pick(parietal_chs[:3])
                    features['parietal_alpha'] = parietal_spectrum.get_data(fmin=8, fmax=13).mean()
            except:
                features['parietal_alpha'] = features['power_alpha'] * 0.9  # Fallback
        
        # 6. Variability (değişkenlik)
        data = raw.get_data()
        features['signal_variance'] = np.var(data)
        features['signal_mean'] = np.mean(np.abs(data))
        
        # 7. Asimetri (frontal asymmetry DEHB'de önemli)
        features['frontal_asymmetry'] = 0.0  # Varsayılan değer
        if len(eeg_channels) >= 2:
            left_chs = [ch for ch in eeg_channels if '3' in ch or 'Fp1' in ch]
            right_chs = [ch for ch in eeg_channels if '4' in ch or 'Fp2' in ch]
            if left_chs and right_chs:
                try:
                    left_spectrum = spectrum.copy().pick(left_chs[:2])
                    right_spectrum = spectrum.copy().pick(right_chs[:2])
                    left_alpha = left_spectrum.get_data(fmin=8, fmax=13).mean()
                    right_alpha = right_spectrum.get_data(fmin=8, fmax=13).mean()
                    features['frontal_asymmetry'] = (right_alpha - left_alpha) / (right_alpha + left_alpha + 1e-10)
                except:
                    features['frontal_asymmetry'] = 0.0  # Fallback
        
        return features
    
    def generate_synthetic_data(self, n_samples=200):
        """Sentetik eğitim verisi oluştur (demo amaçlı)"""
        print("\n" + "="*60)
        print("SENTETİK EĞİTİM VERİSİ OLUŞTURULUYOR")
        print("="*60)
        print("\n⚠️  UYARI: Bu sentetik veridir, gerçek teşhis için kullanılamaz!")
        print("   Gerçek bir sistem için klinik veri seti gerekir.\n")
        
        np.random.seed(42)
        features_list = []
        labels = []
        
        # Özellik isimleri
        feature_names = [
            'power_delta', 'power_theta', 'power_alpha', 'power_beta', 'power_gamma',
            'theta_beta_ratio', 'alpha_peak_freq', 'total_power',
            'frontal_alpha', 'central_beta', 'parietal_alpha',
            'signal_variance', 'signal_mean', 'frontal_asymmetry'
        ]
        
        for i in range(n_samples):
            # Sağlıklı kontrol grubu (%50)
            if i < n_samples // 2:
                label = 0  # Sağlıklı
                # Normal EEG özellikleri
                features = {
                    'power_delta': np.random.normal(2.5, 0.5),
                    'power_theta': np.random.normal(3.0, 0.6),
                    'power_alpha': np.random.normal(4.5, 0.8),
                    'power_beta': np.random.normal(3.5, 0.7),
                    'power_gamma': np.random.normal(2.0, 0.4),
                    'theta_beta_ratio': np.random.normal(0.85, 0.15),  # Normal oran
                    'alpha_peak_freq': np.random.normal(10.5, 1.0),
                    'total_power': np.random.normal(15.5, 2.0),
                    'frontal_alpha': np.random.normal(4.0, 0.7),
                    'central_beta': np.random.normal(3.2, 0.6),
                    'parietal_alpha': np.random.normal(5.0, 0.9),
                    'signal_variance': np.random.normal(0.5, 0.1),
                    'signal_mean': np.random.normal(2.0, 0.3),
                    'frontal_asymmetry': np.random.normal(0.0, 0.1)  # Simetrik
                }
            else:
                # DEHB grubu (%50)
                label = 1  # DEHB
                # DEHB karakteristik özellikleri
                features = {
                    'power_delta': np.random.normal(2.8, 0.6),
                    'power_theta': np.random.normal(4.5, 0.8),  # Artmış theta
                    'power_alpha': np.random.normal(3.5, 0.7),   # Azalmış alpha
                    'power_beta': np.random.normal(2.8, 0.6),   # Azalmış beta
                    'power_gamma': np.random.normal(2.2, 0.5),
                    'theta_beta_ratio': np.random.normal(1.6, 0.3),  # Yüksek oran (DEHB işareti)
                    'alpha_peak_freq': np.random.normal(9.5, 1.2),   # Düşük peak
                    'total_power': np.random.normal(16.8, 2.5),
                    'frontal_alpha': np.random.normal(3.0, 0.6),      # Azalmış
                    'central_beta': np.random.normal(2.5, 0.5),     # Azalmış
                    'parietal_alpha': np.random.normal(4.0, 0.8),
                    'signal_variance': np.random.normal(0.7, 0.15),  # Artmış değişkenlik
                    'signal_mean': np.random.normal(2.3, 0.4),
                    'frontal_asymmetry': np.random.normal(0.15, 0.2)  # Asimetri
                }
            
            features_list.append([features[name] for name in feature_names])
            labels.append(label)
        
        X = np.array(features_list)
        y = np.array(labels)
        
        self.feature_names = feature_names
        
        print(f"✓ {n_samples} sentetik örnek oluşturuldu")
        print(f"  - Sağlıklı: {np.sum(y == 0)} örnek")
        print(f"  - DEHB: {np.sum(y == 1)} örnek")
        
        return X, y
    
    def train(self, X, y):
        """Modeli eğit"""
        print("\n" + "="*60)
        print("MODEL EĞİTİMİ")
        print("="*60)
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Ölçeklendir
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modeli eğit
        print("\n1. Model eğitiliyor...")
        self.model.fit(X_train_scaled, y_train)
        
        # Test
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Model eğitildi!")
        print(f"  - Test doğruluğu: {accuracy*100:.2f}%")
        print(f"\n2. Sınıflandırma raporu:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Sağlıklı', 'DEHB']))
        
        self.is_trained = True
        
        # Önemli özellikleri göster
        self.plot_feature_importance()
        
        return accuracy
    
    def plot_feature_importance(self):
        """Özellik önemini görselleştir"""
        if not self.is_trained:
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Özellik Önemi (Feature Importance)', fontsize=14, fontweight='bold')
        plt.barh(range(len(importances)), importances[indices])
        plt.yticks(range(len(importances)), 
                  [self.feature_names[i] for i in indices])
        plt.xlabel('Önem Skoru', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def diagnose(self, raw):
        """EEG verisini analiz et ve teşhis yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi! Önce train() metodunu çağırın.")
        
        print("\n" + "="*60)
        print("TEŞHİS ANALİZİ")
        print("="*60)
        
        # Özellikleri çıkar
        features = self.extract_features(raw)
        
        # DataFrame'e dönüştür
        feature_vector = np.array([[features[name] for name in self.feature_names]])
        
        # Ölçeklendir
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Tahmin yap
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        
        # Sonuçları göster
        print("\n" + "="*60)
        print("TEŞHİS SONUÇLARI")
        print("="*60)
        
        if prediction == 1:
            diagnosis = "Dikkat Eksikliği ve Hiperaktivite Bozukluğu (DEHB)"
            probability = probabilities[1] * 100
        else:
            diagnosis = "Sağlıklı (Normal EEG Paterni)"
            probability = probabilities[0] * 100
        
        print(f"\n🔍 Teşhis: {diagnosis}")
        print(f"📊 Olasılık: %{probability:.2f}")
        
        print(f"\n📈 Detaylı Olasılıklar:")
        print(f"  - Sağlıklı: %{probabilities[0]*100:.2f}")
        print(f"  - DEHB: %{probabilities[1]*100:.2f}")
        
        # Önemli özellikleri göster
        print(f"\n🔬 Analiz Edilen Özellikler:")
        print(f"  - Theta/Beta Oranı: {features['theta_beta_ratio']:.3f}")
        if features['theta_beta_ratio'] > 1.3:
            print("    ⚠️  Yüksek oran (DEHB işareti olabilir)")
        print(f"  - Alpha Peak Frekansı: {features['alpha_peak_freq']:.2f} Hz")
        print(f"  - Frontal Asimetri: {features['frontal_asymmetry']:.3f}")
        print(f"  - Toplam Güç: {features['total_power']:.2f}")
        
        # Uyarı
        print("\n" + "="*60)
        print("⚠️  ÖNEMLİ UYARI")
        print("="*60)
        print("Bu sistem sadece eğitim ve demo amaçlıdır.")
        print("Gerçek tıbbi teşhis için kullanılamaz!")
        print("Tıbbi teşhis için mutlaka uzman doktora başvurun.")
        print("="*60)
        
        return {
            'diagnosis': diagnosis,
            'probability': probability,
            'probabilities': probabilities,
            'features': features
        }

def load_sample_eeg():
    """Örnek EEG verisini yükle"""
    print("="*60)
    print("EEG VERİSİ YÜKLEME")
    print("="*60)
    
    sample_data_folder = mne.datasets.sample.data_path()
    data_path = os.path.join(sample_data_folder, 'MEG', 'sample')
    raw_fname = os.path.join(data_path, 'sample_audvis_raw.fif')
    
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.pick_types(eeg=True, stim=True)
    raw.filter(l_freq=0.1, h_freq=40, method='iir', picks='eeg', verbose=False)
    
    print(f"\n✓ Veri yüklendi!")
    print(f"  - Kanal sayısı: {len(raw.ch_names)}")
    print(f"  - Örnekleme frekansı: {raw.info['sfreq']} Hz")
    print(f"  - Veri süresi: {raw.times[-1]:.2f} saniye")
    
    return raw

def main():
    """Ana fonksiyon"""
    print("\n" + "="*70)
    print("YAPAY ZEKA DESTEKLİ EEG TEŞHİS SİSTEMİ (DEMO)")
    print("="*70)
    
    try:
        # AI sistemi oluştur
        ai_system = EEGDiagnosticAI()
        
        # Sentetik veri oluştur ve modeli eğit
        X, y = ai_system.generate_synthetic_data(n_samples=200)
        accuracy = ai_system.train(X, y)
        
        # Örnek EEG verisini yükle
        print("\n" + "="*60)
        raw = load_sample_eeg()
        
        # Teşhis yap
        result = ai_system.diagnose(raw)
        
        print("\n✅ Analiz tamamlandı!")
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

