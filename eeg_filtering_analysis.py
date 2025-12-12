"""
MNE-Python ile EEG Verisi Filtreleme ve Görselleştirme
Ham veri ve filtrelenmiş veriyi yan yana karşılaştırır
"""

import mne
import os
import matplotlib.pyplot as plt
import numpy as np

def load_eeg_data():
    """EEG verisini yükle"""
    print("="*60)
    print("EEG VERİSİ YÜKLEME")
    print("="*60)
    
    # Örnek veri setini indir ve yükle
    print("\n1. Örnek veri seti indiriliyor...")
    sample_data_folder = mne.datasets.sample.data_path()
    data_path = os.path.join(sample_data_folder, 'MEG', 'sample')
    raw_fname = os.path.join(data_path, 'sample_audvis_raw.fif')
    
    print("2. Veri dosyası okunuyor...")
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    
    # Sadece EEG kanallarını seç
    print("3. EEG kanalları seçiliyor...")
    raw.pick_types(eeg=True, stim=True)
    
    print("\n✓ Veri başarıyla yüklendi!")
    print(f"\nVeri Bilgileri:")
    print(f"  - Kanal sayısı: {len(raw.ch_names)}")
    print(f"  - Örnekleme frekansı: {raw.info['sfreq']} Hz")
    print(f"  - Veri süresi: {raw.times[-1]:.2f} saniye")
    print(f"  - EEG kanalları: {[ch for ch in raw.ch_names if 'EEG' in ch][:5]}...")
    
    return raw

def apply_filters(raw, highpass=0.1, lowpass=40):
    """Veriye filtreleme uygula"""
    print("\n" + "="*60)
    print("FİLTRELEME İŞLEMİ")
    print("="*60)
    
    # Orijinal veriyi kopyala (ham veri korunmalı)
    raw_filtered = raw.copy()
    
    print(f"\n1. Yüksek geçiren filtre uygulanıyor (High-pass: {highpass} Hz)...")
    print("   → Düşük frekanslı gürültüleri (örn: DC offset, drift) temizler")
    raw_filtered.filter(l_freq=highpass, h_freq=None, 
                        method='iir', iir_params=None, 
                        picks='eeg', verbose=False)
    
    print(f"2. Düşük geçiren filtre uygulanıyor (Low-pass: {lowpass} Hz)...")
    print("   → Yüksek frekanslı gürültüleri (örn: kas aktivitesi, 50 Hz şebeke gürültüsü) temizler")
    raw_filtered.filter(l_freq=None, h_freq=lowpass, 
                        method='iir', iir_params=None, 
                        picks='eeg', verbose=False)
    
    print("\n✓ Filtreleme tamamlandı!")
    print(f"  - Filtrelenmiş frekans aralığı: {highpass} - {lowpass} Hz")
    
    return raw_filtered

def visualize_comparison(raw, raw_filtered, duration=10, n_channels=20):
    """Ham ve filtrelenmiş veriyi yan yana görselleştir"""
    print("\n" + "="*60)
    print("GÖRSELLEŞTİRME")
    print("="*60)
    
    print(f"\nİlk {duration} saniye görselleştiriliyor...")
    print("  → Sol: Ham veri (filtrelenmemiş)")
    print("  → Sağ: Filtrelenmiş veri")
    
    # İki ayrı pencere aç
    fig1 = plt.figure(figsize=(16, 10))
    fig1.suptitle('HAM EEG VERİSİ (Filtrelenmemiş)', fontsize=16, fontweight='bold')
    
    # Ham veri
    raw.plot(duration=duration, n_channels=n_channels, scalings='auto', 
             show=False, title='Ham Veri')
    
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle('FİLTRELENMİŞ EEG VERİSİ (0.1-40 Hz)', fontsize=16, fontweight='bold')
    
    # Filtrelenmiş veri
    raw_filtered.plot(duration=duration, n_channels=n_channels, scalings='auto',
                      show=False, title='Filtrelenmiş Veri')
    
    plt.show()

def plot_side_by_side_comparison(raw, raw_filtered, channels=None, duration=5):
    """Belirli kanalları yan yana karşılaştır"""
    if channels is None:
        # İlk 3 EEG kanalını seç
        eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
        channels = eeg_channels[:3]
    
    print(f"\n{len(channels)} kanal için detaylı karşılaştırma...")
    
    # Veriyi çıkar
    data_raw, times = raw.get_data(channels, return_times=True)
    data_filtered, _ = raw_filtered.get_data(channels, return_times=True)
    
    # İlk N saniyeyi al
    end_idx = int(duration * raw.info['sfreq'])
    data_raw = data_raw[:, :end_idx]
    data_filtered = data_filtered[:, :end_idx]
    times = times[:end_idx]
    
    # Grafik çiz
    fig, axes = plt.subplots(len(channels), 2, figsize=(16, 3*len(channels)), 
                             sharex=True, sharey='row')
    if len(channels) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Ham vs Filtrelenmiş EEG Karşılaştırması (İlk {duration} saniye)', 
                 fontsize=16, fontweight='bold')
    
    for i, ch in enumerate(channels):
        # Sol: Ham veri
        axes[i, 0].plot(times, data_raw[i], 'b-', linewidth=0.8, alpha=0.7)
        axes[i, 0].set_ylabel(f'{ch}\n(µV)', fontsize=11)
        axes[i, 0].set_title('HAM VERİ', fontsize=12, fontweight='bold', color='blue')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Sağ: Filtrelenmiş veri
        axes[i, 1].plot(times, data_filtered[i], 'r-', linewidth=0.8, alpha=0.7)
        axes[i, 1].set_ylabel(f'{ch}\n(µV)', fontsize=11)
        axes[i, 1].set_title('FİLTRELENMİŞ VERİ (0.1-40 Hz)', fontsize=12, fontweight='bold', color='red')
        axes[i, 1].grid(True, alpha=0.3)
        
        if i == len(channels) - 1:
            axes[i, 0].set_xlabel('Zaman (s)', fontsize=12)
            axes[i, 1].set_xlabel('Zaman (s)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_frequency_comparison(raw, raw_filtered, fmin=0.1, fmax=50):
    """Frekans spektrumunu karşılaştır"""
    print("\nFrekans spektrumu karşılaştırması...")
    
    # Güç spektral yoğunluğu hesapla
    print("  → Ham veri spektrumu hesaplanıyor...")
    spectrum_raw = raw.compute_psd(fmin=fmin, fmax=fmax, method='welch', 
                                    n_fft=2048, n_overlap=512)
    
    print("  → Filtrelenmiş veri spektrumu hesaplanıyor...")
    spectrum_filtered = raw_filtered.compute_psd(fmin=fmin, fmax=fmax, method='welch',
                                                  n_fft=2048, n_overlap=512)
    
    # Grafik çiz
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sol: Ham veri spektrumu
    spectrum_raw.plot(axes=axes[0], show=False, average=True)
    axes[0].set_title('HAM VERİ - Güç Spektrumu', fontsize=14, fontweight='bold')
    axes[0].axvline(x=0.1, color='green', linestyle='--', linewidth=2, label='0.1 Hz (High-pass)')
    axes[0].axvline(x=40, color='red', linestyle='--', linewidth=2, label='40 Hz (Low-pass)')
    axes[0].legend()
    
    # Sağ: Filtrelenmiş veri spektrumu
    spectrum_filtered.plot(axes=axes[1], show=False, average=True)
    axes[1].set_title('FİLTRELENMİŞ VERİ - Güç Spektrumu (0.1-40 Hz)', 
                      fontsize=14, fontweight='bold')
    axes[1].axvline(x=0.1, color='green', linestyle='--', linewidth=2, label='0.1 Hz')
    axes[1].axvline(x=40, color='red', linestyle='--', linewidth=2, label='40 Hz')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def print_filter_info(highpass, lowpass):
    """Filtre bilgilerini yazdır"""
    print("\n" + "="*60)
    print("FİLTRE BİLGİLERİ")
    print("="*60)
    print(f"\nYüksek Geçiren Filtre (High-pass): {highpass} Hz")
    print("  → Ne yapar: Düşük frekanslı sinyalleri kaldırır")
    print("  → Neden: DC offset, yavaş drift, göz hareketi artefaktlarını temizler")
    print("  → Etkisi: 0.1 Hz altındaki tüm sinyaller zayıflatılır")
    
    print(f"\nDüşük Geçiren Filtre (Low-pass): {lowpass} Hz")
    print("  → Ne yapar: Yüksek frekanslı sinyalleri kaldırır")
    print("  → Neden: Kas aktivitesi, 50 Hz şebeke gürültüsü, yüksek frekanslı artefaktları temizler")
    print("  → Etkisi: 40 Hz üzerindeki tüm sinyaller zayıflatılır")
    
    print(f"\nSonuç: Sadece {highpass}-{lowpass} Hz aralığındaki sinyaller korunur")
    print("  → Bu aralık EEG analizi için standarttır")
    print("  → Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz) bantları korunur")

def main():
    """Ana fonksiyon"""
    print("\n" + "="*70)
    print("MNE-PYTHON İLE EEG VERİSİ FİLTRELEME VE ANALİZİ")
    print("="*70)
    
    # Filtre parametreleri
    HIGH_PASS = 0.1  # Hz
    LOW_PASS = 40    # Hz
    
    try:
        # 1. Veriyi yükle
        raw = load_eeg_data()
        
        # 2. Filtre bilgilerini göster
        print_filter_info(HIGH_PASS, LOW_PASS)
        
        # 3. Filtreleme uygula
        raw_filtered = apply_filters(raw, highpass=HIGH_PASS, lowpass=LOW_PASS)
        
        # 4. Menü
        while True:
            print("\n" + "="*60)
            print("GÖRSELLEŞTİRME SEÇENEKLERİ:")
            print("1. Ham ve filtrelenmiş veriyi yan yana görselleştir (interaktif)")
            print("2. Belirli kanalları yan yana karşılaştır (detaylı)")
            print("3. Frekans spektrumunu karşılaştır")
            print("4. Tüm görselleştirmeleri çalıştır")
            print("0. Çıkış")
            print("="*60)
            
            choice = input("Seçiminiz (0-4): ").strip()
            
            if choice == '0':
                print("\nÇıkılıyor...")
                break
            elif choice == '1':
                visualize_comparison(raw, raw_filtered, duration=10, n_channels=20)
            elif choice == '2':
                plot_side_by_side_comparison(raw, raw_filtered, duration=5)
            elif choice == '3':
                plot_frequency_comparison(raw, raw_filtered)
            elif choice == '4':
                visualize_comparison(raw, raw_filtered, duration=10, n_channels=20)
                plot_side_by_side_comparison(raw, raw_filtered, duration=5)
                plot_frequency_comparison(raw, raw_filtered)
            else:
                print("Geçersiz seçim! Lütfen 0-4 arası bir sayı girin.")
    
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        print("\nOlası çözümler:")
        print("1. MNE-Python kurulu mu kontrol edin: pip install mne")
        print("2. İnternet bağlantınızı kontrol edin (veri seti indirilecek)")
        print("3. Yeterli disk alanı olduğundan emin olun (~100 MB)")

if __name__ == '__main__':
    main()

