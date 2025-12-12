"""
EEG Verisi Analizi Örnekleri
MNE-Python ile EEG verisi üzerinde temel analizler
"""

import mne
import os
import matplotlib.pyplot as plt
import numpy as np

def load_sample_eeg():
    """Örnek EEG verisini yükle"""
    print("Örnek veri seti indiriliyor...")
    sample_data_folder = mne.datasets.sample.data_path()
    data_path = os.path.join(sample_data_folder, 'MEG', 'sample')
    raw_fname = os.path.join(data_path, 'sample_audvis_raw.fif')
    
    # Veriyi oku
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    
    # Sadece EEG kanallarını seç
    raw.pick_types(eeg=True, stim=True)
    
    print("✓ Veri başarıyla yüklendi!")
    print(f"  - Kanal sayısı: {len(raw.ch_names)}")
    print(f"  - Örnekleme frekansı: {raw.info['sfreq']} Hz")
    print(f"  - Veri süresi: {raw.times[-1]:.2f} saniye")
    
    return raw

def visualize_raw_eeg(raw, duration=5, n_channels=30):
    """Ham EEG verisini görselleştir"""
    print(f"\nİlk {duration} saniye görselleştiriliyor...")
    raw.plot(duration=duration, n_channels=n_channels, scalings='auto')
    plt.show()

def plot_power_spectrum(raw, fmin=1, fmax=50):
    """Güç spektrumu çiz"""
    print("\nGüç spektrumu hesaplanıyor...")
    
    # Güç spektral yoğunluğu hesapla
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, method='welch', 
                               n_fft=2048, n_overlap=512)
    
    # Grafik çiz
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum.plot(axes=ax, show=False, average=True)
    ax.set_title('EEG Güç Spektrumu', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frekans (Hz)', fontsize=12)
    ax.set_ylabel('Güç (dB)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return spectrum

def plot_channel_comparison(raw, channels=None, duration=10):
    """Belirli kanalları karşılaştır"""
    if channels is None:
        # İlk 5 EEG kanalını seç
        eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
        channels = eeg_channels[:5]
    
    print(f"\n{len(channels)} kanal karşılaştırılıyor...")
    
    # Veriyi çıkar
    data, times = raw.get_data(channels, return_times=True)
    
    # İlk 10 saniyeyi al
    end_idx = int(duration * raw.info['sfreq'])
    data = data[:, :end_idx]
    times = times[:end_idx]
    
    # Grafik çiz
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2*len(channels)), 
                             sharex=True)
    if len(channels) == 1:
        axes = [axes]
    
    for i, (ch, ax) in enumerate(zip(channels, axes)):
        ax.plot(times, data[i], linewidth=0.5)
        ax.set_ylabel(f'{ch}\n(µV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == len(channels) - 1:
            ax.set_xlabel('Zaman (s)', fontsize=12)
    
    plt.suptitle(f'EEG Kanalları Karşılaştırması (İlk {duration} saniye)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def analyze_frequency_bands(raw):
    """Frekans bantlarını analiz et"""
    print("\nFrekans bantları analiz ediliyor...")
    
    # Frekans bantları tanımla
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-50 Hz)': (30, 50)
    }
    
    # Güç spektrumu hesapla
    spectrum = raw.compute_psd(fmin=0.5, fmax=50, method='welch')
    
    # Her bant için ortalama güç hesapla
    band_powers = {}
    for band_name, (fmin, fmax) in bands.items():
        band_power = spectrum.get_data(fmin=fmin, fmax=fmax).mean()
        band_powers[band_name] = band_power
    
    # Grafik çiz
    fig, ax = plt.subplots(figsize=(12, 6))
    bands_list = list(bands.keys())
    powers = [band_powers[band] for band in bands_list]
    
    bars = ax.bar(bands_list, powers, color=['#4CAF50', '#2196F3', '#FF9800', 
                                              '#f44336', '#9C27B0'], alpha=0.7)
    ax.set_ylabel('Ortalama Güç (dB)', fontsize=12)
    ax.set_title('Frekans Bantlarına Göre Güç Dağılımı', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Değerleri çubukların üzerine yaz
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{power:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return band_powers

def main():
    """Ana fonksiyon"""
    print("="*60)
    print("EEG VERİSİ ANALİZİ")
    print("="*60)
    
    # Veriyi yükle
    raw = load_sample_eeg()
    
    # Menü
    while True:
        print("\n" + "="*60)
        print("Yapılacak Analiz:")
        print("1. Ham EEG verisini görselleştir (5 saniye)")
        print("2. Güç spektrumu çiz")
        print("3. Kanal karşılaştırması")
        print("4. Frekans bantları analizi")
        print("5. Tüm analizleri çalıştır")
        print("0. Çıkış")
        print("="*60)
        
        choice = input("Seçiminiz (0-5): ").strip()
        
        if choice == '0':
            print("Çıkılıyor...")
            break
        elif choice == '1':
            visualize_raw_eeg(raw, duration=5, n_channels=30)
        elif choice == '2':
            plot_power_spectrum(raw)
        elif choice == '3':
            plot_channel_comparison(raw, duration=10)
        elif choice == '4':
            analyze_frequency_bands(raw)
        elif choice == '5':
            visualize_raw_eeg(raw, duration=5, n_channels=30)
            plot_power_spectrum(raw)
            plot_channel_comparison(raw, duration=10)
            analyze_frequency_bands(raw)
        else:
            print("Geçersiz seçim! Lütfen 0-5 arası bir sayı girin.")

if __name__ == '__main__':
    main()

