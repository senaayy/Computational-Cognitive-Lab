"""
EEG Epoklama ve ERP Analizi
Oddball Paradigması: Standart vs Oddball uyaranlarının karşılaştırması
P300 dalgası tespiti
"""

import mne
import os
import matplotlib.pyplot as plt
import numpy as np

def load_and_filter_data():
    """EEG verisini yükle ve filtrele"""
    print("="*60)
    print("VERİ YÜKLEME VE FİLTRELEME")
    print("="*60)
    
    # Örnek veri setini yükle
    print("\n1. Örnek veri seti yükleniyor...")
    sample_data_folder = mne.datasets.sample.data_path()
    data_path = os.path.join(sample_data_folder, 'MEG', 'sample')
    raw_fname = os.path.join(data_path, 'sample_audvis_raw.fif')
    
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    
    # Sadece EEG kanallarını seç
    print("2. EEG kanalları seçiliyor...")
    raw.pick_types(eeg=True, stim=True)
    
    # Filtreleme uygula
    print("3. Filtreleme uygulanıyor (0.1-40 Hz)...")
    raw.filter(l_freq=0.1, h_freq=40, method='iir', picks='eeg', verbose=False)
    
    print("\n✓ Veri hazır!")
    print(f"  - Kanal sayısı: {len(raw.ch_names)}")
    print(f"  - Örnekleme frekansı: {raw.info['sfreq']} Hz")
    print(f"  - Veri süresi: {raw.times[-1]:.2f} saniye")
    
    return raw

def find_events(raw):
    """Uyaran işaretlerini (events) bul"""
    print("\n" + "="*60)
    print("OLAY TESPİTİ (Event Detection)")
    print("="*60)
    
    # Events bul
    print("\n1. Uyaran işaretleri aranıyor...")
    events = mne.find_events(raw, stim_channel='STI 014', min_duration=0.002)
    
    print(f"\n✓ {len(events)} olay bulundu!")
    print(f"\nİlk 10 olay:")
    print(events[:10])
    
    # Event ID'leri göster
    print(f"\n2. Olay tipleri analiz ediliyor...")
    event_ids = np.unique(events[:, 2])
    print(f"  - Bulunan olay tipleri: {event_ids}")
    
    # Olay tiplerini açıkla
    event_dict = {
        1: 'Standart Ses (Sık)',
        2: 'Oddball Ses (Nadir)',
        3: 'Standart Görsel',
        4: 'Oddball Görsel',
        5: 'Buton Basma'
    }
    
    print(f"\n3. Olay tipi açıklamaları:")
    for event_id in event_ids:
        if event_id in event_dict:
            count = len(events[events[:, 2] == event_id])
            print(f"  - Event ID {event_id}: {event_dict[event_id]} ({count} kez)")
        else:
            count = len(events[events[:, 2] == event_id])
            print(f"  - Event ID {event_id}: Bilinmeyen ({count} kez)")
    
    return events, event_dict

def create_epochs(raw, events, event_dict, tmin=-0.2, tmax=0.8):
    """Epoklar oluştur"""
    print("\n" + "="*60)
    print("EPOKLAMA (Epoching)")
    print("="*60)
    
    # Event ID'leri seç (sesli uyaranlar: 1 ve 2)
    # MNE-Python'da event_id dictionary'sinde key'ler string, value'lar integer olmalı
    selected_events = {'Standart Ses': 1, 'Oddball Ses': 2}
    
    print(f"\n1. Epoklar oluşturuluyor...")
    print(f"   - Zaman penceresi: {tmin} saniye ile {tmax} saniye arası")
    print(f"   - Her uyaranın etrafında {abs(tmin)} saniye öncesi ve {tmax} saniye sonrası")
    print(f"   - Event ID'ler: {selected_events}")
    
    # Epoklar oluştur
    epochs = mne.Epochs(raw, events, event_id=selected_events, 
                        tmin=tmin, tmax=tmax, 
                        baseline=(None, 0),  # Baseline: uyaran öncesi
                        preload=True,
                        verbose=False)
    
    print(f"\n✓ Epoklar oluşturuldu!")
    print(f"  - Toplam epok sayısı: {len(epochs)}")
    print(f"  - Standart Ses: {len(epochs['Standart Ses'])} epok")
    print(f"  - Oddball Ses: {len(epochs['Oddball Ses'])} epok")
    print(f"  - Epok süresi: {tmax - tmin} saniye")
    
    return epochs

def compute_erp(epochs):
    """ERP (Event-Related Potential) hesapla"""
    print("\n" + "="*60)
    print("ERP HESAPLAMA")
    print("="*60)
    
    print("\n1. Her uyaran tipi için ortalama tepki hesaplanıyor...")
    
    # Her uyaran tipi için ortalama al
    evoked_standard = epochs['Standart Ses'].average()
    evoked_oddball = epochs['Oddball Ses'].average()
    
    print("\n✓ ERP hesaplandı!")
    print(f"  - Standart Ses ortalaması: {len(evoked_standard.times)} zaman noktası")
    print(f"  - Oddball Ses ortalaması: {len(evoked_oddball.times)} zaman noktası")
    
    return evoked_standard, evoked_oddball

def visualize_erp_comparison(evoked_standard, evoked_oddball):
    """ERP'leri karşılaştırmalı görselleştir"""
    print("\n" + "="*60)
    print("GÖRSELLEŞTİRME")
    print("="*60)
    
    # 1. Zaman serisi karşılaştırması
    print("\n1. Zaman serisi karşılaştırması çiziliyor...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Tüm kanallar için ortalama
    evoked_standard.plot(axes=axes[0], show=False, time_unit='s')
    axes[0].set_title('Standart Ses - Ortalama ERP (Tüm Kanallar)', 
                     fontsize=14, fontweight='bold')
    
    evoked_oddball.plot(axes=axes[1], show=False, time_unit='s')
    axes[1].set_title('Oddball Ses - Ortalama ERP (Tüm Kanallar)', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Fark dalgası (Difference Wave)
    print("2. Fark dalgası hesaplanıyor (Oddball - Standart)...")
    evoked_diff = mne.combine_evoked([evoked_oddball, evoked_standard], 
                                     weights=[1, -1])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    evoked_diff.plot(axes=ax, show=False, time_unit='s')
    ax.set_title('Fark Dalgası: Oddball - Standart (P300 Bileşeni)', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Uyaran')
    ax.axvline(x=0.3, color='red', linestyle='--', linewidth=2, 
               label='P300 Zamanı (~300ms)', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return evoked_diff

def plot_topomaps(evoked_standard, evoked_oddball, times=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """Topografik haritalar çiz"""
    print("\n3. Topografik haritalar çiziliyor...")
    
    # Standart için topomap
    print("  → Standart Ses topomap'leri...")
    fig = evoked_standard.plot_topomap(times=times, 
                                       title='Standart Ses - Topografik Harita',
                                       show=False)
    plt.show()
    
    # Oddball için topomap
    print("  → Oddball Ses topomap'leri...")
    fig = evoked_oddball.plot_topomap(times=times,
                                      title='Oddball Ses - Topografik Harita',
                                      show=False)
    plt.show()
    
    # Fark için topomap
    print("  → Fark dalgası topomap'leri...")
    evoked_diff = mne.combine_evoked([evoked_oddball, evoked_standard], 
                                     weights=[1, -1])
    fig = evoked_diff.plot_topomap(times=times,
                                   title='Fark Dalgası (Oddball - Standart) - Topografik Harita',
                                   show=False)
    plt.show()

def plot_joint_comparison(evoked_standard, evoked_oddball):
    """Joint plot (zaman serisi + topomap) karşılaştırması"""
    print("\n4. Joint plot karşılaştırması çiziliyor...")
    
    # Standart için joint plot
    print("  → Standart Ses joint plot...")
    evoked_standard.plot_joint(times=[0.1, 0.2, 0.3, 0.4, 0.5],
                               title='Standart Ses - Joint Plot',
                               show=False)
    plt.show()
    
    # Oddball için joint plot
    print("  → Oddball Ses joint plot...")
    evoked_oddball.plot_joint(times=[0.1, 0.2, 0.3, 0.4, 0.5],
                              title='Oddball Ses - Joint Plot',
                              show=False)
    plt.show()
    
    # Fark için joint plot
    print("  → Fark dalgası joint plot...")
    evoked_diff = mne.combine_evoked([evoked_oddball, evoked_standard], 
                                     weights=[1, -1])
    evoked_diff.plot_joint(times=[0.1, 0.2, 0.3, 0.4, 0.5],
                           title='Fark Dalgası (Oddball - Standart) - Joint Plot',
                           show=False)
    plt.show()

def analyze_p300(evoked_standard, evoked_oddball):
    """P300 dalgasını analiz et"""
    print("\n" + "="*60)
    print("P300 DALGASI ANALİZİ")
    print("="*60)
    
    # Fark dalgası hesapla
    evoked_diff = mne.combine_evoked([evoked_oddball, evoked_standard], 
                                     weights=[1, -1])
    
    # P300 zaman penceresi (250-400 ms)
    p300_tmin = 0.25
    p300_tmax = 0.40
    
    print(f"\n1. P300 zaman penceresi: {p300_tmin*1000:.0f}-{p300_tmax*1000:.0f} ms")
    
    # P300 zaman penceresindeki ortalama genlik
    p300_window = evoked_diff.copy().crop(tmin=p300_tmin, tmax=p300_tmax)
    p300_amplitude = p300_window.data.mean(axis=0).max()
    
    # P300 zamanı (maksimum genlik zamanı)
    p300_time_idx = np.argmax(p300_window.data.mean(axis=0))
    p300_time = p300_window.times[p300_time_idx]
    
    print(f"\n2. P300 Özellikleri:")
    print(f"  - Maksimum genlik: {p300_amplitude:.2f} µV")
    print(f"  - P300 zamanı: {p300_time*1000:.0f} ms")
    
    # En güçlü P300 kanalı
    channel_amplitudes = p300_window.data.max(axis=1)
    max_channel_idx = np.argmax(channel_amplitudes)
    max_channel = evoked_diff.ch_names[max_channel_idx]
    
    print(f"  - En güçlü kanal: {max_channel}")
    print(f"  - Bu kanaldaki genlik: {channel_amplitudes[max_channel_idx]:.2f} µV")
    
    # P300 görselleştirmesi
    print(f"\n3. P300 görselleştirmesi çiziliyor...")
    
    # En güçlü kanalı seç
    evoked_diff_pick = evoked_diff.copy().pick_channels([max_channel])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    evoked_diff_pick.plot(axes=ax, show=False, time_unit='s')
    ax.axvspan(p300_tmin, p300_tmax, alpha=0.3, color='yellow', 
               label='P300 Zaman Penceresi')
    ax.axvline(x=p300_time, color='red', linestyle='--', linewidth=2,
               label=f'P300 Zamanı ({p300_time*1000:.0f} ms)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(f'P300 Dalgası - {max_channel} Kanalı', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'amplitude': p300_amplitude,
        'time': p300_time,
        'channel': max_channel,
        'window': (p300_tmin, p300_tmax)
    }

def print_summary(epochs, evoked_standard, evoked_oddball, p300_info):
    """Analiz özetini yazdır"""
    print("\n" + "="*60)
    print("ANALİZ ÖZETİ")
    print("="*60)
    
    print(f"\n📊 Epok İstatistikleri:")
    print(f"  - Toplam epok: {len(epochs)}")
    print(f"  - Standart Ses: {len(epochs['Standart Ses'])} epok")
    print(f"  - Oddball Ses: {len(epochs['Oddball Ses'])} epok")
    print(f"  - Epok oranı: {len(epochs['Oddball Ses'])/len(epochs['Standart Ses']):.2%}")
    
    print(f"\n📈 ERP Özellikleri:")
    print(f"  - Standart Ses ortalama genlik: {evoked_standard.data.mean():.2f} µV")
    print(f"  - Oddball Ses ortalama genlik: {evoked_oddball.data.mean():.2f} µV")
    print(f"  - Fark (Oddball - Standart): {(evoked_oddball.data.mean() - evoked_standard.data.mean()):.2f} µV")
    
    print(f"\n🎯 P300 Dalgası:")
    print(f"  - Genlik: {p300_info['amplitude']:.2f} µV")
    print(f"  - Zaman: {p300_info['time']*1000:.0f} ms")
    print(f"  - En güçlü kanal: {p300_info['channel']}")
    print(f"  - Zaman penceresi: {p300_info['window'][0]*1000:.0f}-{p300_info['window'][1]*1000:.0f} ms")
    
    print("\n" + "="*60)

def main():
    """Ana fonksiyon"""
    print("\n" + "="*70)
    print("EEG EPOKLAMA VE ERP ANALİZİ - ODDBALL PARADİGMASI")
    print("="*70)
    
    try:
        # 1. Veriyi yükle ve filtrele
        raw = load_and_filter_data()
        
        # 2. Olayları bul
        events, event_dict = find_events(raw)
        
        # 3. Epoklar oluştur
        epochs = create_epochs(raw, events, event_dict, tmin=-0.2, tmax=0.8)
        
        # 4. ERP hesapla
        evoked_standard, evoked_oddball = compute_erp(epochs)
        
        # 5. Menü
        while True:
            print("\n" + "="*60)
            print("GÖRSELLEŞTİRME SEÇENEKLERİ:")
            print("1. Zaman serisi karşılaştırması")
            print("2. Topografik haritalar (Topomap)")
            print("3. Joint plot (zaman serisi + topomap)")
            print("4. P300 dalgası analizi")
            print("5. Tüm görselleştirmeleri çalıştır")
            print("0. Çıkış")
            print("="*60)
            
            choice = input("Seçiminiz (0-5): ").strip()
            
            if choice == '0':
                print("\nÇıkılıyor...")
                break
            elif choice == '1':
                visualize_erp_comparison(evoked_standard, evoked_oddball)
            elif choice == '2':
                plot_topomaps(evoked_standard, evoked_oddball)
            elif choice == '3':
                plot_joint_comparison(evoked_standard, evoked_oddball)
            elif choice == '4':
                p300_info = analyze_p300(evoked_standard, evoked_oddball)
                print_summary(epochs, evoked_standard, evoked_oddball, p300_info)
            elif choice == '5':
                visualize_erp_comparison(evoked_standard, evoked_oddball)
                plot_topomaps(evoked_standard, evoked_oddball)
                plot_joint_comparison(evoked_standard, evoked_oddball)
                p300_info = analyze_p300(evoked_standard, evoked_oddball)
                print_summary(epochs, evoked_standard, evoked_oddball, p300_info)
            else:
                print("Geçersiz seçim! Lütfen 0-5 arası bir sayı girin.")
    
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        print("\nOlası çözümler:")
        print("1. MNE-Python kurulu mu kontrol edin: pip install mne")
        print("2. İnternet bağlantınızı kontrol edin")
        print("3. Yeterli bellek olduğundan emin olun")

if __name__ == '__main__':
    main()

