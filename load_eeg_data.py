"""
EEG Verisi Yükleme ve Görselleştirme
MNE-Python kullanarak örnek EEG verisini yükler ve görselleştirir
"""

# Gerekli kütüphaneleri içe aktar
import mne
import os
import matplotlib.pyplot as plt

# Veri setini indirmek ve yüklemek için işlev
sample_data_folder = mne.datasets.sample.data_path()
data_path = os.path.join(sample_data_folder, 'MEG', 'sample')
raw_fname = os.path.join(data_path, 'sample_audvis_raw.fif')

# Veriyi oku (bu dosya hem MEG hem de EEG verisi içerir)
# Biz sadece EEG kanallarını kullanacağız.
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Sadece EEG kanallarını seç
# (Örnek veride EEG dışında MEG, EOG, ECG gibi kanallar da bulunur)
raw.pick_types(eeg=True, stim=True)

print("Veri başarıyla yüklendi!")
print(raw.info)

# Ham verinin basit bir görünümünü çizdir (İlk 5 saniye)
raw.plot(duration=5, n_channels=30, scalings='auto')
plt.show()

