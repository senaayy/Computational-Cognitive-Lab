# 🧠 Computational Cognitive Lab (Hesaplamalı Bilişsel Laboratuvar)

## 🌟 Proje Özeti

Bu proje, modern nörobilim araştırmalarında kullanılan üç temel analitik disiplini (Davranışsal Testler, Sinyal İşleme ve Makine Öğrenmesi) birleştiren bir araç setidir. Amaç, insan bilişsel fonksiyonlarını (Dikkat, İnhibisyon, Uyanıklık) hem davranışsal veriler hem de EEG sinyal özellikleri üzerinden analiz ederek Hesaplamalı Nörobilim alanında pratik yeterlilik kazanmaktır.

Bu depo, bir Yazılım Mühendisi'nin biyomedikal ve nöroteknoloji alanına geçişini gösteren güçlü bir portfolyo görevi görür.

## 🔬 Uygulanan Metodolojiler ve Bulgular

### 1. Bilişsel Kontrol (Davranışsal Analiz)

**Araçlar:** HTML/CSS/JavaScript (Frontend), Python (Veri Kayıt ve Analiz), Flask (Backend)

**Testler:** Stroop Testi ve Go/No-Go Testi uygulamaları sıfırdan kodlanmıştır.

**Kanıtlanan Etki (Stroop):** Uyumsuz denemelerde reaksiyon süresinin (RT) ortalama 1169ms daha uzun olduğu istatistiksel olarak kanıtlanmıştır.

**Analiz Detayı:** 
- Reaksiyon zamanı (RT) dağılımı
- Uyumlu/Uyumsuz denemelerdeki doğruluk oranları
- Hata tipi analizi (Kelime hatası, Renk hatası)
- Detaylı zaman kaydı (saat, dakika, saniye, milisaniye)

### 2. Sinyal İşleme (ERP Analizi)

**Araçlar:** MNE-Python

**Veri Seti:** Auditory Oddball (İşitsel Tekillik) veri seti kullanılmıştır.

**Filtreleme:** Ham EEG sinyali (Gürültülü veri), 0.1 Hz Yüksek Geçiren ve 40 Hz Düşük Geçiren filtreler uygulanarak temizlenmiştir.

**Bulgu (ERP):** Nadir (Oddball) ve Standart uyaranlara verilen Olayla İlişkili Potansiyeller (ERP) çıkarılmıştır. Bu analiz, beynin dikkat ve bilgi güncelleme sürecini temsil eden **P300 dalgasının** varlığını görsel olarak kanıtlamıştır.

**Özellikler:**
- Epoklama (Epoching) - Uyaran etrafında zaman pencereleri
- ERP hesaplama ve karşılaştırma
- **P300 Kanıtı - Tek grafikte Oddball vs Standart karşılaştırması**
- Topografik haritalar (Topomap)
- Joint plot görselleştirmesi
- Parietal bölge odaklı P300 analizi

### 3. Yapay Zeka & Biyobelirteç Geliştirme

**Araçlar:** Python, Pandas, Scikit-learn, Makine Öğrenmesi Kütüphaneleri

**Özellik Mühendisliği:** EEG sinyalinden frekans tabanlı özellikler (Delta, Theta, Alpha, Beta, Gamma band güçleri) çıkarılmıştır.

**Model ve Yorumlama:** Bir Makine Öğrenmesi modeli (Random Forest) eğitilmiş ve tahmin için hangi özelliklerin kritik olduğu analiz edilmiştir.

**Biyobelirteç Tespiti:** Modelin, bilişsel durumu tahmin ederken en çok **Theta-Beta Oranı** ile **Theta Gücüne** odaklandığı tespit edilmiştir. Bu bulgu, DEHB ve uyanıklık çalışmalarında kullanılan temel nöral biyobelirteçleri yansıtmaktadır.

⚠️ **UYARI:** AI teşhis sistemi sadece eğitim ve demo amaçlıdır. Gerçek tıbbi teşhis için kullanılamaz!

## 🛠️ Kurulum ve Kullanım

### Gereksinimler

- Python 3.8+
- İnternet bağlantısı (ilk çalıştırmada MNE örnek veri seti indirilecek)

### Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/senaayy/Computational-Cognitive-Lab.git
cd Computational-Cognitive-Lab
```

2. **Sanal ortam oluşturun (önerilir):**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

### Kullanım

#### 1. Reaksiyon Zamanı Testleri (Stroop & Go/No-Go)

**Backend'i başlatın:**
```bash
python app.py
```

**Test arayüzünü açın:**
- `reaction_time_test.html` dosyasını tarayıcıda açın
- Stroop veya Go/No-Go testini seçin ve testi başlatın
- Veriler otomatik olarak `data/` klasörüne CSV formatında kaydedilir

**Veri analizi:**
```bash
# Stroop Etkisi analizi
python analyze_data.py

# Hata tipi analizi
python analyze_errors.py
```

#### 2. EEG Veri Analizi

**Basit EEG yükleme:**
```bash
python load_eeg_data.py
```

**Gelişmiş EEG analizi:**
```bash
python eeg_analysis_example.py
```

**Filtreleme ve görselleştirme:**
```bash
python eeg_filtering_analysis.py
```

**Epoklama ve ERP analizi:**
```bash
python eeg_epoching_erp.py
```

Bu script şunları içerir:
- Olay tespiti (find_events)
- Epoklama (Epoching) - uyaran etrafında zaman pencereleri
- ERP hesaplama (Event-Related Potentials)
- **P300 Kanıtı - Tek grafikte karşılaştırmalı görselleştirme**
- Oddball vs Standart karşılaştırması
- P300 dalgası analizi
- Topografik haritalar (Topomap)
- Joint plot görselleştirmesi

**Yapay Zeka destekli teşhis (Demo):**
```bash
python eeg_ai_diagnosis.py
```

## 📁 Proje Yapısı

```
Computational-Cognitive-Lab/
├── reaction_time_test.html    # Ana test arayüzü
├── app.py                     # Flask backend (veri kayıt)
├── analyze_data.py            # Stroop Etkisi analizi
├── analyze_errors.py          # Hata tipi analizi
├── load_eeg_data.py           # Basit EEG yükleme
├── eeg_analysis_example.py    # Gelişmiş EEG analiz örnekleri
├── eeg_filtering_analysis.py  # EEG filtreleme ve görselleştirme
├── eeg_epoching_erp.py        # Epoklama ve ERP analizi
├── eeg_ai_diagnosis.py        # AI destekli teşhis (demo)
├── update_github_repo.py      # GitHub repository güncelleme scripti
├── requirements.txt           # Python bağımlılıkları
├── README.md                  # Bu dosya
├── GITHUB_SETUP.md            # GitHub kurulum kılavuzu
├── data/                      # CSV veri dosyaları (otomatik oluşturulur)
└── results/                   # Analiz grafikleri (otomatik oluşturulur)
```

## 📊 Özellikler

### Davranışsal Testler
- ✅ Stroop Testi (Renk-kelime uyumsuzluğu)
- ✅ Go/No-Go Testi (Dikkat ve inhibisyon)
- ✅ Otomatik veri kaydı (CSV)
- ✅ Detaylı zaman kaydı (saat, dakika, saniye, milisaniye)
- ✅ Hata tipi sınıflandırması
- ✅ Gerçek zamanlı istatistikler

### EEG Analizi
- ✅ Veri yükleme ve filtreleme
- ✅ Epoklama (Epoching)
- ✅ ERP hesaplama
- ✅ **P300 Kanıtı - Tek grafikte Oddball vs Standart karşılaştırması**
- ✅ P300 dalgası analizi ve tespiti
- ✅ Topografik haritalar (Topomap)
- ✅ Joint plot görselleştirmesi
- ✅ Frekans bantları analizi
- ✅ Parietal bölge odaklı analiz

### Makine Öğrenmesi
- ✅ Özellik çıkarımı (14 özellik)
- ✅ Random Forest modeli
- ✅ Olasılık tabanlı teşhis
- ✅ Özellik önemi analizi

## 🎯 Gelecek Hedefler

1. **Canlı EEG Entegrasyonu:** Geliştirilen bilişsel testler (Stroop/GoNoGo) için kendi EEG kaydını alarak bu projeyi canlı bir sisteme dönüştürmek.

2. **Gelişmiş Yapay Zeka:** CNN, RNN gibi derin öğrenme algoritmaları kullanarak EEG sinyalinden otomatik teşhis yapabilen bir model geliştirmek.

3. **Gerçek Klinik Veri:** Gerçek klinik veri setleri ile modeli eğitmek ve validasyon yapmak.

4. **Web Arayüzü:** Tüm analizleri web üzerinden yapabilen interaktif bir platform geliştirmek.

5. **Real-time Analiz:** Canlı EEG verisi üzerinden gerçek zamanlı analiz yapabilme.

## 📚 Referanslar ve Dokümantasyon

- **MNE-Python:** https://mne.tools/stable/index.html
- **Stroop Testi:** Stroop, J. R. (1935). Studies of interference in serial verbal reactions.
- **P300 Dalgası:** Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b.
- **DEHB ve EEG:** Loo, S. K., & Makeig, S. (2012). Clinical utility of EEG in attention-deficit/hyperactivity disorder.

## ⚠️ Önemli Notlar

- **Tıbbi Teşhis:** Bu proje sadece eğitim ve araştırma amaçlıdır. Gerçek tıbbi teşhis için kullanılamaz.
- **Veri:** AI teşhis sistemi sentetik veri ile eğitilmiştir. Gerçek klinik uygulamalar için doğrulanmış veri setleri gerekir.
- **Etik:** EEG verisi kullanırken etik kurallara ve gizlilik yönetmeliklerine uyulmalıdır.

## 🤝 Katkıda Bulunma

Bu proje eğitim amaçlıdır. Öneriler ve geri bildirimler için issue açabilirsiniz.

## 📄 Lisans

Bu proje eğitim ve araştırma amaçlıdır. Ticari kullanım için izin gerekebilir.

## 👤 Yazar

**Sena Ay**
- GitHub: [@senaayy](https://github.com/senaayy)
- Portfolio: Computational Cognitive Lab

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
