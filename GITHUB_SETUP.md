# GitHub Repository Kurulum Kılavuzu

## Description ve Topics Ekleme

### Yöntem 1: GitHub Web Arayüzü (Önerilen)

1. **Repository'ye gidin:**
   - https://github.com/senaayy/Computational-Cognitive-Lab

2. **Settings'e tıklayın:**
   - Repository sayfasında sağ üstte "Settings" butonuna tıklayın

3. **Description ekleyin:**
   - "About" bölümünde description kutusuna şunu yazın:
   ```
   🧠 Computational Cognitive Lab - Behavioral tests, EEG signal processing, and AI-powered diagnosis system for cognitive neuroscience research
   ```

4. **Topics ekleyin:**
   - "Topics" bölümüne şu topic'leri ekleyin:
   - `neuroscience`
   - `eeg-analysis`
   - `cognitive-science`
   - `machine-learning`
   - `python`
   - `mne-python`
   - `behavioral-testing`
   - `stroop-test`
   - `erp-analysis`
   - `computational-neuroscience`
   - `biomedical-engineering`
   - `neurotechnology`

5. **Website ekleyin (opsiyonel):**
   - `https://github.com/senaayy/Computational-Cognitive-Lab`

6. **Save changes'e tıklayın**

### Yöntem 2: Python Script ile (Otomatik)

1. **GitHub Personal Access Token oluşturun:**
   - https://github.com/settings/tokens
   - "Generate new token (classic)" tıklayın
   - Token'a bir isim verin (örn: "repo-update")
   - `repo` yetkisini seçin
   - "Generate token" tıklayın
   - Token'ı kopyalayın (bir daha gösterilmeyecek!)

2. **Token'ı environment variable olarak ayarlayın:**

   **Windows PowerShell:**
   ```powershell
   $env:GITHUB_TOKEN='your_token_here'
   ```

   **Windows CMD:**
   ```cmd
   set GITHUB_TOKEN=your_token_here
   ```

   **Linux/Mac:**
   ```bash
   export GITHUB_TOKEN='your_token_here'
   ```

3. **Scripti çalıştırın:**
   ```bash
   python update_github_repo.py
   ```

### Yöntem 3: GitHub CLI ile

1. **GitHub CLI kurun:**
   ```bash
   # Windows (Chocolatey)
   choco install gh
   
   # Windows (Scoop)
   scoop install gh
   
   # Linux
   sudo apt install gh
   
   # Mac
   brew install gh
   ```

2. **GitHub'a giriş yapın:**
   ```bash
   gh auth login
   ```

3. **Repository'yi güncelleyin:**
   ```bash
   gh repo edit senaayy/Computational-Cognitive-Lab \
     --description "🧠 Computational Cognitive Lab - Behavioral tests, EEG signal processing, and AI-powered diagnosis system for cognitive neuroscience research" \
     --add-topic neuroscience \
     --add-topic eeg-analysis \
     --add-topic cognitive-science \
     --add-topic machine-learning \
     --add-topic python \
     --add-topic mne-python \
     --add-topic behavioral-testing \
     --add-topic stroop-test \
     --add-topic erp-analysis \
     --add-topic computational-neuroscience \
     --add-topic biomedical-engineering \
     --add-topic neurotechnology
   ```

## Önerilen Repository Ayarları

### Features
- ✅ Issues: Açık
- ✅ Projects: Açık
- ✅ Wiki: Açık
- ✅ Discussions: İsteğe bağlı

### General Settings
- **Repository name:** Computational-Cognitive-Lab
- **Description:** (Yukarıdaki description)
- **Website:** https://github.com/senaayy/Computational-Cognitive-Lab
- **Topics:** (Yukarıdaki 12 topic)

## Güvenlik Notları

⚠️ **ÖNEMLİ:**
- GitHub token'ınızı asla public repository'lere commit etmeyin
- Token'ı `.gitignore`'a ekleyin
- Token'ı sadece güvenli yerlerde saklayın
- Token'ı paylaşmayın

## Kontrol

Güncellemeleri kontrol etmek için:
1. Repository sayfasına gidin
2. "About" bölümünü kontrol edin
3. Topics'ların göründüğünü doğrulayın

