"""
GitHub Repository Güncelleme Scripti
Description ve Topics ekler
"""

import requests
import json
import os

def update_github_repo():
    """GitHub repository'sine description ve topics ekle"""
    
    # Repository bilgileri
    owner = "senaayy"
    repo = "Computational-Cognitive-Lab"
    
    # GitHub Personal Access Token gerekli
    # Token'ı environment variable'dan al veya kullanıcıdan iste
    token = os.getenv('GITHUB_TOKEN')
    
    if not token:
        print("="*60)
        print("GITHUB TOKEN GEREKLİ")
        print("="*60)
        print("\n1. GitHub'da Personal Access Token oluşturun:")
        print("   https://github.com/settings/tokens")
        print("   → 'Generate new token (classic)'")
        print("   → 'repo' yetkisini seçin")
        print("   → Token'ı kopyalayın")
        print("\n2. Token'ı şu şekilde kullanın:")
        print("   Windows PowerShell:")
        print("   $env:GITHUB_TOKEN='your_token_here'")
        print("   python update_github_repo.py")
        print("\n   Veya token'ı doğrudan script içine ekleyin (güvenli değil)")
        return
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Description
    description = "🧠 Computational Cognitive Lab - Behavioral tests, EEG signal processing, and AI-powered diagnosis system for cognitive neuroscience research"
    
    # Topics
    topics = [
        "neuroscience",
        "eeg-analysis",
        "cognitive-science",
        "machine-learning",
        "python",
        "mne-python",
        "behavioral-testing",
        "stroop-test",
        "erp-analysis",
        "computational-neuroscience",
        "biomedical-engineering",
        "neurotechnology"
    ]
    
    # 1. Repository bilgilerini güncelle
    print("="*60)
    print("GITHUB REPOSITORY GÜNCELLENİYOR")
    print("="*60)
    
    url = f"https://api.github.com/repos/{owner}/{repo}"
    
    data = {
        "description": description,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": True,
        "homepage": f"https://github.com/{owner}/{repo}",
        "topics": topics
    }
    
    print(f"\n1. Repository bilgileri güncelleniyor...")
    print(f"   Description: {description[:60]}...")
    print(f"   Topics: {', '.join(topics)}")
    
    try:
        response = requests.patch(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("   ✓ Repository başarıyla güncellendi!")
        else:
            print(f"   ❌ Hata: {response.status_code}")
            print(f"   Mesaj: {response.text}")
            return
    except Exception as e:
        print(f"   ❌ Hata oluştu: {e}")
        return
    
    # 2. Topics'ı ayrıca güncelle (bazı durumlarda gerekli)
    print(f"\n2. Topics güncelleniyor...")
    topics_url = f"https://api.github.com/repos/{owner}/{repo}/topics"
    topics_data = {"names": topics}
    
    try:
        topics_headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.mercy-preview+json"
        }
        response = requests.put(topics_url, headers=topics_headers, json=topics_data)
        
        if response.status_code == 200:
            print("   ✓ Topics başarıyla güncellendi!")
        else:
            print(f"   ⚠️  Topics güncellenemedi: {response.status_code}")
            print(f"   (Repository bilgileri güncellendi, topics manuel eklenebilir)")
    except Exception as e:
        print(f"   ⚠️  Topics güncellenemedi: {e}")
        print(f"   (Repository bilgileri güncellendi, topics manuel eklenebilir)")
    
    print("\n" + "="*60)
    print("✅ GÜNCELLEME TAMAMLANDI")
    print("="*60)
    print(f"\nRepository: https://github.com/{owner}/{repo}")
    print("GitHub'da kontrol edin!")

if __name__ == '__main__':
    update_github_repo()

