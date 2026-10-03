"""
Stroop Test Veri Analizi
Stroop Etkisi hesaplama ve görselleştirme
"""

import pandas as pd
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_stroop_data(data_dir='data'):
    """Stroop test verilerini yükle"""
    csv_files = glob.glob(os.path.join(data_dir, 'stroop_*.csv'))
    
    if not csv_files:
        print("Stroop test verisi bulunamadı!")
        return None
    
    # Tüm CSV dosyalarını birleştir
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    if not dfs:
        return None
    
    data = pd.concat(dfs, ignore_index=True)
    return data

def calculate_stroop_effect(data):
    """Stroop Etkisini hesapla"""
    # Uyumlu denemeler: kelime ve renk aynı
    # Uyumsuz denemeler: kelime ve renk farklı
    
    congruent = data[data['congruent'] == True]
    incongruent = data[data['congruent'] == False]
    
    # Sadece doğru cevapları analiz et
    congruent_correct = congruent[congruent['correct'] == True]
    incongruent_correct = incongruent[incongruent['correct'] == True]
    
    stats = {
        'congruent': {
            'mean_rt': congruent_correct['reactionTime'].mean() if len(congruent_correct) > 0 else 0,
            'median_rt': congruent_correct['reactionTime'].median() if len(congruent_correct) > 0 else 0,
            'std_rt': congruent_correct['reactionTime'].std() if len(congruent_correct) > 0 else 0,
            'count': len(congruent_correct),
            'accuracy': len(congruent_correct) / len(congruent) * 100 if len(congruent) > 0 else 0
        },
        'incongruent': {
            'mean_rt': incongruent_correct['reactionTime'].mean() if len(incongruent_correct) > 0 else 0,
            'median_rt': incongruent_correct['reactionTime'].median() if len(incongruent_correct) > 0 else 0,
            'std_rt': incongruent_correct['reactionTime'].std() if len(incongruent_correct) > 0 else 0,
            'count': len(incongruent_correct),
            'accuracy': len(incongruent_correct) / len(incongruent) * 100 if len(incongruent) > 0 else 0
        }
    }
    
    # Stroop Etkisi = Uyumsuz RT - Uyumlu RT
    stroop_effect = stats['incongruent']['mean_rt'] - stats['congruent']['mean_rt']
    
    stats['stroop_effect'] = stroop_effect

    # Gercek istatistiksel anlamlilik testi (Welch t-testi) + etki buyuklugu
    inc_rt = incongruent_correct['reactionTime'].dropna()
    con_rt = congruent_correct['reactionTime'].dropna()
    if len(inc_rt) >= 2 and len(con_rt) >= 2:
        t_stat, p_value = scipy_stats.ttest_ind(inc_rt, con_rt, equal_var=False)
        n1, n2 = len(inc_rt), len(con_rt)
        pooled_std = (((n1 - 1) * inc_rt.std()**2 + (n2 - 1) * con_rt.std()**2) / (n1 + n2 - 2)) ** 0.5
        cohens_d = stroop_effect / pooled_std if pooled_std > 0 else 0
        stats['t_stat'] = t_stat
        stats['p_value'] = p_value
        stats['cohens_d'] = cohens_d
        stats['significant'] = p_value < 0.05
    else:
        stats['t_stat'] = None
        stats['p_value'] = None
        stats['cohens_d'] = None
        stats['significant'] = None

    return stats

def visualize_stroop_effect(data, stats, output_dir='results'):
    """Stroop Etkisini görselleştir"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Uyumlu vs Uyumsuz Ortalama RT Karşılaştırması
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Stroop Etkisi Analizi', fontsize=16, fontweight='bold')
    
    # Grafik 1: Ortalama Reaksiyon Zamanları
    ax1 = axes[0, 0]
    categories = ['Uyumlu\n(Congruent)', 'Uyumsuz\n(Incongruent)']
    means = [stats['congruent']['mean_rt'], stats['incongruent']['mean_rt']]
    stds = [stats['congruent']['std_rt'], stats['incongruent']['std_rt']]
    
    bars = ax1.bar(categories, means, yerr=stds, capsize=10, 
                   color=['#4CAF50', '#f44336'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Ortalama Reaksiyon Zamanı (ms)', fontsize=12)
    ax1.set_title('Uyumlu vs Uyumsuz Ortalama RT', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Değerleri çubukların üzerine yaz
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + stds[i],
                f'{mean:.0f}ms',
                ha='center', va='bottom', fontweight='bold')
    
    # Stroop Etkisi değerini göster
    ax1.text(0.5, max(means) * 0.9, f"Stroop Etkisi: {stats['stroop_effect']:.0f}ms",
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Grafik 2: RT Dağılımı (Box Plot)
    ax2 = axes[0, 1]
    congruent_rt = data[(data['congruent'] == True) & (data['correct'] == True)]['reactionTime']
    incongruent_rt = data[(data['congruent'] == False) & (data['correct'] == True)]['reactionTime']
    
    box_data = [congruent_rt.dropna(), incongruent_rt.dropna()]
    bp = ax2.boxplot(box_data, labels=['Uyumlu', 'Uyumsuz'], 
                     patch_artist=True, widths=0.6)
    
    # Box plot renklendirme
    colors = ['#4CAF50', '#f44336']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Reaksiyon Zamanı (ms)', fontsize=12)
    ax2.set_title('RT Dağılımı (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Grafik 3: Doğruluk Oranları
    ax3 = axes[1, 0]
    accuracies = [stats['congruent']['accuracy'], stats['incongruent']['accuracy']]
    bars3 = ax3.bar(categories, accuracies, color=['#4CAF50', '#f44336'], 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Doğruluk Oranı (%)', fontsize=12)
    ax3.set_title('Uyumlu vs Uyumsuz Doğruluk', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 100])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars3, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Grafik 4: Zaman Serisi (Deneme numarasına göre RT)
    ax4 = axes[1, 1]
    congruent_trials = data[(data['congruent'] == True) & (data['correct'] == True)].copy()
    incongruent_trials = data[(data['congruent'] == False) & (data['correct'] == True)].copy()
    
    if len(congruent_trials) > 0:
        ax4.scatter(congruent_trials['trial'], congruent_trials['reactionTime'], 
                   alpha=0.6, color='#4CAF50', label='Uyumlu', s=50)
    
    if len(incongruent_trials) > 0:
        ax4.scatter(incongruent_trials['trial'], incongruent_trials['reactionTime'], 
                   alpha=0.6, color='#f44336', label='Uyumsuz', s=50)
    
    ax4.set_xlabel('Deneme Numarası', fontsize=12)
    ax4.set_ylabel('Reaksiyon Zamanı (ms)', fontsize=12)
    ax4.set_title('RT Zaman Serisi', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Kaydet
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'stroop_analysis_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Grafik kaydedildi: {output_file}")
    
    plt.show()
    
    return output_file

def print_statistics(stats):
    """İstatistikleri yazdır"""
    print("\n" + "="*60)
    print("STROOP ETKİSİ ANALİZ SONUÇLARI")
    print("="*60)
    
    print("\n📊 UYUMLU DENEMELER (Congruent):")
    print(f"   Ortalama RT: {stats['congruent']['mean_rt']:.2f} ms")
    print(f"   Medyan RT: {stats['congruent']['median_rt']:.2f} ms")
    print(f"   Standart Sapma: {stats['congruent']['std_rt']:.2f} ms")
    print(f"   Doğru Cevap Sayısı: {stats['congruent']['count']}")
    print(f"   Doğruluk Oranı: {stats['congruent']['accuracy']:.2f}%")
    
    print("\n📊 UYUMSUZ DENEMELER (Incongruent):")
    print(f"   Ortalama RT: {stats['incongruent']['mean_rt']:.2f} ms")
    print(f"   Medyan RT: {stats['incongruent']['median_rt']:.2f} ms")
    print(f"   Standart Sapma: {stats['incongruent']['std_rt']:.2f} ms")
    print(f"   Doğru Cevap Sayısı: {stats['incongruent']['count']}")
    print(f"   Doğruluk Oranı: {stats['incongruent']['accuracy']:.2f}%")
    
    print("\n" + "="*60)
    print(f"🎯 STROOP ETKİSİ: {stats['stroop_effect']:.2f} ms")
    print("="*60)
    
    if stats.get('p_value') is None:
        print(f"\n⚠️  Anlamlılık testi için yeterli veri yok (en az 2 denek/deneme gerekir).")
    elif stats['stroop_effect'] > 0 and stats['significant']:
        print(f"\n✅ Stroop Etkisi tespit edildi ve istatistiksel olarak anlamlı!")
        print(f"   Uyumsuz denemeler, uyumlu denemelerden {stats['stroop_effect']:.2f} ms daha yavaş.")
        print(f"   t({stats['congruent']['count'] + stats['incongruent']['count'] - 2}) = {stats['t_stat']:.3f}, p = {stats['p_value']:.4f}, Cohen's d = {stats['cohens_d']:.3f}")
    elif stats['stroop_effect'] > 0 and not stats['significant']:
        print(f"\n⚠️  Uyumsuz denemeler {stats['stroop_effect']:.2f} ms daha yavaş görünüyor ama bu fark istatistiksel olarak anlamlı DEĞİL.")
        print(f"   t({stats['congruent']['count'] + stats['incongruent']['count'] - 2}) = {stats['t_stat']:.3f}, p = {stats['p_value']:.4f} (p >= 0.05)")
    else:
        print(f"\n⚠️  Beklenmeyen sonuç: Stroop Etkisi negatif veya sıfır.")
    
    print()

def main():
    """Ana analiz fonksiyonu"""
    print("Stroop Test Veri Analizi Başlatılıyor...")
    
    # Veriyi yükle
    data = load_stroop_data()
    
    if data is None or len(data) == 0:
        print("Analiz için yeterli veri bulunamadı!")
        print("Lütfen önce testi çalıştırıp veri toplayın.")
        return
    
    print(f"Toplam {len(data)} deneme yüklendi.")
    
    # Stroop Etkisini hesapla
    stats = calculate_stroop_effect(data)
    
    # İstatistikleri yazdır
    print_statistics(stats)
    
    # Görselleştir
    visualize_stroop_effect(data, stats)
    
    print("\n✅ Analiz tamamlandı!")

if __name__ == '__main__':
    main()

