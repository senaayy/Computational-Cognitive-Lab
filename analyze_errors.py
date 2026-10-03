"""
Hata Tipi Analizi - Stroop ve Go/No-Go Testleri
Yanlış kelime, yanlış renk, false alarm, missed go gibi hata tiplerini analiz eder
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import numpy as np

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_test_data(data_dir='data', test_type='stroop'):
    """Test verilerini yükle"""
    csv_files = glob.glob(os.path.join(data_dir, f'{test_type}_*.csv'))
    
    if not csv_files:
        print(f"{test_type.upper()} test verisi bulunamadı!")
        return None
    
    # Tüm CSV dosyalarını birleştir
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Uyarı: {file} yüklenemedi: {e}")
    
    if not dfs:
        return None
    
    data = pd.concat(dfs, ignore_index=True)
    return data

def analyze_stroop_errors(data):
    """Stroop testi hata tiplerini analiz et"""
    if data is None or len(data) == 0:
        return None
    
    # Hata tipi dağılımı
    error_counts = data['errorType'].value_counts()
    
    # Doğru ve yanlış cevapları ayır
    correct_trials = data[data['correct'] == True]
    incorrect_trials = data[data['correct'] == False]
    
    # Hata tiplerine göre grupla
    error_analysis = {
        'correct': {
            'count': len(data[data['errorType'] == 'correct']),
            'percentage': len(data[data['errorType'] == 'correct']) / len(data) * 100,
            'mean_rt': data[data['errorType'] == 'correct']['reactionTime'].mean() if len(data[data['errorType'] == 'correct']) > 0 else 0
        },
        'word_error': {
            'count': len(data[data['errorType'] == 'word_error']),
            'percentage': len(data[data['errorType'] == 'word_error']) / len(data) * 100,
            'mean_rt': data[data['errorType'] == 'word_error']['reactionTime'].mean() if len(data[data['errorType'] == 'word_error']) > 0 else 0
        },
        'color_error': {
            'count': len(data[data['errorType'] == 'color_error']),
            'percentage': len(data[data['errorType'] == 'color_error']) / len(data) * 100,
            'mean_rt': data[data['errorType'] == 'color_error']['reactionTime'].mean() if len(data[data['errorType'] == 'color_error']) > 0 else 0
        }
    }
    
    # Uyumlu/uyumsuz denemelere göre hata dağılımı
    congruent_errors = data[(data['congruent'] == True) & (data['correct'] == False)]
    incongruent_errors = data[(data['congruent'] == False) & (data['correct'] == False)]
    
    # Renklere göre hata dağılımı
    color_error_dist = data[data['correct'] == False].groupby('color')['errorType'].value_counts()
    
    return {
        'error_analysis': error_analysis,
        'error_counts': error_counts,
        'congruent_errors': len(congruent_errors),
        'incongruent_errors': len(incongruent_errors),
        'color_error_dist': color_error_dist,
        'total_trials': len(data),
        'correct_trials': len(correct_trials),
        'incorrect_trials': len(incorrect_trials)
    }

def analyze_gonogo_errors(data):
    """Go/No-Go testi hata tiplerini analiz et"""
    if data is None or len(data) == 0:
        return None
    
    # Hata tipi dağılımı
    error_counts = data['errorType'].value_counts()
    
    # Hata tiplerine göre analiz
    error_analysis = {}
    for error_type in ['correct', 'missed_go', 'false_alarm']:
        error_data = data[data['errorType'] == error_type]
        if len(error_data) > 0:
            error_analysis[error_type] = {
                'count': len(error_data),
                'percentage': len(error_data) / len(data) * 100,
                'mean_rt': error_data['reactionTime'].mean() if error_type != 'false_alarm' else None
            }
    
    # Go ve No-Go denemelerine göre hata dağılımı
    go_trials = data[data['isGo'] == True]
    nogo_trials = data[data['isGo'] == False]
    
    go_errors = go_trials[go_trials['correct'] == False]
    nogo_errors = nogo_trials[nogo_trials['correct'] == False]
    
    incorrect_total = len(go_errors) + len(nogo_errors)

    return {
        'error_analysis': error_analysis,
        'error_counts': error_counts,
        'go_errors': len(go_errors),
        'nogo_errors': len(nogo_errors),
        'total_trials': len(data),
        'go_trials': len(go_trials),
        'nogo_trials': len(nogo_trials),
        'correct_trials': len(data) - incorrect_total,
        'incorrect_trials': incorrect_total
    }

def visualize_stroop_errors(data, analysis, output_dir='results'):
    """Stroop hata analizini görselleştir"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Stroop Testi - Hata Tipi Analizi', fontsize=16, fontweight='bold')
    
    # Grafik 1: Hata Tipi Dağılımı (Pasta Grafiği)
    ax1 = axes[0, 0]
    error_types = ['correct', 'word_error', 'color_error']
    error_labels = ['Doğru', 'Kelime Hatası', 'Renk Hatası']
    error_values = [analysis['error_analysis'][et]['count'] for et in error_types]
    colors_pie = ['#4CAF50', '#FF9800', '#f44336']
    
    ax1.pie(error_values, labels=error_labels, autopct='%1.1f%%', 
            colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Hata Tipi Dağılımı', fontsize=14, fontweight='bold')
    
    # Grafik 2: Hata Tipine Göre Ortalama RT
    ax2 = axes[0, 1]
    error_types_rt = [et for et in error_types if analysis['error_analysis'][et]['mean_rt'] > 0]
    rt_values = [analysis['error_analysis'][et]['mean_rt'] for et in error_types_rt]
    rt_labels = ['Doğru' if et == 'correct' else 'Kelime Hatası' if et == 'word_error' else 'Renk Hatası' 
                 for et in error_types_rt]
    
    bars = ax2.bar(rt_labels, rt_values, color=[colors_pie[error_types.index(et)] for et in error_types_rt], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Ortalama Reaksiyon Zamanı (ms)', fontsize=12)
    ax2.set_title('Hata Tipine Göre Ortalama RT', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, rt_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.0f}ms',
                ha='center', va='bottom', fontweight='bold')
    
    # Grafik 3: Uyumlu vs Uyumsuz Hata Oranları
    ax3 = axes[1, 0]
    congruent_total = len(data[data['congruent'] == True])
    incongruent_total = len(data[data['congruent'] == False])
    congruent_error_rate = (analysis['congruent_errors'] / congruent_total * 100) if congruent_total > 0 else 0
    incongruent_error_rate = (analysis['incongruent_errors'] / incongruent_total * 100) if incongruent_total > 0 else 0
    
    categories = ['Uyumlu\n(Congruent)', 'Uyumsuz\n(Incongruent)']
    error_rates = [congruent_error_rate, incongruent_error_rate]
    
    bars3 = ax3.bar(categories, error_rates, color=['#4CAF50', '#f44336'], 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Hata Oranı (%)', fontsize=12)
    ax3.set_title('Uyumlu vs Uyumsuz Hata Oranları', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, max(error_rates) * 1.2 if max(error_rates) > 0 else 100])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars3, error_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Grafik 4: Renklere Göre Hata Dağılımı
    ax4 = axes[1, 1]
    color_error_data = data[data['correct'] == False]
    if len(color_error_data) > 0:
        color_error_counts = color_error_data['color'].value_counts()
        colors_bar = ['#f44336', '#2196F3', '#4CAF50', '#FFC107']
        color_names = color_error_counts.index.tolist()
        
        bars4 = ax4.bar(color_names, color_error_counts.values, 
                       color=[colors_bar[['red', 'blue', 'green', 'yellow'].index(c)] if c in ['red', 'blue', 'green', 'yellow'] else '#999' 
                              for c in color_names],
                       alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Hata Sayısı', fontsize=12)
        ax4.set_xlabel('Renk', fontsize=12)
        ax4.set_title('Renklere Göre Hata Dağılımı', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars4, color_error_counts.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}',
                    ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Hata verisi yok', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Renklere Göre Hata Dağılımı', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Kaydet
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'stroop_error_analysis_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Grafik kaydedildi: {output_file}")
    
    plt.show()
    
    return output_file

def visualize_gonogo_errors(data, analysis, output_dir='results'):
    """Go/No-Go hata analizini görselleştir"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Go/No-Go Testi - Hata Tipi Analizi', fontsize=16, fontweight='bold')
    
    # Grafik 1: Hata Tipi Dağılımı
    ax1 = axes[0, 0]
    error_types = ['correct', 'missed_go', 'false_alarm']
    error_labels = ['Doğru', 'Kaçırılan Go', 'Yanlış Alarm']
    error_values = [analysis['error_analysis'].get(et, {}).get('count', 0) for et in error_types]
    colors_pie = ['#4CAF50', '#FF9800', '#f44336']
    
    # Sadece sıfır olmayan değerleri göster
    non_zero = [(label, value, color) for label, value, color in zip(error_labels, error_values, colors_pie) if value > 0]
    if non_zero:
        labels, values, colors = zip(*non_zero)
        ax1.pie(values, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Hata Tipi Dağılımı', fontsize=14, fontweight='bold')
    
    # Grafik 2: Go vs No-Go Hata Oranları
    ax2 = axes[0, 1]
    go_error_rate = (analysis['go_errors'] / analysis['go_trials'] * 100) if analysis['go_trials'] > 0 else 0
    nogo_error_rate = (analysis['nogo_errors'] / analysis['nogo_trials'] * 100) if analysis['nogo_trials'] > 0 else 0
    
    categories = ['Go\n(Kaçırılan)', 'No-Go\n(Yanlış Alarm)']
    error_rates = [go_error_rate, nogo_error_rate]
    
    bars = ax2.bar(categories, error_rates, color=['#FF9800', '#f44336'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Hata Oranı (%)', fontsize=12)
    ax2.set_title('Go vs No-Go Hata Oranları', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(error_rates) * 1.2 if max(error_rates) > 0 else 100])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars, error_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Grafik 3: Hata Tipine Göre RT (sadece Go denemeleri için)
    ax3 = axes[1, 0]
    go_correct = data[(data['isGo'] == True) & (data['correct'] == True)]
    go_missed = data[(data['isGo'] == True) & (data['correct'] == False)]
    
    if len(go_correct) > 0 or len(go_missed) > 0:
        categories_rt = []
        rt_values = []
        
        if len(go_correct) > 0:
            categories_rt.append('Doğru Go')
            rt_values.append(go_correct['reactionTime'].mean())
        
        if len(go_missed) > 0:
            categories_rt.append('Kaçırılan Go')
            rt_values.append(go_missed['reactionTime'].mean() if len(go_missed[go_missed['reactionTime'].notna()]) > 0 else 0)
        
        if categories_rt:
            bars3 = ax3.bar(categories_rt, rt_values, color=['#4CAF50', '#FF9800'], 
                           alpha=0.7, edgecolor='black', linewidth=1.5)
            ax3.set_ylabel('Ortalama RT (ms)', fontsize=12)
            ax3.set_title('Go Denemelerinde RT Karşılaştırması', fontsize=14, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            
            for bar, value in zip(bars3, rt_values):
                if value > 0:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.0f}ms',
                            ha='center', va='bottom', fontweight='bold')
    
    # Grafik 4: Zaman İçinde Hata Dağılımı
    ax4 = axes[1, 1]
    if 'trial' in data.columns:
        error_trials = data[data['correct'] == False]['trial']
        correct_trials = data[data['correct'] == True]['trial']
        
        if len(error_trials) > 0:
            ax4.scatter(error_trials, [1] * len(error_trials), 
                       alpha=0.6, color='#f44336', label='Hata', s=50)
        if len(correct_trials) > 0:
            ax4.scatter(correct_trials, [0] * len(correct_trials), 
                       alpha=0.6, color='#4CAF50', label='Doğru', s=50)
        
        ax4.set_xlabel('Deneme Numarası', fontsize=12)
        ax4.set_ylabel('Sonuç', fontsize=12)
        ax4.set_title('Zaman İçinde Hata Dağılımı', fontsize=14, fontweight='bold')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Doğru', 'Hata'])
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Kaydet
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'gonogo_error_analysis_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Grafik kaydedildi: {output_file}")
    
    plt.show()
    
    return output_file

def print_error_statistics(analysis, test_type='stroop'):
    """Hata istatistiklerini yazdır"""
    print("\n" + "="*60)
    print(f"{test_type.upper()} TESTİ - HATA TİPİ ANALİZ SONUÇLARI")
    print("="*60)
    
    if test_type == 'stroop':
        print("\n📊 HATA TİPİ DAĞILIMI:")
        for error_type, label in [('correct', 'Doğru'), ('word_error', 'Kelime Hatası'), ('color_error', 'Renk Hatası')]:
            if error_type in analysis['error_analysis']:
                stats = analysis['error_analysis'][error_type]
                print(f"   {label}:")
                print(f"      Sayı: {stats['count']}")
                print(f"      Yüzde: {stats['percentage']:.2f}%")
                if stats['mean_rt'] > 0:
                    print(f"      Ortalama RT: {stats['mean_rt']:.2f} ms")
        
        print(f"\n📊 UYUMLU/UYUMSUZ HATA KARŞILAŞTIRMASI:")
        print(f"   Uyumlu Denemelerde Hata: {analysis['congruent_errors']}")
        print(f"   Uyumsuz Denemelerde Hata: {analysis['incongruent_errors']}")
        
    elif test_type == 'gonogo':
        print("\n📊 HATA TİPİ DAĞILIMI:")
        for error_type, label in [('correct', 'Doğru'), ('missed_go', 'Kaçırılan Go'), ('false_alarm', 'Yanlış Alarm')]:
            if error_type in analysis['error_analysis']:
                stats = analysis['error_analysis'][error_type]
                print(f"   {label}:")
                print(f"      Sayı: {stats['count']}")
                print(f"      Yüzde: {stats['percentage']:.2f}%")
                if stats['mean_rt']:
                    print(f"      Ortalama RT: {stats['mean_rt']:.2f} ms")
        
        print(f"\n📊 GO/NO-GO HATA KARŞILAŞTIRMASI:")
        print(f"   Go Denemelerinde Hata: {analysis['go_errors']} / {analysis['go_trials']}")
        print(f"   No-Go Denemelerinde Hata: {analysis['nogo_errors']} / {analysis['nogo_trials']}")
    
    print(f"\n📊 GENEL İSTATİSTİKLER:")
    print(f"   Toplam Deneme: {analysis['total_trials']}")
    print(f"   Doğru Cevap: {analysis['correct_trials'] if 'correct_trials' in analysis else 'N/A'}")
    print(f"   Yanlış Cevap: {analysis['incorrect_trials'] if 'incorrect_trials' in analysis else 'N/A'}")
    print(f"   Genel Doğruluk: {(analysis['correct_trials'] / analysis['total_trials'] * 100) if 'correct_trials' in analysis and analysis['total_trials'] > 0 else 'N/A':.2f}%")
    
    print("="*60)
    print()

def main():
    """Ana analiz fonksiyonu"""
    print("Hata Tipi Analizi Başlatılıyor...")
    
    # Stroop testi analizi
    print("\n" + "="*60)
    print("STROOP TESTİ ANALİZİ")
    print("="*60)
    stroop_data = load_test_data(test_type='stroop')
    
    if stroop_data is not None and len(stroop_data) > 0:
        print(f"Toplam {len(stroop_data)} Stroop denemesi yüklendi.")
        stroop_analysis = analyze_stroop_errors(stroop_data)
        if stroop_analysis:
            print_error_statistics(stroop_analysis, 'stroop')
            visualize_stroop_errors(stroop_data, stroop_analysis)
    else:
        print("Stroop test verisi bulunamadı veya yeterli değil.")
    
    # Go/No-Go testi analizi
    print("\n" + "="*60)
    print("GO/NO-GO TESTİ ANALİZİ")
    print("="*60)
    gonogo_data = load_test_data(test_type='gonogo')
    
    if gonogo_data is not None and len(gonogo_data) > 0:
        print(f"Toplam {len(gonogo_data)} Go/No-Go denemesi yüklendi.")
        gonogo_analysis = analyze_gonogo_errors(gonogo_data)
        if gonogo_analysis:
            print_error_statistics(gonogo_analysis, 'gonogo')
            visualize_gonogo_errors(gonogo_data, gonogo_analysis)
    else:
        print("Go/No-Go test verisi bulunamadı veya yeterli değil.")
    
    print("\n✅ Hata analizi tamamlandı!")

if __name__ == '__main__':
    main()

