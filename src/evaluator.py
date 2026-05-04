"""
evaluator.py — Framework Evaluasi untuk Klasifikasi Sentimen
=============================================================
Metrics, visualisasi, dan analisis error untuk evaluasi model.

Fungsi utama:
    - evaluate_model(): Hitung semua metrics
    - plot_confusion_matrix(): Confusion matrix heatmap
    - plot_comparison_quadrant(): F1 vs Training Time scatter
    - error_analysis(): False positive/negative analysis
    - generate_comparison_table(): Tabel ringkasan semua eksperimen

Author: Text Mining Project — CoretTax Sentiment Classification
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')


def evaluate_model(y_true, y_pred, y_proba=None,
                   labels=None, target_names=None) -> Dict:
    """
    Prompt 5.1 — Evaluasi komprehensif satu model.

    Returns dict dengan:
        - weighted_f1, macro_f1
        - precision, recall (per-class dan weighted)
        - accuracy
        - roc_auc (jika y_proba tersedia)
        - classification_report (string)
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        accuracy_score, classification_report, roc_auc_score
    )

    results = {
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted'),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted'),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels)
    if target_names and len(target_names) == len(per_class_f1):
        for name, score in zip(target_names, per_class_f1):
            results[f'f1_{name}'] = score

    # ROC-AUC (jika probabilitas tersedia)
    if y_proba is not None:
        try:
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                results['roc_auc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                )
            else:
                results['roc_auc'] = roc_auc_score(y_true, y_proba)
        except (ValueError, TypeError):
            results['roc_auc'] = None

    # Classification report (string)
    results['classification_report'] = classification_report(
        y_true, y_pred, target_names=target_names, digits=4
    )

    return results


def measure_inference_time(model_wrapper, X_test, n_runs: int = 3) -> float:
    """
    Ukur rata-rata inference time per 1000 sampel (ms).
    """
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = model_wrapper.predict(X_test)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    per_1000 = (avg_time / len(X_test) if hasattr(X_test, '__len__')
                else avg_time / X_test.shape[0]) * 1000 * 1000  # ms per 1000

    return round(per_1000, 2)


def plot_confusion_matrix(y_true, y_pred, target_names=None,
                          title: str = 'Confusion Matrix',
                          save_path: Optional[str] = None):
    """
    Plot confusion matrix heatmap.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names or 'auto',
                yticklabels=target_names or 'auto',
                ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")

    plt.show()
    plt.close()


def plot_comparison_quadrant(results_df: pd.DataFrame,
                             save_path: Optional[str] = None):
    """
    Prompt 5.1 — Quadrant Plot: Weighted F1 vs Training Time.

    Warna berdasarkan kategori feature extractor:
    - Merah: Statistik (TF-IDF, BM25)
    - Hijau: Embeddings (Word2Vec, GloVe, FastText)
    - Biru: Transformer (DistilBERT, BERT, RoBERTa)
    """
    import matplotlib.pyplot as plt

    # Mapping warna
    color_map = {
        'TF-IDF': '#E74C3C', 'BM25': '#E74C3C',
        'Word2Vec': '#2ECC71', 'GloVe': '#2ECC71', 'FastText': '#27AE60',
        'DistilBERT': '#3498DB', 'IndoBERT': '#2980B9', 'RoBERTa': '#2471A3',
    }
    category_colors = {
        'Statistik': '#E74C3C',
        'Embeddings': '#2ECC71',
        'Transformer': '#3498DB',
    }

    def _get_category(extractor):
        if extractor in ('TF-IDF', 'BM25'):
            return 'Statistik'
        elif extractor in ('Word2Vec', 'GloVe', 'FastText'):
            return 'Embeddings'
        else:
            return 'Transformer'

    fig, ax = plt.subplots(figsize=(12, 8))

    for _, row in results_df.iterrows():
        extractor = row['feature_extractor']
        category = _get_category(extractor)
        color = category_colors.get(category, '#95A5A6')

        ax.scatter(
            row['train_time_s'], row['weighted_f1'],
            c=color, s=100, alpha=0.7, edgecolors='white', linewidth=1
        )
        ax.annotate(
            f"{extractor}\n+{row['model']}",
            (row['train_time_s'], row['weighted_f1']),
            fontsize=7, ha='center', va='bottom', alpha=0.8
        )

    # Legend
    for cat, color in category_colors.items():
        ax.scatter([], [], c=color, s=100, label=cat, alpha=0.7)
    ax.legend(fontsize=10, loc='lower right')

    ax.set_xlabel('Training Time (detik)', fontsize=12)
    ax.set_ylabel('Weighted F1-Score', fontsize=12)
    ax.set_title('🎯 Quadrant Plot: Performance vs. Efficiency',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")

    plt.show()
    plt.close()


def error_analysis(y_true, y_pred, texts, target_names=None,
                   n_examples: int = 10) -> Dict:
    """
    Prompt 5.2 — Analisis error: False Positive & False Negative.

    Returns dict dengan contoh-contoh kesalahan prediksi per kelas.
    """
    errors = {
        'false_positives': [],
        'false_negatives': [],
        'total_errors': 0,
        'error_rate': 0.0,
    }

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Total errors
    error_mask = y_true != y_pred
    errors['total_errors'] = error_mask.sum()
    errors['error_rate'] = errors['total_errors'] / len(y_true)

    # Kumpulkan contoh error
    error_indices = np.where(error_mask)[0]
    for idx in error_indices[:n_examples * 3]:  # Ambil lebih banyak, filter nanti
        true_label = target_names[y_true[idx]] if target_names else str(y_true[idx])
        pred_label = target_names[y_pred[idx]] if target_names else str(y_pred[idx])
        text = texts[idx] if idx < len(texts) else "N/A"

        entry = {
            'text': str(text)[:200],
            'true_label': true_label,
            'pred_label': pred_label,
        }

        # Categorize: FP = prediksi Positif padahal sebenarnya bukan
        if y_pred[idx] == 2:  # Positif
            errors['false_positives'].append(entry)
        else:
            errors['false_negatives'].append(entry)

    # Trim to n_examples
    errors['false_positives'] = errors['false_positives'][:n_examples]
    errors['false_negatives'] = errors['false_negatives'][:n_examples]

    return errors


def generate_comparison_table(all_results: List[Dict],
                              save_path: str = 'results/comparison_table.csv') -> pd.DataFrame:
    """
    Prompt 5.1 — Generate tabel perbandingan semua eksperimen.

    Kolom: feature_extractor, model, weighted_f1, macro_f1, auc,
           train_time_s, infer_time_ms
    """
    rows = []
    for result in all_results:
        rows.append({
            'feature_extractor': result.get('feature_extractor', ''),
            'model': result.get('model', ''),
            'weighted_f1': result.get('weighted_f1', 0),
            'macro_f1': result.get('macro_f1', 0),
            'roc_auc': result.get('roc_auc', None),
            'accuracy': result.get('accuracy', 0),
            'train_time_s': result.get('train_time_s', 0),
            'infer_time_ms': result.get('infer_time_ms', 0),
            'best_params': str(result.get('best_params', {})),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('weighted_f1', ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # Rank mulai dari 1
    df.index.name = 'Rank'

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"\n💾 Comparison table saved: {save_path}")

    return df


def print_top_results(comparison_df: pd.DataFrame, n: int = 10):
    """Print top N kombinasi terbaik."""
    print(f"\n🏆 Top {n} Kombinasi Terbaik (Weighted F1):")
    print("=" * 90)
    print(f"{'Rank':>4} | {'Feature Extractor':>18} | {'Model':>22} | "
          f"{'W-F1':>6} | {'M-F1':>6} | {'AUC':>6} | {'Train(s)':>8}")
    print("-" * 90)

    for idx, row in comparison_df.head(n).iterrows():
        auc_str = f"{row['roc_auc']:.4f}" if row['roc_auc'] else 'N/A'
        print(f"{idx:>4} | {row['feature_extractor']:>18} | {row['model']:>22} | "
              f"{row['weighted_f1']:.4f} | {row['macro_f1']:.4f} | "
              f"{auc_str:>6} | {row['train_time_s']:>8.1f}")


def plot_rating_distribution(df: pd.DataFrame,
                             save_path: Optional[str] = None):
    """Prompt 0.3 — Bar chart distribusi rating."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 5))
    rating_counts = df['rating'].value_counts().sort_index()

    colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#27AE60']
    bars = ax.bar(rating_counts.index, rating_counts.values, color=colors,
                  edgecolor='white', linewidth=1.5)

    # Tambahkan label di atas bar
    for bar, count in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Jumlah Review', fontsize=12)
    ax.set_title('📊 Distribusi Rating Ulasan CoretTax', fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['⭐1', '⭐2', '⭐3', '⭐4', '⭐5'])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_monthly_trend(df: pd.DataFrame, save_path: Optional[str] = None):
    """Prompt 0.3 — Grafik tren jumlah review per bulan."""
    import matplotlib.pyplot as plt

    monthly = df.groupby('year_month').size().reset_index(name='count')
    monthly = monthly.sort_values('year_month')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly['year_month'], monthly['count'],
            marker='o', linewidth=2, color='#3498DB', markersize=6)
    ax.fill_between(monthly['year_month'], monthly['count'],
                    alpha=0.1, color='#3498DB')

    ax.set_xlabel('Bulan', fontsize=12)
    ax.set_ylabel('Jumlah Review', fontsize=12)
    ax.set_title('📈 Tren Jumlah Review per Bulan', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_wordcloud(texts, title: str = 'Word Cloud',
                   save_path: Optional[str] = None):
    """Prompt 0.3 — Word cloud visualisasi."""
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    text = ' '.join(texts)
    wc = WordCloud(
        width=1200, height=600,
        background_color='white',
        max_words=200,
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()
