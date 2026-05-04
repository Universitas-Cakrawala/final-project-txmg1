"""
experiment_runner.py — Loop Eksperimen Otomatis untuk Kombinasi Fitur × Model
==============================================================================
Menjalankan semua kombinasi valid antara feature extractors dan model ML,
mencatat metrics ke results/comparison_table.csv.

Usage:
    from src.experiment_runner import run_experiments
    results_df = run_experiments(df, subset='priority')

Author: Text Mining Project — CoretTax Sentiment Classification
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def run_experiments(df: pd.DataFrame,
                    text_col: str = 'review_clean',
                    label_col: str = 'sentiment_encoded',
                    subset: str = 'priority',
                    test_size: float = 0.2,
                    random_state: int = 42,
                    n_iter: int = 5,
                    cv: int = 3,
                    save_features: bool = True,
                    save_dir: str = 'results') -> pd.DataFrame:
    """
    Jalankan semua kombinasi feature extractor × model ML.

    Args:
        df: DataFrame dengan kolom teks dan label
        text_col: Nama kolom teks yang sudah dipreprocessing
        label_col: Nama kolom label terenkode
        subset: 'priority' untuk subset penting, 'all' untuk semua
        test_size: Proporsi test split
        random_state: Random seed
        save_features: Simpan feature matrices ke disk
        save_dir: Direktori output

    Returns:
        DataFrame comparison table yang sudah di-sort berdasarkan Weighted F1
    """
    from src.feature_extractors import get_all_extractors, BM25Extractor
    from src.models import get_all_models, get_feature_type_for_extractor
    from src.evaluator import (
        evaluate_model, measure_inference_time,
        generate_comparison_table, print_top_results
    )

    print("\n" + "=" * 70)
    print("🚀 EXPERIMENT RUNNER — Klasifikasi Sentimen CoretTax")
    print("=" * 70)

    # Persiapkan data
    texts = df[text_col].fillna('').tolist()
    labels = df[label_col].values
    target_names = ['Negatif', 'Netral', 'Positif']

    # Train-test split (stratified)
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size,
        random_state=random_state, stratify=labels
    )

    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(texts_train)} | Test: {len(texts_test)}")
    print(f"   Train distribution: {np.bincount(y_train)}")
    print(f"   Test distribution:  {np.bincount(y_test)}")

    # Get extractors & models
    extractors = get_all_extractors(subset=subset)
    all_models = get_all_models(subset=subset)

    print(f"\n🔧 Extractors: {list(extractors.keys())}")
    print(f"🤖 Models: {list(all_models.keys())}")

    # Storage untuk hasil
    all_results = []
    feature_cache = {}  # Cache agar tidak re-extract

    total_combos = sum(
        1 for ext_name in extractors
        for model_name, model in all_models.items()
        if model.is_compatible(get_feature_type_for_extractor(ext_name))
    )
    print(f"\n📈 Total kombinasi valid: {total_combos}")
    print("-" * 70)

    combo_idx = 0

    for ext_name, extractor in extractors.items():
        print(f"\n{'='*60}")
        print(f"📦 Feature Extractor: {ext_name}")
        print(f"{'='*60}")

        # Extract features (atau ambil dari cache)
        if ext_name not in feature_cache:
            try:
                ext_start = time.time()

                # BM25 membutuhkan labels
                if isinstance(extractor, BM25Extractor):
                    X_train, X_test = extractor.fit_transform(
                        texts_train, texts_test, train_labels=y_train
                    )
                else:
                    X_train, X_test = extractor.fit_transform(
                        texts_train, texts_test
                    )

                ext_time = time.time() - ext_start
                feature_cache[ext_name] = (X_train, X_test, ext_time)

                # Simpan feature matrices
                if save_features:
                    _save_features(X_train, X_test, ext_name, save_dir)

                print(f"   ⏱️  Extraction time: {ext_time:.1f}s")

            except Exception as e:
                print(f"   ❌ Error extracting {ext_name}: {e}")
                continue
        else:
            X_train, X_test, ext_time = feature_cache[ext_name]
            print(f"   📋 Using cached features")

        feature_type = get_feature_type_for_extractor(ext_name)

        # Jalankan semua model yang kompatibel
        for model_name, model_wrapper in all_models.items():
            if not model_wrapper.is_compatible(feature_type):
                print(f"\n   ⏭️  Skip: {model_name} (tidak kompatibel dengan {feature_type})")
                continue

            combo_idx += 1
            print(f"\n   [{combo_idx}/{total_combos}] 🤖 {ext_name} + {model_name}")

            try:
                # Fit model
                model_wrapper.fit(X_train, y_train, cv=cv, n_iter=n_iter)

                # Predict
                y_pred = model_wrapper.predict(X_test)
                y_proba = model_wrapper.predict_proba(X_test)

                # Evaluate
                metrics = evaluate_model(
                    y_test, y_pred, y_proba,
                    labels=[0, 1, 2],
                    target_names=target_names
                )

                # Inference time
                infer_time = measure_inference_time(model_wrapper, X_test)

                # Compile result
                result = {
                    'feature_extractor': ext_name,
                    'model': model_name,
                    'weighted_f1': metrics['weighted_f1'],
                    'macro_f1': metrics['macro_f1'],
                    'roc_auc': metrics.get('roc_auc'),
                    'accuracy': metrics['accuracy'],
                    'train_time_s': round(model_wrapper.train_time, 2),
                    'infer_time_ms': infer_time,
                    'best_params': model_wrapper.best_params,
                    'classification_report': metrics['classification_report'],
                }
                all_results.append(result)

                print(f"      ✅ W-F1={metrics['weighted_f1']:.4f} | "
                      f"M-F1={metrics['macro_f1']:.4f} | "
                      f"Time={model_wrapper.train_time:.1f}s")

            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue

    # Generate comparison table
    print("\n" + "=" * 70)
    comparison_df = generate_comparison_table(
        all_results,
        save_path=os.path.join(save_dir, 'comparison_table.csv')
    )

    print_top_results(comparison_df, n=10)

    return comparison_df


def _save_features(X_train, X_test, name: str, save_dir: str):
    """Simpan feature matrices ke disk."""
    feat_dir = os.path.join(save_dir, 'feature_matrices')
    os.makedirs(feat_dir, exist_ok=True)

    safe_name = name.lower().replace(' ', '_').replace('/', '_')

    if sp.issparse(X_train):
        sp.save_npz(os.path.join(feat_dir, f'X_train_{safe_name}.npz'), X_train)
        sp.save_npz(os.path.join(feat_dir, f'X_test_{safe_name}.npz'), X_test)
    else:
        np.save(os.path.join(feat_dir, f'X_train_{safe_name}.npy'), X_train)
        np.save(os.path.join(feat_dir, f'X_test_{safe_name}.npy'), X_test)

    print(f"   💾 Features saved: {safe_name}")
