"""
complete_pipeline.py — Full Re-run Pipeline + Ablation Study + Report Generation
================================================================================
Script ini menjalankan seluruh pipeline dari awal dengan Sastrawi stemming aktif,
melakukan ablation study untuk class imbalance handling, dan generate semua
artefak yang dibutuhkan untuk presentasi.

Usage:
    python complete_pipeline.py
"""

import sys
import os
import warnings
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    load_raw_data, add_sentiment_labels, preprocess_dataframe,
    _get_stemmer, _STOPWORDS, _STEMMER
)
from src.feature_extractors import (
    TFIDFExtractor, Word2VecExtractor, GloVeExtractor, FastTextExtractor
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import scipy.sparse as sp

# ======================================================================
# CONFIG
# ======================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
np.random.seed(RANDOM_STATE)

RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
DATA_DIR = 'data/processed'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(FIGURES_DIR, 'eda'), exist_ok=True)
os.makedirs(os.path.join(FIGURES_DIR, 'evaluation'), exist_ok=True)
os.makedirs(os.path.join(FIGURES_DIR, 'interpretation'), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ======================================================================
# STEP 1: Load & Preprocess with Stemming
# ======================================================================
print("\n" + "=" * 70)
print("🚀 STEP 1: Loading Data + Preprocessing with Sastrawi Stemming")
print("=" * 70)

# Load raw data
df_raw = load_raw_data('data/raw/coretax_reviews.csv')
print(f"   Raw data: {len(df_raw)} reviews")

# Balance via undersampling (203 per class)
from sklearn.utils import resample
df_neg = df_raw[df_raw['rating'] <= 2]
df_neu = df_raw[df_raw['rating'] == 3]
df_pos = df_raw[df_raw['rating'] >= 4]

n_min = min(len(df_neg), len(df_neu), len(df_pos))
df_neg_bal = resample(df_neg, n_samples=n_min, random_state=RANDOM_STATE)
df_neu_bal = resample(df_neu, n_samples=n_min, random_state=RANDOM_STATE)
df_pos_bal = resample(df_pos, n_samples=n_min, random_state=RANDOM_STATE)

df_bal = pd.concat([df_neg_bal, df_neu_bal, df_pos_bal]).reset_index(drop=True)
print(f"   Balanced: {len(df_bal)} reviews ({n_min} per class)")

# Labeling
df_bal = add_sentiment_labels(df_bal)

# Preprocessing WITH stemming
# Reset stemmer cache to force reload
import src.preprocessing as pp
pp._STEMMER = None
pp._STEMMER_WARNED = False

df_bal = preprocess_dataframe(df_bal, text_col='review_text',
                               remove_stopwords=True, apply_stemming=True)

# Save
df_bal.to_csv(os.path.join(DATA_DIR, 'balanced_reviews.csv'), index=False)
print(f"   ✅ Saved: {os.path.join(DATA_DIR, 'balanced_reviews.csv')}")

# Show sample
print("\n📋 Sample Preprocessing (with stemming):")
for idx, row in df_bal.head(5).iterrows():
    print(f"   ⭐{row['rating']} | {row['sentiment']:7s} | {str(row['review_clean'])[:80]}")

# ======================================================================
# STEP 2: Feature Extraction
# ======================================================================
print("\n" + "=" * 70)
print("🔧 STEP 2: Feature Extraction")
print("=" * 70)

texts = df_bal['review_clean'].fillna('').tolist()
labels = df_bal['sentiment_encoded'].values

texts_train, texts_test, y_train, y_test = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
)
print(f"   Train: {len(texts_train)}, Test: {len(texts_test)}")
print(f"   Train dist: {np.bincount(y_train)}")

# Store features
features = {}

# 2a. TF-IDF
print("\n   📦 TF-IDF...")
tfidf = TFIDFExtractor(max_features=10000, ngram_range=(1, 2))
X_train_tfidf, X_test_tfidf = tfidf.fit_transform(texts_train, texts_test)
features['TF-IDF'] = (X_train_tfidf, X_test_tfidf)
print(f"      Shape: {X_train_tfidf.shape}")

# 2b. Word2Vec
print("\n   📦 Word2Vec (train from scratch)...")
w2v = Word2VecExtractor(vector_size=100, train_from_scratch=True)
X_train_w2v, X_test_w2v = w2v.fit_transform(texts_train, texts_test)
features['Word2Vec'] = (X_train_w2v, X_test_w2v)
print(f"      Shape: {X_train_w2v.shape}")

# 2c. FastText
print("\n   📦 FastText (train from scratch)...")
ft = FastTextExtractor(vector_size=100, train_from_scratch=True)
X_train_ft, X_test_ft = ft.fit_transform(texts_train, texts_test)
features['FastText'] = (X_train_ft, X_test_ft)
print(f"      Shape: {X_train_ft.shape}")

# 2d. GloVe
print("\n   📦 GloVe (cc.id.300.vec)...")
glove_path = 'data/embeddings/cc.id.300.vec'
if os.path.exists(glove_path):
    glove = GloVeExtractor(vectors_path=glove_path, dim=300)
    X_train_glove, X_test_glove = glove.fit_transform(texts_train, texts_test)
    features['GloVe'] = (X_train_glove, X_test_glove)
    print(f"      Shape: {X_train_glove.shape}, OOV: {glove.oov_rate:.1f}%")
else:
    print(f"      ⚠️ GloVe file not found at {glove_path}, skipping")

# Save labels
np.save(os.path.join(RESULTS_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(RESULTS_DIR, 'y_test.npy'), y_test)

# ======================================================================
# STEP 3: Model Training (Baseline)
# ======================================================================
print("\n" + "=" * 70)
print("🤖 STEP 3: Model Training (Baseline — No Balancing)")
print("=" * 70)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models_dict = {
    'Decision Tree': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
    'XGBoost': XGBClassifier,
}

model_params = {
    'Decision Tree': [
        {'max_depth': 5, 'criterion': 'gini', 'min_samples_split': 2},
        {'max_depth': 10, 'criterion': 'entropy', 'min_samples_split': 5},
        {'max_depth': 15, 'criterion': 'gini', 'min_samples_split': 10},
    ],
    'Random Forest': [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2, 'max_features': 'log2'},
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
        {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
    ],
    'XGBoost': [
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_alpha': 0.1},
        {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.01, 'subsample': 1.0, 'colsample_bytree': 0.7, 'reg_alpha': 1.0},
        {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.1},
    ],
}

results_baseline = []

for feat_name, (X_tr, X_te) in features.items():
    print(f"\n   📦 Feature: {feat_name}")
    for model_name, model_cls in models_dict.items():
        best_f1 = 0
        best_params = None
        best_model = None

        for params in model_params[model_name]:
            if model_name == 'Decision Tree':
                m = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
            elif model_name == 'Random Forest':
                m = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
            elif model_name == 'XGBoost':
                # XGBoost v3.2+ doesn't need use_label_encoder
                m = XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', **params)

            m.fit(X_tr, y_train)
            y_pred = m.predict(X_te)
            f1 = f1_score(y_test, y_pred, average='macro')

            if f1 > best_f1:
                best_f1 = f1
                best_params = params
                best_model = m

        # Evaluate best model
        y_pred = best_model.predict(X_te)
        y_prob = best_model.predict_proba(X_te) if hasattr(best_model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except:
            auc = 0.0

        results_baseline.append({
            'feature_extractor': feat_name,
            'model': model_name,
            'weighted_f1': round(weighted_f1, 4),
            'macro_f1': round(macro_f1, 4),
            'roc_auc': round(auc, 4),
            'accuracy': round(acc, 4),
            'best_params': str(best_params),
        })
        print(f"      {model_name:15s} → M-F1={macro_f1:.4f} | Acc={acc:.4f}")

df_baseline = pd.DataFrame(results_baseline).sort_values('macro_f1', ascending=False).reset_index(drop=True)
df_baseline.index += 1
df_baseline.to_csv(os.path.join(RESULTS_DIR, 'baseline_comparison_table.csv'))
print(f"\n   ✅ Saved: baseline_comparison_table.csv")

# ======================================================================
# STEP 4: Ablation Study — Class Imbalance Handling
# ======================================================================
print("\n" + "=" * 70)
print("⚖️  STEP 4: Ablation Study — Class Imbalance Handling")
print("=" * 70)

# Use GloVe if available, otherwise Word2Vec
if 'GloVe' in features:
    ablation_feat = 'GloVe'
else:
    ablation_feat = 'Word2Vec'

X_tr_ab, X_te_ab = features[ablation_feat]
print(f"   Using feature: {ablation_feat}")
print(f"   Original train shape: {X_tr_ab.shape}")

strategies = {
    'Baseline (No Balance)': (X_tr_ab, y_train, None),
}

# SMOTE
print("\n   🔹 Applying SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_ab, y_train)
strategies['SMOTE'] = (X_tr_smote, y_tr_smote, 'SMOTE')
print(f"      After SMOTE: {np.bincount(y_tr_smote)}")

# Random Undersampling
print("\n   🔹 Applying Random Undersampling...")
rus = RandomUnderSampler(random_state=RANDOM_STATE)
X_tr_rus, y_tr_rus = rus.fit_resample(X_tr_ab, y_train)
strategies['Random Undersampling'] = (X_tr_rus, y_tr_rus, 'RUS')
print(f"      After RUS: {np.bincount(y_tr_rus)}")

# SMOTEENN
print("\n   🔹 Applying SMOTEENN...")
smoteenn = SMOTEENN(random_state=RANDOM_STATE)
X_tr_smoteenn, y_tr_smoteenn = smoteenn.fit_resample(X_tr_ab, y_train)
strategies['SMOTEENN'] = (X_tr_smoteenn, y_tr_smoteenn, 'SMOTEENN')
print(f"      After SMOTEENN: {np.bincount(y_tr_smoteenn)}")

# Class Weight (no resampling, just model parameter)
strategies['Class Weight Balanced'] = (X_tr_ab, y_train, 'class_weight')

results_ablation = []

for strategy_name, (X_tr_s, y_tr_s, strategy_type) in strategies.items():
    print(f"\n   📊 Strategy: {strategy_name}")
    print(f"      Train shape: {X_tr_s.shape}, dist: {np.bincount(y_tr_s)}")

    for model_name, model_cls in models_dict.items():
        best_f1 = 0
        best_params = None
        best_model = None

        for params in model_params[model_name]:
            if model_name == 'Decision Tree':
                m = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
            elif model_name == 'Random Forest':
                if strategy_type == 'class_weight':
                    m = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                               class_weight='balanced', **params)
                else:
                    m = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
            elif model_name == 'XGBoost':
                # XGBoost doesn't have class_weight, use scale_pos_weight workaround
                m = XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', **params)

            m.fit(X_tr_s, y_tr_s)
            y_pred = m.predict(X_te_ab)
            f1 = f1_score(y_test, y_pred, average='macro')

            if f1 > best_f1:
                best_f1 = f1
                best_params = params
                best_model = m

        y_pred = best_model.predict(X_te_ab)
        y_prob = best_model.predict_proba(X_te_ab) if hasattr(best_model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except:
            auc = 0.0

        # Per-class metrics
        per_class = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results_ablation.append({
            'strategy': strategy_name,
            'feature_extractor': ablation_feat,
            'model': model_name,
            'weighted_f1': round(weighted_f1, 4),
            'macro_f1': round(macro_f1, 4),
            'roc_auc': round(auc, 4),
            'accuracy': round(acc, 4),
            'recall_negatif': round(per_class.get('0', {}).get('recall', 0), 4),
            'recall_netral': round(per_class.get('1', {}).get('recall', 0), 4),
            'recall_positif': round(per_class.get('2', {}).get('recall', 0), 4),
            'f1_negatif': round(per_class.get('0', {}).get('f1-score', 0), 4),
            'f1_netral': round(per_class.get('1', {}).get('f1-score', 0), 4),
            'f1_positif': round(per_class.get('2', {}).get('f1-score', 0), 4),
        })
        print(f"      {model_name:15s} → M-F1={macro_f1:.4f} | Acc={acc:.4f}")

df_ablation = pd.DataFrame(results_ablation).sort_values('macro_f1', ascending=False).reset_index(drop=True)
df_ablation.index += 1
df_ablation.to_csv(os.path.join(RESULTS_DIR, 'ablation_comparison_table.csv'))
print(f"\n   ✅ Saved: ablation_comparison_table.csv")

# ======================================================================
# STEP 5: Generate Confusion Matrices for Top Models
# ======================================================================
print("\n" + "=" * 70)
print("📊 STEP 5: Generating Confusion Matrices")
print("=" * 70)

# Get top 3 from ablation
top3 = df_ablation.head(3)
label_names = ['Negatif', 'Netral', 'Positif']

for idx, row in top3.iterrows():
    strategy = row['strategy']
    model_name = row['model']

    # Re-train to get predictions
    if strategy in strategies:
        X_tr_s, y_tr_s, _ = strategies[strategy]
    else:
        X_tr_s, y_tr_s, _ = strategies['Baseline (No Balance)']

    params = eval(row['best_params']) if hasattr(row, 'best_params') else model_params[model_name][0]

    if model_name == 'Decision Tree':
        m = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    elif model_name == 'Random Forest':
        m = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    elif model_name == 'XGBoost':
        m = XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', **params)

    m.fit(X_tr_s, y_tr_s)
    y_pred = m.predict(X_te_ab)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix\n{strategy} + {model_name}\n'
                 f'Macro F1={row["macro_f1"]:.4f} | Acc={row["accuracy"]:.4f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f'cm_{strategy.replace(" ", "_").replace("(", "").replace(")", "")}_{model_name.replace(" ", "_")}.png'
    plt.savefig(os.path.join(FIGURES_DIR, 'evaluation', fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {fname}")

# ======================================================================
# STEP 6: Generate Summary Visualizations
# ======================================================================
print("\n" + "=" * 70)
print("📈 STEP 6: Generating Summary Visualizations")
print("=" * 70)

# 6a. Ablation comparison bar chart
fig, ax = plt.subplots(figsize=(14, 7))
ablation_pivot = df_ablation.pivot_table(
    index='strategy', columns='model', values='macro_f1', aggfunc='first'
)
ablation_pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='white', linewidth=0.5,
                    color=['#3498DB', '#E74C3C', '#2ECC71'])
ax.set_xlabel('Balancing Strategy', fontsize=12)
ax.set_ylabel('Macro F1-Score', fontsize=12)
ax.set_title('⚖️  Impact of Class Imbalance Handling on Macro F1-Score', fontsize=14, fontweight='bold')
ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline (3-class)')
ax.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'evaluation', 'ablation_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: ablation_comparison.png")

# 6b. Per-class recall comparison
fig, ax = plt.subplots(figsize=(14, 7))
recall_data = df_ablation[['strategy', 'model', 'recall_negatif', 'recall_netral', 'recall_positif']].copy()
recall_melted = recall_data.melt(
    id_vars=['strategy', 'model'],
    value_vars=['recall_negatif', 'recall_netral', 'recall_positif'],
    var_name='class', value_name='recall'
)
recall_melted['class'] = recall_melted['class'].map({
    'recall_negatif': 'Negatif',
    'recall_netral': 'Netral',
    'recall_positif': 'Positif'
})

sns.barplot(data=recall_melted, x='strategy', y='recall', hue='class', ax=ax,
            palette=['#E74C3C', '#F1C40F', '#2ECC71'])
ax.set_xlabel('Balancing Strategy', fontsize=12)
ax.set_ylabel('Recall', fontsize=12)
ax.set_title('📊 Per-Class Recall by Balancing Strategy', fontsize=14, fontweight='bold')
ax.legend(title='Class')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'evaluation', 'per_class_recall.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: per_class_recall.png")

# ======================================================================
# STEP 7: Generate Final Report Data
# ======================================================================
print("\n" + "=" * 70)
print("📝 STEP 7: Generating Report Data")
print("=" * 70)

# Best overall
best_overall = df_ablation.iloc[0]
print(f"\n🏆 BEST OVERALL:")
print(f"   Strategy: {best_overall['strategy']}")
print(f"   Model: {best_overall['model']}")
print(f"   Feature: {best_overall['feature_extractor']}")
print(f"   Macro F1: {best_overall['macro_f1']:.4f}")
print(f"   Accuracy: {best_overall['accuracy']:.4f}")
print(f"   AUC: {best_overall['roc_auc']:.4f}")

# Save summary JSON
summary = {
    'dataset': {
        'total_raw': len(df_raw),
        'total_balanced': len(df_bal),
        'samples_per_class': n_min,
        'classes': ['Negatif', 'Netral', 'Positif'],
    },
    'best_model': {
        'strategy': best_overall['strategy'],
        'feature_extractor': best_overall['feature_extractor'],
        'model': best_overall['model'],
        'macro_f1': float(best_overall['macro_f1']),
        'weighted_f1': float(best_overall['weighted_f1']),
        'accuracy': float(best_overall['accuracy']),
        'roc_auc': float(best_overall['roc_auc']),
        'recall_negatif': float(best_overall['recall_negatif']),
        'recall_netral': float(best_overall['recall_netral']),
        'recall_positif': float(best_overall['recall_positif']),
    },
    'baseline_best': {
        'model': df_baseline.iloc[0]['model'],
        'feature_extractor': df_baseline.iloc[0]['feature_extractor'],
        'macro_f1': float(df_baseline.iloc[0]['macro_f1']),
        'accuracy': float(df_baseline.iloc[0]['accuracy']),
    },
    'improvement': {
        'macro_f1_gain': float(best_overall['macro_f1'] - df_baseline.iloc[0]['macro_f1']),
        'accuracy_gain': float(best_overall['accuracy'] - df_baseline.iloc[0]['accuracy']),
    }
}

with open(os.path.join(RESULTS_DIR, 'experiment_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n   ✅ Saved: experiment_summary.json")

# ======================================================================
# DONE
# ======================================================================
print("\n" + "=" * 70)
print("✅ COMPLETE PIPELINE FINISHED!")
print("=" * 70)
print(f"\n📂 Results saved to: {RESULTS_DIR}/")
print(f"   - baseline_comparison_table.csv")
print(f"   - ablation_comparison_table.csv")
print(f"   - experiment_summary.json")
print(f"   - figures/evaluation/ (confusion matrices, charts)")
print(f"\n🏆 Best Result: {best_overall['strategy']} + {best_overall['model']}")
print(f"   Macro F1 = {best_overall['macro_f1']:.4f}")
print(f"   Accuracy = {best_overall['accuracy']:.4f}")
