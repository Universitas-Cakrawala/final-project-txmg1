"""
models.py — Model Machine Learning untuk Klasifikasi Sentimen
==============================================================
9 model ML yang diimplementasikan dengan wrapper konsisten:

Tree-based:
    1. Decision Tree
    2. Random Forest

Boosting:
    3. XGBoost
    4. LightGBM
    5. CatBoost

Linear:
    6. Logistic Regression
    7. SVM (LinearSVC)

Probabilistic:
    8. Naive Bayes (MultinomialNB / GaussianNB / ComplementNB)

Neural:
    9. MLP (Multi-Layer Perceptron)

Usage:
    from src.models import get_all_models
    models = get_all_models(feature_type='sparse')

Author: Text Mining Project — CoretTax Sentiment Classification
"""

import time
import warnings
import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings('ignore')


class ModelWrapper:
    """
    Wrapper konsisten untuk semua model ML.

    Mendukung:
    - Hyperparameter tuning via RandomizedSearchCV
    - Training time tracking
    - Kompatibilitas check (sparse vs dense features)
    """

    def __init__(self, name: str, model, param_grid: dict,
                 compatible_features: list = None,
                 needs_dense: bool = False):
        """
        Args:
            name: Nama model (untuk display)
            model: Sklearn-compatible model instance
            param_grid: Dictionary hyperparameter untuk tuning
            compatible_features: List tipe feature yang kompatibel
                                 ('sparse', 'dense_low', 'dense_high')
            needs_dense: Jika True, konversi sparse ke dense sebelum fit
        """
        self.name = name
        self.model = model
        self.param_grid = param_grid
        self.compatible_features = compatible_features or ['sparse', 'dense_low', 'dense_high']
        self.needs_dense = needs_dense
        self.best_model = None
        self.best_params = None
        self.train_time = 0.0

    def is_compatible(self, feature_type: str) -> bool:
        """Check apakah model kompatibel dengan tipe feature tertentu."""
        return feature_type in self.compatible_features

    def fit(self, X_train, y_train, cv: int = 3, n_iter: int = 20,
            scoring: str = 'f1_weighted') -> 'ModelWrapper':
        """
        Fit model dengan hyperparameter tuning.

        Args:
            X_train: Feature matrix (sparse atau dense)
            y_train: Labels
            cv: Jumlah cross-validation folds
            n_iter: Jumlah iterasi random search
            scoring: Metric untuk optimisasi

        Returns:
            self (fitted)
        """
        # Konversi sparse ke dense jika diperlukan
        X = X_train
        if self.needs_dense and sp.issparse(X):
            X = X.toarray()

        start_time = time.time()

        if self.param_grid:
            search = RandomizedSearchCV(
                self.model,
                self.param_grid,
                n_iter=min(n_iter, self._count_combinations()),
                cv=cv,
                scoring=scoring,
                random_state=42,
                n_jobs=-1,
                error_score='raise'
            )
            search.fit(X, y_train)
            self.best_model = search.best_estimator_
            self.best_params = search.best_params_
        else:
            self.model.fit(X, y_train)
            self.best_model = self.model
            self.best_params = {}

        self.train_time = time.time() - start_time
        return self

    def predict(self, X_test) -> np.ndarray:
        """Predict labels."""
        X = X_test
        if self.needs_dense and sp.issparse(X):
            X = X.toarray()
        return self.best_model.predict(X)

    def predict_proba(self, X_test) -> Optional[np.ndarray]:
        """Predict probabilities (jika model mendukung)."""
        X = X_test
        if self.needs_dense and sp.issparse(X):
            X = X.toarray()
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        elif hasattr(self.best_model, 'decision_function'):
            return self.best_model.decision_function(X)
        return None

    def _count_combinations(self) -> int:
        """Hitung total kombinasi hyperparameter."""
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total

    def __repr__(self):
        return f"ModelWrapper(name='{self.name}', fitted={self.best_model is not None})"


# ======================================================================
# Model Factory Functions
# ======================================================================

def _decision_tree() -> ModelWrapper:
    """Prompt 4.1 — Decision Tree."""
    from sklearn.tree import DecisionTreeClassifier

    return ModelWrapper(
        name='Decision Tree',
        model=DecisionTreeClassifier(class_weight='balanced', random_state=42),
        param_grid={
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None],
        },
        needs_dense=True  # Fix ambiguous length error
    )


def _random_forest() -> ModelWrapper:
    """Prompt 4.2 — Random Forest."""
    from sklearn.ensemble import RandomForestClassifier

    return ModelWrapper(
        name='Random Forest',
        model=RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        param_grid={
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, None],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [1, 2, 4],
        },
        needs_dense=True  # Fix ambiguous length error
    )


def _xgboost() -> ModelWrapper:
    """Prompt 4.3 — XGBoost."""
    from xgboost import XGBClassifier

    return ModelWrapper(
        name='XGBoost',
        model=XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        ),
        param_grid={
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
        },
        needs_dense=True  # Fix ambiguous length error
    )


def _lightgbm() -> ModelWrapper:
    """Prompt 4.7 — LightGBM."""
    from lightgbm import LGBMClassifier

    return ModelWrapper(
        name='LightGBM',
        model=LGBMClassifier(
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        param_grid={
            'num_leaves': [31, 63, 127],
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.7, 0.8, 1.0],
            'min_child_samples': [10, 20, 50],
        },
        needs_dense=True  # Fix ambiguous length error
    )


def _catboost() -> ModelWrapper:
    """Prompt 4.9 — CatBoost."""
    from catboost import CatBoostClassifier

    return ModelWrapper(
        name='CatBoost',
        model=CatBoostClassifier(
            auto_class_weights='Balanced',
            random_state=42,
            verbose=0
        ),
        param_grid={
            'iterations': [200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5],
        },
        compatible_features=['dense_low', 'dense_high'],  # Tidak efisien untuk sparse
        needs_dense=True
    )


def _logistic_regression() -> ModelWrapper:
    """Prompt 4.5 — Logistic Regression."""
    from sklearn.linear_model import LogisticRegression

    return ModelWrapper(
        name='Logistic Regression',
        model=LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        param_grid={
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
        },
        needs_dense=True  # Fix ambiguous length error
    )


def _svm() -> ModelWrapper:
    """Prompt 4.4 — LinearSVC."""
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    # CalibratedClassifierCV agar mendukung predict_proba
    base_svm = LinearSVC(
        class_weight='balanced',
        max_iter=2000,
        random_state=42
    )

    return ModelWrapper(
        name='SVM (LinearSVC)',
        model=CalibratedClassifierCV(base_svm, cv=3),
        param_grid={
            'estimator__C': [0.01, 0.1, 1, 10, 100],
        },
        needs_dense=True  # Paksa ke dense untuk menghindari error 'ambiguous length' pada CalibratedClassifierCV
    )


def _naive_bayes_multinomial() -> ModelWrapper:
    """Prompt 4.6 — MultinomialNB (hanya untuk sparse non-negative features)."""
    from sklearn.naive_bayes import MultinomialNB

    return ModelWrapper(
        name='Naive Bayes (Multinomial)',
        model=MultinomialNB(),
        param_grid={
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
        },
        compatible_features=['sparse']  # Hanya TF-IDF, BM25
    )


def _naive_bayes_gaussian() -> ModelWrapper:
    """Prompt 4.6 — GaussianNB (untuk dense embeddings)."""
    from sklearn.naive_bayes import GaussianNB

    return ModelWrapper(
        name='Naive Bayes (Gaussian)',
        model=GaussianNB(),
        param_grid={},  # Tidak ada hyperparameter signifikan
        compatible_features=['dense_low', 'dense_high'],
        needs_dense=True
    )


def _mlp() -> ModelWrapper:
    """Prompt 4.8 — MLP Neural Network."""
    from sklearn.neural_network import MLPClassifier

    return ModelWrapper(
        name='MLP',
        model=MLPClassifier(
            activation='relu',
            solver='adam',
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ),
        param_grid={
            'hidden_layer_sizes': [(256,), (256, 128), (512, 256, 128)],
            'alpha': [0.0001, 0.001, 0.01],
        },
        needs_dense=True
    )


# ======================================================================
# Registry — Semua Models
# ======================================================================

def get_all_models(subset: str = 'priority') -> Dict[str, ModelWrapper]:
    """
    Return dictionary of all model wrappers.

    Args:
        subset: 'priority' untuk subset (LR, SVM, XGBoost, LightGBM)
                'all' untuk semua 9+ models

    Returns:
        Dict[str, ModelWrapper]
    """
    if subset == 'priority':
        return {
            'Decision Tree': _decision_tree(),
            'Random Forest': _random_forest(),
            'XGBoost': _xgboost(),
        }
    else:
        return {
            'Decision Tree': _decision_tree(),
            'Random Forest': _random_forest(),
            'XGBoost': _xgboost(),
            'LightGBM': _lightgbm(),
            'CatBoost': _catboost(),
            'Logistic Regression': _logistic_regression(),
            'SVM': _svm(),
            'NB-Multinomial': _naive_bayes_multinomial(),
            'NB-Gaussian': _naive_bayes_gaussian(),
            'MLP': _mlp(),
        }


def get_feature_type_for_extractor(extractor_name: str) -> str:
    """
    Map nama extractor ke tipe feature.

    Returns:
        'sparse' untuk TF-IDF, BM25
        'dense_low' untuk Word2Vec, GloVe, FastText (100-300D)
        'dense_high' untuk BERT, DistilBERT, RoBERTa (768D)
    """
    sparse_extractors = {'TF-IDF', 'BM25'}
    dense_high_extractors = {'DistilBERT', 'IndoBERT', 'BERT-IndoBERT', 'RoBERTa'}

    if extractor_name in sparse_extractors:
        return 'sparse'
    elif extractor_name in dense_high_extractors:
        return 'dense_high'
    else:
        return 'dense_low'
