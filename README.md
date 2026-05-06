# 📊 CoreTax Sentiment Classification Project

Proyek ini bertujuan untuk melakukan klasifikasi sentimen pada ulasan aplikasi **CoreTax/M-Pajak** menggunakan berbagai teknik Text Mining dan Machine Learning. Proyek ini telah berhasil mencapai akurasi **>86%** dengan strategi klasifikasi biner yang dioptimasi.

## 📁 Struktur Proyek

- `notebooks/`: File Jupyter Notebook yang terbagi menjadi dua alur (Baseline & Optimized).
- `src/`: Modul Python modular untuk preprocessing, feature extraction, dan modeling.
- `data/`: Folder penyimpanan data (Raw, Processed, & Feature Matrices).
- `results/`: Output eksperimen (Tabel komparasi & Visualisasi).
- `scripts/`: Script otomatisasi untuk rebuild dataset dan training.

---

## 🛠️ Persiapan (Setup)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Pilih Kernel**:
   Gunakan environment `.venv` atau kernel Python yang sesuai.
3. **Download Embeddings (Opsional)**:
   Jika ingin menggunakan Word2Vec/FastText/GloVe, pastikan file embedding tersedia di `data/embeddings/`.
   ```bash
   python scripts/download_embeddings.py
   ```

---

## 🚀 Pilihan Alur Kerja (Workflow)

Proyek ini menyediakan dua pendekatan berbeda tergantung pada kebutuhan analisis:

### 🅰️ Workflow A: Klasifikasi Ternary (Baseline)
Fokus pada pemetaan 3 kelas sentimen asli. Berguna untuk analisis granular namun memiliki tantangan akurasi yang lebih besar karena ambiguitas rating 3.
- **Notebooks**: `01`, `02`, `03`, `04`
- **Kelas**: Negatif (1-2), Netral (3), Positif (4-5)
- **Estimasi Akurasi**: ~65-70%

### 🅱️ Workflow B: Klasifikasi Biner (Optimized) - **REKOMENDASI**
Fokus pada akurasi tinggi dan sinyal sentimen yang kuat dengan mengeliminasi data "noise" (Rating 3) dan ulasan yang terlalu pendek.
- **Notebooks**: `01.1`, `02.2`, `03.3`, `04.4`
- **Kelas**: Negatif (1-2) vs Positif (4-5)
- **Optimasi**: Drop Rating 3, filter ulasan < 3 token, dataset balancing.
- **Estimasi Akurasi**: **86% - 90.6%**

---

## 📝 Detail Alur Kerja Baseline (Workflow A)

Jalankan notebook tanpa akhiran `.x` secara berurutan:

1. **`01_eda_and_preparation.ipynb`**: Pembersihan awal dan pemetaan distribusi rating pengguna.
2. **`02_labeling_and_preprocessing.ipynb`**: 
   - Labeling Ternary: Menentukan kelas Negatif, Netral, dan Positif.
   - Preprocessing dasar: Tokenizing, filtering, dan stemming.
3. **`03_feature_engineering.ipynb`**: Transformasi teks ke numerik dengan TF-IDF dan Baseline Embeddings.
4. **`04_model_training.ipynb`**: Pelatihan model pada data 3-kelas dan evaluasi metrik dasar.

### 🏆 Hasil Baseline (Workflow A)
Berdasarkan ulasan ternary (3-kelas):

| Rank | Feature Extractor | Model | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| 1 | FastText | XGBoost | 68.0% | 0.681 |
| 2 | Word2Vec | Random Forest | 67.2% | 0.673 |
| 3 | FastText | Random Forest | 65.6% | 0.658 |

---

## 📝 Detail Alur Kerja Optimized (Workflow B)

Jalankan notebook dengan akhiran `.x` secara berurutan:

1. **`01.1_eda_and_preparation.ipynb`**: Eksplorasi data 8.000+ ulasan dan persiapan dataset awal.
2. **`02.2_labeling_and_preprocessing.ipynb`**: 
   - Konversi ke Binary (Drop Rating 3).
   - Preprocessing: Case folding, slang removal, stopword removal, dan Sastrawi stemming.
   - Filter Noise: Menghapus ulasan dengan < 3 kata.
3. **`03.3_feature_engineering.ipynb`**: 
   - Ekstraksi fitur menggunakan TF-IDF (Word & Char), Word2Vec, GloVe, dan FastText.
4. **`04.4_model_training.ipynb`**:
   - Eksperimen berbagai model (Logistic Regression, SVM, Random Forest, XGBoost).
   - Hyperparameter tuning menggunakan GridSearchCV.
5. **`05.5_comparison_analysis.ipynb`**:
   - Analisis performa final dan Error Analysis.
   - Topic Modeling (LDA) untuk menemukan keluhan utama pengguna.

---

## 🏆 Ringkasan Hasil (Workflow B)

Berdasarkan eksperimen pada dataset biner yang diseimbangkan:

| Rank | Feature Extractor | Model | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **TF-IDF** | **Logistic Regression** | **90.6%** | **0.906** |
| 2 | Word2Vec | XGBoost | 90.3% | 0.903 |
| 3 | TF-IDF | XGBoost | 90.3% | 0.903 |
| 4 | Word2Vec | Random Forest | 90.3% | 0.903 |
| 5 | TF-IDF | Random Forest | 89.4% | 0.894 |

**Insight Utama:** Kombinasi **TF-IDF** dan **Logistic Regression** terbukti paling efisien (cepat dan akurasi tertinggi) untuk karakteristik teks ulasan aplikasi CoreTax.

---
**Text Mining Final Project — Universitas Cakrawala**
