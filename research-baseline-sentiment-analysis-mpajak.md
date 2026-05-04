# Research Baseline: Sentiment Analysis of M-Pajak / Coretax App Reviews

> **Compiled for:** UAS Text Mining — Final Project  
> **Date:** May 2026  
> **Context:** ~8,100 rows Google Play reviews, 79.8% Negative / 17% Positive / ~3.2% Neutral

---

## 1. Masalah Utama yang Diidentifikasi

### 1.1 Class Imbalance (Ketidakseimbangan Kelas)
- **Negatif:** ~79.8% (≈6,464 rows)
- **Positif:** ~17% (≈1,377 rows)
- **Netral:** ~3.2% (≈259 rows)

**Dampak:** Model akan bias ke kelas mayoritas (negatif). Akurasi tinggi tapi meaningless — model cukup tebak "negatif" untuk semua data dan dapat ~80% accuracy. Metrik yang relevan: **Macro F1-Score**, **Recall per kelas**, **Confusion Matrix**.

### 1.2 Ketidakjelasan Tujuan
Pertanyaan: **Sentiment Analysis saja, atau Aspect-Based Sentiment Analysis (ABSA)?**

**Rekomendasi:** Lakukan **keduanya secara bertahap**:
1. **Document-level Sentiment Analysis** → klasifikasi review ke Positif/Negatif/Netral
2. **Aspect-Based Sentiment Analysis** → identifikasi aspek spesifik (UI, fitur, performa, customer service) + sentimen per aspek

---

## 2. Paper Baseline (Sangat Relevan dengan Case Anda)

### 2.1 ⭐ PRIMARY BASELINE — Coretax Sentiment Analysis (Paling Mirip!)

| Detail | Informasi |
|--------|-----------|
| **Judul** | Comparative Analysis of Resampling Techniques to Improve SVM and Random Forest Performance in Coretax Sentiment Analysis |
| **Penulis** | H. Oktafiandi, F. Panjaitan, M.F. Ramadhan |
| **Tahun** | 2025 |
| **Publikasi** | International Conference on Informatics, Multimedia, Cyber (IEEE) |
| **Dataset** | 435 komentar Twitter & YouTube tentang Coretax |
| **Labeling** | Lexicon-based |
| **Embedding** | TF-IDF |
| **Model** | SVM, Random Forest |
| **Balancing** | SMOTE |
| **Hasil Sebelum SMOTE** | SVM: 45.82%, RF: 37.53% |
| **Hasil Setelah SMOTE** | SVM: **96.89%**, RF: **92.33%** |
| **Link** | [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11326920/) |

**Mengapa ini baseline utama:**
- Objek penelitian SAMA: Coretax/M-Pajak
- Masalah SAMA: class imbalance
- Solusi yang digunakan: SMOTE + SVM/RF
- **Novelty Anda:** Anda bisa improve dengan menambahkan Word2Vec, GloVe, FastText, dan model lain (XGBoost, LightGBM, dll) yang tidak ada di paper ini

---

### 2.2 ⭐ SECONDARY BASELINE — Governmental Apps Sentiment Analysis

| Detail | Informasi |
|--------|-----------|
| **Judul** | An Improved Sentiment Classification Approach for Measuring User Satisfaction toward Governmental Services' Mobile Apps Using Machine Learning Methods with Feature Engineering and SMOTE Technique |
| **Penulis** | M. Hadwan, M. Al-Sarem, F. Saeed, M.A. Al-Hagery |
| **Tahun** | 2022 |
| **Publikasi** | Applied Sciences (MDPI), Q1 Journal |
| **Dataset** | 51,000+ reviews dari 6 aplikasi pemerintah Saudi Arabia (Google Play & App Store) |
| **Embedding** | BoW, TF-IDF, Word2Vec, **Concatenated Features** (BoW+TF-IDF+Word2Vec+Lexicons) |
| **Balancing** | SMOTE |
| **Model** | SVM, Random Forest, Bagging, Logistic Regression, Naïve Bayes |
| **Hasil Terbaik** | SVM + SMOTE + Concatenated Features = **94.38% accuracy** |
| **Link** | [MDPI Full Paper (Open Access)](https://www.mdpi.com/2076-3417/12/11/5547) |
| **DOI** | https://doi.org/10.3390/app12115547 |

**Takeaway untuk Anda:**
- Paper ini membuktikan bahwa **concatenated features** (gabungan beberapa embedding) menghasilkan performa terbaik
- SMOTE sangat efektif untuk dataset imbalance
- Anda bisa mengadopsi pendekatan ini: gabungkan TF-IDF + Word2Vec + GloVe + FastText

---

### 2.3 Aspect-Level Sentiment Analysis + Word Embeddings Comparison

| Detail | Informasi |
|--------|-----------|
| **Judul** | Aspect-level Sentiment Analysis on GoPay App Reviews Using Multilayer Perceptron and Word Embeddings |
| **Penulis** | H. Juandri, Hasmawati, Bunyamin |
| **Tahun** | 2024 |
| **Publikasi** | Kinetik Journal, Vol. 9 No. 4 |
| **Dataset** | 15,000 reviews GoPay dari Google Play Store |
| **Embedding** | **fastText vs Word2Vec** (comparison) |
| **Balancing** | SMOTE, Random Oversampling |
| **Model** | Multilayer Perceptron (MLP) |
| **Hasil Terbaik** | fastText + MLP + Balanced = **F1-Score 98%** (sentiment), **97%** (aspect) |
| **Link** | [Kinetik UMM](https://kinetik.umm.ac.id/index.php/kinetik/article/view/2041) |
| **DOI** | https://doi.org/10.22219/kinetik.v9i4.2041 |

**Takeaway:**
- **fastText mengungguli Word2Vec** dalam eksperimen ini (karena fastText menangani OOV/out-of-vocabulary words lebih baik)
- Aspect-level analysis memberikan insight lebih mendalam dibanding document-level saja
- SMOTE lebih efektif daripada Random Oversampling

---

### 2.4 Comparison: Word2Vec vs GloVe vs FastText + SVM

| Detail | Informasi |
|--------|-----------|
| **Judul** | Comparison Performance of Word2Vec, GloVe, Fasttext Using Support Vector Machine Method for Sentiment Analysis |
| **Penulis** | M. Anjani, H.N. Irmanda |
| **Tahun** | 2024 |
| **Publikasi** | Jurnal Teknik Informatika (JUTIF) |
| **Dataset** | Reviews aplikasi Spotify dari Google Play Store |
| **Embedding** | Word2Vec, GloVe, FastText |
| **Model** | SVM |
| **Hasil** | **GloVe mengungguli** Word2Vec dan FastText untuk dataset ini |
| **Link** | [JUTIF UNSOED](http://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/1366) |

**Takeaway:** GloVe bisa jadi yang terbaik tergantung dataset — ini justifikasi mengapa Anda perlu membandingkan ketiganya secara empiris.

---

### 2.5 M-Pajak Specific Paper

| Detail | Informasi |
|--------|-----------|
| **Judul** | Using SVM and KNN for Predicting Customer Response Sentiment of M-PAJAK Application |
| **Penulis** | MTRA Wijaya, I. Widaningrum, et al. |
| **Tahun** | 2025 |
| **Publikasi** | Jurnal Ilmu Komputer (JIKA), UMS |
| **Dataset** | Respons customer aplikasi M-Pajak |
| **Embedding** | TF-IDF |
| **Model** | SVM, KNN |
| **Catatan** | Menyebutkan masalah imbalance data |
| **Link** | [JIKA UMS](https://journals2.ums.ac.id/index.php/jika/article/view/Using+SVM+and+KNN+for+Predicting+Customer+Response+Sentiment+of+M-PAJAK+Application) |

---

### 2.6 Additional Relevant Papers

| # | Judul | Tahun | Link | Key Finding |
|---|-------|-------|------|-------------|
| 6 | Aspect-Based Sentiment Analysis for Mobile App Review Using CNN and Word2Vec (Lestari et al.) | 2024 | [IEEE](https://ieeexplore.ieee.org/abstract/document/10828541/) | CNN + Word2Vec + TF-IDF untuk ABSA |
| 7 | Sentiment Analysis of Transportation Application Reviews with SVM on Handling Imbalanced Data Using SMOTE (Lestari et al.) | 2025 | [IEEE](https://ieeexplore.ieee.org/document/Sentiment+Analysis+of+Transportation+Application+Reviews+with+SVM) | SMOTE + SMOTEENN untuk transport app reviews |
| 8 | Optimizing Sentiment Analysis in Multilingual Balanced Datasets (Jakha et al.) | 2025 | [MDPI](https://www.mdpi.com/2571-5577/8/4/104) | TF-IDF, Word2Vec, FastText, BERT + SMOTE |
| 9 | Optimizing Sentiment Analysis on Imbalanced Hotel Review Data Using SMOTE and Ensemble ML (Putra et al.) | 2025 | [Bright Journal](https://bright-journal.org) | SMOTE + Ensemble techniques |
| 10 | Aspect-based Sentiment Analysis Using Smart Government Review Data (Alqaryouti et al.) | 2024 | [Emerald](https://www.emerald.com/insight/content/doi/10.1108/ACI-07-2023-0037) | ABSA untuk smart government apps |
| 11 | Machine Learning Approach to Evaluate Public Perception: Sentiment Analysis of Mobile Government App User Reviews (Enhartana et al.) | 2025 | [IEEE](https://ieeexplore.ieee.org/document/Machine+Learning+Approach+to+Evaluate+Public+Perception) | SMOTE untuk government app reviews |
| 12 | Improving the Review Classification of Google Apps Using Combined Feature Embedding and Deep CNN (Aslam et al.) | 2023 | [Springer](https://link.springer.com/article/10.1007/s10515-023-00397-7) | **FastText + GloVe combined** + CNN |

---

## 3. Solusi untuk Class Imbalance

### 3.1 Resampling Techniques (Paling Umum & Efektif)

| Teknik | Tipe | Deskripsi | Kapan Gunakan |
|--------|------|-----------|---------------|
| **SMOTE** | Oversampling | Membuat sampel sintetis untuk minoritas class | ✅ **REKOMENDASI UTAMA** — terbukti di semua paper baseline |
| **ADASYN** | Oversampling | Versi改进 SMOTE, fokus pada sampel yang sulit | Ketika SMOTE belum optimal |
| **Random Oversampling** | Oversampling | Duplikasi random sampel minoritas | Baseline comparison |
| **Random Undersampling** | Undersampling | Hapus random sampel mayoritas | Ketika data sangat besar |
| **SMOTEENN** | Hybrid | SMOTE + Edited Nearest Neighbors (clean noise) | Ketika ada overlap antar kelas |
| **SMOTETomek** | Hybrid | SMOTE + Tomek Links (clean boundary) | Alternatif SMOTEENN |

**Rekomendasi untuk project Anda:**
```
Primary: SMOTE (sudah terbukti di paper Coretax → 45% → 96%)
Comparison: SMOTE vs ADASYN vs SMOTEENN vs Random Oversampling
```

### 3.2 Algorithm-Level Solutions

| Teknik | Deskripsi |
|--------|-----------|
| **Class Weights** | Berikan bobot lebih tinggi ke minoritas class (built-in di sklearn: `class_weight='balanced'`) |
| **Threshold Moving** | Adjust decision threshold setelah training |
| **Ensemble Methods** | Balanced Random Forest, EasyEnsemble, RUSBoost |

### 3.3 Evaluation Metrics yang HARUS Digunakan

Jangan hanya pakai Accuracy! Gunakan:

| Metrik | Mengapa |
|--------|---------|
| **Macro F1-Score** | Rata-rata F1 semua kelas (tidak bias ke mayoritas) |
| **Weighted F1-Score** | F1 yang di-weight oleh support tiap kelas |
| **Recall per Kelas** | Seberapa baik model mendeteksi tiap kelas |
| **Precision per Kelas** | Seberapa akurat prediksi tiap kelas |
| **Confusion Matrix** | Visualisasi misclassification |
| **ROC-AUC (OvR)** | Area under curve untuk multi-class |
| **Cohen's Kappa** | Agreement score yang memperhitungkan chance |

---

## 4. Rekomendasi Tujuan & Scope Penelitian

### 4.1 Tujuan yang Direkomendasikan

Berdasarkan paper-paper baseline, berikut tujuan yang paling tepat:

> **"Analisis sentimen review aplikasi M-Pajak/Coretax menggunakan perbandingan word embedding (TF-IDF, Word2Vec, GloVe, FastText) dan machine learning models dengan penanganan class imbalance menggunakan SMOTE untuk mengidentifikasi aspek-aspek yang paling dikritik oleh pengguna."**

### 4.2 Research Questions

1. **RQ1:** Word embedding mana (TF-IDF, Word2Vec, GloVe, FastText) yang menghasilkan performa terbaik untuk sentiment analysis review M-Pajak/Coretax?
2. **RQ2:** Bagaimana pengaruh teknik handling class imbalance (SMOTE, ADASYN, SMOTEENN) terhadap performa model?
3. **RQ3:** Model machine learning mana (Decision Tree, Random Forest, XGBoost, SVM, LightGBM, MLP) yang paling optimal?
4. **RQ4:** Apakah concatenated features (gabungan 2+ embedding) menghasilkan performa lebih baik dibanding single embedding?
5. **RQ5:** Aspek apa saja yang paling banyak mendapat sentimen negatif dari pengguna? (ABSA)

### 4.3 Scope yang Disarankan

```
┌─────────────────────────────────────────────────────┐
│  EXPERIMENT DESIGN                                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Phase 1: Document-level Sentiment Analysis          │
│  ├── Embeddings: TF-IDF, Word2Vec, GloVe, FastText   │
│  ├── Models: DT, RF, XGBoost, SVM, LightGBM, MLP     │
│  ├── Balancing: Original, SMOTE, ADASYN, SMOTEENN    │
│  └── Metrics: Macro F1, Weighted F1, Recall, AUC     │
│                                                      │
│  Phase 2: Feature Combination (Novelty)              │
│  ├── TF-IDF + Word2Vec                               │
│  ├── TF-IDF + FastText                               │
│  ├── Word2Vec + GloVe                                │
│  └── TF-IDF + Word2Vec + FastText                    │
│                                                      │
│  Phase 3: Aspect-Based Sentiment Analysis (Optional) │
│  ├── Aspect extraction (LDA / manual categories)     │
│  ├── Aspects: UI, Fitur, Performa, Customer Service  │
│  └── Sentiment per aspect                            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 5. Novelty / Improvement dari Paper Baseline

### 5.1 Apa yang Belum Ada di Paper Baseline

| Paper Baseline | Yang Mereka Lakukan | Yang BELUM Mereka Lakukan (Peluang Anda) |
|----------------|---------------------|------------------------------------------|
| Oktafiandi et al. (2025) — Coretax | TF-IDF + SVM/RF + SMOTE | ❌ Tidak compare Word2Vec, GloVe, FastText |
| | | ❌ Tidak coba XGBoost, LightGBM, MLP |
| | | ❌ Tidak coba concatenated features |
| | | ❌ Tidak ada Aspect-Based analysis |
| Hadwan et al. (2022) — Gov Apps | BoW, TF-IDF, Word2Vec + SMOTE | ❌ Tidak include GloVe & FastText |
| | | ❌ Tidak coba XGBoost/LightGBM |
| | | ❌ Dataset Arab, bukan Indonesia |
| Juandri et al. (2024) — GoPay | fastText vs Word2Vec + MLP | ❌ Tidak include GloVe & TF-IDF |
| | | ❌ Tidak compare dengan tree-based models |

### 5.2 Novelty yang Bisa Anda Claim

1. **Komparasi 4 embedding + 6+ model pada dataset Coretax/M-Pajak** — belum ada paper yang melakukan ini secara komprehensif
2. **Concatenated features** (gabungan TF-IDF + Word2Vec + FastText) — terbukti efektif di Hadwan et al. tapi belum di konteks Indonesia
3. **Perbandingan multiple resampling techniques** (SMOTE vs ADASYN vs SMOTEENN) pada domain pajak digital Indonesia
4. **Aspect-Based Sentiment Analysis** — mengidentifikasi aspek spesifik (UI, fitur, performa, CS) yang paling dikritik
5. **Dataset Google Play Store** yang lebih besar (8,100 rows) dibanding paper Coretax sebelumnya (435 rows dari Twitter/YouTube)

---

## 6. Pipeline Implementasi yang Disarankan

```
Data Collection (Google Play Reviews)
         │
         ▼
Preprocessing
├── Case folding
├── Cleaning (remove URL, mention, special chars)
├── Normalization (slang → formal)
├── Stopword removal
├── Tokenization
└── Stemming (Sastrawi for Indonesian)
         │
         ▼
Labeling
├── Manual annotation (gold standard)
├── OR Lexicon-based (IndoLexicon)
└── OR Star-rating mapping (≥4=Pos, 3=Neu, ≤2=Neg)
         │
         ▼
Train-Test Split (80:20, stratified)
         │
         ▼
Apply Resampling (ONLY on training data!)
├── SMOTE
├── ADASYN
└── SMOTEENN
         │
         ▼
Feature Extraction
├── TF-IDF
├── Word2Vec (train on corpus or use pre-trained IndoWord2Vec)
├── GloVe (pre-trained, may need Indonesian-specific)
├── FastText (pre-trained Indonesian — available!)
└── Concatenated combinations
         │
         ▼
Model Training
├── Decision Tree
├── Random Forest
├── XGBoost
├── SVM
├── LightGBM
└── MLP / Neural Network
         │
         ▼
Evaluation
├── Macro F1-Score (PRIMARY METRIC)
├── Weighted F1-Score
├── Per-class Precision, Recall, F1
├── Confusion Matrix
├── ROC-AUC (One-vs-Rest)
└── Statistical significance test (McNemar / Wilcoxon)
         │
         ▼
Analysis & Insights
├── Best embedding + model combination
├── Impact of resampling techniques
├── Top negative aspects (if ABSA)
└── Recommendations for DJP
```

---

## 7. Resources & Tools

### 7.1 Pre-trained Indonesian Word Embeddings

| Embedding | Link | Size |
|-----------|------|------|
| **FastText Indonesian** | [facebookresearch/fastText — cc.id.300.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz) | 300 dim, 1.3M words |
| **IndoWord2Vec** | [github.com/IndoNLP/indonlp](https://github.com/IndoNLP/indonlp) | Various |
| **IndoBERT** (for advanced) | [github.com/indobenchmark/indobenchmark](https://github.com/indobenchmark/indobenchmark) | Contextual |

### 7.2 Python Libraries

```python
# Preprocessing
import sastrawi  # Indonesian stemming
import nltk
import re

# Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
import gensim  # Word2Vec, FastText
# GloVe — load from file

# Resampling
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# Evaluation
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, cohen_kappa_score)
```

### 7.3 Indonesian Sentiment Lexicons

| Lexicon | Link |
|---------|------|
| **IndoLexicon** | [github.com/fajri91/IndoLexicon](https://github.com/fajri91/IndoLexicon) |
| **InSet Lexicon** | Paper by Nurminah et al. |
| **VADER Indonesian** | Adapted from VADER |

---

## 8. Summary Tabel Perbandingan Paper

| Paper | Dataset | Embedding | Model | Balancing | Best Result |
|-------|---------|-----------|-------|-----------|-------------|
| **Oktafiandi 2025** (Coretax) | 435 (Twitter/YT) | TF-IDF | SVM, RF | SMOTE | SVM 96.89% |
| **Hadwan 2022** (Gov Apps) | 51K (Google Play) | BoW, TF-IDF, W2V, Concat | SVM, RF, NB, LR, Bagging | SMOTE | SVM 94.38% |
| **Juandri 2024** (GoPay) | 15K (Google Play) | fastText, Word2Vec | MLP | SMOTE, ROS | fastText 98% F1 |
| **Anjani 2024** (Spotify) | Spotify reviews | W2V, GloVe, FastText | SVM | — | GloVe best |
| **Wijaya 2025** (M-Pajak) | M-Pajak responses | TF-IDF | SVM, KNN | — | — |
| **Anda (Proposed)** | 8.1K (Google Play) | TF-IDF, W2V, GloVe, FastText, Concat | DT, RF, XGB, SVM, LGBM, MLP | SMOTE, ADASYN, SMOTEENN | **TBD — Target >97%** |

---

## 9. Referensi Lengkap

1. Oktafiandi, H., Panjaitan, F., & Ramadhan, M.F. (2025). Comparative Analysis of Resampling Techniques to Improve SVM and Random Forest Performance in Coretax Sentiment Analysis. *IEEE International Conference on Informatics, Multimedia, Cyber*. https://ieeexplore.ieee.org/abstract/document/11326920/

2. Hadwan, M., Al-Sarem, M., Saeed, F., & Al-Hagery, M.A. (2022). An Improved Sentiment Classification Approach for Measuring User Satisfaction toward Governmental Services' Mobile Apps Using Machine Learning Methods with Feature Engineering and SMOTE Technique. *Applied Sciences, 12*(11), 5547. https://doi.org/10.3390/app12115547

3. Juandri, H., Hasmawati, & Bunyamin. (2024). Aspect-level Sentiment Analysis on GoPay App Reviews Using Multilayer Perceptron and Word Embeddings. *Kinetik, 9*(4). https://doi.org/10.22219/kinetik.v9i4.2041

4. Anjani, M., & Irmanda, H.N. (2024). Comparison Performance of Word2Vec, GloVe, Fasttext Using Support Vector Machine Method for Sentiment Analysis. *JUTIF*. http://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/1366

5. Wijaya, M.T.R.A., Widaningrum, I., et al. (2025). Using SVM and KNN for Predicting Customer Response Sentiment of M-PAJAK Application. *JIKA, UMS*. https://journals2.ums.ac.id/index.php/jika/article/view/Using+SVM+and+KNN+for+Predicting+Customer+Response+Sentiment+of+M-PAJAK+Application

6. Lestari, N.I., Taib, S.M., Wibowo, W., & Aziz, I.A. (2024). Aspect-Based Sentiment Analysis for Mobile App Review Using CNN and Word2Vec. *IEEE 7th International Conference*. https://ieeexplore.ieee.org/abstract/document/10828541/

7. Alqaryouti, O., Siyam, N., & Abdel Monem, A. (2024). Aspect-based Sentiment Analysis Using Smart Government Review Data. *Applied Computing and Informatics, Emerald*. https://www.emerald.com/insight/content/doi/10.1108/ACI-07-2023-0037

8. Jakha, H., El Houssaini, S., et al. (2025). Optimizing Sentiment Analysis in Multilingual Balanced Datasets. *Applied System Innovation, 8*(4), 104. https://www.mdpi.com/2571-5577/8/4/104

9. Aslam, N., Alzamzami, O., Xia, K., & Sadiq, S. (2023). Improving the Review Classification of Google Apps Using Combined Feature Embedding and Deep CNN. *Journal of Ambient Intelligence*. https://link.springer.com/article/10.1007/s10515-023-00397-7

10. Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR, 16*, 321-357. https://doi.org/10.1613/jair.953

---

## 10. Next Steps

1. ✅ Tentukan tujuan: **Document-level SA + Aspect-based SA**
2. ✅ Pilih baseline paper: **Oktafiandi 2025 (Coretax)** sebagai primary, **Hadwan 2022** sebagai secondary
3. ✅ Definisikan novelty: **4 embeddings × 6 models × 3 balancing techniques + concatenated features**
4. 🔲 Mulai preprocessing dataset
5. 🔲 Implementasi pipeline
6. 🔲 Eksperimen & evaluasi
7. 🔲 Tulis paper

---

> **Good luck!** 🎯 Dokumentasi ini bisa langsung jadi bagian **Bab 2 (Literature Review)** dan **Bab 3 (Methodology)** dari paper/skripsi Anda.
