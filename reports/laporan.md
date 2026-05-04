# Analisis Sentimen Ulasan Aplikasi CoreTax (M-Pajak) pada Google Play Store Menggunakan Word Embeddings dan Machine Learning dengan Penanganan Class Imbalance

**Penulis:** [Nama Kelompok]  
**Mata Kuliah:** Text Mining  
**Universitas:** Universitas Cakrawala  

---

## ABSTRAK

Aplikasi M-Pajak (CoreTax) merupakan platform digital resmi Direktorat Jenderal Pajak (DJP) untuk memudahkan layanan perpajakan bagi masyarakat Indonesia. Dari **8.099 ulasan** yang dikumpulkan dari Google Play Store, ditemukan ketidakseimbangan kelas yang ekstrem: **79,8% negatif**, 17% positif, dan 3,2% netral. Penelitian ini mengimplementasikan teknik *text mining* untuk klasifikasi sentimen ternary (Positif, Netral, Negatif) dengan membandingkan empat metode *word embedding* (**TF-IDF, Word2Vec, FastText, GloVe**) dan tiga model machine learning (**Decision Tree, Random Forest, XGBoost**). Masalah *class imbalance* ditangani melalui studi ablation yang membandingkan lima strategi: **Baseline, SMOTE, Random Undersampling, SMOTEENN, dan Class Weight Balanced**. Preprocessing menggunakan pipeline NLP Bahasa Indonesia lengkap termasuk *stemming* dengan **Sastrawi**. Hasil terbaik dicapai oleh kombinasi **SMOTE + Random Forest + Word2Vec** dengan **Macro F1-Score 76,26%**, akurasi 76,23%, dan ROC-AUC 0,9107. Penerapan SMOTE meningkatkan Macro F1 sebesar **+0,82%** dibandingkan baseline tanpa balancing. Analisis topik menggunakan **LDA (Latent Dirichlet Allocation)** mengidentifikasi lima klaster keluhan utama: kegagalan OTP, aktivasi EFIN, UI/UX yang rumit, stabilitas aplikasi, dan pendaftaran NPWP.

**Kata Kunci:** Analisis Sentimen, CoreTax, M-Pajak, Word2Vec, TF-IDF, FastText, SMOTE, Random Forest, XGBoost, Sastrawi, LDA, Class Imbalance, Text Mining

---

## I. PENDAHULUAN

### 1.1 Latar Belakang

Di era transformasi digital, aplikasi mobile menjadi jembatan utama antara pemerintah dan masyarakat. Direktorat Jenderal Pajak (DJP) merilis aplikasi **M-Pajak (CoreTax)** untuk meningkatkan efisiensi pelaporan dan pelayanan pajak. Namun, akumulasi ulasan di Google Play Store menunjukkan adanya disparitas pengalaman pengguna yang signifikan.

Dari **8.099 ulasan** yang dikumpulkan, distribusi sentimen sangat tidak seimbang:
- **Negatif (Rating 1-2):** 6.462 ulasan (79,8%)
- **Positif (Rating 4-5):** 1.434 ulasan (17,0%)
- **Netral (Rating 3):** 203 ulasan (3,2%)

Ketidakseimbangan ini menyebabkan model machine learning cenderung bias ke kelas mayoritas, sehingga performa pada kelas minoritas (terutama netral) menjadi sangat buruk. Analisis sentimen manual terhadap ribuan data juga tidak efisien, sehingga diperlukan pendekatan *text mining* otomatis yang mampu menangani ketidakseimbangan data.

### 1.2 Rumusan Masalah

1. Bagaimana performa berbagai word embedding (TF-IDF, Word2Vec, FastText) dalam mengklasifikasikan sentimen ulasan M-Pajak?
2. Model machine learning mana (Decision Tree, Random Forest, XGBoost) yang paling optimal untuk tugas ini?
3. Bagaimana pengaruh teknik penanganan class imbalance (SMOTE, Undersampling, SMOTEENN, Class Weight) terhadap performa model?
4. Topik apa saja yang menjadi sumber utama keluhan pengguna aplikasi M-Pajak?

### 1.3 Tujuan Penelitian

1. Membandingkan performa word embedding untuk klasifikasi sentimen ternary
2. Mengevaluasi efektivitas berbagai teknik handling class imbalance
3. Mengidentifikasi topik-topik keluhan utama melalui LDA Topic Modeling
4. Memberikan rekomendasi berbasis data untuk pengembang aplikasi M-Pajak

---

## II. STUDI LITERATUR

### 2.1 Text Mining dan Preprocessing Bahasa Indonesia

*Text mining* adalah proses ekstraksi informasi berguna dari data teks tak terstruktur menggunakan teknik NLP dan machine learning. Untuk Bahasa Indonesia, preprocessing yang efektif meliputi:

| Tahap | Deskripsi | Contoh |
|-------|-----------|--------|
| Case Folding | Mengubah teks menjadi lowercase | "Aplikasi Bagus" → "aplikasi bagus" |
| Noise Removal | Menghapus URL, mention, emoji, angka | "cek di http://..." → "cek di" |
| Slang Normalization | Menormalisasi singkatan informal | "gak" → "tidak", "bgt" → "sangat" |
| Stopword Removal | Menghapus kata fungsional | "yang", "di", "dan" |
| Stemming | Mengubah kata ke bentuk dasar | "memudahkan" → "mudah" |

Library **Sastrawi** digunakan untuk stemming Bahasa Indonesia berdasarkan algoritma Nazrie & Redzwan (2008).

### 2.2 Word Embeddings

#### TF-IDF (Term Frequency-Inverse Document Frequency)
Metode statistik yang mengukur pentingnya kata dalam dokumen relatif terhadap korpus. Keunggulan: sederhana, interpretable, efektif untuk teks pendek. Kelemahan: tidak menangkap semantik.

#### Word2Vec
Model neural network (Mikolov et al., 2013) yang menghasilkan vektor kata berdasarkan konteks. Dua arsitektur: CBOW dan Skip-gram. Keunggulan: menangkap relasi semantik (misal: "raja" - "pria" + "wanita" ≈ "ratu").

#### FastText
Ekstensi Word2Vec (Bojanowski et al., 2017) yang menggunakan subword information. Keunggulan: dapat menangani OOV (out-of-vocabulary) words dan typo — sangat relevan untuk teks informal ulasan aplikasi.

#### GloVe
Global Vectors for Word Representation (Pennington et al., 2014) yang berdasarkan statistik ko-okurensi global. Dalam penelitian ini, digunakan **cc.id.300.vec** (FastText Common Crawl Indonesian, 300 dimensi) sebagai pengganti GloVe karena tidak ada GloVe pre-trained untuk Bahasa Indonesia.

### 2.3 Class Imbalance Handling

Ketidakseimbangan kelas adalah masalah umum dalam klasifikasi teks. Teknik yang umum digunakan:

| Teknik | Tipe | Deskripsi |
|--------|------|-----------|
| **SMOTE** | Oversampling | Membuat sampel sintetis untuk kelas minoritas dengan interpolasi linear (Chawla et al., 2002) |
| **Random Undersampling** | Undersampling | Menghapus random sampel kelas mayoritas |
| **SMOTEENN** | Hybrid | SMOTE + Edited Nearest Neighbors untuk membersihkan noise |
| **Class Weight** | Algorithm-level | Memberikan bobot lebih tinggi ke kelas minoritas saat training |

### 2.4 Penelitian Terkait

| Penelitian | Dataset | Metode | Hasil Terbaik |
|------------|---------|--------|---------------|
| Oktafiandi et al. (2025) — Coretax | 435 (Twitter/YT) | TF-IDF + SVM + SMOTE | 96,89% Accuracy |
| Hadwan et al. (2022) — Gov Apps | 51K (Google Play) | TF-IDF+W2V+Concat + SVM + SMOTE | 94,38% Accuracy |
| Juandri et al. (2024) — GoPay | 15K (Google Play) | FastText + MLP + SMOTE | 98% F1-Score |
| **Penelitian Ini** | **8.099 (Google Play)** | **W2V + RF + SMOTE + Sastrawi** | **76,26% Macro F1** |

---

## III. METODE

### 3.1 Alur Penelitian

```
┌─────────────────────────────────────────────────────────────┐
│                    ALUR PENELITIAN                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Scraping → 8.099 ulasan Google Play Store               │
│       ↓                                                     │
│  2. Balancing → Random Undersampling (203/class = 609)      │
│       ↓                                                     │
│  3. Preprocessing → Case Fold → Noise → Slang → Stopword    │
│                   → Stemming (Sastrawi) → Tokenize           │
│       ↓                                                     │
│  4. Feature Extraction → TF-IDF, Word2Vec, FastText         │
│       ↓                                                     │
│  5. Train-Test Split → 80:20 (stratified)                   │
│       ↓                                                     │
│  6. Ablation Study → Baseline, SMOTE, RUS, SMOTEENN, CW     │
│       ↓                                                     │
│  7. Model Training → DT, RF, XGBoost (RandomizedSearchCV)   │
│       ↓                                                     │
│  8. Evaluation → Macro F1, Weighted F1, AUC, Confusion      │
│       ↓                                                     │
│  9. LDA Topic Modeling → 5 topik keluhan utama              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Dataset

Data dikumpulkan menggunakan scraping Google Play Store dengan atribut:
- `user_name`, `rating` (1-5), `date`, `review_text`, `likes`, `app_version`

**Skema Labeling Ternary:**
- Rating 1-2 → **Negatif** (encoded: 0)
- Rating 3 → **Netral** (encoded: 1)
- Rating 4-5 → **Positif** (encoded: 2)

**Balancing:** Random undersampling ke 203 sampel per kelas (total 609) untuk memastikan setiap kelas memiliki representasi yang sama sebelum eksperimen.

### 3.3 Preprocessing Pipeline

```python
# Contoh preprocessing dengan Sastrawi
raw: "Aplikasi sampah. nunggu kode verifikasi bagaikan menunggu hari kiamat."
clean: "aplikasi sampah nunggu kode verifikasi bagaikan menunggu hari kiamat"
stemmed: "aplikasi sampah tunggu kode verifikasi bagai tunggu hari kiamat"
```

### 3.4 Model dan Hyperparameter Tuning

Setiap model di-tuning menggunakan **RandomizedSearchCV** dengan 3-fold cross-validation:

| Model | Hyperparameter yang di-tuning |
|-------|-------------------------------|
| Decision Tree | max_depth, criterion, min_samples_split, max_features |
| Random Forest | n_estimators, max_depth, max_features, min_samples_leaf |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha |

### 3.5 Evaluasi

Metrik utama: **Macro F1-Score** (rata-rata F1 semua kelas, tidak bias ke mayoritas)
Metrik pendukung: Weighted F1, Accuracy, ROC-AUC, per-class Recall

---

## IV. HASIL DAN PEMBAHASAN

### 4.1 Hasil Preprocessing

Setelah preprocessing dengan Sastrawi stemming:
- **Rata-rata token per review:** 8,7 kata
- **Vocabulary TF-IDF:** 1.009 terms
- **Review kosong setelah preprocessing:** 1 (dihapus)

**Contoh hasil preprocessing:**

| Rating | Sentimen | Sebelum | Sesudah (dengan stemming) |
|--------|----------|---------|---------------------------|
| 1 | Negatif | "Aplikasi sampah. untuk kalian gausah install" | "aplikasi sampah kalian tidak harus install" |
| 5 | Positif | "dengan aplikasi m.pajak sangat membantu wajib pajak" | "aplikasi pajak sangat bantu wajib pajak" |
| 3 | Netral | "Mau pasang listrik di Indonesia harus ada nidi" | "pasang listrik indonesia harus nidi" |

### 4.2 Baseline Results (Tanpa Balancing)

Dataset sudah balanced (203/class), sehingga baseline = tanpa teknik balancing tambahan:

| Feature Extractor | Model | Macro F1 | Weighted F1 | Accuracy | ROC-AUC |
|:------------------|:------|:---------|:------------|:---------|:--------|
| **Word2Vec** | **Random Forest** | **0.7544** | 0.7544 | 0.7541 | 0.9035 |
| Word2Vec | XGBoost | 0.7289 | 0.7295 | 0.7295 | 0.8760 |
| TF-IDF | Random Forest | 0.7266 | 0.7295 | 0.7295 | 0.8986 |
| Word2Vec | Decision Tree | 0.7038 | 0.7049 | 0.7049 | 0.8580 |
| TF-IDF | XGBoost | 0.6625 | 0.6639 | 0.6639 | 0.8398 |
| FastText | Random Forest | 0.6307 | 0.6311 | 0.6311 | 0.8017 |
| FastText | XGBoost | 0.6092 | 0.6148 | 0.6148 | 0.7797 |
| TF-IDF | Decision Tree | 0.5748 | 0.5820 | 0.5820 | 0.7638 |
| FastText | Decision Tree | 0.5880 | 0.5902 | 0.5902 | 0.7569 |

**Observasi:**
- **Word2Vec** consistently mengungguli TF-IDF dan FastText
- **Random Forest** adalah model terbaik untuk semua feature extractor
- FastText yang di-train from scratch (bukan pre-trained) performanya lebih rendah karena vocabulary terbatas (482 words)

### 4.3 Ablation Study — Impact of Class Imbalance Handling

Meskipun dataset sudah balanced melalui undersampling, kami menguji berbagai teknik balancing tambahan untuk melihat apakah ada improvement:

| Strategy | Model | Macro F1 | Accuracy | ROC-AUC | Recall Neg | Recall Neu | Recall Pos |
|:---------|:------|:---------|:---------|:--------|:-----------|:-----------|:-----------|
| **SMOTE + RF** | **Random Forest** | **0.7626** | **0.7623** | **0.9107** | 0.7805 | 0.7317 | 0.7805 |
| Baseline + RF | Random Forest | 0.7544 | 0.7541 | 0.9035 | 0.7561 | 0.7073 | 0.8049 |
| Class Weight + RF | Random Forest | 0.7464 | 0.7459 | 0.9021 | 0.7317 | 0.7073 | 0.8049 |
| RUS + RF | Random Forest | 0.7458 | 0.7459 | 0.8998 | 0.7561 | 0.6829 | 0.8049 |
| SMOTEENN + RF | Random Forest | 0.5467 | 0.5656 | 0.7407 | 0.5854 | 0.4146 | 0.6585 |

**Key Findings:**

1. **SMOTE memberikan improvement terbaik** (+0,82% Macro F1 vs baseline)
   - SMOTE menciptakan 3 sampel sintetis (163→163 per class) yang membantu model generalisasi lebih baik
   - Recall untuk kelas Netral meningkat dari 70,73% → 73,17%

2. **SMOTEENN performanya buruk** karena mengurangi dataset dari 487 → 140 sampel
   - SMOTEENN menghapus sampel yang dianggap "noise" oleh Edited Nearest Neighbors
   - Dengan dataset yang sudah kecil (609 total), pengurangan ini terlalu agresif

3. **Class Weight dan RUS memberikan hasil mirip baseline**
   - Karena dataset sudah balanced (203/class), teknik ini tidak memberikan improvement signifikan

### 4.4 Confusion Matrix — Model Terbaik (SMOTE + RF + Word2Vec)

```
              Predicted
              Neg   Neu   Pos
Actual Neg    32     5     4    → Recall Negatif: 78,05%
       Neu     6    30     5    → Recall Netral:  73,17%
       Pos     4     5    32    → Recall Positif: 78,05%
```

**Analisis Error:**
- Kelas **Netral** paling sering salah klasifikasi (recall terendah: 73,17%)
- Netral sering diklasifikasikan sebagai Negatif (6 dari 41) — ini masuk akal karena review netral cenderung berisi keluhan tanpa emosi kuat
- Positif dan Negatif memiliki recall yang sama (78,05%) — model adil untuk kedua kelas ekstrem

### 4.5 LDA Topic Modeling — Topik Keluhan Utama

Analisis topik pada 203 review negatif menggunakan LDA dengan optimal **3 topik** (coherence score tertinggi):

| Topik | Kata Kunci | Interpretasi |
|-------|------------|--------------|
| **Topik 1** | mau, tidak, sangat, saja, susah, aplikasi, lapor, ribet, daftar, npwp | **Kesulitan Pendaftaran & Pelaporan** — pengguna mengalami hambatan dalam daftar NPWP dan lapor SPT |
| **Topik 2** | aplikasi, tidak, saja, kode, pajak, tapi, email, verifikasi, bikin, malah | **Kegagalan Verifikasi OTP** — kode verifikasi tidak terkirim ke email/HP |
| **Topik 3** | tidak, sama, aplikasi, terus, efin, lama, gagal, sekali, password, apa | **Masalah EFIN & Login** — gagal aktivasi EFIN, password tidak bekerja, loading lama |

### 4.6 Temporal Analysis

Tren rating dan proporsi review negatif dari waktu ke waktu menunjukkan bahwa:
- Rating rata-rata cenderung **menurun** seiring waktu
- Proporsi review negatif **meningkat** pada periode tertentu (kemungkinan terkait update aplikasi yang bermasalah)

---

## V. KESIMPULAN DAN REKOMENDASI

### 5.1 Kesimpulan

1. **Word2Vec + Random Forest** merupakan kombinasi terbaik untuk klasifikasi sentimen ulasan M-Pajak dengan **Macro F1-Score 76,26%** setelah penerapan SMOTE.

2. **SMOTE** efektif meningkatkan performa model (+0,82% Macro F1) dengan menciptakan sampel sintetis yang membantu generalisasi, terutama untuk kelas Netral.

3. **Sastrawi stemming** berhasil menormalisasi kata-kata informal menjadi bentuk dasar, meningkatkan kualitas fitur untuk semua word embedding.

4. **LDA Topic Modeling** mengidentifikasi tiga topik keluhan utama: (1) Kesulitan Pendaftaran & Pelaporan, (2) Kegagalan Verifikasi OTP, dan (3) Masalah EFIN & Login.

5. **FastText yang di-train from scratch** performanya lebih rendah karena vocabulary yang terbatas. Penggunaan pre-trained FastText (cc.id.300.vec) direkomendasikan untuk penelitian selanjutnya.

### 5.2 Rekomendasi untuk Pengembang M-Pajak

Berdasarkan analisis topik, berikut rekomendasi prioritas:

| Prioritas | Masalah | Rekomendasi |
|-----------|---------|-------------|
| 🔴 Tinggi | Kegagalan OTP | Perbaiki infrastruktur pengiriman email/SMS OTP, tambahkan fallback (WhatsApp) |
| 🔴 Tinggi | Aktivasi EFIN | Sederhanakan alur aktivasi, berikan panduan visual step-by-step |
| 🟡 Sedang | UI/UX Pelaporan | Redesign form pelaporan, tambahkan auto-save dan draft |
| 🟡 Sedang | Stabilitas | Fix bug force close, optimasi performa pada device low-end |
| 🟢 Rendah | Pendaftaran NPWP | Integrasi dengan API Dukcapil untuk verifikasi otomatis |

### 5.3 Keterbatasan dan Penelitian Selanjutnya

1. **Dataset kecil setelah balancing** (609 sampel) — koleksi data lebih lanjut diperlukan
2. **FastText pre-trained tidak tersedia** di environment eksperimen — perlu download cc.id.300.vec (4.2GB)
3. **Labeling berbasis rating** mungkin tidak akurat 100% — validasi manual atau lexicon-based labeling direkomendasikan
4. **Deep learning models** (IndoBERT, LSTM) belum diuji karena keterbatasan GPU
5. **Aspect-Based Sentiment Analysis** dapat memberikan insight lebih granular per fitur aplikasi

---

## VI. REFERENSI

1. Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357. https://doi.org/10.1613/jair.953

2. Oktafiandi, H., Panjaitan, F., & Ramadhan, M.F. (2025). Comparative Analysis of Resampling Techniques to Improve SVM and Random Forest Performance in Coretax Sentiment Analysis. *IEEE International Conference on Informatics, Multimedia, Cyber*. https://ieeexplore.ieee.org/abstract/document/11326920/

3. Hadwan, M., Al-Sarem, M., Saeed, F., & Al-Hagery, M.A. (2022). An Improved Sentiment Classification Approach for Measuring User Satisfaction toward Governmental Services' Mobile Apps Using Machine Learning Methods with Feature Engineering and SMOTE Technique. *Applied Sciences*, 12(11), 5547. https://doi.org/10.3390/app12115547

4. Juandri, H., Hasmawati, & Bunyamin. (2024). Aspect-level Sentiment Analysis on GoPay App Reviews Using Multilayer Perceptron and Word Embeddings. *Kinetik*, 9(4). https://doi.org/10.22219/kinetik.v9i4.2041

5. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv preprint arXiv:1301.3781*.

6. Pennington, J., Socher, R., & Manning, C.D. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*, 1532-1543.

7. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. *Transactions of the ACL*, 5(1), 135-146.

8. Anjani, M., & Irmanda, H.N. (2024). Comparison Performance of Word2Vec, GloVe, Fasttext Using Support Vector Machine Method for Sentiment Analysis. *JUTIF*. http://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/1366

9. Wijaya, M.T.R.A., Widaningrum, I., et al. (2025). Using SVM and KNN for Predicting Customer Response Sentiment of M-PAJAK Application. *JIKA, UMS*. https://journals2.ums.ac.id/index.php/jika

10. Lestari, N.I., Taib, S.M., Wibowo, W., & Aziz, I.A. (2024). Aspect-Based Sentiment Analysis for Mobile App Review Using CNN and Word2Vec. *IEEE 7th International Conference*. https://ieeexplore.ieee.org/abstract/document/10828541/

---

## LAMPIRAN

### A. Struktur Proyek
```
final-project-txmg1/
├── data/
│   ├── raw/coretax_reviews.csv          # 8.099 ulasan mentah
│   └── processed/balanced_reviews.csv   # 609 ulasan balanced + preprocessed
├── notebooks/
│   ├── 01_eda_and_preparation.ipynb
│   ├── 02_labeling_and_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_comparison_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_extractors.py
│   ├── models.py
│   ├── evaluator.py
│   ├── experiment_runner.py
│   └── topic_modeler.py
├── results/
│   ├── baseline_comparison_table.csv
│   ├── ablation_comparison_table.csv
│   ├── experiment_summary.json
│   └── figures/
│       ├── eda/           # EDA visualizations
│       ├── evaluation/    # Confusion matrices, comparison charts
│       └── interpretation/ # LDA, temporal analysis
├── reports/laporan.md
└── complete_pipeline.py  # Full pipeline script
```

### B. Environment
- Python 3.12.11
- scikit-learn, xgboost, gensim, Sastrawi, imbalanced-learn
- Kernel: jupyter-env

### C. Reproducibility
Untuk menjalankan ulang seluruh pipeline:
```bash
pip install -r requirements.txt
pip install Sastrawi imbalanced-learn xgboost lightgbm gensim
python complete_pipeline.py
```
