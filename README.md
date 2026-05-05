# 📊 CoretTax Sentiment Classification Project

Proyek ini bertujuan untuk melakukan analisis sentimen ternary (Positif, Netral, Negatif) pada ulasan aplikasi **CoretTax/M-Pajak** menggunakan berbagai teknik Text Mining dan Machine Learning.

## 📁 Struktur Proyek

- `notebooks/`: File Jupyter Notebook untuk alur eksperimen.
- `src/`: Modul Python modular (preprocessing, modeling, dll).
- `data/`: Folder penyimpanan data (Raw & Processed).
- `results/`: Output eksperimen (Tabel komparasi & Visualisasi).

---

## 🛠️ Persiapan (Setup)

1. **Install Dependencies**:
   Buka terminal di folder root proyek dan jalankan:

   ```bash
   pip install -r requirements.txt
   ```
2. **Pilih Kernel**:
   Pastikan Anda menggunakan kernel Python yang benar (disarankan menggunakan environment `.venv` proyek ini).

3. **Download FastText Pre-trained Embedding**:
   Sebelum menjalankan alur kerja, Anda harus mengunduh file embedding GloVe (cc.id.300.vec):

   ```bash
   python scripts/download_cc_id_300_vec.py
   ```

   File ini (~4.21GB) akan disimpan di `data/embeddings/cc.id.300.vec` dan digunakan untuk feature extraction.
   
   **Catatan:** Tidak perlu `chmod +x` untuk menjalankan dengan command `python` atau `bash`. Hanya diperlukan jika ingin menjalankan langsung dengan `./scripts/download_cc_id_300_vec.sh`.

---

## 🚀 Alur Kerja (Step-by-Step)

Anda harus menjalankan notebook di folder `notebooks/` secara berurutan:

### 1️⃣ `01_eda_and_preparation.ipynb`

- **Tujuan**: Eksplorasi data awal dan pembersihan struktur data.
- **Output**: `data/processed/reviews_prepared.csv` (Data awal yang sudah rapi).

### 2️⃣ `02_labeling_and_preprocessing.ipynb`

- **Tujuan**: Melakukan labeling sentimen ternary (Rating 1-2=Negatif, 3=Netral, 4-5=Positif) dan menjalankan pipeline NLP (Slang removal, Stopwords, Stemming).
- **Output**: Menambahkan kolom `sentiment` dan `review_clean` pada dataset.

### 3️⃣ `03_feature_engineering.ipynb`

- **Tujuan**: Mengekstrak fitur teks menjadi matriks numerik.
- **Metode**: TF-IDF, FastText, dan Word2Vec.
- **Catatan**: Jika spek laptop terbatas, hindari menjalankan bagian **IndoBERT** karena sangat berat di CPU.

### 4️⃣ `04_model_training.ipynb`

- **Tujuan**: Melatih model Machine Learning dan mencari parameter terbaik.
- **Model**: Decision Tree, Random Forest, dan XGBoost (Konfigurasi ringan).
- **Output**: `results/comparison_table.csv` (Tabel hasil performa semua model).

### 5️⃣ `05_comparison_analysis.ipynb`

- **Tujuan**: Analisis hasil akhir, visualisasi perbandingan, Error Analysis, dan Topic Modeling (LDA).
- **Output**: Grafik perbandingan dan insight domain (masalah utama pengguna).

---

**Text Mining Final Project — Universitas Cakrawala**
