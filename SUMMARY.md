# 📊 Deep Analysis: Perbandingan Workflow Analisis Sentimen CoreTax

Dokumen ini menyajikan analisis mendalam mengenai evolusi sistem klasifikasi sentimen untuk aplikasi CoreTax/M-Pajak, membandingkan sistem baseline (**Old Workflow**) dengan sistem yang telah dioptimasi (**New Workflow**).

## 1. Evolusi Strategi Workflow

Peralihan dari Workflow lama ke baru bukan hanya sekadar penggantian model, melainkan perombakan total pada pipa pemrosesan data (data pipeline).

### 🔄 Perbandingan Teknis

| Fitur                     | Workflow Lama (Baseline)                                            | Workflow Baru (Optimized)                                               |
| :------------------------ | :------------------------------------------------------------------ | :---------------------------------------------------------------------- |
| **Notebooks**       | `01_eda_and_preparation.ipynb` - `05_comparison_analysis.ipynb` | `01.1_eda_and_preparation.ipynb` - `05.5_comparison_analysis.ipynb` |
| **Rentang Waktu**   | 2021 - 2026                                                         | **2024 - 2026** (2 Tahun Terakhir)                                |
| **Labeling**        | 3 Kelas (Rating 1-5)                                                | **2 Kelas** (Rating 1-2 & 4-5)                                    |
| **Noise Filtering** | Tanpa filter panjang ulasan                                         | **Minimal 3 Token**                                               |
| **Preprocessing**   | Basic cleaning                                                      | Advanced cleaning + Sastrawi Stemming                                   |
| **Dataset State**   | Unbalanced (Dominan Negatif)                                        | **Balanced** (Strategi Resampling)                                |

---

## 2. Analisis Sumber Dataset (Dataset Source)

Salah satu faktor kunci keberhasilan **Workflow Baru** adalah penyempurnaan pada pemilihan dan pembersihan dataset. Berikut adalah perbedaan mendasar antara kedua dataset yang digunakan:

### 📂 Dataset Workflow Lama (Baseline)

- **Path:** [`data/processed/reviews_prepared.csv`](file://data/processed/reviews_prepared.csv)
- **Cakupan Data:** Mengambil seluruh riwayat ulasan dari tahun **2021 hingga 2026**.
- **Karakteristik:** Dataset ini masih mempertahankan rating **Netral (Rating 3)**. Keberadaan data netral ini terbukti menjadi "noise" yang signifikan karena ambiguitas kata yang tumpang tindih antara ulasan positif dan negatif, sehingga akurasi model sulit menembus angka 70%.

### 📂 Dataset Workflow Baru (Optimized)

- **Path:** [`data/processed/reviews_prepared_new.csv`](file://data/processed/reviews_prepared_new.csv)
- **Cakupan Data:** Difokuskan pada data **2 tahun terakhir (2024 - 2026)**.
- **Karakteristik:** Dilakukan strategi **Drop Rating Netral (3)** u ntuk mempertajam perbedaan antara sentimen positif dan negatif. Selain itu, dataset ini telah diseimbangkan (balanced) agar model tidak bias terhadap satu kelas tertentu. Pemilihan data terbaru (2024-2026) juga memastikan model lebih relevan dengan kondisi aplikasi saat ini.

---

## 3. Analisis Hasil Evaluasi

Workflow baru menunjukkan lompatan performa yang signifikan, memvalidasi bahwa penghapusan kelas "Netral" (Rating 3) sangat krusial untuk akurasi model.

### 📈 Metrik Performa Utama

- **Workflow Lama:** Akurasi puncak **68.0%** (FastText + XGBoost).
- **Workflow Baru:** Akurasi puncak **90.6%** (TF-IDF + Logistic Regression).

### 🖼️ Visualisasi Evaluasi

Perhatikan perbedaan distribusi skor F1 antara kedua workflow di bawah ini:

#### A. Grouped Bar Chart (Performa Model)

Grafik ini membandingkan skor **Weighted F1-Score** (keseimbangan antara ketepatan dan cakupan prediksi) untuk berbagai kombinasi teknik ekstraksi fitur dan model mesin pencari.

![Grouped Bar F1 - Old](results/figures/evaluation/grouped_bar_f1.png)
*Gambar 1: Workflow Lama - Anda dapat melihat bahwa sebagian besar batang grafik tertahan di bawah angka 0.7 (70%). Ini menunjukkan model kesulitan memberikan prediksi yang konsisten karena adanya data "Netral" yang membingungkan.*

![Grouped Bar F1 - New](results/figures/evaluation/grouped_bar_f1_new.png)
*Gambar 2: Workflow Baru - Terjadi lonjakan tinggi pada hampir seluruh model. Model terbaik (**TF-IDF + Logistic Regression**) mencapai angka 0.9 (90%), membuktikan bahwa pembersihan data dan fokus pada dua kategori (Positif/Negatif) membuat model jauh lebih cerdas.*

#### B. Quadrant Analysis (F1 vs Time)

Analisis kuadran ini digunakan untuk mencari model yang tidak hanya akurat, tetapi juga efisien (cepat). Sumbu **Y (Vertikal)** menunjukkan Akurasi/F1, dan sumbu **X (Horizontal)** menunjukkan waktu training.

![Quadrant - Old](results/figures/evaluation/quadrant_f1_vs_time.png)
![Quadrant - New](results/figures/evaluation/quadrant_f1_vs_time_new.png)
*Gambar 3 & 4: Pada Workflow Baru, titik-titik model berkumpul di **Kuadran Kiri Atas (Sweet Spot)**. Artinya, model mampu memberikan hasil sangat akurat dengan waktu proses yang sangat singkat (di bawah 5 detik), terutama pada model Logistic Regression.*

---

## 4. Analisis Interpretasi & Topic Modeling

Selain akurasi, interpretasi terhadap apa yang dibicarakan pengguna juga mengalami perubahan kualitas.

### 🧠 Topic Modeling (LDA)

Kami menggunakan teknik LDA untuk mengelompokkan kata-kata yang sering muncul bersama menjadi "Topik". Metrik **Coherence Score** digunakan untuk menentukan jumlah topik yang paling logis bagi manusia.

![LDA Coherence - Old](results/figures/interpretation/lda_coherence.png)
![LDA Coherence - New](results/figures/interpretation/lda_coherence_new.png)
*Gambar 5 & 6: Grafik Coherence pada Workflow Baru menunjukkan tren yang lebih stabil dan tinggi dibandingkan versi lama. Ini berarti topik-topik yang ditemukan oleh AI (seperti masalah login, error sistem, atau pujian UI) menjadi lebih jelas dan tidak tumpang tindih.*

### 🕒 Analisis Temporal & Sentimen Versi

Analisis ini membantu tim pengembang melihat kapan sentimen negatif memuncak dan pada versi aplikasi mana masalah sering terjadi.

#### Perbandingan Visual:
- **Sentiment by Version:** Menunjukkan perbandingan jumlah sentimen untuk setiap update aplikasi.
- **Temporal Analysis:** Menunjukkan fluktuasi emosi pengguna dari bulan ke bulan.

![Sentiment by Version - Old](results/figures/interpretation/sentiment_by_version.png)
![Temporal Analysis - Old](results/figures/interpretation/temporal_analysis.png)
*Gambar 7 & 8: Pada Workflow Lama, kelas "Netral" (warna abu-abu) mendominasi dan menutupi tren asli, sehingga sulit menentukan apakah aplikasi sebenarnya membaik atau memburuk.*

![Sentiment by Version - New](results/figures/interpretation/sentiment_by_version_new.png)
![Temporal Analysis - New](results/figures/interpretation/temporal_analysis_new.png)
*Gambar 9 & 10: Pada Workflow Baru, grafik menjadi sangat kontras (Hanya Biru/Positif dan Merah/Negatif). Kita dapat melihat dengan jelas titik-titik di mana sentimen negatif melonjak, yang biasanya bertepatan dengan adanya bug sistem pada tanggal tertentu.*

---

## 5. Kesimpulan Akhir

Berdasarkan analisis di atas:

1. **Workflow Baru (`_new`)** adalah pemenang mutlak dengan akurasi **90.6%**.
2. **TF-IDF + Logistic Regression** adalah kombinasi paling direkomendasikan karena mencapai akurasi tertinggi dengan waktu komputasi yang sangat efisien.
3. Penyebab utama rendahnya akurasi pada workflow lama adalah **Rating 3 (Netral)** yang bersifat ambigu, di mana teks ulasannya seringkali memiliki nada yang sama dengan ulasan negatif (misal: "Aplikasi lambat tapi oke lah").

---

**Text Mining Project — Universitas Cakrawala**
