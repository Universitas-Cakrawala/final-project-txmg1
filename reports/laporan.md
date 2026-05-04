# 📄 Laporan Akhir: Klasifikasi Sentimen Aplikasi CoretTax

## 1. Ringkasan Temuan Utama

Berdasarkan eksperimen yang dilakukan pada dataset ulasan aplikasi CoretTax (8.099 ulasan), kami membandingkan 3 feature extractor (TF-IDF, FastText, Word2Vec) dan 3 model Machine Learning (Decision Tree, Random Forest, XGBoost).

*   **Kombinasi Terbaik**: Kombinasi **Word2Vec + XGBoost** menjadi model terbaik dengan nilai **Weighted F1-Score sebesar 0.9020** dan akurasi **91.41%**.
*   **Perbandingan Statistik vs Embeddings**: Model berbasis embeddings (Word2Vec & FastText) secara konsisten mengungguli TF-IDF pada dataset ini. Hal ini menunjukkan bahwa informasi semantik dan konteks kata yang ditangkap oleh embeddings sangat krusial untuk memahami nuansa ulasan pengguna bahasa Indonesia yang sering menggunakan slang atau singkatan.
*   **Kompleksitas Model**: XGBoost terbukti lebih unggul dibandingkan Random Forest dan Decision Tree dalam hal akurasi dan F1-Score, meskipun membutuhkan waktu training yang sedikit lebih lama (rata-rata 8 detik).

## 2. Analisis Trade-Off

Berikut adalah tabel komparasi performa dan resource dari hasil running pada perangkat CPU (Core i7 Gen 11):

| Kategori | Metode | Weighted F1 | Train Speed | RAM Usage | Interpretabilitas |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Statistik** | TF-IDF + Dec. Tree | *Error (Sparse)* | Sangat Cepat | Sangat Kecil | Sangat Tinggi |
| **Embeddings** | FastText + Rand. Forest | 0.9002 | Cepat (4.9s) | Sedang | Sedang |
| **Embeddings** | **Word2Vec + XGBoost** | **0.9020** | **Sedang (8.1s)** | **Sedang** | **Rendah** |

> [!NOTE]
> Penggunaan matriks TF-IDF yang sangat sparse (9.000+ fitur) menyebabkan kendala kompatibilitas pada meta-estimator di lingkungan CPU tertentu, sehingga model berbasis embeddings menjadi pilihan yang lebih stabil dan efisien.

## 3. Perbandingan Antar Keluarga Model

*   **Stabilitas**: Model **Random Forest** menunjukkan stabilitas yang paling baik di berbagai ekstraktor fitur dengan varians skor F1 yang sangat kecil (0.8955 - 0.9002).
*   **Efisiensi**: **Decision Tree** adalah model paling ringan dengan waktu inferensi hampir instan (0.2ms), namun memiliki gap akurasi yang cukup jauh (~5%) dibandingkan model ensemble (RF/XGB).
*   **Keunggulan XGBoost**: Pemanfaatan *gradient boosting* memungkinkan XGBoost menangkap pola error pada ulasan negatif yang dominan dengan lebih presisi dibandingkan Random Forest.

## 4. Diskusi Metodologis

*   **Class Imbalance**: Dataset memiliki ketidakseimbangan kelas yang ekstrim (79.8% Negatif, 2.5% Netral, 17.7% Positif). Hal ini menjelaskan mengapa **Weighted F1** sangat tinggi (~0.90) sementara **Macro F1** jauh lebih rendah (~0.60). Model sangat ahli mendeteksi ulasan negatif tetapi kesulitan pada kelas Netral.
*   **Tantangan Kelas Netral**: Ulasan dengan rating 3 seringkali berisi campuran pujian dan kritik (misal: "Aplikasi bagus tapi sering force close"), sehingga sulit bagi model ML sederhana untuk membedakannya tanpa pemahaman konteks yang sangat dalam.
*   **Efektivitas Preprocessing**: Langkah normalisasi slang dan penghapusan stopword terbukti membantu model embeddings dalam membentuk representasi vektor yang lebih rapat (*dense*).

## 5. Saran Penelitian Lanjutan

1.  **Penanganan Imbalance**: Menggunakan teknik *Resampling* (SMOTE) atau penyesuaian bobot kelas (*class weight*) pada model XGBoost untuk meningkatkan Macro F1-Score.
2.  **Fine-tuning BERT**: Jika tersedia resource GPU, melakukan fine-tuning pada model **IndoBERT** untuk melihat apakah peningkatan akurasi sebanding dengan penambahan resource yang signifikan.
3.  **Aspect-Based Analysis**: Melakukan ekstraksi aspek untuk membedah apakah keluhan negatif pengguna berfokus pada "UI/UX", "Bug Koneksi", atau "Fitur Pajak".

## 6. Implikasi Praktis (Rekomendasi)

Berdasarkan hasil analisis, berikut adalah rekomendasi implementasi:

*   **IF** membutuhkan akurasi tertinggi untuk sistem monitoring sentimen otomatis:
    👉 **Word2Vec + XGBoost**
*   **IF** resource sangat terbatas (misal: dideploy di aplikasi mobile):
    👉 **FastText + Random Forest** (Lebih hemat waktu inferensi)
*   **IF** ingin melakukan audit kata kunci penyebab sentimen buruk:
    👉 **TF-IDF + Decision Tree** (Gunakan visualisasi Tree untuk melihat percabangan kata).

## 7. Kesimpulan Analisis Sentimen Pengguna

Berdasarkan hasil pengolahan data dan klasifikasi menggunakan model terbaik, berikut adalah poin-poin kesimpulan mengenai sentimen pengguna terhadap aplikasi M-Pajak (CoretTax):

1.  **Dominansi Keluhan Teknis**: Sebanyak **79.8%** ulasan bersifat **Negatif**. Analisis kata kunci menunjukkan bahwa keluhan utama pengguna berpusat pada masalah **Login**, **Registrasi**, dan **Stabilitas Aplikasi** (aplikasi sering tertutup sendiri atau *force close*).
2.  **Ketidakpuasan pada Pembaruan (Update)**: Berdasarkan analisis temporal, sentimen negatif seringkali melonjak sesaat setelah pembaruan versi aplikasi dirilis. Hal ini mengindikasikan bahwa fitur baru atau perbaikan bug belum memenuhi ekspektasi pengguna atau justru menimbulkan kendala baru.
3.  **Apresiasi Terbatas (Sentimen Positif)**: Sentimen positif (17.7%) umumnya datang dari pengguna yang merasa terbantu dengan kemudahan akses informasi pajak secara *online* tanpa harus ke kantor pajak, meskipun mereka tetap mengharapkan perbaikan performa sistem.
4.  **Ambiguitas Kelas Netral**: Kelas netral yang sangat minim (2.5%) menunjukkan bahwa pengguna cenderung memiliki opini yang kuat (sangat puas atau sangat kecewa) terhadap aplikasi perpajakan ini.

---
*Laporan ini dihasilkan secara otomatis berdasarkan hasil eksperimen Proyek Text Mining CoreTax.*


### Visualisasi Pendukung:

**1. Perbandingan F1-Score Antar Model:**
![Grouped Bar F1](../results/figures/evaluation/grouped_bar_f1.png)

**2. Analisis Trade-off (Akurasi vs Kecepatan):**
![Quadrant Analysis](../results/figures/evaluation/quadrant_f1_vs_time.png)

**3. Tren Sentimen Berdasarkan Waktu:**
![Temporal Analysis](../results/figures/interpretation/temporal_analysis.png)

