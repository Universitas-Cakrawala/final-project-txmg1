# ANALISIS SENTIMEN REVIEW APLIKASI M-PAJAK (CORETAX) BERDASARKAN HASIL EKSPERIMEN TERBARU

## A. Abstrak

Penelitian ini bertujuan untuk menganalisis sentimen pengguna terhadap aplikasi M-Pajak berdasarkan review pada Google Play Store dengan pendekatan *text mining*. Data yang digunakan terdiri atas korpus hasil persiapan sebanyak 8.099 review dan dataset klasifikasi seimbang sebanyak 609 review yang terbagi merata ke dalam tiga kelas sentimen, yaitu negatif, netral, dan positif. Eksperimen klasifikasi dilakukan dengan membandingkan tiga representasi fitur, yaitu Word2Vec, FastText, dan GloVe, serta tiga algoritma klasifikasi, yaitu Decision Tree, Random Forest, dan XGBoost. Hasil eksperimen terbaru menunjukkan bahwa kombinasi **Word2Vec + Random Forest** memberikan performa terbaik dengan nilai **weighted F1 sebesar 0,7462**, **macro F1 sebesar 0,7464**, **akurasi sebesar 74,59%**, dan **ROC-AUC sebesar 0,8966**. Analisis lanjutan menunjukkan bahwa keluhan pengguna paling banyak berkaitan dengan hambatan login dan verifikasi akun, kesulitan pelaporan atau pembayaran pajak, serta kendala registrasi dan NPWP. Temuan ini menunjukkan bahwa keberhasilan model tidak hanya ditentukan oleh algoritma klasifikasi, tetapi juga oleh kesesuaian representasi fitur dengan karakter korpus review yang relatif singkat dan padat keluhan fungsional.

**Kata Kunci:** analisis sentimen, M-Pajak, Word2Vec, Random Forest, text mining, LDA.

## B. Pendahuluan

Aplikasi M-Pajak merupakan salah satu sarana digital yang digunakan masyarakat untuk mengakses layanan perpajakan. Sebagai aplikasi layanan publik, kualitas pengalaman pengguna menjadi faktor yang sangat penting karena berhubungan langsung dengan persepsi masyarakat terhadap efektivitas dan kemudahan layanan yang diberikan. Salah satu sumber informasi yang dapat digunakan untuk menilai pengalaman tersebut adalah review pengguna pada Google Play Store. Review ini memuat penilaian, kritik, hambatan, dan apresiasi yang disampaikan secara langsung oleh pengguna setelah berinteraksi dengan aplikasi.

Permasalahan utama dalam membaca review secara manual terletak pada jumlah data yang besar dan sifat teks yang tidak terstruktur. Oleh karena itu, pendekatan *text mining* diperlukan agar pola sentimen dapat dikenali secara sistematis. Dalam penelitian ini, analisis sentimen dilakukan pada tiga kelas, yaitu negatif, netral, dan positif. Selain menentukan model klasifikasi terbaik, penelitian ini juga memanfaatkan analisis komparatif dan topik untuk memahami konteks keluhan pengguna secara lebih mendalam.

## C. Data dan Tahapan Penelitian

Penelitian ini memanfaatkan dua lapis data. Pertama, korpus hasil persiapan data yang berjumlah 8.099 review. Kedua, dataset klasifikasi seimbang sebanyak 609 review, dengan distribusi 203 review untuk masing-masing kelas sentimen. Dataset seimbang digunakan pada tahap eksperimen klasifikasi agar evaluasi model tidak terdistorsi oleh dominasi salah satu kelas.

Tahapan penelitian yang dilakukan dapat diringkas sebagai berikut.

1. Mengumpulkan review aplikasi M-Pajak dari Google Play Store.
2. Melakukan pembersihan teks dan persiapan data.
3. Menyusun dataset klasifikasi seimbang untuk tiga kelas sentimen.
4. Mengekstraksi fitur menggunakan Word2Vec, FastText, dan GloVe.
5. Melatih model Decision Tree, Random Forest, dan XGBoost.
6. Mengevaluasi model menggunakan weighted F1, macro F1, akurasi, ROC-AUC, dan waktu training.
7. Melakukan analisis lanjutan terhadap pola topik, tren waktu, dan distribusi sentimen per versi aplikasi.

Pada tahap evaluasi, metrik weighted F1 dan macro F1 menjadi perhatian utama karena keduanya lebih representatif untuk menilai performa model pada klasifikasi multi-kelas dibanding sekadar akurasi.

## D. Hasil Eksperimen Klasifikasi

### 1. Perbandingan Model

Hasil eksperimen terbaru yang digunakan dalam laporan ini mengacu pada `results/comparison_table.csv`, yang juga telah disinkronkan ke dalam analisis komparatif pada Notebook 5. Tabel berikut menunjukkan peringkat model berdasarkan weighted F1.

| Peringkat | Feature Extractor | Model | Weighted F1 | Macro F1 | Accuracy | ROC-AUC | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Word2Vec** | **Random Forest** | **0,7462** | **0,7464** | **74,59%** | **0,8966** | **0,83 s** |
| 2 | Word2Vec | XGBoost | 0,7117 | 0,7114 | 71,31% | 0,9007 | 10,69 s |
| 3 | FastText | Random Forest | 0,6570 | 0,6564 | 65,57% | 0,8135 | 3,00 s |
| 4 | Word2Vec | Decision Tree | 0,6377 | 0,6371 | 63,93% | 0,7598 | 0,07 s |
| 5 | FastText | XGBoost | 0,6067 | 0,6059 | 60,66% | 0,8335 | 8,58 s |
| 6 | FastText | Decision Tree | 0,5320 | 0,5315 | 53,28% | 0,6677 | 2,70 s |

![Perbandingan Performa Model](../results/figures/evaluation/grouped_bar_f1.png)

*Gambar 1. Perbandingan weighted F1 seluruh kombinasi model berdasarkan hasil eksperimen terbaru.*

Tabel tersebut menunjukkan bahwa kombinasi **Word2Vec + Random Forest** menempati posisi pertama, disusul oleh **Word2Vec + XGBoost**. Pola ini menandakan bahwa representasi Word2Vec memiliki kesesuaian yang lebih baik terhadap karakter data dibanding extractor lain pada eksperimen yang tersedia saat ini.

### 2. Interpretasi Hasil Model Terbaik

Keunggulan kombinasi Word2Vec + Random Forest dapat dijelaskan dari dua sisi. Dari sisi fitur, Word2Vec mampu merepresentasikan hubungan semantik antaristilah yang sering muncul secara berulang dalam review pengguna, seperti istilah yang berkaitan dengan login, error, verifikasi, pendaftaran, dan pelaporan. Dari sisi model, Random Forest cenderung stabil pada dataset berukuran menengah karena memanfaatkan banyak pohon keputusan yang bekerja secara kolektif, sehingga lebih tahan terhadap variasi lokal dibanding pohon tunggal.

Selain itu, kombinasi ini menunjukkan keseimbangan yang baik antara kualitas prediksi dan efisiensi komputasi. Waktu training yang relatif singkat menunjukkan bahwa model tidak hanya unggul secara metrik, tetapi juga praktis untuk diimplementasikan ulang pada eksperimen lanjutan.

### 3. Posisi Model Lain

Word2Vec + XGBoost tetap menunjukkan performa yang kuat, terutama karena nilai ROC-AUC-nya sedikit lebih tinggi dibanding Random Forest. Namun weighted F1 dan macro F1 yang lebih rendah menunjukkan bahwa konsistensi klasifikasi lintas kelas masih berada di bawah Word2Vec + Random Forest. Sementara itu, FastText belum mampu melampaui Word2Vec pada eksperimen ini. Hal tersebut mengindikasikan bahwa karakter korpus review yang digunakan lebih diuntungkan oleh representasi vektor kata yang stabil pada level kata utuh dibanding keunggulan subword yang menjadi ciri utama FastText.

## E. Analisis Lanjutan terhadap Data Review

### 1. Topik Keluhan Pengguna

Analisis topik pada review negatif menunjukkan bahwa keluhan pengguna terkonsentrasi pada tiga kelompok utama. Pertama, hambatan login, kode verifikasi, email, dan identitas akun. Kedua, kesulitan pelaporan dan pembayaran pajak. Ketiga, kendala registrasi, pendaftaran, dan NPWP. Pola ini menunjukkan bahwa sumber keluhan terbesar berasal dari alur inti penggunaan aplikasi, bukan dari fitur tambahan.

### 2. Tren Waktu dan Versi Aplikasi

Analisis temporal dan analisis per versi aplikasi menunjukkan adanya variasi tekanan sentimen negatif pada periode dan versi tertentu. Temuan ini berguna untuk mengidentifikasi kemungkinan adanya perubahan fitur, regresi teknis, atau titik friksi baru pada rilis tertentu. Meskipun demikian, hasil analisis domain ini tetap perlu dibaca sebagai indikasi pola analitis pada dataset yang digunakan, bukan sebagai generalisasi mutlak terhadap seluruh populasi review.

## F. Pembahasan

Hasil penelitian ini menunjukkan bahwa pemilihan representasi fitur memiliki peranan yang sangat penting dalam klasifikasi sentimen. Meskipun beberapa pendekatan lain dapat memberikan performa yang cukup baik, Word2Vec terbukti paling konsisten pada eksperimen terbaru. Temuan ini menunjukkan bahwa karakter review aplikasi M-Pajak cenderung cukup cocok direpresentasikan dengan embedding berbasis konteks distribusional yang sederhana namun stabil.

Random Forest sebagai model terbaik juga memperlihatkan bahwa pendekatan ensemble berbasis pohon masih sangat kompetitif pada klasifikasi teks berdimensi sedang, khususnya ketika ukuran dataset belum terlalu besar. Dalam konteks penelitian ini, model tersebut mampu menjaga keseimbangan antara sensitivitas terhadap variasi data dan kestabilan prediksi antar kelas.

Di sisi lain, analisis topik memperlihatkan bahwa masalah utama pengguna masih berkisar pada hambatan akses awal dan alur utama layanan. Dengan demikian, hasil eksperimen klasifikasi dan hasil analisis topik saling melengkapi. Model terbaik membantu mengidentifikasi polaritas sentimen secara kuantitatif, sedangkan analisis topik menjelaskan substansi masalah yang melatarbelakangi sentimen negatif tersebut.

## G. Keterbatasan Penelitian

Beberapa keterbatasan penelitian ini perlu dicatat. Pertama, dataset klasifikasi yang digunakan hanya berjumlah 609 review seimbang, sehingga hasil model belum tentu identik dengan perilaku seluruh populasi review. Kedua, repo sempat menyimpan lebih dari satu file hasil eksperimen, sehingga sinkronisasi antar notebook perlu dilakukan agar pembacaan hasil tetap konsisten. Ketiga, analisis topik, waktu, dan versi aplikasi bersifat pelengkap dan tetap memerlukan validasi lebih lanjut apabila akan digunakan untuk pengambilan keputusan operasional yang lebih spesifik.

## H. Kesimpulan

Berdasarkan hasil eksperimen terbaru, kombinasi **Word2Vec + Random Forest** merupakan model terbaik untuk klasifikasi sentimen review aplikasi M-Pajak pada data yang tersedia, dengan weighted F1 sebesar 0,7462, macro F1 sebesar 0,7464, akurasi sebesar 74,59%, dan ROC-AUC sebesar 0,8966. Hasil ini menunjukkan bahwa kombinasi representasi fitur yang sesuai dengan karakter korpus dan model ensemble yang stabil dapat memberikan performa yang lebih baik dibanding kombinasi lain pada eksperimen yang dilakukan.

Analisis lanjutan juga menunjukkan bahwa sentimen negatif pengguna paling banyak berkaitan dengan hambatan login dan verifikasi akun, kesulitan pelaporan atau pembayaran pajak, serta kendala registrasi dan NPWP. Oleh karena itu, perbaikan kualitas aplikasi sebaiknya diprioritaskan pada alur inti penggunaan yang langsung memengaruhi keberhasilan pengguna dalam menyelesaikan proses perpajakan.

## I. Referensi

1. Artefak internal repo: `notebooks/04_model_training.ipynb`, `notebooks/05_comparison_analysis.ipynb`, `results/comparison_table.csv`, `data/processed/balanced_reviews.csv`, dan `data/processed/reviews_prepared.csv`.
2. Mikolov, T., Chen, K., Corrado, G., and Dean, J. 2013. Efficient Estimation of Word Representations in Vector Space.
3. Breiman, L. 2001. Random Forests.
4. Blei, D. M., Ng, A. Y., and Jordan, M. I. 2003. Latent Dirichlet Allocation.
