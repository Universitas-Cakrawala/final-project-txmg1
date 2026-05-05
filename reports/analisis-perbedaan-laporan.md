# ANALISIS PERBEDAAN ANTARA LAPORAN SEBELUMNYA DAN LAPORAN REVISI

## A. Tujuan Dokumen

Dokumen ini disusun untuk membandingkan isi `laporan.md` versi sebelumnya dengan laporan revisi yang baru disusun berdasarkan artefak eksperimen final di repo. Fokus utamanya bukan untuk mencari kesalahan penulisan lama, melainkan untuk menjelaskan mengapa isi laporan bisa berbeda setelah seluruh notebook, file hasil, dan jejak eksperimen dibaca ulang secara menyeluruh.

Kesimpulan singkatnya adalah sebagai berikut: **perbedaan laporan tidak terutama disebabkan oleh perubahan teori, tetapi oleh perbedaan fase artefak yang dijadikan dasar penulisan.** Repo ini menyimpan hasil eksperimen legacy dan hasil eksperimen final secara berdampingan. Jika keduanya tidak dipisahkan dengan tegas, narasi hasil akan mudah bergeser.

## B. Perbandingan Inti

| Aspek | Laporan Sebelumnya | Laporan Revisi | Mengapa Berbeda |
| :--- | :--- | :--- | :--- |
| Sumber tabel performa | Langsung menonjolkan `GloVe + XGBoost` sebagai hasil utama tanpa menjelaskan evolusi artefak | Menetapkan `results/balanced_comparison_table.csv` sebagai sumber performa final dan `results/comparison_table.csv` sebagai artefak legacy | Repo memuat dua tabel hasil dari fase eksperimen yang berbeda |
| Status GloVe pada fase awal | Tidak dijelaskan | Dijelaskan bahwa eksperimen awal gagal memuat GloVe karena path `data/embeddings/cc.id.300.vec` tidak ditemukan | `notebooks/04_model_training.ipynb` menyimpan jejak error tersebut secara eksplisit |
| Jumlah topik LDA | Menyebut 5 topik dan menempatkan OTP sebagai isu paling dominan | Menetapkan 3 topik sebagai hasil optimal, dengan fokus pada login atau verifikasi, lapor atau bayar, dan registrasi atau NPWP | Output notebook 05 menunjukkan `best_n = 3`, bukan 5 |
| Basis analisis temporal dan versi | Terkesan berbicara tentang keseluruhan korpus review | Menyatakan eksplisit bahwa analisis temporal dan versi di notebook 05 dihitung dari `balanced_reviews.csv` | Notebook 05 memuat `balanced_reviews.csv` sebelum analisis topik, waktu, dan versi dijalankan |
| Deskripsi extractor GloVe | Cenderung dibaca sebagai GloVe murni | Dijelaskan sebagai GloVe-style mean pooling di atas pretrained Indonesian vectors `cc.id.300.vec` | Implementasi aktual di `src/feature_extractors.py` memakai vector pretrained tersebut |
| Klaim akurasi 91% | Ada klaim penurunan akurasi dari 91% ke 71% | Klaim ini dihapus dari laporan revisi | Tidak ada artefak hasil di repo yang mendukung angka 91% sebagai hasil eksperimen final |
| Pemisahan korpus dan sampel eksperimen | Angka 8.000-an review dan 609 review belum dipisahkan secara tegas | Korpus penuh 8.099 review dipisahkan jelas dari dataset klasifikasi final 609 review | Pemisahan ini diperlukan agar pembaca tahu basis setiap kesimpulan |

## C. Penjelasan Detail Perbedaan

### 1. Repo Menyimpan Dua Lapisan Hasil

Perbedaan pertama datang dari struktur repo itu sendiri. Pada satu sisi, ada `results/comparison_table.csv` yang masih merekam fase eksperimen ketika GloVe belum berhasil dimuat. Pada sisi lain, ada `results/balanced_comparison_table.csv` yang justru memuat konfigurasi final setelah hasil balanced yang memuat GloVe tersedia. Laporan lama cenderung langsung mengutip hasil terbaik final, tetapi belum menerangkan bahwa repo menyimpan fase sebelumnya secara berdampingan.

Akibatnya, pembaca yang membuka repo tanpa konteks bisa melihat dua cerita yang tampak berbeda. Satu file memperlihatkan Word2Vec dan FastText sebagai kandidat utama. File lain memperlihatkan GloVe sebagai pemenang. Setelah notebook training diperiksa, perbedaan ini menjadi masuk akal: pada fase awal, GloVe memang gagal masuk ke tabel karena file vector tidak ditemukan.

### 2. Error Path GloVe Menjelaskan Mengapa Tabel Lama Bisa Berbeda

`notebooks/04_model_training.ipynb` menyimpan pesan error yang sangat penting: extractor GloVe mencoba memuat file dari `data/embeddings/cc.id.300.vec`, lalu gagal karena file tersebut tidak ada. Ini menjelaskan kenapa `comparison_table.csv` tidak dapat diperlakukan sebagai representasi akhir dari seluruh extractor yang dirancang dalam eksperimen. Dengan kata lain, tabel lama berbeda bukan karena model Word2Vec tiba-tiba lebih unggul secara mutlak, melainkan karena satu kandidat utama belum sempat masuk secara valid ke kompetisi pada fase itu.

### 3. Laporan Lama Terlalu Cepat Mengunci Interpretasi Topik

Laporan sebelumnya menyebut 5 topik LDA dan menempatkan OTP sebagai isu sistemik paling dominan. Setelah output notebook 05 diperiksa ulang, kesimpulan itu terlalu jauh. Notebook justru menunjukkan coherence terbaik pada 3 topik. Kata-kata dominan yang keluar juga lebih aman dibaca sebagai tiga kelompok besar: autentikasi atau verifikasi, proses lapor atau bayar, serta registrasi atau NPWP. Istilah OTP mungkin masih berkaitan dengan tema verifikasi, tetapi output topik yang tersedia tidak cukup kuat untuk menjadikan OTP sebagai label tunggal paling dominan tanpa tambahan pembacaan manual terhadap review mentah.

### 4. Basis Analisis Domain pada Notebook Bukan Korpus Penuh

Perbedaan penting lain adalah basis data untuk analisis domain. Pada notebook 05, variabel `df` untuk error analysis, topic modeling, temporal analysis, dan version analysis berasal dari `balanced_reviews.csv`. Ini berarti bulan-bulan yang ditampilkan, distribusi per versi, dan jumlah review negatif yang dipakai LDA semuanya berasal dari sampel seimbang, bukan dari populasi asli 8.099 review. Laporan lama belum menekankan batas ini secara cukup keras, sehingga pembaca bisa mengira tren temporal yang terlihat sudah identik dengan populasi review secara keseluruhan.

### 5. Revisi Menghapus Klaim yang Tidak Memiliki Jejak Artefak

Laporan sebelumnya menyatakan bahwa akurasi turun dari 91% pada data timpang menjadi 71% pada data seimbang. Setelah pencarian repo dilakukan ulang, angka 91% tersebut tidak muncul sebagai hasil eksperimen yang didukung file hasil, notebook, atau tabel evaluasi lain. Karena itu, laporan revisi tidak mempertahankan klaim tersebut. Dalam pelaporan eksperimen, angka yang tidak punya jejak artefak sebaiknya dianggap sebagai klaim yang belum tervalidasi, bukan sebagai fakta akhir.

## D. Makna Metodologis dari Perbedaan Ini

Perbedaan antara laporan lama dan laporan revisi memberi satu pelajaran metodologis yang cukup penting. Dalam proyek berbasis notebook, hasil akhir sangat mudah bergeser jika repo menyimpan banyak artefak transisi. Satu notebook bisa masih memuat output lama. Notebook lain bisa sudah memakai file hasil baru. Sementara laporan markdown bisa menulis narasi yang berada di tengah-tengah keduanya. Jika ketiga lapisan ini tidak disinkronkan, pembaca akan melihat perbedaan yang sebenarnya bukan berasal dari perubahan model, melainkan dari perubahan konteks eksperimen.

Karena itu, laporan revisi mengambil posisi yang lebih ketat: setiap klaim utama harus punya jejak yang jelas di artefak repo. Jika tidak ada jejaknya, klaim tersebut tidak dipakai sebagai dasar kesimpulan.

## E. Kesimpulan Perbandingan

Secara substantif, laporan revisi tidak membatalkan inti penelitian sebelumnya. Arah umumnya tetap sama: aplikasi M-Pajak menerima cukup banyak keluhan pengguna, dan kombinasi fitur semantik yang lebih kaya dengan model ensemble memberikan hasil terbaik. Yang berubah adalah **ketelitian pembacaan artefak**. Laporan revisi memperbaiki basis bukti, memperjelas fase hasil, mengoreksi jumlah topik LDA, menjelaskan error GloVe pada fase awal, dan menghapus klaim angka yang tidak punya dukungan artefak.

Kalau diringkas dalam satu kalimat, maka penyebab utama perbedaan laporan adalah ini: **laporan lama menulis cerita yang masuk akal, tetapi laporan revisi menulis cerita yang lebih ketat terhadap bukti yang benar-benar tersimpan di repo.**

## F. Rekomendasi Praktis untuk Repo

Supaya perbedaan serupa tidak terulang pada iterasi berikutnya, ada beberapa langkah praktis yang layak dilakukan.

1. Arsipkan atau beri nama yang lebih eksplisit pada `comparison_table.csv`, misalnya `comparison_table_legacy_missing_glove.csv`.
2. Tetapkan satu file hasil kanonik untuk pelaporan final, misalnya `final_comparison_table.csv`.
3. Samakan konfigurasi path GloVe di seluruh kode agar tidak ada lagi perbedaan antara `data/embeddings` dan `data/processed`.
4. Pisahkan dengan jelas notebook analisis pada dataset seimbang dan notebook analisis pada korpus penuh agar pembaca tidak mencampur keduanya.