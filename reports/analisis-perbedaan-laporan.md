# ANALISIS PERBEDAAN ANTARA LAPORAN SEBELUMNYA DAN LAPORAN REVISI TERBARU

## A. Tujuan Dokumen

Dokumen ini menjelaskan mengapa laporan terbaru sekarang berbeda dari versi sebelumnya. Fokus utamanya adalah menunjukkan sumber perbedaan pembacaan hasil eksperimen, agar tidak terjadi lagi kebingungan antara hasil Model Training dan hasil yang muncul pada notebook Comparison Analysis.

Secara singkat, sumber utama perbedaan adalah ini: **versi laporan sebelumnya terlalu terpengaruh oleh `notebooks/05_comparison_analysis.ipynb`, sedangkan laporan terbaru sekarang dipaksa kembali mengikuti `notebooks/04_model_training.ipynb` dan `results/comparison_table.csv`.**

## B. Perbedaan Inti

| Aspek | Laporan Sebelumnya | Laporan Terbaru | Penyebab Perbedaan |
| :--- | :--- | :--- | :--- |
| Sumber ranking model | Lebih mengandalkan hasil pada comparison analysis | Mengandalkan hasil pada model training | Notebook 05 membaca file hasil yang berbeda dari Notebook 04 |
| Model terbaik | GloVe + XGBoost | Word2Vec + Random Forest | `comparison_table.csv` menempatkan Word2Vec + Random Forest di peringkat pertama |
| Fungsi Notebook 05 | Dipakai sebagai dasar utama hasil | Dipakai hanya sebagai analisis tambahan | Notebook 05 cocok untuk analisis lanjutan, bukan penentuan pemenang model pada laporan ini |
| Struktur laporan | Ada artefak patch dan duplikasi di bagian bawah | Dibersihkan total dan ditulis ulang | Patch sebelumnya sempat ikut tertulis ke isi file Markdown |

## C. Mengapa Hasilnya Bisa Berbeda

Perbedaan hasil muncul karena repo ini menyimpan lebih dari satu artefak evaluasi. `results/comparison_table.csv` memuat ranking yang selaras dengan Notebook 04, sedangkan Notebook 05 pada salah satu tahap memakai `results/balanced_comparison_table.csv`. Dari sinilah muncul dua narasi berbeda.

Kalau pembacaan dimulai dari Notebook 04, maka model terbaik yang terlihat adalah **Word2Vec + Random Forest**. Akan tetapi, kalau pembacaan dimulai dari Notebook 05 tanpa pemisahan konteks, maka pembaca bisa terdorong pada simpulan lain, yaitu **GloVe + XGBoost**. Jadi, masalah utamanya bukan pada teori modelnya, melainkan pada **artefak mana yang dipilih sebagai sumber utama laporan**.

## D. Kenapa Laporan Sebelumnya Sulit Dibaca

Selain masalah sumber hasil, versi laporan sebelumnya juga memang sulit dibaca karena file `laporan.md` sempat berisi sisa teks patch. Akibatnya, setelah bagian referensi selesai, isi file masih berlanjut dengan teks teknis seperti `*** Add File` dan potongan dokumen lain. Itu yang menimbulkan kesan duplikat dan membuat alur bacanya kacau.

Pada revisi terbaru, file tersebut sudah dibersihkan dengan pendekatan paling aman, yaitu **menulis ulang seluruh laporan utama dari awal**. Jadi, bukan hanya mengganti beberapa paragraf, tetapi menghapus artefak duplikat sampai tuntas.

## E. Posisi Notebook 05 dalam Laporan Terbaru

Notebook 05 tidak dibuang. Fungsinya tetap penting, tetapi statusnya diturunkan menjadi bahan analisis tambahan. Dengan kata lain:

1. Notebook 04 dipakai untuk menentukan model terbaik.
2. Notebook 05 dipakai untuk membantu membaca topik keluhan, tren waktu, dan distribusi sentimen secara deskriptif.

Pemisahan ini membuat laporan lebih masuk akal. Hasil training tidak lagi berubah hanya karena notebook analisis lanjutan membaca file hasil lain.

## F. Kesimpulan

Perbedaan antara laporan sebelumnya dan laporan terbaru sekarang berasal dari dua hal. Pertama, pemilihan sumber hasil yang tidak konsisten. Kedua, adanya artefak patch yang membuat isi file Markdown menjadi berduplikasi. Setelah keduanya diperbaiki, laporan terbaru sekarang mengikuti arah yang lebih jelas: **Word2Vec + Random Forest menjadi model terbaik berdasarkan Model Training, sedangkan Comparison Analysis hanya dipakai sebagai bahan pembahasan tambahan.**