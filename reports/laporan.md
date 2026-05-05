# LAPORAN AKHIR ANALISIS SENTIMEN REVIEW APLIKASI M-PAJAK (CORETAX)

## A. Ringkasan Eksekutif

Dokumen ini merupakan laporan akhir yang disusun berdasarkan artefak eksperimen yang benar-benar tersedia di repo, terutama `notebooks/04_model_training.ipynb`, `notebooks/05_comparison_analysis.ipynb`, `results/balanced_comparison_table.csv`, `results/comparison_table.csv`, dan file data pada folder `data/processed`. Fokus utama laporan ini adalah menjawab dua pertanyaan. Pertama, kombinasi fitur dan model mana yang memberikan performa terbaik untuk klasifikasi sentimen review M-Pajak. Kedua, mengapa hasil akhir eksperimen dapat terlihat seperti sekarang, termasuk mengapa hasil tersebut dapat berbeda dari laporan yang lebih lama.

Secara ringkas, eksperimen supervised final dijalankan pada dataset seimbang berukuran 609 review, terdiri atas 203 review negatif, 203 review netral, dan 203 review positif. Dari sembilan kombinasi fitur dan model pada tabel final, kombinasi **GloVe-style embeddings + XGBoost** menjadi konfigurasi terbaik dengan **accuracy 71,31%**, **macro F1 0,7144**, dan **ROC-AUC 0,8560**. Pada sisi interpretasi domain, notebook analisis menunjukkan bahwa review negatif paling kuat mengelompok pada tiga area masalah: hambatan login dan verifikasi, kesulitan alur lapor atau bayar pajak, serta friksi registrasi atau pendaftaran NPWP. Dengan kata lain, sumber ketidakpuasan pengguna tidak hanya bersifat teknis, tetapi juga terkait friksi pada alur penggunaan inti aplikasi.

Laporan ini juga menegaskan satu hal penting. Tidak semua artefak hasil di repo merepresentasikan fase eksperimen yang sama. Ada tabel hasil legacy yang masih tersimpan dari fase ketika GloVe belum berhasil dimuat karena path file vector salah. Karena itu, untuk kepentingan pelaporan akhir, sumber performa model yang dipakai dalam dokumen ini adalah `results/balanced_comparison_table.csv`, sedangkan `results/comparison_table.csv` diperlakukan sebagai jejak eksperimen antara yang tetap berguna untuk menjelaskan evolusi hasil.

## B. Konteks Penelitian

Analisis sentimen terhadap review aplikasi M-Pajak penting karena review pengguna di Google Play Store berfungsi sebagai sinyal langsung atas kualitas layanan digital perpajakan. Dalam konteks ini, review bukan sekadar opini bebas. Review adalah representasi pengalaman pengguna ketika berhadapan dengan proses login, registrasi, pelaporan, pembayaran, dan stabilitas teknis aplikasi. Ketika volume review sudah mencapai ribuan entri, pembacaan manual menjadi tidak efisien. Karena itu, pendekatan *text mining* diperlukan agar pola sentimen dan sumber friksi dapat diidentifikasi secara sistematis.

Repositori proyek ini menunjukkan bahwa penelitian dilakukan dalam dua lapis. Lapis pertama adalah persiapan korpus review dalam skala penuh. Lapis kedua adalah eksperimen supervised pada sampel yang sudah diseimbangkan. Pemisahan dua lapis ini penting, karena tanpa pemisahan yang tegas, angka performa model dan interpretasi domain dapat terlihat seolah-olah berasal dari basis data yang sama, padahal tidak selalu demikian.

## C. Sumber Data dan Artefak yang Dipakai

Tabel berikut merangkum sumber data yang benar-benar dipakai untuk menyusun laporan ini.

| Artefak | Fungsi | Temuan Kunci |
| :--- | :--- | :--- |
| `data/processed/reviews_prepared.csv` | Korpus hasil persiapan data | Berisi 8.099 review yang sudah dibersihkan secara awal, tetapi belum memuat label sentimen final untuk eksperimen supervised. |
| `data/processed/balanced_reviews.csv` | Dataset klasifikasi final | Berisi 609 review dengan distribusi seimbang: 203 negatif, 203 netral, 203 positif. |
| `results/balanced_comparison_table.csv` | Tabel performa final | Memuat 9 kombinasi valid dan menjadi dasar utama evaluasi klasifikasi pada laporan ini. |
| `results/comparison_table.csv` | Tabel hasil legacy | Memuat hasil fase sebelumnya ketika GloVe belum ikut masuk ke tabel final karena error pemuatan vector. |
| `notebooks/04_model_training.ipynb` | Jejak proses training | Menunjukkan bahwa eksperimen awal gagal memuat GloVe karena file `data/embeddings/cc.id.300.vec` tidak ditemukan. |
| `notebooks/05_comparison_analysis.ipynb` | Analisis komparatif dan domain | Menunjukkan bahwa visual final performa memakai `balanced_comparison_table.csv`, sedangkan analisis topik, temporal, dan versi dijalankan di atas `balanced_reviews.csv`. |

Struktur artefak di atas menjelaskan bahwa laporan final tidak boleh hanya membaca satu file hasil secara terpisah. Ia harus membaca hubungan antarfile. Tanpa itu, laporan mudah jatuh pada kesimpulan yang tampak masuk akal, tetapi kurang presisi secara metodologis.

## D. Metodologi Eksperimen

### 1. Alur Data

Penelitian dimulai dari review aplikasi M-Pajak yang kemudian melewati tahap pembersihan dan persiapan teks. Tahap ini mencakup pembersihan karakter yang tidak relevan, normalisasi teks, penyusunan kolom waktu dan versi aplikasi, serta pembentukan teks bersih untuk pemodelan. Setelah itu, data untuk klasifikasi disusun ke dalam dataset seimbang agar model tidak terlalu bias ke kelas tertentu.

Secara praktis, repo ini memperlihatkan dua ukuran data yang harus dibedakan sejak awal:

1. Korpus review siap olah: 8.099 review.
2. Dataset klasifikasi seimbang: 609 review.

Pemisahan ini sangat penting, karena angka 8.099 berbicara tentang cakupan korpus, sedangkan angka 609 berbicara tentang basis evaluasi supervised yang dipakai untuk membandingkan model.

### 2. Representasi Fitur

Eksperimen final membandingkan tiga extractor utama:

1. **GloVe-style embeddings**
2. **FastText**
3. **Word2Vec**

Ada satu nuansa implementasi yang perlu dicatat secara eksplisit. Di level kode, extractor yang dinamai `GloVe` pada repo ini sesungguhnya menggunakan pendekatan **GloVe-style mean pooling** di atas file vector pretrained Bahasa Indonesia `cc.id.300.vec`. Jadi, istilah `GloVe` dipakai sebagai nama eksperimen yang konsisten dengan repo, tetapi sumber vector-nya adalah pretrained Indonesian word vectors berukuran 300 dimensi. Nuansa ini penting karena performa yang tinggi pada konfigurasi `GloVe` tidak datang dari training embedding dari nol pada 609 review, melainkan dari transfer pengetahuan semantik yang sudah dilatih di korpus yang jauh lebih besar.

### 3. Model Klasifikasi

Tiga model yang dibandingkan pada fase final adalah:

1. **Decision Tree**
2. **Random Forest**
3. **XGBoost**

Ketiganya dipilih karena memberikan spektrum yang cukup jelas: Decision Tree sebagai baseline pohon tunggal, Random Forest sebagai ensemble yang lebih stabil, dan XGBoost sebagai boosting model yang biasanya lebih kuat dalam menangkap boundary klasifikasi yang tidak sederhana.

### 4. Metrik Evaluasi

Evaluasi mengacu pada beberapa metrik sekaligus, yaitu `accuracy`, `weighted_f1`, `macro_f1`, `roc_auc`, `train_time_s`, dan `infer_time_ms`. Dari semua metrik tersebut, **macro F1** menjadi metrik yang paling penting untuk membaca kualitas model pada dataset multi-kelas yang sengaja diseimbangkan. Alasannya sederhana: macro F1 menilai setiap kelas secara setara, sehingga model tidak bisa terlihat bagus hanya karena dominan pada satu kelas saja.

## E. Hasil Eksperimen Klasifikasi

### 1. Tabel Hasil Final

Berikut adalah tabel hasil final yang diambil dari `results/balanced_comparison_table.csv`.

| Peringkat | Feature Extractor | Model | Accuracy | Macro F1 | ROC-AUC | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **GloVe** | **XGBoost** | **71,31%** | **0,7144** | **0,8560** | **20,97 s** |
| 2 | **GloVe** | **Random Forest** | **69,67%** | **0,6999** | **0,8479** | **1,97 s** |
| 3 | Word2Vec | XGBoost | 63,93% | 0,6413 | 0,7989 | 3,77 s |
| 4 | Word2Vec | Random Forest | 63,93% | 0,6389 | 0,8040 | 0,91 s |
| 5 | Word2Vec | Decision Tree | 63,11% | 0,6084 | 0,7839 | 0,04 s |
| 6 | GloVe | Decision Tree | 54,10% | 0,5405 | 0,6607 | 1,84 s |
| 7 | FastText | Random Forest | 53,28% | 0,5362 | 0,6864 | 0,89 s |
| 8 | FastText | Decision Tree | 50,82% | 0,5143 | 0,6486 | 0,25 s |
| 9 | FastText | XGBoost | 49,18% | 0,4924 | 0,6954 | 4,27 s |

![Perbandingan Performa Model](../results/figures/evaluation/grouped_bar_f1.png)

*Gambar 1. Perbandingan weighted F1 antar kombinasi fitur dan model pada dataset seimbang.*

Dari tabel di atas, pola utamanya sangat jelas. Kombinasi berbasis `GloVe` berada di dua posisi teratas, sementara kombinasi berbasis `FastText` justru mengisi tiga posisi terbawah. Ini berarti kualitas representasi fitur sangat menentukan hasil akhir, bahkan sebelum kita berbicara tentang model klasifikasi apa yang dipakai.

### 2. Pembacaan Hasil per Keluarga Model

**Pertama, XGBoost memberikan hasil terbaik ketika dipasangkan dengan fitur yang kaya secara semantik.** Pada extractor `GloVe`, XGBoost mencapai macro F1 tertinggi, sekaligus ROC-AUC tertinggi. Ini menunjukkan bahwa boosting mampu memanfaatkan kualitas embedding pretrained secara lebih agresif dibanding model lain.

**Kedua, Random Forest tampil sebagai opsi yang lebih efisien namun tetap kuat.** Performa `GloVe + Random Forest` hanya sedikit di bawah `GloVe + XGBoost`, tetapi waktu latihnya jauh lebih singkat. Secara praktis, konfigurasi ini layak dipertimbangkan bila kebutuhan komputasi menjadi kendala.

**Ketiga, Decision Tree konsisten lebih lemah.** Ketika hanya satu pohon keputusan yang dipakai, model menjadi lebih rentan terhadap split lokal yang tidak stabil, terutama pada dataset yang relatif kecil. Hasil `GloVe + Decision Tree` memperlihatkan hal tersebut dengan sangat jelas: representasi fiturnya bagus, tetapi kapasitas generalisasi modelnya tidak cukup kuat.

### 3. Pembacaan Hasil per Feature Extractor

**GloVe-style embeddings** unggul karena tidak belajar dari nol pada korpus kecil. Ia membawa pengetahuan semantik dari vector pretrained Bahasa Indonesia, sehingga kata-kata yang maknanya berdekatan masih bisa diproyeksikan dalam ruang vektor yang lebih informatif. Dalam domain review aplikasi, hal ini sangat berguna karena banyak kata keluhan muncul dalam bentuk yang bervariasi, misalnya *error*, *eror*, *tidak bisa*, *gagal*, *susah*, atau *lambat*.

**Word2Vec** berada di tengah. Hasilnya masih kompetitif, khususnya ketika dipasangkan dengan XGBoost dan Random Forest, tetapi tetap tertinggal dari GloVe. Salah satu penjelasan yang paling masuk akal adalah keterbatasan korpus latih. Pada eksperimen awal di notebook training, Word2Vec yang dilatih dari nol menunjukkan *out-of-vocabulary rate* yang tinggi. Artinya, banyak token yang tidak punya representasi kuat atau tidak cukup sering muncul untuk membentuk vektor yang stabil.

**FastText** justru tampil paling lemah pada tabel final. Secara teori, FastText biasanya membantu ketika bentuk kata sangat beragam karena ia memanfaatkan informasi subword. Namun pada repo ini, versi FastText yang dipakai pada eksperimen final dilatih dari nol pada korpus yang jauh lebih kecil dibanding pretrained Indonesian vectors yang dipakai oleh extractor `GloVe`. Akibatnya, keunggulan teoretis subword tidak cukup untuk menutup keterbatasan data latih.

## F. Mengapa Hasilnya Bisa Seperti Ini

Bagian ini adalah inti interpretasi eksperimen. Angka performa yang baik tidak terjadi secara kebetulan. Ia muncul karena interaksi antara ukuran data, jenis fitur, dan karakter model yang dipakai.

### 1. Dataset Supervised Relatif Kecil

Dataset klasifikasi final hanya memuat 609 review. Ukuran ini cukup untuk eksperimen pembanding yang rapi, tetapi belum cukup besar untuk membuat embedding yang dilatih dari nol selalu stabil. Karena itu, model yang bertumpu pada representasi pretrained memiliki keuntungan alami. GloVe-style embeddings menang bukan semata karena nama algoritmanya lebih terkenal, melainkan karena ia membawa informasi semantik yang sudah matang dari luar dataset eksperimen kecil ini.

### 2. XGBoost Cocok untuk Boundary Kelas yang Tidak Linear

Review pengguna tidak memiliki pola batas kelas yang sederhana. Satu kalimat bisa berisi pujian untuk satu fitur sekaligus keluhan keras untuk fitur lain. Dalam situasi seperti ini, boundary antar kelas sering kali tidak linear. XGBoost unggul karena ia membangun banyak pohon secara bertahap untuk memperbaiki kesalahan pohon-pohon sebelumnya. Ketika dipadukan dengan embedding yang sudah informatif, model ini memperoleh fondasi fitur yang baik sekaligus mekanisme klasifikasi yang fleksibel.

### 3. Random Forest Lebih Stabil, Tetapi Sedikit Kurang Tajam

Random Forest memanfaatkan banyak pohon yang dilatih secara paralel. Pendekatan ini membuatnya stabil dan cepat, tetapi tidak selalu setajam boosting ketika harus membedakan kasus-kasus yang lebih halus. Itu sebabnya `GloVe + Random Forest` masih sangat baik, tetapi tetap berada sedikit di bawah `GloVe + XGBoost`.

### 4. FastText dan Word2Vec Versi Scratch Terbatas oleh Korpus

Word2Vec dan FastText pada eksperimen final tidak mendapatkan keuntungan sebesar GloVe karena keduanya bertumpu lebih besar pada korpus proyek yang jauh lebih kecil. Untuk domain review aplikasi, variasi istilah, typo, dan frasa informal sangat tinggi. Jika ukuran korpus latih terbatas, embedding scratch mudah kehilangan banyak sinyal penting. Inilah alasan kenapa performa Word2Vec masih cukup baik tetapi belum menyaingi GloVe, dan kenapa FastText justru turun cukup jauh di bawah keduanya.

### 5. Waktu Latih Tinggi pada GloVe + XGBoost Adalah Trade-off yang Wajar

Konfigurasi terbaik bukan yang paling cepat. `GloVe + XGBoost` memerlukan waktu latih paling besar di tabel final karena dua hal: dimensi fitur lebih tinggi dan boosting menggunakan banyak estimator. Namun trade-off ini masih dapat diterima, karena peningkatan kualitas klasifikasinya paling konsisten dibandingkan kombinasi lain.

## G. Analisis Domain: Topik Keluhan, Waktu, dan Versi Aplikasi

### 1. Topic Modeling pada Review Negatif

Notebook analisis menunjukkan bahwa sampel review negatif yang dipakai untuk LDA berjumlah 203 review, yaitu seluruh kelas negatif pada `balanced_reviews.csv`. Dari pengujian coherence score, jumlah topik terbaik adalah **3 topik**, bukan 5 topik. Kata-kata kunci yang muncul untuk masing-masing topik mengarah pada tiga kelompok masalah berikut.

| Topik | Kata Kunci Dominan | Interpretasi |
| :--- | :--- | :--- |
| 1 | `tidak`, `login`, `aplikasi`, `email`, `kode`, `nomor` | Friksi autentikasi, login, verifikasi, dan identitas akun. |
| 2 | `pajak`, `lapor`, `sulit`, `susah`, `bayar`, `eror` | Hambatan proses inti aplikasi, terutama pelaporan dan pembayaran. |
| 3 | `aplikasi`, `daftar`, `npwp`, `bikin`, `terus` | Kendala registrasi, onboarding, dan pendaftaran NPWP atau akun baru. |

![LDA Coherence](../results/figures/interpretation/lda_coherence.png)

*Gambar 2. Coherence score LDA yang menunjukkan jumlah topik optimal berada pada tiga topik.*

Interpretasi ini penting karena ia menunjukkan bahwa keluhan pengguna tidak tersebar acak. Keluhan berkumpul di area yang sangat dekat dengan *core task* aplikasi. Jika pengguna gagal login, gagal verifikasi, sulit mendaftar, atau terhambat saat lapor dan bayar, maka aplikasi dianggap gagal justru pada fungsi yang paling vital.

### 2. Analisis Temporal

Notebook 05 menghitung analisis temporal dari dataset seimbang yang sama. Karena itu, pembacaan temporal di bawah ini harus dilihat sebagai **indikasi pola pada sampel analitis**, bukan sebagai potret populasi asli 8.099 review.

Ringkasan bulan-bulan terakhir menunjukkan pola berikut:

1. Januari 2026 memiliki proporsi review negatif tertinggi yang cukup menonjol, yaitu 85,71% pada sampel bulan tersebut.
2. Desember 2025 juga menunjukkan tekanan negatif yang tinggi, yaitu 66,67%.
3. Maret 2026 masih berada di level negatif yang cukup tinggi, yaitu 57,14%.
4. Februari 2026 dan April 2026 terlihat lebih terkendali dibanding periode puncak tadi.

![Analisis Temporal](../results/figures/interpretation/temporal_analysis.png)

*Gambar 3. Tren rata-rata rating dan proporsi review negatif per bulan pada sampel seimbang yang dianalisis notebook.*

Pola ini masuk akal jika dibaca bersama hasil LDA. Ketika friksi autentikasi, pendaftaran, atau alur pelaporan belum stabil, lonjakan sentimen negatif pada bulan tertentu dapat muncul bersamaan dengan momen penggunaan yang tinggi atau rilis yang kurang mulus.

### 3. Analisis per App Version

Notebook juga menunjukkan distribusi sentimen pada versi aplikasi yang memiliki lebih dari 10 review pada sampel analisis. Tiga temuan yang paling penting adalah sebagai berikut:

1. Versi `2.0.6` dan `3.0.6` menunjukkan proporsi sentimen negatif di atas 50%, sehingga layak dicurigai sebagai versi dengan friksi pengguna yang cukup tinggi.
2. Versi `2.0.3` juga menunjukkan tekanan negatif yang besar, yaitu 48,21%, dan hanya 17,86% review positif.
3. Versi `1.0` dan `1.3.0` terlihat lebih positif dibanding beberapa versi sesudahnya.

![Sentimen per Version](../results/figures/interpretation/sentiment_by_version.png)

*Gambar 4. Distribusi sentimen per versi aplikasi pada sampel yang lolos ambang minimal jumlah review.*

Temuan ini tidak otomatis berarti versi tertentu pasti buruk secara absolut. Namun ia cukup kuat untuk dijadikan alarm analitis: ketika proporsi negatif naik tajam pada suatu versi, ada kemungkinan perubahan fitur, perubahan alur, atau regresi teknis pada rilis tersebut memang dirasakan langsung oleh pengguna.

## H. Keterbatasan yang Harus Dicatat

Laporan yang baik tidak hanya menunjukkan hasil terbaik, tetapi juga menjelaskan batas validitas hasil tersebut. Pada repo ini, setidaknya ada empat keterbatasan yang harus dicatat secara terbuka.

1. **Analisis klasifikasi final menggunakan dataset seimbang berukuran 609 review**, bukan seluruh 8.099 review. Ini baik untuk evaluasi antarkelas, tetapi tidak identik dengan distribusi asli populasi review.
2. **Analisis temporal, topik, dan versi di notebook 05 juga berjalan di atas dataset seimbang**, sehingga temuan domainnya bersifat indikatif dan tidak boleh dibaca sebagai statistik populasi penuh tanpa verifikasi tambahan.
3. **Nama extractor `GloVe` di repo merujuk pada implementasi GloVe-style dengan pretrained Indonesian vectors**, bukan murni pretrained GloVe Bahasa Indonesia. Dari sisi pelaporan, hal ini tetap sah, tetapi harus dijelaskan agar tidak menimbulkan salah tafsir teknis.
4. **Repo masih menyimpan artefak hasil legacy dan artefak hasil final secara berdampingan.** Jika pembaca tidak hati-hati, ia bisa mengambil tabel yang salah sebagai sumber kesimpulan utama.

## I. Kesimpulan

Berdasarkan seluruh eksperimen yang berhasil dijalankan dan artefak hasil yang tersedia di repo, kombinasi **GloVe-style embeddings + XGBoost** merupakan konfigurasi terbaik untuk klasifikasi sentimen review aplikasi M-Pajak pada dataset seimbang. Keunggulan utamanya terletak pada perpaduan antara representasi semantik pretrained yang kaya dan model boosting yang mampu menangkap boundary kelas secara lebih fleksibel. Kombinasi **GloVe + Random Forest** berada sangat dekat di belakangnya dan layak dipertimbangkan sebagai opsi yang lebih hemat komputasi.

Di sisi lain, hasil interpretasi domain memperlihatkan bahwa sumber ketidakpuasan pengguna paling banyak berkisar pada tiga area: autentikasi atau verifikasi akun, kesulitan proses inti seperti lapor dan bayar pajak, serta friksi registrasi atau pendaftaran NPWP. Dengan demikian, perbaikan kualitas aplikasi seharusnya tidak hanya difokuskan pada stabilitas teknis umum, tetapi juga pada alur inti yang langsung menentukan berhasil atau tidaknya pengguna menyelesaikan tugas perpajakan mereka.

Secara metodologis, pelajaran terbesar dari repo ini adalah pentingnya menjaga konsistensi antara artefak eksperimen, notebook analisis, dan laporan akhir. Ketika satu file hasil legacy tetap tersimpan berdampingan dengan hasil final, laporan bisa bergeser tanpa disadari. Karena itu, laporan revisi ini menetapkan `balanced_comparison_table.csv` sebagai sumber performa final, sekaligus menjelaskan konteks historis artefak lain agar pembacaan hasil menjadi lebih jujur dan lebih presisi.

## J. Referensi

1. Pennington, J., Socher, R., and Manning, C. 2014. GloVe: Global Vectors for Word Representation.
2. Chen, T., and Guestrin, C. 2016. XGBoost: A Scalable Tree Boosting System.
3. Blei, D. M., Ng, A. Y., and Jordan, M. I. 2003. Latent Dirichlet Allocation.
4. Artefak internal repo: `notebooks/04_model_training.ipynb`, `notebooks/05_comparison_analysis.ipynb`, `results/balanced_comparison_table.csv`, `results/comparison_table.csv`, `data/processed/balanced_reviews.csv`, dan `data/processed/reviews_prepared.csv`.

*** Add File: c:\Users\Fauzan\Downloads\05_College\Text Mining\UAS\final-project-txmg1\reports\analisis-perbedaan-laporan.md
# ANALISIS PERBEDAAN ANTARA LAPORAN SEBELUMNYA DAN LAPORAN REVISI

## A. Tujuan Dokumen

Dokumen ini disusun untuk membandingkan isi `laporan.md` versi sebelumnya dengan laporan revisi yang baru disusun berdasarkan artefak eksperimen final di repo. Fokus utamanya bukan untuk mencari kesalahan penulisan lama, melainkan untuk menjelaskan mengapa isi laporan bisa berbeda setelah seluruh notebook, file hasil, dan jejak eksperimen dibaca ulang secara menyeluruh.

Kesimpulan singkatnya adalah sebagai berikut: **perbedaan laporan tidak terutama disebabkan oleh perubahan teori, tetapi oleh perbedaan fase artefak yang dijadikan dasar penulisan.** Repo ini menyimpan hasil eksperimen legacy dan hasil eksperimen final secara berdampingan. Jika keduanya tidak dipisahkan dengan tegas, narasi hasil akan mudah bergeser.

## B. Perbandingan Inti

| Aspek | Laporan Sebelumnya | Laporan Revisi | Mengapa Berbeda |
| :--- | :--- | :--- | :--- |
| Sumber tabel performa | Langsung menonjolkan `GloVe + XGBoost` sebagai hasil utama tanpa menjelaskan evolusi artefak | Menetapkan `results/balanced_comparison_table.csv` sebagai sumber performa final dan `results/comparison_table.csv` sebagai artefak legacy | Repo memuat dua tabel hasil dari fase eksperimen yang berbeda |
| Status GloVe pada fase awal | Tidak dijelaskan | Dijelaskan bahwa eksperimen awal gagal memuat GloVe karena path `data/embeddings/cc.id.300.vec` tidak ditemukan | `notebooks/04_model_training.ipynb` menyimpan jejak error tersebut secara eksplisit |
| Jumlah topik LDA | Menyebut 5 topik dan menempatkan OTP sebagai isu paling dominan | Menetapkan 3 topik sebagai hasil optimal, dengan fokus pada login/verifikasi, lapor-bayar, dan registrasi/NPWP | Output notebook 05 menunjukkan `best_n = 3`, bukan 5 |
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
