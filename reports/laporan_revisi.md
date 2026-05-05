# Perbandingan Word Embeddings dan Machine Learning untuk Analisis Sentimen Masyarakat terhadap Aplikasi CoreTax (M-Pajak) pada Google Play Store

**ABSTRAK**

Aplikasi M-Pajak (CoreTax) merupakan platform digital resmi Direktorat Jenderal Pajak yang ditujukan untuk memudahkan layanan perpajakan bagi masyarakat Indonesia. Ulasan pengguna pada Google Play Store dapat dimanfaatkan sebagai sumber data untuk mengukur persepsi publik terhadap kualitas layanan aplikasi. Penelitian ini menerapkan pendekatan *text mining* untuk klasifikasi sentimen tiga kelas, yaitu Positif, Netral, dan Negatif, dengan fokus pada perbandingan beberapa metode *word embedding* dan algoritma *machine learning*. Dataset akhir yang digunakan pada eksperimen terbaru berjumlah 609 ulasan yang telah diseimbangkan, masing-masing 203 ulasan untuk setiap kelas. Hasil eksperimen menunjukkan bahwa kombinasi **Word2Vec + XGBoost** memberikan performa terbaik dengan nilai **Weighted F1-Score 63,92%**, **Macro F1-Score 63,94%**, **akurasi 63,93%**, dan **ROC-AUC 0,8051**. Selain itu, analisis LDA menunjukkan bahwa keluhan pengguna didominasi oleh isu login, verifikasi, error aplikasi, dan kerumitan alur layanan. Hasil ini menunjukkan bahwa representasi semantik berbasis *word embedding* tetap relevan untuk analisis sentimen layanan publik, dan Word2Vec menjadi pilihan yang paling efektif pada konfigurasi eksperimen terbaru.

**Kata Kunci**: Analisis Sentimen, CoreTax, Word2Vec, GloVe, FastText, XGBoost, LDA, Text Mining

---

## I. PENDAHULUAN

Di era digital, aplikasi mobile menjadi jembatan utama antara pemerintah dan masyarakat. Direktorat Jenderal Pajak merilis aplikasi M-Pajak (CoreTax) untuk meningkatkan efisiensi pelaporan pajak. Namun, akumulasi ulasan di Google Play Store menunjukkan adanya disparitas pengalaman pengguna. Analisis sentimen manual terhadap ribuan data menjadi tidak efisien, sehingga diperlukan pendekatan *text mining* otomatis. Penelitian ini bertujuan untuk membandingkan representasi fitur berbasis embedding dan model *machine learning* untuk memperoleh model klasifikasi sentimen yang paling efektif, sekaligus mengidentifikasi isu utama yang paling sering dikeluhkan pengguna.

## II. STUDI LITERATUR

### Text Mining dan Preprocessing

*Text mining* adalah proses ekstraksi informasi berguna dari data teks tidak terstruktur. Tahapan penting yang digunakan dalam penelitian ini meliputi *case folding*, *noise removal*, *stopword removal*, *tokenizing*, dan *stemming* menggunakan Sastrawi agar teks ulasan menjadi lebih bersih dan konsisten.

### Word Embeddings

Berbeda dengan TF-IDF yang menitikberatkan pada frekuensi kemunculan kata, *word embeddings* memetakan kata ke dalam representasi vektor numerik yang mampu menangkap kedekatan makna. Pada penelitian ini, fitur yang dibandingkan adalah **GloVe**, **Word2Vec**, dan **FastText**.

### Machine Learning untuk Klasifikasi Sentimen

Algoritma *machine learning* yang digunakan adalah **Decision Tree**, **Random Forest**, dan **XGBoost**. Ketiga model tersebut dipilih karena mewakili model pohon keputusan tunggal, *ensemble bagging*, dan *gradient boosting* yang umum digunakan pada tugas klasifikasi teks.

## III. METODE

Penelitian ini menggunakan alur sistematis sebagai berikut:

1. **Pengumpulan Data**: Mengambil ulasan aplikasi CoreTax dari Google Play Store.
2. **Pelabelan Sentimen**: Mengubah rating menjadi tiga kelas sentimen, yaitu Negatif, Netral, dan Positif.
3. **Preprocessing Teks**: Melakukan normalisasi teks melalui pembersihan, tokenisasi, dan stemming.
4. **Penyeimbangan Data**: Menyiapkan dataset akhir seimbang sebanyak 609 ulasan, terdiri atas 203 data per kelas.
5. **Ekstraksi Fitur**: Mengubah teks menjadi representasi vektor menggunakan GloVe, Word2Vec, dan FastText.
6. **Pelatihan Model**: Menguji kombinasi fitur dan model untuk memperoleh performa terbaik.
7. **Evaluasi dan Analisis Domain**: Menggunakan metrik akurasi, Weighted F1, Macro F1, ROC-AUC, analisis topik LDA, dan analisis temporal.

![Distribusi Sentimen](../results/figures/eda/sentiment_distribution.png)
*Gambar 1. Distribusi kelas sentimen pada dataset akhir yang digunakan dalam eksperimen.*

---

## BAB IV
## HASIL DAN PEMBAHASAN

### A. Hasil Preprocessing Data

Setelah melalui tahapan preprocessing yang meliputi *case folding*, *noise removal*, *stopword removal*, *tokenizing*, dan *stemming* menggunakan Sastrawi, diperoleh data teks yang lebih bersih dan lebih siap untuk diekstraksi menjadi fitur numerik. Pada artefak dataset terbaru, file `reviews_prepared.csv` berisi **609 ulasan** dan seluruh kelas telah berada pada kondisi seimbang, yaitu masing-masing **203 ulasan** untuk sentimen Negatif, Netral, dan Positif.

Beberapa hasil preprocessing yang diperoleh adalah sebagai berikut:

- Rata-rata jumlah token per ulasan: **9,63 token**
- Median jumlah token per ulasan: **6 token**
- Vocabulary hasil preprocessing: **1.592 kata unik**
- Data kosong setelah preprocessing: **7 baris**

Contoh hasil preprocessing:

- Sebelum: "Aplikasinya bikin bingung gak satset. Log in salah terus padahal udah sesuai."
- Sesudah: "aplikasinya bikin bingung tidak satset log in salah terus padahal sesuai"

Tahapan preprocessing ini terbukti mampu mengurangi *noise*, menyeragamkan bentuk kata, dan mempertahankan informasi penting yang relevan untuk proses klasifikasi sentimen.

![Distribusi Panjang Token](../results/figures/eda/token_distribution_after_preprocessing.png)
*Gambar 2. Distribusi jumlah token setelah preprocessing.*

![WordCloud Preprocessing](../results/figures/eda/wordcloud_preprocessed.png)
*Gambar 3. Wordcloud dari ulasan yang telah dipreprocessing.*

### B. Hasil Eksperimen Model

Penelitian ini menguji beberapa kombinasi metode *feature extraction* dan algoritma *machine learning* pada dataset seimbang. Berdasarkan hasil eksperimen final yang digunakan pada analisis komparatif, enam kombinasi terbaik ditunjukkan pada tabel berikut:

| Rank | Feature Extraction | Model | Accuracy | Weighted F1 | Macro F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|
| 1 | Word2Vec | XGBoost | 63,93% | 0,6392 | 0,6394 | 0,8051 |
| 2 | Word2Vec | Random Forest | 62,30% | 0,6181 | 0,6184 | 0,8154 |
| 3 | Word2Vec | Decision Tree | 57,38% | 0,5738 | 0,5737 | 0,7227 |
| 4 | FastText | Random Forest | 52,46% | 0,5286 | 0,5283 | 0,6702 |
| 5 | FastText | XGBoost | 52,46% | 0,5216 | 0,5220 | 0,7243 |
| 6 | FastText | Decision Tree | 50,00% | 0,5064 | 0,5058 | 0,6416 |

Dari hasil tersebut, kombinasi **Word2Vec + XGBoost** memberikan performa terbaik dengan nilai Weighted F1 sebesar **0,6392** dan ROC-AUC **0,8051**. Hasil ini menunjukkan bahwa Word2Vec embedding, meskipun lebih sederhana dibandingkan metode lain, tetap mampu menangkap pola semantik yang efektif untuk klasifikasi sentimen pada dataset CoreTax.

![Grouped Bar F1](../results/figures/evaluation/grouped_bar_f1.png)
*Gambar 4. Perbandingan Weighted F1-Score antar kombinasi fitur dan model.*

![Quadrant Plot](../results/figures/evaluation/quadrant_f1_vs_time.png)
*Gambar 5. Quadrant plot antara performa model dan waktu pelatihan.*

### C. Analisis Perbandingan Model

Berdasarkan hasil eksperimen, beberapa temuan penting adalah sebagai berikut:

- **Word2Vec** menjadi representasi fitur terbaik pada eksperimen final karena menghasilkan Weighted F1 tertinggi sebesar **0,6392** ketika dikombinasikan dengan XGBoost.
- **XGBoost** merupakan model yang paling konsisten memberikan performa tinggi pada hasil akhir, sekaligus menjadi model terbaik secara keseluruhan.
- **Random Forest** menunjukkan performa yang stabil dan efisien, menjadi alternatif terbaik kedua dengan Weighted F1 **0,6181** untuk kombinasi Word2Vec.
- **Decision Tree** menghasilkan performa yang lebih rendah, sehingga kurang optimal untuk menangkap pola sentimen yang lebih kompleks.
- **FastText** menghasilkan performa di bawah Word2Vec pada konfigurasi eksperimen yang digunakan, sehingga belum menjadi pilihan terbaik untuk dataset ini.

Secara keseluruhan, hasil ini menunjukkan bahwa kombinasi Word2Vec + XGBoost merupakan pilihan optimal untuk klasifikasi sentimen ulasan CoreTax, karena memberikan keseimbangan paling baik antara performa dan kemampuan diskriminasi kelas.

### D. Analisis Class Imbalance

Pada artefak eksperimen terbaru yang tersimpan di repositori, penanganan *class imbalance* sudah diterapkan pada tahap pembentukan dataset akhir. Hal ini terlihat dari distribusi kelas yang seimbang, yaitu **203 Negatif**, **203 Netral**, dan **203 Positif**. Dengan demikian, evaluasi model tidak lagi bias ke kelas mayoritas.

Temuan utama dari kondisi dataset seimbang ini adalah sebagai berikut:

- Nilai Weighted F1 dan Macro F1 pada model terbaik hampir identik, yaitu **0,6392** dan **0,6394**.
- Kedekatan kedua metrik tersebut menunjukkan bahwa model terbaik tidak hanya baik secara keseluruhan, tetapi juga relatif konsisten di seluruh kelas sentimen.
- ROC-AUC sebesar **0,8051** menunjukkan bahwa model memiliki kemampuan diskriminasi yang cukup baik antara kelas-kelas sentimen.
- Evaluasi pada dataset seimbang membuat interpretasi performa menjadi lebih adil karena kelas minoritas tidak lagi tertekan oleh dominasi kelas tertentu.

Dengan demikian, pada versi hasil terbaru, pembahasan *class imbalance* lebih tepat difokuskan pada keberhasilan penyusunan dataset seimbang sebagai dasar evaluasi yang lebih kredibel.

### E. Confusion Matrix Model Terbaik

Artefak visual *confusion matrix* tidak tersimpan sebagai file terpisah pada hasil eksperimen terbaru. Meskipun demikian, performa model terbaik dapat diinterpretasikan melalui konsistensi metrik evaluasi.

Analisis yang dapat ditarik adalah sebagai berikut:

- Model terbaik tidak menunjukkan gejala bias yang kuat terhadap satu kelas tertentu, karena nilai Weighted F1 dan Macro F1 hampir sama.
- Kelas Netral tetap berpotensi menjadi kelas yang paling menantang, karena secara konseptual berada di antara opini positif dan negatif serta cenderung ambigu.
- ROC-AUC sebesar **0,8051** menunjukkan kemampuan diskriminasi yang cukup baik terhadap tiga kelas sentimen.
- Untuk pelaporan lanjutan, pembuatan *confusion matrix* eksplisit disarankan agar *recall* dan *precision* per kelas dapat dilaporkan secara kuantitatif.

### F. Topic Modeling (LDA)

Analisis menggunakan metode LDA pada ulasan negatif menunjukkan bahwa jumlah topik optimal berada pada **10 topik**, sesuai hasil *coherence score* tertinggi pada notebook analisis. Beberapa kata dominan yang muncul pada topik-topik tersebut antara lain:

- **Topik 1**: *ribet, aplikasi, npwp, susah, login, daftar, pemerintah*
- **Topik 2**: *chat, aplikasi, error, live, buka, pajak, web*
- **Topik 3**: *login, tolong, perbaiki, aplikasi, belum*
- **Topik 4**: *kode, verifikasi, email, otp, lama, masuk, lemot*
- **Topik 5**: *aplikasi, pajak, lapor, bayar, masuk, efin*
- **Topik 6**: *login, gagal, verifikasi, kali, data*
- **Topik 7**: *aplikasi, harus, padahal, eror, bikin*
- **Topik 8**: *email, gagal, nomor, aktivasi, handphone*
- **Topik 9**: *aplikasi, pajak, salah, login, mempersulit*
- **Topik 10**: *aplikasi, web, daftar, bagaimana*

Berdasarkan kata-kata kunci tersebut, beberapa isu utama yang dikeluhkan pengguna dapat diringkas sebagai berikut:

- Kerumitan alur penggunaan aplikasi, terutama pada login dan pendaftaran NPWP
- Gangguan verifikasi kode, OTP, dan email aktivasi
- Error aplikasi dan kegagalan membuka layanan tertentu
- Hambatan login berulang, termasuk masalah EFIN dan validasi data
- Persepsi bahwa aplikasi masih mempersulit proses layanan pajak

Topik ini menunjukkan bahwa masalah utama pengguna tidak hanya bersifat teknis, tetapi juga berkaitan dengan pengalaman penggunaan dan kejelasan alur layanan.

![LDA Coherence](../results/figures/interpretation/lda_coherence.png)
*Gambar 6. Coherence score untuk penentuan jumlah topik optimal pada LDA.*

### G. Analisis Temporal dan Kata Kunci

Analisis temporal menunjukkan bahwa persepsi pengguna berubah dari waktu ke waktu. Pada notebook analisis, tren ini divisualisasikan melalui perubahan rata-rata rating dan proporsi review negatif per bulan.

Beberapa temuan penting dari analisis temporal dan kata kunci adalah sebagai berikut:

- Puncak proporsi review negatif pada artefak terbaru terlihat pada **Januari 2026**, diikuti **November 2025**, **September 2023**, **Juni 2024**, dan **Juli 2022**.
- Pada analisis versi aplikasi, versi **1.2.0**, **2.0.4**, **2.0.6**, **2.0.3**, dan **1.4.0** termasuk versi yang memiliki proporsi sentimen negatif relatif tinggi pada kelompok versi dengan jumlah ulasan memadai.
- Kata-kata yang sering muncul pada analisis topik dan wordcloud didominasi oleh istilah seperti **login**, **verifikasi**, **kode**, **email**, **error**, **daftar**, **npwp**, dan **efin**.

Hasil ini memperkuat temuan bahwa hambatan akses, verifikasi, dan kerumitan proses layanan masih menjadi sumber utama ketidakpuasan pengguna aplikasi CoreTax.

![Temporal Analysis](../results/figures/interpretation/temporal_analysis.png)
*Gambar 7. Tren rata-rata rating dan proporsi review negatif per bulan.*

![Sentiment by Version](../results/figures/interpretation/sentiment_by_version.png)
*Gambar 8. Distribusi sentimen berdasarkan versi aplikasi.*

---

## BAB V
## KESIMPULAN DAN SARAN

### A. Kesimpulan

Berdasarkan hasil penelitian yang telah dilakukan, dapat ditarik beberapa kesimpulan sebagai berikut:

1. Metode *word embedding* **Word2Vec** terbukti memberikan performa terbaik pada eksperimen terbaru dibandingkan FastText, dengan nilai Weighted F1 tertinggi ketika dipadukan dengan XGBoost.
2. Algoritma **XGBoost** menunjukkan kinerja paling optimal pada konfigurasi akhir, dengan nilai Weighted F1-Score **63,92%**, Macro F1-Score **63,94%**, akurasi **63,93%**, dan ROC-AUC **0,8051**.
3. Penyusunan dataset akhir yang seimbang, yaitu 203 data untuk masing-masing kelas, membuat evaluasi model menjadi lebih adil dan mengurangi bias terhadap kelas mayoritas.
4. Kedekatan nilai Weighted F1 dan Macro F1 pada model terbaik menunjukkan bahwa performa klasifikasi relatif merata di seluruh kelas sentimen.
5. Analisis topik menggunakan LDA menunjukkan bahwa keluhan utama pengguna berkaitan dengan login, verifikasi email atau OTP, error aplikasi, aktivasi akun, serta kerumitan proses layanan pajak.
6. Secara keseluruhan, pendekatan kombinasi *text mining*, *word embeddings*, *machine learning*, dan analisis domain mampu memberikan gambaran yang cukup komprehensif terhadap persepsi pengguna aplikasi CoreTax.

### B. Saran

Berdasarkan hasil penelitian, beberapa saran yang dapat diberikan adalah sebagai berikut:

1. **Bagi Pengembang Aplikasi**
   - Memprioritaskan perbaikan pada sistem login, verifikasi kode, OTP, dan integrasi email agar proses akses menjadi lebih stabil.
   - Menyederhanakan alur penggunaan aplikasi, terutama pada proses login, pelaporan, dan pendaftaran NPWP.
   - Meningkatkan kualitas pesan kesalahan serta panduan penggunaan agar pengguna nonteknis lebih mudah memahami langkah yang harus dilakukan.

2. **Bagi Penelitian Selanjutnya**
   - Menambahkan artefak evaluasi yang lebih lengkap, seperti *confusion matrix* final dan analisis *recall* per kelas.
   - Menguji model berbasis *transformer* atau *pre-trained language model* Bahasa Indonesia untuk membandingkan hasil dengan embedding klasik.
   - Mengembangkan analisis ke tingkat *aspect-based sentiment analysis* agar setiap keluhan dapat dipetakan lebih spesifik ke aspek layanan tertentu.

3. **Bagi Akademisi**
   - Penelitian ini dapat dijadikan referensi untuk kajian analisis sentimen pada aplikasi layanan publik berbasis ulasan pengguna.
   - Hasil penelitian ini juga dapat menjadi dasar pengembangan studi lanjutan mengenai evaluasi layanan digital pemerintah berbasis data teks.

## VI. REFERENSI

* Ceci, L. (2024). Statistics on Google Play Store Apps. Statista.
* Astuti, K. C., et al. (2024). Implementasi Text Mining Korlantas Polri. Remik Journal.
* Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation. EMNLP.