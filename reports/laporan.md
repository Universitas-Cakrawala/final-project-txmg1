# Implementasi GloVe Embeddings dan XGBoost untuk Analisis Sentimen Masyarakat terhadap Aplikasi CoreTax (M-Pajak) pada Google Play Store

**ABSTRAK**
Aplikasi M-Pajak (CoreTax) merupakan platform digital resmi Direktorat Jenderal Pajak untuk memudahkan layanan perpajakan bagi masyarakat Indonesia. Banyaknya ulasan pengguna di Google Play Store menjadi indikator kepuasan layanan yang perlu dianalisis. Penelitian ini mengimplementasikan teknik *text mining* untuk klasifikasi sentimen (Positif, Netral, Negatif) dengan mengatasi masalah ketimpangan data (*class imbalance*) menggunakan teknik *Random Undersampling*. Fitur ekstraksi yang digunakan adalah **GloVe Embeddings** yang mampu menangkap relasi semantik antar kata secara global. Pengujian dilakukan terhadap tiga model ML, di mana algoritma **XGBoost** menunjukkan performa terbaik pada dataset seimbang dengan nilai **Macro F1-Score sebesar 71,44%** dan akurasi 71,31%. Selain itu, analisis topik menggunakan **LDA** mengungkap lima klaster keluhan utama, dengan masalah verifikasi OTP sebagai kendala sistemik yang paling sering dilaporkan.

**Kata Kunci**: Analisis Sentimen, CoreTax, GloVe Embeddings, XGBoost, LDA, Text Mining

---

## I. PENDAHULUAN
Di era digital, aplikasi mobile menjadi jembatan utama antara pemerintah dan masyarakat. Direktorat Jenderal Pajak merilis aplikasi M-Pajak (CoreTax) untuk meningkatkan efisiensi pelaporan pajak. Namun, akumulasi ulasan di Google Play Store menunjukkan adanya disparitas pengalaman pengguna. Analisis sentimen manual terhadap ribuan data menjadi tidak efisien, sehingga diperlukan pendekatan *text mining* otomatis. Penelitian ini bertujuan untuk membangun model klasifikasi yang adil dan mendalam, serta membedah akar permasalahan melalui *Topic Modeling*.

## II. STUDI LITERATUR
### Text Mining & Preprocessing
*Text mining* adalah proses ekstraksi informasi berguna dari data teks tak terstruktur. Tahapan krusial meliputi *Case Folding*, *Filtering* (Stopword), *Tokenizing*, dan *Stemming* menggunakan library Sastrawi untuk membersihkan noise pada bahasa tidak baku (slang).
### GloVe Embeddings
Berbeda dengan TF-IDF yang bersifat frekuensi-sentris, **GloVe (Global Vectors for Word Representation)** adalah model yang menghasilkan representasi vektor kata berdasarkan statistik ko-okurensi global. Hal ini memungkinkan model memahami konteks semantik (seperti kedekatan kata "eror" dan "bug").
### XGBoost
*Extreme Gradient Boosting* (XGBoost) merupakan algoritma *ensemble learning* berbasis pohon keputusan yang dioptimalkan untuk kecepatan dan performa tinggi melalui regularisasi untuk mencegah *overfitting*.

## III. METODE
Penelitian ini menggunakan alur sistematis sebagai berikut:
1.  **Scraping Data**: Mengambil ~8.000 ulasan dari Google Play Store.
2.  **Balancing Data**: Menerapkan *undersampling* untuk mendapatkan 609 data seimbang (203 sampel per kelas).
3.  **Preprocessing**: Pembersihan teks, normalisasi slang, dan penghapusan kata umum.
4.  **Feature Extraction**: Mengonversi teks menjadi vektor 300-dimensi menggunakan GloVe.
5.  **Klasifikasi**: Melatih model Decision Tree, Random Forest, dan XGBoost.
6.  **Evaluasi**: Menggunakan *Confusion Matrix* dan metrik Macro F1-Score untuk menjamin keadilan kelas.

![Metode Penelitian](../results/figures/eda/sentiment_distribution.png)
*Gambar 1: Alur Penyeimbangan Dataset (Undersampling) untuk Menghindari Bias Akurasi.*

## IV. HASIL DAN PEMBAHASAN
### Perbandingan Performa Model
Berdasarkan pengujian pada dataset seimbang, kombinasi fitur embeddings menunjukkan performa yang lebih stabil dibandingkan metode statistik tradisional.

| Feature Extractor | Model | Accuracy | Macro F1 | ROC-AUC | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GloVe** | **XGBoost** | **71,31%** | **0,7144** | **0,8560** | **20.9s** |
| GloVe | Random Forest | 69,67% | 0,6999 | 0,8479 | 1.9s |
| Word2Vec | XGBoost | 63,93% | 0,6413 | 0,7989 | 3.7s |

![Comparison Chart](../results/figures/evaluation/grouped_bar_f1.png)
*Gambar 2: Grafik Perbandingan F1-Score antar Kombinasi Fitur dan Model.*

### Deep Analysis (LDA Topic Modeling)
Analisis terhadap ulasan negatif menggunakan LDA mengidentifikasi 5 topik utama yang menjadi sumber ketidakpuasan:
1.  **Kegagalan OTP**: Masalah pengiriman kode verifikasi (Kendala paling dominan).
2.  **Aktivasi EFIN**: Kesulitan teknis pada sinkronisasi profil perpajakan.
3.  **UI/UX (Birokrasi)**: Keluhan alur pelaporan yang dianggap kurang intuitif.
4.  **Stabilitas (Force Close)**: Masalah teknis pada stabilitas aplikasi.
5.  **Pendaftaran NPWP**: Kendala pada fitur pendaftaran akun baru.

![LDA Coherence](../results/figures/interpretation/lda_coherence.png)
*Gambar 3: Grafik Coherence Score untuk Penentuan Jumlah Topik Optimal pada LDA.*

### Analisis Temporal dan Kata Kunci
Data menunjukkan bahwa sentimen negatif tidak tersebar merata, namun seringkali berkumpul pada periode pembaruan aplikasi tertentu. Kata-kata seperti "OTP", "Daftar", dan "EFIN" mendominasi awan kata (*wordcloud*) keluhan pengguna.

![Temporal Analysis](../results/figures/interpretation/temporal_analysis.png)
*Gambar 4: Tren Sentimen Pengguna dari Waktu ke Waktu.*

![WordCloud](../results/figures/eda/wordcloud_preprocessed.png)
*Gambar 5: Awan Kata (WordCloud) dari Ulasan yang Telah Dipreprocessing.*

## V. KESIMPULAN
Penelitian ini membuktikan bahwa penggunaan **GloVe Embeddings** dikombinasikan dengan **XGBoost** mampu menghasilkan klasifikasi sentimen yang jauh lebih presisi (Macro F1 0,7144) pada dataset seimbang. Meskipun akurasi menurun dari 91% (pada data timpang) menjadi 71% (pada data seimbang), hasil ini lebih valid secara ilmiah karena model terbukti mampu mengenali kelas "Netral" dan "Positif" dengan tingkat keberhasilan yang sama baiknya dengan kelas "Negatif". Rekomendasi utama bagi pengembang adalah optimalisasi infrastruktur verifikasi OTP untuk menekan volume sentimen negatif secara signifikan.

## VI. REFERENSI
*   Ceci, L. (2024). Statistics on Google Play Store Apps. Statista.
*   Astuti, K. C., et al. (2024). Implementasi Text Mining Korlantas Polri. Remik Journal.
*   Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation. EMNLP.
