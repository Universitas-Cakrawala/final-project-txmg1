# Implementasi GloVe Embeddings dan XGBoost untuk Analisis Sentimen Masyarakat terhadap Aplikasi CoreTax (M-Pajak) pada Google Play Store

**ABSTRAK**
Aplikasi M-Pajak (CoreTax) merupakan platform digital resmi Direktorat Jenderal Pajak untuk memudahkan layanan perpajakan bagi masyarakat Indonesia. Banyaknya ulasan pengguna di Google Play Store menjadi indikator kepuasan layanan yang perlu dianalisis. Penelitian ini mengimplementasikan teknik *text mining* untuk klasifikasi sentimen (Positif, Netral, Negatif) dengan mengatasi masalah ketimpangan data (*class imbalance*) menggunakan teknik *Random Undersampling*. Fitur ekstraksi yang digunakan adalah **GloVe Embeddings** yang mampu menangkap relasi semantik antar kata secara global. Pengujian dilakukan terhadap tiga model ML, di mana algoritma **XGBoost** menunjukkan performa terbaik pada dataset seimbang dengan nilai **Macro F1-Score sebesar 71,4%** dan akurasi 71,3%. Selain itu, analisis topik menggunakan **LDA (Latent Dirichlet Allocation)** mengungkap lima klaster keluhan utama, dengan masalah verifikasi OTP sebagai kendala sistemik yang paling sering dilaporkan.

**Kata Kunci**: Analisis Sentimen, CoreTax, GloVe Embeddings, XGBoost, LDA, Text Mining

---

## I. PENDAHULUAN
Di era digital, aplikasi mobile menjadi jembatan utama antara pemerintah dan masyarakat. Direktorat Jenderal Pajak merilis aplikasi M-Pajak (CoreTax) untuk meningkatkan efisiensi pelaporan pajak. Namun, akumulasi ulasan di Google Play Store menunjukkan adanya disparitas pengalaman pengguna. Analisis sentimen manual terhadap ribuan data menjadi tidak efisien, sehingga diperlukan pendekatan *text mining* otomatis. Penelitian ini bertujuan untuk membangun model klasifikasi yang adil dan mendalam, tidak hanya sekadar menghitung akurasi mentah, tetapi juga membedah akar permasalahan melalui *Topic Modeling*.

## II. STUDI LITERATUR
### Text Mining & Preprocessing
*Text mining* adalah proses ekstraksi informasi berguna dari data teks tak terstruktur. Tahapan krusial meliputi *Case Folding*, *Filtering* (Stopword), *Tokenizing*, dan *Stemming* menggunakan library Sastrawi.
### GloVe Embeddings
Berbeda dengan TF-IDF yang bersifat frekuensi-sentris, **GloVe (Global Vectors for Word Representation)** adalah model *unsupervised learning* yang menghasilkan representasi vektor kata berdasarkan statistik ko-okurensi global. Hal ini memungkinkan model memahami konteks semantik yang lebih kaya.
### XGBoost
*Extreme Gradient Boosting* (XGBoost) merupakan algoritma *ensemble learning* berbasis pohon keputusan yang dioptimalkan untuk kecepatan dan performa tinggi melalui regularisasi dan penanganan *missing values*.

## III. METODE
Penelitian ini menggunakan alur sistematis sebagai berikut:
1.  **Scraping Data**: Mengambil ~8.000 ulasan dari Google Play Store.
2.  **Balancing Data**: Menerapkan *undersampling* untuk mendapatkan 609 data seimbang (203 sampel per kelas).
3.  **Preprocessing**: Pembersihan teks dan normalisasi.
4.  **Feature Extraction**: Mengonversi teks menjadi vektor menggunakan GloVe.
5.  **Klasifikasi**: Melatih model Decision Tree, Random Forest, dan XGBoost dengan *Class Weighting*.
6.  **Evaluasi**: Menggunakan *Confusion Matrix* dan metrik Macro F1-Score.

![Metode Penelitian](../results/figures/eda/sentiment_distribution.png)
*Gambar 1: Alur Penyeimbangan Dataset (Undersampling).*

## IV. HASIL DAN PEMBAHASAN
### Perbandingan Performa Model
Berdasarkan pengujian pada dataset seimbang, kombinasi fitur embeddings menunjukkan performa yang lebih stabil dibandingkan metode statistik.

| Feature Extractor | Model | Accuracy | Macro F1 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **GloVe** | **XGBoost** | **71,3%** | **71,4%** | **0,856** |
| GloVe | Random Forest | 69,7% | 70,0% | 0,848 |
| Word2Vec | XGBoost | 63,9% | 0.641% | 0,799 |

![Comparison Chart](../results/figures/evaluation/grouped_bar_f1.png)
*Gambar 2: Grafik Perbandingan F1-Score.*

### Deep Analysis (LDA Topic Modeling)
Analisis mendalam terhadap ulasan negatif menggunakan LDA mengidentifikasi 5 topik utama yang menjadi sumber ketidakpuasan pengguna:
1.  **Kegagalan OTP**: Kode verifikasi tidak terkirim atau pulsa terpotong tanpa hasil.
2.  **Aktivasi EFIN**: Kendala teknis pada sinkronisasi profil.
3.  **UI/UX (Birokrasi)**: Keluhan "ribet" pada alur pelaporan.
4.  **Stabilitas (Force Close)**: Bug teknis pada aplikasi.
5.  **Pendaftaran NPWP**: Masalah pada fitur registrasi baru.

![LDA Coherence](../results/figures/interpretation/lda_coherence.png)
*Gambar 3: Grafik Coherence Score untuk penentuan jumlah topik optimal (LDA).*

## V. KESIMPULAN
Penelitian ini membuktikan bahwa penggunaan **GloVe Embeddings** dikombinasikan dengan **XGBoost** mampu menghasilkan klasifikasi sentimen yang jauh lebih presisi dan adil (Macro F1 71,4%) pada dataset yang diseimbangkan. Penurunan akurasi dari 91% (data timpang) ke 71% (data seimbang) merupakan indikator bahwa model telah berhasil mengatasi bias kelas mayoritas. Rekomendasi utama bagi pengembang adalah perbaikan mendesak pada infrastruktur pengiriman OTP untuk meningkatkan kepuasan pengguna secara signifikan.

## VI. REFERENSI
*   Ceci, L. (2024). Statistics on Google Play Store Apps. Statista.
*   Astuti, K. C., et al. (2024). Implementasi Text Mining Korlantas Polri. Remik Journal.
*   Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation. EMNLP.
