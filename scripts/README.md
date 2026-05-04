# 📝 Scripts Directory

Folder ini berisi utility scripts untuk project sentiment classification CoretTax.

## 📥 Download Scripts

### 1. Download FastText Indonesian Vectors (`download_cc_id_300_vec.py`)

**Script Python untuk download `cc.id.300.vec` dari HuggingFace**

```bash
# Cara 1: Jalankan dari scripts folder
cd scripts
python download_cc_id_300_vec.py

# Cara 2: Jalankan dari root project
python scripts/download_cc_id_300_vec.py
```

**Fitur:**
- ✅ Progress bar dengan kecepatan dan estimasi waktu
- ✅ Verifikasi otomatis ukuran file
- ✅ Deteksi file rusak
- ✅ Opsi skip jika file sudah ada
- ✅ User-friendly interface dalam Bahasa Indonesia

**Output:**
- 📁 File: `data/processed/cc.id.300.vec` (4.21GB)
- ⏱️ Waktu: ~7-8 menit (tergantung kecepatan internet)
- 📊 Ukuran: 4.21 GB (pre-trained Indonesian FastText embeddings)

---

### 2. Download FastText Indonesian Vectors - Bash Version (`download_cc_id_300_vec.sh`)

**Script Bash untuk download, alternatif untuk script Python**

```bash
# Cara 1: Dari scripts folder
cd scripts
bash download_cc_id_300_vec.sh
# atau
./download_cc_id_300_vec.sh

# Cara 2: Dari root project
bash scripts/download_cc_id_300_vec.sh
```

**Fitur:**
- ✅ Menggunakan curl atau wget (otomatis pilih yang tersedia)
- ✅ Progress bar
- ✅ Verifikasi file
- ✅ Cross-platform compatibility (Linux, macOS)

**Persyaratan:**
- `curl` atau `wget` (biasanya sudah terinstall)
- `bash` shell

---

## 🚀 Quick Start

### Opsi 1: Python Script (Recommended)

```bash
python scripts/download_cc_id_300_vec.py
```

Proses:
1. ✅ Otomatis membuat folder `data/processed` jika belum ada
2. ✅ Download dari HuggingFace
3. ✅ Verifikasi ukuran file
4. ✅ Siap digunakan

### Opsi 2: Bash Script

```bash
bash scripts/download_cc_id_300_vec.sh
```

Proses sama dengan Python script, tapi menggunakan sistem tools (curl/wget).

---

## 📊 File Information

| Property | Value |
|----------|-------|
| **File** | `cc.id.300.vec` |
| **Size** | 4.21 GB |
| **Type** | FastText Word Embeddings (binary format) |
| **Dimensions** | 300 |
| **Vocabulary** | ~2.3 million words |
| **Language** | Indonesian + other languages |
| **Source** | HuggingFace Hub - restyaaa/OptimasiFasttextGridSearch |
| **Output Location** | `data/processed/cc.id.300.vec` |
| **Download Speed** | ~9-10 MB/s (tergantung koneksi) |
| **Download Time** | ~7-8 menit |

---

## 💻 Usage Example

Setelah download selesai, gunakan dalam Python:

```python
from gensim.models import KeyedVectors

# Load model
model = KeyedVectors.load_word2vec_format(
    'data/processed/cc.id.300.vec',
    binary=False
)

# Get word vector
vector = model['aplikasi']
print(vector.shape)  # (300,)

# Find similar words
similar = model.most_similar('bagus', topn=5)
for word, score in similar:
    print(f"{word}: {score:.4f}")

# Calculate similarity
sim = model.similarity('baik', 'bagus')
print(f"Similarity: {sim:.4f}")
```

---

## 🔗 Integration dengan Notebook

Dalam `02_feature_extraction_advanced.ipynb`, Part 3.4:

```python
# Part 3.4: GloVe/FastText Indo (using cc.id.300.vec)
from gensim.models import KeyedVectors

# Load pre-trained model
try:
    vectors = KeyedVectors.load_word2vec_format(
        'data/processed/cc.id.300.vec',
        binary=False
    )
    print(f"✅ Loaded FastText model")
    print(f"   Vocabulary: {len(vectors):,} words")
    print(f"   Dimensions: {vectors.vector_size}")
except FileNotFoundError:
    print("❌ Model file not found. Run: python scripts/download_cc_id_300_vec.py")
    vectors = None

# Extract features using mean pooling
if vectors:
    X_train_fasttext = extract_fasttext_features(
        X_train_tokenized,
        vectors,
        vector_size=300
    )
    print(f"✅ Extracted FastText features: {X_train_fasttext.shape}")
```

---

## 🐛 Troubleshooting

### Download Terlalu Lambat
- Coba lagi nanti (load server berkurang pada off-peak hours)
- Periksa kecepatan internet: `speedtest-cli`
- Gunakan VPN jika diperlukan

### "File not found" atau "Network error"
```bash
# Coba lagi
python scripts/download_cc_id_300_vec.py

# Jika gagal lagi, gunakan bash version
bash scripts/download_cc_id_300_vec.sh
```

### "File terlalu kecil / rusak"
```bash
# Script akan bertanya apakah mau menghapus
# Pilih 'y' untuk menghapus dan unduh ulang
rm data/processed/cc.id.300.vec
python scripts/download_cc_id_300_vec.py
```

### "Permission denied" (saat run script)
```bash
# Beri permission execute
chmod +x scripts/download_cc_id_300_vec.sh

# Kemudian jalankan
./scripts/download_cc_id_300_vec.sh
```

---

## 📚 References

- **HuggingFace Hub:** https://huggingface.co/restyaaa/OptimasiFasttextGridSearch
- **FastText Official:** https://fasttext.cc/
- **Gensim KeyedVectors:** https://radimrehurek.com/gensim/models/keyedvectors.html

---

## 📋 Script Comparison

| Feature | Python Script | Bash Script |
|---------|---------------|------------|
| **Ease of Use** | Easy | Easy |
| **Dependencies** | Python 3.6+ | bash, curl/wget |
| **Progress Bar** | ✅ Detailed | ✅ Basic |
| **Cross-platform** | ✅ Yes | ✅ Linux/macOS |
| **Error Handling** | ✅ Excellent | ✅ Good |
| **Recommended** | ✅ Yes | Alternative |

---

## ✅ Checklist

- [ ] Run `python scripts/download_cc_id_300_vec.py`
- [ ] Wait for download to complete (~7-8 minutes)
- [ ] Verify `data/processed/cc.id.300.vec` exists
- [ ] File size should be ~4.21GB
- [ ] Ready to use in notebooks!

---

**Status:** ✅ READY TO USE  
**Last Updated:** May 4, 2026  
**Tested:** Yes ✓
