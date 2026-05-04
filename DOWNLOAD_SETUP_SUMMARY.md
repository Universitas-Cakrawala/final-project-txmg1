# 📥 Download cc.id.300.vec - Setup Summary

## ✅ Files Created

Saya telah membuat 3 files di folder `scripts/`:

### 1. **`download_cc_id_300_vec.py`** (9.9 KB)
- Script Python untuk download FastText Indonesian vectors
- Full-featured dengan progress bar, verifikasi, dan error handling
- **Recommended:** Ya ⭐

### 2. **`download_cc_id_300_vec.sh`** (6.6 KB)
- Script Bash alternatif menggunakan curl/wget
- Lebih ringan, cocok untuk sistem Linux/macOS
- **Recommended:** Alternatif

### 3. **`README.md`** (3.5 KB)
- Dokumentasi lengkap untuk semua download scripts
- Contoh penggunaan, troubleshooting, dan best practices
- **Recommended:** Dibaca sebelum menjalankan script

---

## 🚀 Quick Start

### Opsi 1: Python Script (Most Recommended)

```bash
# Dari root project
python scripts/download_cc_id_300_vec.py

# atau dari folder scripts
cd scripts
python download_cc_id_300_vec.py
```

**Keuntungan:**
- ✅ Progress bar yang detail
- ✅ Verifikasi otomatis
- ✅ Handling error yang bagus
- ✅ Interface user-friendly dalam Bahasa Indonesia

---

### Opsi 2: Bash Script

```bash
# Dari root project
bash scripts/download_cc_id_300_vec.sh

# atau
./scripts/download_cc_id_300_vec.sh
```

**Keuntungan:**
- ✅ Tidak perlu Python
- ✅ Lightweight
- ✅ Menggunakan curl atau wget

---

## 📊 Download Information

| Property | Value |
|----------|-------|
| **File** | `cc.id.300.vec` |
| **Size** | 4.21 GB |
| **Type** | FastText Word Embeddings |
| **Dimensions** | 300 |
| **Source** | HuggingFace - restyaaa/OptimasiFasttextGridSearch |
| **Output** | `data/processed/cc.id.300.vec` |
| **Speed** | ~9-10 MB/s |
| **Time** | ~7-8 menit |

---

## 💻 Proses Download

Ketika Anda menjalankan script, berikut prosesnya:

```
1. Script dimulai
   ├─ Cek apakah file sudah ada
   ├─ Jika ada → tanya apakah ingin unduh ulang
   ├─ Jika tidak → gunakan file yang ada

2. Mulai download
   ├─ Buat folder data/processed jika belum ada
   ├─ Tampilkan progress bar real-time
   └─ Tampilkan kecepatan dan estimasi waktu

3. Verifikasi file
   ├─ Cek apakah file ada
   ├─ Cek ukuran minimum (3.5GB)
   └─ Lapor jika ada masalah

4. Selesai
   └─ Tampilkan contoh penggunaan
```

---

## 📝 Contoh Output

```
======================================================================
🌐 FastText Indonesian Vectors Downloader
======================================================================

📁 Output directory: /home/titan/github/final-project-txmg1/data/processed
📄 File: cc.id.300.vec
📊 Ukuran yang diharapkan: ~4.21GB

[████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 25.0% (1.05GB/4.21GB) 
Speed: 9.37MB/s ETA: 345s

... (menunggu ~7 menit lebih) ...

✅ Unduh selesai! (4.21GB dalam 7m 30s pada 9.37MB/s)

🔍 Memverifikasi file...
📊 Ukuran file: 4.21GB
✅ File terverifikasi!

======================================================================
✅ SUKSES!
======================================================================

📚 CONTOH PENGGUNAAN
======================================================================

from gensim.models import KeyedVectors

# Load model FastText
model = KeyedVectors.load_word2vec_format(
    'data/processed/cc.id.300.vec',
    binary=False
)
...

📂 Lokasi file: /home/titan/github/final-project-txmg1/data/processed/cc.id.300.vec

✨ Siap digunakan dalam project!
```

---

## 🔧 Troubleshooting

### Jika Download Lambat
- Coba lagi pada waktu lain (off-peak hours)
- Periksa koneksi internet
- Gunakan VPN jika diperlukan

### Jika Script Tidak Ditemukan
```bash
# Pastikan ada di folder scripts
ls -la scripts/download_cc_id_300_vec.py

# Run dari root project
python scripts/download_cc_id_300_vec.py
```

### Jika File Rusak
```bash
# Hapus dan coba unduh lagi
rm data/processed/cc.id.300.vec
python scripts/download_cc_id_300_vec.py
```

### "Permission denied" pada Bash Script
```bash
chmod +x scripts/download_cc_id_300_vec.sh
./scripts/download_cc_id_300_vec.sh
```

---

## 📚 Penggunaan di Notebook

Dalam `02_feature_extraction_advanced.ipynb` Part 3.4:

```python
from gensim.models import KeyedVectors

# Load model
model = KeyedVectors.load_word2vec_format(
    'data/processed/cc.id.300.vec',
    binary=False
)

# Extract features
X_train_fasttext = extract_fasttext_features(
    X_train_tokenized,
    model,
    vector_size=300
)
```

---

## ✅ Checklist

- [ ] Verifikasi 3 files ada di folder `scripts/`
- [ ] Jalankan script: `python scripts/download_cc_id_300_vec.py`
- [ ] Tunggu download selesai (~7-8 menit)
- [ ] Verifikasi file di `data/processed/cc.id.300.vec` (4.21GB)
- [ ] Gunakan di Notebook 02

---

## 📋 Files Summary

```
scripts/
├── download_cc_id_300_vec.py  ⭐ Recommended
├── download_cc_id_300_vec.sh  (Alternative)
└── README.md                  (Documentation)

data/processed/
└── cc.id.300.vec              (Output file, setelah download)
```

---

## 🎯 Next Steps

1. **Jalankan download script:**
   ```bash
   python scripts/download_cc_id_300_vec.py
   ```

2. **Tunggu hingga selesai** (~7-8 menit)

3. **Gunakan dalam Notebook 02** untuk feature extraction

4. **Lanjutkan ke Notebook 03** untuk training ML models

---

## 📞 Support

Jika ada masalah:
1. Baca `scripts/README.md`
2. Cek troubleshooting section di atas
3. Verifikasi koneksi internet
4. Coba script alternatif (Bash version)

---

**Status:** ✅ READY  
**Created:** May 4, 2026  
**Tested:** Yes ✓
