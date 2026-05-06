"""
preprocessing.py — Text Preprocessing Pipeline untuk Bahasa Indonesia
======================================================================
Pipeline preprocessing lengkap untuk ulasan CoreTax/M-Pajak.

Tahapan:
    1. Loading & type conversion
    2. Normalisasi ringan (newline, control chars, double spaces)
    3. Feature engineering (word_count, char_count, year_month, is_duplicate)
    4. Full NLP pipeline (case folding, noise removal, slang normalization,
       stopword removal, stemming, tokenization)

Usage:
    from src.preprocessing import load_and_prepare, preprocess_text

Author: Text Mining Project — CoreTax Sentiment Classification
"""

import re
import os
import warnings
import pandas as pd
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")

# ======================================================================
# Kamus Normalisasi Slang/Singkatan Indonesia
# Sumber referensi: github.com/nasalsabila/kamus-alay (diperkecil)
# ======================================================================
SLANG_DICT = {
    # Negasi & kata penghubung
    "gk": "tidak",
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "engga": "tidak",
    "enggak": "tidak",
    "tdk": "tidak",
    "gx": "tidak",
    "g": "tidak",
    "kaga": "tidak",
    "kagak": "tidak",
    "gabisa": "tidak bisa",
    "gabsa": "tidak bisa",
    "gaisa": "tidak bisa",
    "gbs": "tidak bisa",
    "gbsa": "tidak bisa",
    "gk bisa": "tidak bisa",
    "blm": "belum",
    "blom": "belum",
    "blum": "belum",
    "tp": "tapi",
    "tpi": "tapi",
    "krn": "karena",
    "karna": "karena",
    "krna": "karena",
    "utk": "untuk",
    "buat": "untuk",
    "u/": "untuk",
    "utuk": "untuk",
    "dgn": "dengan",
    "dg": "dengan",
    "dng": "dengan",
    "yg": "yang",
    "yng": "yang",
    "sdh": "sudah",
    "udh": "sudah",
    "udah": "sudah",
    "uda": "sudah",
    "lg": "lagi",
    "lgi": "lagi",
    "jg": "juga",
    "jga": "juga",
    "sm": "sama",
    "sma": "sama",
    "dr": "dari",
    "dri": "dari",
    "pd": "pada",
    "aj": "saja",
    "aja": "saja",
    "doang": "saja",
    "doank": "saja",
    # Intensifier & adverb
    "bgt": "sangat",
    "banget": "sangat",
    "bngt": "sangat",
    "bngtt": "sangat",
    "bner": "benar",
    "bnr": "benar",
    "skrg": "sekarang",
    "skrng": "sekarang",
    "skrang": "sekarang",
    "sblm": "sebelum",
    "sblmnya": "sebelumnya",
    "msh": "masih",
    "masi": "masih",
    "msih": "masih",
    "slalu": "selalu",
    "sllu": "selalu",
    "sgt": "sangat",
    "sngat": "sangat",
    # Kata kerja umum
    "bs": "bisa",
    "bsa": "bisa",
    "jdi": "jadi",
    "jd": "jadi",
    "hrs": "harus",
    "hrus": "harus",
    "mksd": "maksud",
    "mksud": "maksud",
    "tau": "tahu",
    "tw": "tahu",
    "mau": "mau",
    "mo": "mau",
    "mw": "mau",
    "pke": "pakai",
    "pake": "pakai",
    "pk": "pakai",
    "tlg": "tolong",
    "tlng": "tolong",
    "tlong": "tolong",
    "cb": "coba",
    "cba": "coba",
    "trs": "terus",
    "trus": "terus",
    "dtg": "datang",
    "dtng": "datang",
    "krj": "kerja",
    "krja": "kerja",
    "bkin": "bikin",
    "bkn": "bukan",
    "dpt": "dapat",
    "dpat": "dapat",
    "ksh": "kasih",
    "gmn": "bagaimana",
    "gmna": "bagaimana",
    "gimana": "bagaimana",
    "knp": "kenapa",
    "knpa": "kenapa",
    "dmn": "dimana",
    "dmna": "dimana",
    "gini": "begini",
    "gitu": "begitu",
    # Kata benda & domain
    "app": "aplikasi",
    "apk": "aplikasi",
    "apl": "aplikasi",
    "aplksi": "aplikasi",
    "hp": "handphone",
    "hape": "handphone",
    "no": "nomor",
    "nmr": "nomor",
    "org": "orang",
    "orng": "orang",
    "info": "informasi",
    "srtifikat": "sertifikat",
    "srtfkt": "sertifikat",
    "akun": "akun",
    # Kata sifat
    "bgus": "bagus",
    "bgs": "bagus",
    "jlek": "jelek",
    "jlk": "jelek",
    "gmpng": "gampang",
    "gmpang": "gampang",
    "ssh": "susah",
    "lma": "lama",
    "lm": "lama",
    "cpet": "cepat",
    "cpt": "cepat",
    "byk": "banyak",
    "bnyk": "banyak",
    "sdkt": "sedikit",
    "sdkit": "sedikit",
    "kecil": "kecil",
    "kcl": "kecil",
    "bsr": "besar",
    # Sapaan & ekspresi
    "dong": "",
    "deh": "",
    "sih": "",
    "nih": "",
    "loh": "",
    "lah": "",
    "ya": "",
    "yaa": "",
    "yah": "",
    "woy": "",
    "woii": "",
    "woi": "",
    "pls": "tolong",
    "please": "tolong",
    "thx": "terima kasih",
    "tq": "terima kasih",
    "makasih": "terima kasih",
    "mksh": "terima kasih",
    "mkasih": "terima kasih",
    "sorry": "maaf",
    "sory": "maaf",
    "sori": "maaf",
    # Kata teknis (domain CoreTax/pajak)
    "npwp": "npwp",  # pertahankan sebagai entitas
    "efin": "efin",  # pertahankan sebagai entitas
    "spt": "spt",  # surat pemberitahuan tahunan
    "djp": "djp",  # direktorat jenderal pajak
}


def load_raw_data(filepath: str = "data/raw/coretax_reviews.csv") -> pd.DataFrame:
    """
    Prompt 0.1 — Loading & Eksplorasi Dasar
    Load raw CSV dan lakukan type conversion dasar.

    Args:
        filepath: Path ke file CSV raw

    Returns:
        DataFrame dengan kolom yang sudah di-type-convert
    """
    # Load CSV
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    # Type conversions
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["app_version"] = df["app_version"].fillna("unknown").astype(str)
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df["rating"] = df["rating"].astype(int)
    df["likes"] = df["likes"].fillna(0).astype(int)

    # Hapus baris dengan review_text kosong setelah strip
    empty_mask = df["review_text"].isin(["", "nan", "None"])
    n_empty = empty_mask.sum()
    if n_empty > 0:
        print(f"⚠️  Menghapus {n_empty} baris dengan review_text kosong")
        df = df[~empty_mask].reset_index(drop=True)

    return df


def normalize_text_basic(text: str) -> str:
    """
    Prompt 0.2 — Normalisasi Ringan
    Pembersihan teks tingkat dasar TANPA NLP agresif.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Ubah newline jadi spasi
    text = text.replace("\n", " ").replace("\r", " ")

    # 2. Hapus karakter kontrol yang tidak terlihat (tapi pertahankan spasi biasa)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # 3. Hapus spasi ganda
    text = re.sub(r"\s+", " ", text).strip()

    return text


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prompt 0.2 — Feature Engineering Sederhana
    Tambahkan fitur pendukung analisis.
    """
    df = df.copy()

    # Word count & char count
    df["word_count"] = df["review_text"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["review_text"].apply(lambda x: len(str(x)))

    # Year-month period
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # Duplicate flag
    dup_counts = df["review_text"].value_counts()
    df["is_duplicate"] = df["review_text"].map(lambda x: dup_counts.get(x, 0) > 1)

    return df


def prepare_dataset(
    filepath: str = "data/raw/coretax_reviews.csv",
    output_path: str = "data/processed/reviews_prepared.csv",
) -> pd.DataFrame:
    """
    Pipeline lengkap Bagian 0: Load → Normalize → Feature Engineering → Save.

    Returns:
        DataFrame yang sudah siap untuk tahap labeling & preprocessing
    """
    print("📦 Bagian 0 — Data Processing (Persiapan Data)")
    print("=" * 60)

    # Step 1: Load
    print("\n🔄 Step 1: Loading data...")
    df = load_raw_data(filepath)
    print(f"   ✅ Loaded {len(df)} baris")
    print(f"   📊 Distribusi rating:")
    for rating, count in df["rating"].value_counts().sort_index().items():
        print(f"      ⭐ {rating}: {count} ({count/len(df)*100:.1f}%)")

    # Step 2: Normalize
    print("\n🔄 Step 2: Normalisasi ringan...")
    df["review_text"] = df["review_text"].apply(normalize_text_basic)
    print("   ✅ Normalisasi teks dasar selesai")

    # Step 3: Feature engineering
    print("\n🔄 Step 3: Feature engineering...")
    df = add_engineered_features(df)
    n_dup = df["is_duplicate"].sum()
    print(f"   ✅ Fitur ditambahkan: word_count, char_count, year_month, is_duplicate")
    print(f"   📊 Duplikat teks: {n_dup} baris ({n_dup/len(df)*100:.1f}%)")

    # Step 4: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset tersimpan: {output_path}")
    print(f"   Total baris: {len(df)}")

    return df


# ======================================================================
# Full NLP Preprocessing Pipeline (Bagian 2 — Prompt 2.1)
# ======================================================================


def _load_stopwords():
    """Load Indonesian stopwords dari NLTK + custom additions."""
    try:
        import nltk

        try:
            stopwords_id = set(nltk.corpus.stopwords.words("indonesian"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stopwords_id = set(nltk.corpus.stopwords.words("indonesian"))
    except ImportError:
        # Fallback: common Indonesian stopwords
        stopwords_id = {
            "yang",
            "di",
            "ke",
            "dari",
            "dan",
            "ini",
            "itu",
            "dengan",
            "untuk",
            "pada",
            "adalah",
            "juga",
            "saya",
            "sudah",
            "ada",
            "akan",
            "atau",
            "bisa",
            "lebih",
            "kalau",
            "se",
            "nya",
            "aku",
            "kamu",
            "dia",
            "mereka",
            "kami",
            "kita",
        }

    # PENTING: Pertahankan kata negasi — krusial untuk sentimen
    negation_words = {"tidak", "bukan", "belum", "jangan", "tanpa", "tak"}
    stopwords_id -= negation_words

    return stopwords_id


def _load_stemmer():
    """Load Sastrawi stemmer."""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        factory = StemmerFactory()
        return factory.create_stemmer()
    except (ImportError, AttributeError):
        print(
            "⚠️  Sastrawi tidak terinstall atau versi tidak sesuai. Stemming dinonaktifkan."
        )
        print("   Install: pip install Sastrawi")
        return None


# Cache global agar tidak load berulang kali
_STOPWORDS = None
_STEMMER = None
_STEMMER_WARNED = False  # Flag untuk menghindari spam warning


def _get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = _load_stopwords()
    return _STOPWORDS


def _get_stemmer():
    global _STEMMER, _STEMMER_WARNED
    if _STEMMER is None and not _STEMMER_WARNED:
        _STEMMER = _load_stemmer()
        if _STEMMER is None:
            _STEMMER_WARNED = True
    return _STEMMER


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    apply_stemming: bool = True,
    return_tokens: bool = False,
):
    """
    Prompt 2.1 — Pipeline Preprocessing Bahasa Indonesia

    Full NLP preprocessing pipeline:
    1. Case folding
    2. Noise removal (HTML, URL, mention, hashtag, angka)
    3. Emoji removal
    4. Slang/singkatan normalization
    5. Stopword removal (dengan preservasi negasi)
    6. Stemming (Sastrawi)
    7. Tokenisasi

    Args:
        text: Teks mentah
        remove_stopwords: Apakah menghapus stopwords (default True)
        apply_stemming: Apakah menerapkan stemming (default True)
        return_tokens: Jika True, return list token; jika False, return string

    Returns:
        Teks yang sudah dipreprocessing (str atau list[str])
    """
    if not isinstance(text, str) or not text.strip():
        return [] if return_tokens else ""

    # 1. Case folding
    text = text.lower()

    # 2. Hapus noise
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    text = re.sub(r"http\S+|www\.\S+", "", text)  # URL
    text = re.sub(r"@\w+", "", text)  # Mention
    text = re.sub(r"#\w+", "", text)  # Hashtag
    text = re.sub(r"\d+", "", text)  # Angka

    # 3. Hapus emoji dan karakter non-alfabet
    text = re.sub(r"[^\w\s]", " ", text)  # Tanda baca & simbol
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Non-ASCII (emoji, etc.)

    # 4. Normalisasi slang/singkatan
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        if token in SLANG_DICT:
            replacement = SLANG_DICT[token]
            if replacement:  # Skip empty replacements (filler words)
                normalized_tokens.extend(replacement.split())
        else:
            normalized_tokens.append(token)
    tokens = normalized_tokens

    # 5. Stopword removal
    if remove_stopwords:
        stopwords = _get_stopwords()
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]

    # 6. Stemming
    if apply_stemming:
        stemmer = _get_stemmer()
        if stemmer:
            tokens = [stemmer.stem(t) for t in tokens]

    # 7. Bersihkan token kosong
    tokens = [t.strip() for t in tokens if t.strip()]

    if return_tokens:
        return tokens
    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "review_text",
    remove_stopwords: bool = True,
    apply_stemming: bool = True,
) -> pd.DataFrame:
    """
    Apply preprocessing pipeline ke seluruh DataFrame.

    Menambahkan kolom:
        - review_clean: teks setelah full preprocessing
        - review_tokens: list token hasil preprocessing

    Args:
        df: DataFrame input
        text_col: Nama kolom teks
        remove_stopwords: Apakah menghapus stopwords
        apply_stemming: Apakah menerapkan stemming

    Returns:
        DataFrame dengan kolom review_clean dan review_tokens
    """
    df = df.copy()

    print(f"🔄 Preprocessing {len(df)} teks...")
    print(f"   Stopword removal: {'✅' if remove_stopwords else '❌'}")
    print(f"   Stemming: {'✅' if apply_stemming else '❌'}")

    # Preprocess teks
    df["review_clean"] = df[text_col].apply(
        lambda x: preprocess_text(
            x,
            remove_stopwords=remove_stopwords,
            apply_stemming=apply_stemming,
            return_tokens=False,
        )
    )

    # Tokenize
    df["review_tokens"] = df["review_clean"].apply(lambda x: x.split() if x else [])

    # Statistik
    empty_count = (df["review_clean"] == "").sum()
    avg_tokens = df["review_tokens"].apply(len).mean()
    print(f"   ✅ Selesai!")
    print(f"   📊 Rata-rata token per review: {avg_tokens:.1f}")
    print(f"   ⚠️  Review kosong setelah preprocessing: {empty_count}")

    return df


# ======================================================================
# Labeling (Bagian 1 — Prompt 1.1)
# ======================================================================


def add_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prompt 1.1 — Skema Labeling Ternary dari Rating

    Mapping:
        Rating 1-2 → Negatif
        Rating 3   → Netral
        Rating 4-5 → Positif

    PENTING: Rating 3 TIDAK di-drop, tetap masuk sebagai kelas Netral.
    """
    df = df.copy()

    def map_sentiment(rating):
        if rating <= 2:
            return "Negatif"
        elif rating == 3:
            return "Netral"
        else:
            return "Positif"

    df["sentiment"] = df["rating"].apply(map_sentiment)

    # Encode numerik untuk model ML
    label_map = {"Negatif": 0, "Netral": 1, "Positif": 2}
    df["sentiment_encoded"] = df["sentiment"].map(label_map)

    # Print distribusi
    print("\n🏷️  Labeling Ternary:")
    print("=" * 40)
    dist = df["sentiment"].value_counts()
    for label in ["Negatif", "Netral", "Positif"]:
        count = dist.get(label, 0)
        pct = count / len(df) * 100
        print(f"   {label:10s}: {count:5d} ({pct:.1f}%)")

    # Rasio imbalance
    max_class = dist.max()
    min_class = dist.min()
    print(f"\n   Rasio imbalance (max/min): {max_class/min_class:.1f}x")
    print(f"   ⚠️  Kelas Netral sangat minoritas ({dist.get('Netral', 0)} ulasan)")

    return df


# ======================================================================
# Utility functions
# ======================================================================


def show_preprocessing_examples(df: pd.DataFrame, n: int = 5):
    """Tampilkan contoh teks sebelum dan sesudah preprocessing."""
    print("\n📋 Contoh Preprocessing (Before → After):")
    print("=" * 80)
    sample = df.sample(n=min(n, len(df)), random_state=42)
    for idx, row in sample.iterrows():
        before = str(row.get("review_text", ""))[:80]
        after = str(row.get("review_clean", ""))[:80]
        print(
            f"\n  ⭐ Rating {row.get('rating', '?')} | Sentimen: {row.get('sentiment', '?')}"
        )
        print(f"  📝 Before: {before}")
        print(f"  ✅ After : {after}")
