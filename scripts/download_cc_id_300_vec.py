#!/usr/bin/env python3
"""
HuggingFace FastText Indonesian Vectors Downloader
====================================================
Download cc.id.300.vec dari HuggingFace untuk feature extraction

Sumber: https://huggingface.co/restyaaa/OptimasiFasttextGridSearch
File: cc.id.300.vec (Indonesian FastText Embeddings - 300 dimensions)

Author: Text Mining Project
Date: 2026
"""

import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ==================== Configuration ====================

# Output settings
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings")
OUTPUT_FILE = "cc.id.300.vec"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# HuggingFace URL (convert dari blob ke resolve)
# Original: https://huggingface.co/restyaaa/OptimasiFasttextGridSearch/blob/3a2d6f5a8f4ed3f06b0209a93fa0c66889b3e12c/cc.id.300.vec
# Download: https://huggingface.co/restyaaa/OptimasiFasttextGridSearch/resolve/3a2d6f5a8f4ed3f06b0209a93fa0c66889b3e12c/cc.id.300.vec

DOWNLOAD_URL = "https://huggingface.co/restyaaa/OptimasiFasttextGridSearch/resolve/3a2d6f5a8f4ed3f06b0209a93fa0c66889b3e12c/cc.id.300.vec"

# File specifications
EXPECTED_SIZE = 4.21 * 1024 * 1024 * 1024  # Approximately 4.21GB
MIN_SIZE = 3.5 * 1024 * 1024 * 1024  # Minimum 3.5GB to detect corruption

# ==================== Helper Functions ====================


def format_size(bytes_size):
    """Konversi bytes ke format yang dapat dibaca manusia."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f}TB"


def format_time(seconds):
    """Konversi detik ke format waktu yang dapat dibaca."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def get_file_size(url):
    """Dapatkan ukuran file dari URL tanpa mendownload."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        response = urllib.request.urlopen(req, timeout=10)
        size = response.headers.get("Content-Length")
        return int(size) if size else None
    except Exception as e:
        return None


def download_file(url, output_path):
    """Download file dengan progress bar."""
    print("\n" + "=" * 70)
    print("📥 MENGUNDUH FILE")
    print("=" * 70)
    print(f"\n🔗 URL: {url}")
    print(f"💾 Menyimpan ke: {output_path}")
    print("-" * 70)

    # Cek apakah file sudah ada
    if os.path.exists(output_path):
        existing_size = os.path.getsize(output_path)
        print(f"\n⚠️  File sudah ada ({format_size(existing_size)})")
        response = input("📝 Unduh ulang? (y/n): ").strip().lower()
        if response != "y":
            print("✅ Menggunakan file yang ada")
            return True
        print("🔄 Menimpa file yang ada...\n")
        os.remove(output_path)

    # Buat direktori jika perlu
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set up headers
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )

    try:
        # Dapatkan ukuran total
        response = urllib.request.urlopen(req, timeout=30)
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size else None

        if total_size:
            print(f"📊 Ukuran total: {format_size(total_size)}\n")
        else:
            print("⚠️  Tidak dapat menentukan ukuran file\n")

        # Download dengan progress bar
        start_time = datetime.now()
        downloaded = 0
        block_size = 1024 * 1024  # 1MB blocks

        with open(output_path, "wb") as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break

                f.write(buffer)
                downloaded += len(buffer)

                # Tampilkan progress
                if total_size:
                    percent = (downloaded / total_size) * 100
                    filled = int(40 * downloaded / total_size)

                    # Hitung kecepatan dan estimasi waktu
                    elapsed = (datetime.now() - start_time).total_seconds()
                    speed = downloaded / elapsed / (1024 * 1024) if elapsed > 0 else 0
                    remaining = (
                        (total_size - downloaded) / (1024 * 1024) / speed
                        if speed > 0
                        else 0
                    )

                    bar = "█" * filled + "░" * (40 - filled)
                    print(
                        f"\r[{bar}] {percent:5.1f}% ({format_size(downloaded)}/{format_size(total_size)}) "
                        f"Speed: {speed:6.2f}MB/s ETA: {format_time(remaining):8s}",
                        end="",
                        flush=True,
                    )

        print("\n")
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_speed = downloaded / elapsed / (1024 * 1024) if elapsed > 0 else 0
        print(
            f"✅ Unduh selesai! ({format_size(downloaded)} dalam {format_time(elapsed)} pada {avg_speed:.2f}MB/s)"
        )

        return True

    except urllib.error.HTTPError as e:
        print(f"\n❌ HTTP Error {e.code}: {e.reason}")
        if e.code == 404:
            print("   File tidak ditemukan di sumber")
        elif e.code == 403:
            print("   Akses ditolak (sumber memblokir download otomatis)")
        elif e.code == 401:
            print("   Tidak terotorisasi (mungkin perlu token HuggingFace)")
        return False
    except urllib.error.URLError as e:
        print(f"\n❌ Error Koneksi: {e.reason}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def verify_file(file_path):
    """Verifikasi file yang diunduh - hanya cek ukuran, tanpa input."""
    if not os.path.exists(file_path):
        print(f"❌ File tidak ditemukan: {file_path}")
        return False

    file_size = os.path.getsize(file_path)

    # Cek ukuran minimum (tanpa pertanyaan, hanya return status)
    if file_size < MIN_SIZE:
        print(
            f"❌ Ukuran file tidak valid: {format_size(file_size)} (minimum: {format_size(MIN_SIZE)})"
        )
        return False

    return True


def show_usage_example():
    """Tampilkan contoh penggunaan."""
    print("\n" + "=" * 70)
    print("📚 CONTOH PENGGUNAAN")
    print("=" * 70)
    print("""
from gensim.models import KeyedVectors

# Load model FastText
model = KeyedVectors.load_word2vec_format(
    'data/embeddings/cc.id.300.vec',
    binary=False
)

# Dapatkan vector untuk sebuah kata
vector = model['aplikasi']
print(f"Shape: {vector.shape}")  # (300,)

# Cari kata yang mirip
similar = model.most_similar('bagus', topn=5)
for word, score in similar:
    print(f"{word:20} {score:.4f}")

# Hitung kesamaan antara dua kata
similarity = model.similarity('baik', 'bagus')
print(f"Similarity: {similarity:.4f}")

# Test dengan kata-kata Indonesia
test_words = ['aplikasi', 'buruk', 'bagus', 'error', 'lambat']
for word in test_words:
    if word in model:
        print(f"✓ '{word}' ditemukan dalam model")
    else:
        print(f"✗ '{word}' tidak ditemukan")
    """)
    print("=" * 70)


def main():
    """Fungsi utama."""
    print("\n" + "=" * 70)
    print("🌐 FastText Indonesian Vectors Downloader")
    print("=" * 70)
    print(f"\n📁 Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📄 File: {OUTPUT_FILE}")
    print(f"📊 Ukuran yang diharapkan: ~{format_size(EXPECTED_SIZE)}")
    print()

    # ==================== Pengecekan File ====================
    if os.path.exists(OUTPUT_PATH):
        existing_size = os.path.getsize(OUTPUT_PATH)
        print(f"✅ File sudah ada: {format_size(existing_size)}")

        # Cek apakah ukuran file valid
        if existing_size >= MIN_SIZE:
            print(f"✓ Ukuran valid (minimum: {format_size(MIN_SIZE)})")
            print("\n📝 Verifikasi file...")

            if verify_file(OUTPUT_PATH):
                print("\n" + "=" * 70)
                print("✅ FILE SUDAH LENGKAP - TIDAK PERLU DOWNLOAD!")
                print("=" * 70)
                show_usage_example()
                print(f"\n📂 Lokasi file: {os.path.abspath(OUTPUT_PATH)}")
                print("\n✨ Siap digunakan dalam project!")
                return True
            else:
                print("❌ Verifikasi file gagal")
                response = input("\n📝 Unduh file baru? (y/n): ").strip().lower()
                if response != "y":
                    print("❌ Pembatalan oleh user")
                    return False
                print("🔄 Menghapus file lama dan download baru...")
                os.remove(OUTPUT_PATH)
        else:
            # File ada tapi ukurannya tidak valid
            print(f"⚠️  File terlalu kecil (minimum: {format_size(MIN_SIZE)})")
            response = input("\n📝 Hapus dan unduh ulang? (y/n): ").strip().lower()
            if response != "y":
                print("❌ Pembatalan oleh user")
                return False
            print("🗑️  Menghapus file lama...")
            os.remove(OUTPUT_PATH)

    # ==================== Download File ====================
    # Unduh file
    if download_file(DOWNLOAD_URL, OUTPUT_PATH):
        if verify_file(OUTPUT_PATH):
            print("\n" + "=" * 70)
            print("✅ SUKSES!")
            print("=" * 70)
            show_usage_example()
            print(f"\n📂 Lokasi file: {os.path.abspath(OUTPUT_PATH)}")
            print("\n✨ Siap digunakan dalam project!")
            return True
        else:
            print("\n❌ Verifikasi file gagal")
            return False
    else:
        print("\n" + "=" * 70)
        print("❌ UNDUH GAGAL")
        print("=" * 70)
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Periksa koneksi internet")
        print("2. Coba lagi nanti (sumber mungkin sedang tidak tersedia)")
        print("3. Periksa URL di browser:")
        print(f"   {DOWNLOAD_URL}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download dibatalkan oleh user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
