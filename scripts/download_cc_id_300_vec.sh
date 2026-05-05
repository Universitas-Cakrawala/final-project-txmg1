#!/bin/bash
#
# HuggingFace FastText Indonesian Vectors Downloader (Bash version)
# ==================================================================
# Download cc.id.300.vec dari HuggingFace ke data/processed/
#
# Usage: bash download_cc_id_300_vec.sh
# atau:  ./download_cc_id_300_vec.sh
#

# ==================== Configuration ====================

# Output settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/data/embeddings"
OUTPUT_FILE="cc.id.300.vec"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

# HuggingFace URL
DOWNLOAD_URL="https://huggingface.co/restyaaa/OptimasiFasttextGridSearch/resolve/3a2d6f5a8f4ed3f06b0209a93fa0c66889b3e12c/cc.id.300.vec"

# File specifications
EXPECTED_SIZE="4.21GB"
MIN_SIZE=$((3500 * 1024 * 1024))  # 3.5GB minimum

# ==================== Helper Functions ====================

format_size() {
    local bytes=$1
    if [ "$bytes" -lt 1024 ]; then
        echo "${bytes}B"
    elif [ "$bytes" -lt $((1024 * 1024)) ]; then
        echo "$((bytes / 1024))KB"
    elif [ "$bytes" -lt $((1024 * 1024 * 1024)) ]; then
        echo "$((bytes / (1024 * 1024)))MB"
    else
        echo "$(echo "scale=2; $bytes / (1024 * 1024 * 1024)" | bc)GB"
    fi
}

print_header() {
    echo ""
    echo "======================================================================"
    echo "🌐 FastText Indonesian Vectors Downloader (Bash)"
    echo "======================================================================"
}

print_section() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

# ==================== Main Functions ====================

check_dependencies() {
    # Check for curl or wget
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        echo "❌ Diperlukan curl atau wget"
        echo "   Install dengan: sudo apt-get install curl"
        exit 1
    fi
    
    # Check for bc (untuk perhitungan)
    if ! command -v bc &> /dev/null; then
        echo "⚠️  bc tidak ditemukan, beberapa feature mungkin tidak bekerja sempurna"
    fi
}

check_existing_file() {
    if [ -f "$OUTPUT_PATH" ]; then
        local file_size=$(stat -f%z "$OUTPUT_PATH" 2>/dev/null || stat -c%s "$OUTPUT_PATH" 2>/dev/null)
        echo "⚠️  File sudah ada ($(format_size $file_size))"
        read -p "📝 Unduh ulang? (y/n): " choice
        if [ "$choice" != "y" ]; then
            echo "✅ Menggunakan file yang ada"
            return 1
        fi
        echo "🔄 Menimpa file..."
        rm -f "$OUTPUT_PATH"
    fi
    return 0
}

download_with_curl() {
    echo ""
    echo "📥 Mengunduh dengan curl..."
    echo "🔗 URL: $DOWNLOAD_URL"
    echo "💾 Menyimpan ke: $OUTPUT_PATH"
    echo "-----"
    
    mkdir -p "$OUTPUT_DIR"
    
    if curl -L \
        -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
        --progress-bar \
        --output "$OUTPUT_PATH" \
        "$DOWNLOAD_URL"; then
        return 0
    else
        return 1
    fi
}

download_with_wget() {
    echo ""
    echo "📥 Mengunduh dengan wget..."
    echo "🔗 URL: $DOWNLOAD_URL"
    echo "💾 Menyimpan ke: $OUTPUT_PATH"
    echo "-----"
    
    mkdir -p "$OUTPUT_DIR"
    
    if wget \
        -U "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
        -O "$OUTPUT_PATH" \
        --show-progress \
        "$DOWNLOAD_URL"; then
        return 0
    else
        return 1
    fi
}

download_file() {
    if command -v curl &> /dev/null; then
        download_with_curl
    elif command -v wget &> /dev/null; then
        download_with_wget
    else
        echo "❌ curl atau wget tidak ditemukan"
        return 1
    fi
}

verify_file() {
    echo ""
    echo "🔍 Memverifikasi file..."
    
    if [ ! -f "$OUTPUT_PATH" ]; then
        echo "❌ File tidak ditemukan: $OUTPUT_PATH"
        return 1
    fi
    
    local file_size=$(stat -f%z "$OUTPUT_PATH" 2>/dev/null || stat -c%s "$OUTPUT_PATH" 2>/dev/null)
    echo "📊 Ukuran file: $(format_size $file_size)"
    
    if [ "$file_size" -lt "$MIN_SIZE" ]; then
        echo "⚠️  File terlalu kecil! Minimum: $(format_size $MIN_SIZE)"
        read -p "Hapus file yang rusak? (y/n): " choice
        if [ "$choice" = "y" ]; then
            rm -f "$OUTPUT_PATH"
            echo "🗑️  Dihapus. Silakan coba unduh lagi."
        fi
        return 1
    fi
    
    echo "✅ File terverifikasi! Ukuran: $(format_size $file_size)"
    return 0
}

show_usage_example() {
    print_section "📚 CONTOH PENGGUNAAN"
    cat << 'EOF'
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
EOF
    echo "======================================================================"
}

# ==================== Main Script ====================

main() {
    print_header
    
    echo ""
    echo "📁 Output directory: $OUTPUT_DIR"
    echo "📄 File: $OUTPUT_FILE"
    echo "📊 Ukuran yang diharapkan: $EXPECTED_SIZE"
    
    # Check dependencies
    check_dependencies
    
    # Check existing file
    if ! check_existing_file; then
        verify_file
        if [ $? -eq 0 ]; then
            print_section "✅ SUKSES"
            show_usage_example
            echo ""
            echo "📂 Lokasi file: $OUTPUT_PATH"
            echo ""
            echo "✨ Siap digunakan dalam project!"
            exit 0
        fi
    fi
    
    # Download file
    if download_file; then
        if verify_file; then
            print_section "✅ SUKSES"
            show_usage_example
            echo ""
            echo "📂 Lokasi file: $OUTPUT_PATH"
            echo ""
            echo "✨ Siap digunakan dalam project!"
            exit 0
        else
            print_section "❌ VERIFIKASI GAGAL"
            exit 1
        fi
    else
        print_section "❌ UNDUH GAGAL"
        echo ""
        echo "🔧 TROUBLESHOOTING:"
        echo "1. Periksa koneksi internet"
        echo "2. Coba lagi nanti"
        echo "3. Gunakan script Python untuk opsi lebih banyak:"
        echo "   python scripts/download_cc_id_300_vec.py"
        exit 1
    fi
}

# Run main
main
