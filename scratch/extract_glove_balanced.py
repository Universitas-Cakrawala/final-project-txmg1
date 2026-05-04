import pandas as pd
import numpy as np
import os
from src.feature_extractors import GloVeExtractor

# Load balanced data
df = pd.read_csv('data/processed/balanced_reviews.csv')
texts = df['review_clean'].fillna('').tolist()

# Initialize GloVe (menggunakan file yang sudah ada di data/embeddings jika tersedia, atau default)
# Jika file tidak ada, GloVeExtractor akan menggunakan default path
glove = GloVeExtractor(vectors_path='data/embeddings/cc.id.300.vec')

print("\n🚀 Mengekstrak fitur GloVe pada dataset seimbang...")
X_glove = glove.fit_transform(texts)

# Simpan matrix
os.makedirs('results/feature_matrices', exist_ok=True)
np.save('results/feature_matrices/X_train_balanced_glove.npy', X_glove)

print(f"✅ GloVe extraction selesai. Shape: {X_glove.shape}")
