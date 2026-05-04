"""
feature_extractors.py — Metode Ekstraksi Fitur untuk Klasifikasi Sentimen
==========================================================================
8 metode feature extraction yang diimplementasikan secara modular:

Statistik:
    1. TF-IDF (unigram + bigram)
    2. BM25 (retrieval-based)

Word Embeddings:
    3. Word2Vec (pre-trained / train from scratch)
    4. GloVe (menggunakan FastText Indonesian vectors)
    5. FastText (subword-aware)

Transformer:
    6. DistilBERT (multilingual)
    7. BERT / IndoBERT
    8. RoBERTa / XLM-RoBERTa

Usage:
    from src.feature_extractors import TFIDFExtractor, FastTextExtractor
    extractor = TFIDFExtractor()
    X_train, X_test = extractor.fit_transform(train_texts, test_texts)

Author: Text Mining Project — CoretTax Sentiment Classification
"""

import os
import warnings
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, List
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')


class BaseExtractor(ABC):
    """Base class untuk semua feature extractors."""

    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def fit_transform(self, train_texts: List[str],
                      test_texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit pada train data dan transform kedua split."""
        pass

    def get_feature_type(self) -> str:
        """Return 'sparse' atau 'dense'."""
        return 'dense'

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


# ======================================================================
# 1. TF-IDF (Prompt 3.1)
# ======================================================================

class TFIDFExtractor(BaseExtractor):
    """
    TF-IDF Feature Extraction.

    Mendukung unigram dan bigram, dengan kontrol max_features.
    Output: sparse matrix.
    """

    def __init__(self, max_features: int = 10000,
                 ngram_range: tuple = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95):
        super().__init__(name='TF-IDF')
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None

    def fit_transform(self, train_texts, test_texts):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True  # Menggunakan 1 + log(tf), lebih baik untuk teks
        )

        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)
        self._is_fitted = True

        print(f"   TF-IDF: vocab={len(self.vectorizer.vocabulary_)}, "
              f"shape=({X_train.shape[0]}, {X_train.shape[1]})")

        return X_train, X_test

    def get_feature_type(self):
        return 'sparse'

    def get_top_terms(self, n: int = 20) -> list:
        """Return top N terms berdasarkan IDF score."""
        if not self._is_fitted:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        idf_scores = self.vectorizer.idf_
        top_indices = np.argsort(idf_scores)[::-1][:n]
        return [(feature_names[i], idf_scores[i]) for i in top_indices]


# ======================================================================
# 2. BM25 (Prompt 3.5)
# ======================================================================

class BM25Extractor(BaseExtractor):
    """
    BM25 Retrieval-Based Feature Extraction.

    Adaptasi BM25 untuk klasifikasi:
    - Buat corpus per kelas (Negatif, Netral, Positif)
    - Untuk setiap dokumen, hitung BM25 score terhadap setiap corpus kelas
    - Fitur = [score_negatif, score_netral, score_positif]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__(name='BM25')
        self.k1 = k1
        self.b = b
        self._class_models = {}

    def fit_transform(self, train_texts, test_texts,
                      train_labels=None):
        from rank_bm25 import BM25Okapi

        if train_labels is None:
            raise ValueError("BM25Extractor membutuhkan train_labels untuk membuat corpus per kelas")

        # Tokenize
        train_tokenized = [text.split() for text in train_texts]
        test_tokenized = [text.split() for text in test_texts]

        # Buat BM25 model per kelas
        unique_labels = sorted(set(train_labels))
        self._class_models = {}

        for label in unique_labels:
            class_docs = [train_tokenized[i] for i in range(len(train_labels))
                          if train_labels[i] == label]
            self._class_models[label] = BM25Okapi(class_docs, k1=self.k1, b=self.b)

        # Compute features
        def _compute_features(tokenized_docs):
            features = np.zeros((len(tokenized_docs), len(unique_labels)))
            for i, doc in enumerate(tokenized_docs):
                for j, label in enumerate(unique_labels):
                    scores = self._class_models[label].get_scores(doc)
                    features[i, j] = np.mean(scores) if len(scores) > 0 else 0.0
            return features

        X_train = _compute_features(train_tokenized)
        X_test = _compute_features(test_tokenized)
        self._is_fitted = True

        print(f"   BM25: shape=({X_train.shape[0]}, {X_train.shape[1]})")

        return X_train, X_test

    def get_feature_type(self):
        return 'dense'


# ======================================================================
# 3. Word2Vec (Prompt 3.6)
# ======================================================================

class Word2VecExtractor(BaseExtractor):
    """
    Word2Vec Feature Extraction.

    Dua mode:
    - Pre-trained: Load model yang sudah ada
    - From scratch: Train pada corpus ulasan
    """

    def __init__(self, vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 2,
                 pretrained_path: Optional[str] = None,
                 train_from_scratch: bool = True):
        super().__init__(name='Word2Vec')
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.pretrained_path = pretrained_path
        self.train_from_scratch = train_from_scratch
        self.model = None

    def fit_transform(self, train_texts, test_texts):
        from gensim.models import Word2Vec

        train_tokenized = [text.split() for text in train_texts]
        test_tokenized = [text.split() for text in test_texts]

        if self.pretrained_path and os.path.exists(self.pretrained_path):
            # Load pre-trained
            from gensim.models import KeyedVectors
            self.model = KeyedVectors.load(self.pretrained_path)
            print(f"   Word2Vec: loaded pre-trained from {self.pretrained_path}")
        else:
            # Train from scratch
            self.model = Word2Vec(
                sentences=train_tokenized,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=4,
                epochs=20,
                sg=1  # Skip-gram (lebih baik untuk rare words)
            )
            self.model = self.model.wv
            print(f"   Word2Vec: trained from scratch, vocab={len(self.model)}")

        X_train = self._texts_to_vectors(train_tokenized)
        X_test = self._texts_to_vectors(test_tokenized)
        self._is_fitted = True

        # OOV stats
        all_tokens = set(t for tokens in train_tokenized for t in tokens)
        oov = sum(1 for t in all_tokens if t not in self.model)
        print(f"   Word2Vec: OOV rate = {oov}/{len(all_tokens)} ({oov/max(len(all_tokens),1)*100:.1f}%)")
        print(f"   Word2Vec: shape=({X_train.shape[0]}, {X_train.shape[1]})")

        return X_train, X_test

    def _texts_to_vectors(self, tokenized_texts):
        """Mean pooling of word vectors."""
        dim = self.model.vector_size if hasattr(self.model, 'vector_size') else self.vector_size
        vectors = np.zeros((len(tokenized_texts), dim))
        for i, tokens in enumerate(tokenized_texts):
            valid_vectors = [self.model[t] for t in tokens if t in self.model]
            if valid_vectors:
                vectors[i] = np.mean(valid_vectors, axis=0)
        return vectors


# ======================================================================
# 4. GloVe (Prompt 3.2) — Menggunakan FastText Indonesian vectors
# ======================================================================

class GloVeExtractor(BaseExtractor):
    """
    GloVe-style Feature Extraction menggunakan FastText Indonesian vectors.

    Menggunakan cc.id.300.vec (Common Crawl Indonesian, 300 dimensi)
    sebagai pengganti GloVe karena tidak ada GloVe pre-trained untuk Bahasa Indonesia.
    """

    def __init__(self, vectors_path: str = "data/processed/cc.id.300.vec",
                 dim: int = 300,
                 max_vocab: int = 200000):
        super().__init__(name='GloVe')
        self.vectors_path = vectors_path
        self.dim = dim
        self.max_vocab = max_vocab
        self.word_vectors = {}

    def fit_transform(self, train_texts, test_texts):
        # Load vectors
        print(f"   GloVe: Loading vectors dari {self.vectors_path}...")
        self._load_vectors()

        train_tokenized = [text.split() for text in train_texts]
        test_tokenized = [text.split() for text in test_texts]

        X_train = self._texts_to_vectors(train_tokenized)
        X_test = self._texts_to_vectors(test_tokenized)
        self._is_fitted = True

        # OOV stats
        all_tokens = set(t for tokens in train_tokenized for t in tokens)
        oov = sum(1 for t in all_tokens if t not in self.word_vectors)
        print(f"   GloVe: vocab loaded={len(self.word_vectors)}")
        print(f"   GloVe: OOV rate = {oov}/{len(all_tokens)} ({oov/max(len(all_tokens),1)*100:.1f}%)")
        print(f"   GloVe: shape=({X_train.shape[0]}, {X_train.shape[1]})")

        return X_train, X_test

    def _load_vectors(self):
        """Load word vectors dari file .vec (format teks)."""
        self.word_vectors = {}
        if not os.path.exists(self.vectors_path):
            raise FileNotFoundError(
                f"File vectors tidak ditemukan: {self.vectors_path}. "
                "Unduh dulu dengan `python scripts/download_cc_id_300_vec.py`."
            )
        with open(self.vectors_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip header line (jika ada)
            first_line = f.readline().strip().split()
            if len(first_line) == 2:
                pass  # Header: vocab_size dim
            else:
                # Bukan header, parse sebagai word vector
                word = first_line[0]
                try:
                    vec = np.array(first_line[1:], dtype=np.float32)
                    if len(vec) == self.dim:
                        self.word_vectors[word] = vec
                except ValueError:
                    pass

            for line_num, line in enumerate(f):
                if line_num >= self.max_vocab:
                    break
                parts = line.strip().split()
                if len(parts) != self.dim + 1:
                    continue
                word = parts[0]
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                    self.word_vectors[word] = vec
                except ValueError:
                    continue

    def _texts_to_vectors(self, tokenized_texts):
        """Mean pooling of word vectors."""
        vectors = np.zeros((len(tokenized_texts), self.dim))
        for i, tokens in enumerate(tokenized_texts):
            valid_vectors = [self.word_vectors[t] for t in tokens if t in self.word_vectors]
            if valid_vectors:
                vectors[i] = np.mean(valid_vectors, axis=0)
        return vectors


# ======================================================================
# 5. FastText (Prompt 3.3)
# ======================================================================

class FastTextExtractor(BaseExtractor):
    """
    FastText Feature Extraction.

    Keunggulan: subword information — dapat menangani typo dan OOV words
    yang sangat umum di teks ulasan informal.
    """

    def __init__(self, pretrained_path: Optional[str] = None,
                 vector_size: int = 100,
                 train_from_scratch: bool = True):
        super().__init__(name='FastText')
        self.pretrained_path = pretrained_path
        self.vector_size = vector_size
        self.train_from_scratch = train_from_scratch
        self.model = None

    def fit_transform(self, train_texts, test_texts):
        train_tokenized = [text.split() for text in train_texts]
        test_tokenized = [text.split() for text in test_texts]

        if self.pretrained_path and os.path.exists(self.pretrained_path):
            # Load pre-trained FastText binary
            import fasttext
            self.model = fasttext.load_model(self.pretrained_path)
            self.vector_size = self.model.get_dimension()
            print(f"   FastText: loaded pre-trained (dim={self.vector_size})")
            X_train = self._texts_to_vectors_pretrained(train_texts)
            X_test = self._texts_to_vectors_pretrained(test_texts)
        else:
            # Train from scratch using gensim
            from gensim.models import FastText as GensimFastText
            self.model = GensimFastText(
                sentences=train_tokenized,
                vector_size=self.vector_size,
                window=5,
                min_count=2,
                workers=4,
                epochs=20,
                min_n=3,  # minimum subword length
                max_n=6   # maximum subword length
            )
            print(f"   FastText: trained from scratch, vocab={len(self.model.wv)}")
            X_train = self._texts_to_vectors_gensim(train_tokenized)
            X_test = self._texts_to_vectors_gensim(test_tokenized)

        self._is_fitted = True
        print(f"   FastText: shape=({X_train.shape[0]}, {X_train.shape[1]})")

        return X_train, X_test

    def _texts_to_vectors_pretrained(self, texts):
        """Menggunakan model fasttext official."""
        vectors = np.zeros((len(texts), self.vector_size))
        for i, text in enumerate(texts):
            vectors[i] = self.model.get_sentence_vector(text)
        return vectors

    def _texts_to_vectors_gensim(self, tokenized_texts):
        """Menggunakan model gensim FastText (mendukung OOV via subwords)."""
        vectors = np.zeros((len(tokenized_texts), self.vector_size))
        for i, tokens in enumerate(tokenized_texts):
            valid_vectors = []
            for t in tokens:
                try:
                    valid_vectors.append(self.model.wv[t])
                except KeyError:
                    continue
            if valid_vectors:
                vectors[i] = np.mean(valid_vectors, axis=0)
        return vectors


# ======================================================================
# 6. DistilBERT (Prompt 3.4)
# ======================================================================

class DistilBERTExtractor(BaseExtractor):
    """
    DistilBERT Feature Extraction (Frozen Encoder).

    Menggunakan [CLS] token embedding sebagai fixed feature vector.
    Model: distilbert-base-multilingual-cased
    """

    def __init__(self, model_name: str = 'distilbert-base-multilingual-cased',
                 max_length: int = 128,
                 batch_size: int = 32):
        super().__init__(name='DistilBERT')
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

    def fit_transform(self, train_texts, test_texts):
        import torch
        from transformers import AutoTokenizer, AutoModel

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   DistilBERT: device={device}, model={self.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(device)
        model.eval()

        X_train = self._extract_embeddings(train_texts, tokenizer, model, device)
        X_test = self._extract_embeddings(test_texts, tokenizer, model, device)
        self._is_fitted = True

        print(f"   DistilBERT: shape=({X_train.shape[0]}, {X_train.shape[1]})")

        # Cleanup GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return X_train, X_test

    def _extract_embeddings(self, texts, tokenizer, model, device):
        """Batch extraction of [CLS] embeddings."""
        import torch

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)

            # [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

            if (i // self.batch_size) % 10 == 0:
                print(f"      Batch {i//self.batch_size + 1}/"
                      f"{(len(texts)-1)//self.batch_size + 1}")

        return np.vstack(all_embeddings)


# ======================================================================
# 7. BERT / IndoBERT (Prompt 3.7)
# ======================================================================

class BERTExtractor(BaseExtractor):
    """
    BERT / IndoBERT Feature Extraction.

    Dua mode:
    - MODE A: [CLS] token embedding (default)
    - MODE B: Mean pooling of all token embeddings

    Model: indobenchmark/indobert-base-p1
    """

    def __init__(self, model_name: str = 'indobenchmark/indobert-base-p1',
                 max_length: int = 128,
                 batch_size: int = 32,
                 pooling: str = 'cls'):
        super().__init__(name='BERT-IndoBERT')
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling  # 'cls' atau 'mean'

    def fit_transform(self, train_texts, test_texts):
        import torch
        from transformers import AutoTokenizer, AutoModel

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   BERT: device={device}, model={self.model_name}, pooling={self.pooling}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(device)
        model.eval()

        X_train = self._extract_embeddings(train_texts, tokenizer, model, device)
        X_test = self._extract_embeddings(test_texts, tokenizer, model, device)
        self._is_fitted = True

        print(f"   BERT: shape=({X_train.shape[0]}, {X_train.shape[1]})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return X_train, X_test

    def _extract_embeddings(self, texts, tokenizer, model, device):
        import torch

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)

            if self.pooling == 'mean':
                # Mean pooling with attention mask
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()
                ).float()
                embeddings = (
                    torch.sum(token_embeddings * input_mask_expanded, 1) /
                    torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                )
            else:
                # [CLS] token
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

            if (i // self.batch_size) % 10 == 0:
                print(f"      Batch {i//self.batch_size + 1}/"
                      f"{(len(texts)-1)//self.batch_size + 1}")

        return np.vstack(all_embeddings)


# ======================================================================
# 8. RoBERTa / XLM-RoBERTa (Prompt 3.8)
# ======================================================================

class RoBERTaExtractor(BaseExtractor):
    """
    RoBERTa / XLM-RoBERTa Feature Extraction.

    Model: xlm-roberta-base (multilingual, proven baik untuk low-resource languages)
    """

    def __init__(self, model_name: str = 'xlm-roberta-base',
                 max_length: int = 128,
                 batch_size: int = 32,
                 pooling: str = 'cls'):
        super().__init__(name='RoBERTa')
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling

    def fit_transform(self, train_texts, test_texts):
        import torch
        from transformers import AutoTokenizer, AutoModel

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   RoBERTa: device={device}, model={self.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(device)
        model.eval()

        X_train = self._extract_embeddings(train_texts, tokenizer, model, device)
        X_test = self._extract_embeddings(test_texts, tokenizer, model, device)
        self._is_fitted = True

        print(f"   RoBERTa: shape=({X_train.shape[0]}, {X_train.shape[1]})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return X_train, X_test

    def _extract_embeddings(self, texts, tokenizer, model, device):
        import torch

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)

            if self.pooling == 'mean':
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()
                ).float()
                embeddings = (
                    torch.sum(token_embeddings * input_mask_expanded, 1) /
                    torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                )
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


# ======================================================================
# Registry — Semua Extractors
# ======================================================================

def get_all_extractors(subset: str = 'priority') -> dict:
    """
    Return dictionary of all feature extractors.

    Args:
        subset: 'priority' untuk subset prioritas (TF-IDF, FastText, IndoBERT)
                'all' untuk semua 8 extractors

    Returns:
        Dict[str, BaseExtractor]
    """
    if subset == 'priority':
        return {
            'GloVe': GloVeExtractor(),
            'FastText': FastTextExtractor(train_from_scratch=True),
            'Word2Vec': Word2VecExtractor(train_from_scratch=True),
        }
    else:
        return {
            'TF-IDF': TFIDFExtractor(max_features=10000, ngram_range=(1, 2)),
            'BM25': BM25Extractor(),
            'Word2Vec': Word2VecExtractor(vector_size=100, train_from_scratch=True),
            'GloVe': GloVeExtractor(vectors_path='data/embeddings/cc.id.300.vec'),
            'FastText': FastTextExtractor(vector_size=100, train_from_scratch=True),
            'DistilBERT': DistilBERTExtractor(),
            'IndoBERT': BERTExtractor(
                model_name='indobenchmark/indobert-base-p1',
                pooling='cls'
            ),
            'RoBERTa': RoBERTaExtractor(),
        }
