import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import re

def perform_lda_analysis(csv_path, num_topics=5, num_words=10):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter negative reviews only
    neg_reviews = df[df['sentiment'] == 'Negatif']['review_clean'].fillna('').tolist()
    
    # Tokenize
    texts = [text.split() for text in neg_reviews if len(text.split()) > 2]
    
    # Create Dictionary
    dictionary = corpora.Dictionary(texts)
    
    # Filter extremes (words appearing in < 5 reviews or > 50% of reviews)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    # Create Corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Build LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, 
                         num_topics=num_topics, random_state=42, 
                         passes=10, alpha='auto')
    
    # Extract topics
    topics = lda_model.print_topics(num_words=num_words)
    
    return topics, lda_model

if __name__ == "__main__":
    print("\n🔍 Menjalankan LDA Topic Modeling pada ulasan Negatif...")
    topics, model = perform_lda_analysis('data/processed/reviews_prepared.csv')
    
    print("\n🏆 Topik Utama Keluhan Pengguna (Negatif):")
    print("=" * 60)
    for idx, topic in topics:
        print(f"Topik #{idx+1}: {topic}")
    print("=" * 60)
