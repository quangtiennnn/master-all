import os
import pickle
import csv
import numpy as np
import pandas as pd

def save_corpus_embedding(corpus_embedding, file_name):
    # Ensure the corpus_embedding is in the correct format
    if isinstance(corpus_embedding, list):
        if not all(isinstance(i, list) for i in corpus_embedding):
            raise ValueError("Each element in the corpus_embedding list should be a list.")
    elif isinstance(corpus_embedding, np.ndarray):
        corpus_embedding = corpus_embedding.tolist()
    else:
        raise ValueError("Corpus embedding should be a list of lists or a numpy array.")

    # Open the file with write mode
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write each embedding as a row in the CSV file
        for embedding in corpus_embedding:
            writer.writerow(embedding)
# save_corpus_embedding(corpus_embeddings, 'embeddings/vietnamese_vi_corpus_embedding.csv')

def load_corpus_embedding(file_name):
    corpus_embedding = []

    # Open the file with read mode
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        
        # Read each row and convert to a list of floats
        for row in reader:
            corpus_embedding.append([float(value) for value in row])
    
    return corpus_embedding
# load_corpus_embeddings = load_corpus_embedding('embeddings/vietnamese_vi_corpus_embedding.csv')

def save_model(model, model_name):
    # Ensure the 'models' directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    # Define the full path for the model file
    file_path = os.path.join('models', model_name + '.pkl')
    # Save the model using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def load_model(model_name):
    # Define the full path for the model file
    file_path = os.path.join('models', model_name + '.pkl')
    # Load the model using pickle
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# 
from umap import UMAP
from hdbscan import HDBSCAN
from stopwordsiso import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech


# vectorizer_model=CountVectorizer(stop_words=list(stop_word_vi), min_df=2, ngram_range=(1,2))
# Stop words
stop_word_mul = stopwords(['vi','en'])

# UMAP
umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

# HDBSCAN
hdbscan_model=HDBSCAN(min_cluster_size=80, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Count Vectorizer
vectorizer_model=CountVectorizer(stop_words=list(stop_word_mul), min_df=2, ngram_range=(1,2))

# Presentation 
keybert_model=KeyBERTInspired()
pos_model=PartOfSpeech('en_core_web_sm')
mmr_model=MaximalMarginalRelevance(diversity=0.3)

representation_model={
    'KeyBERT':keybert_model,
    'MMR':mmr_model,
    'POS':pos_model
}