import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import spacy
import re
import pandas as pd
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    """
    Preprocesses the input text by removing stopwords, lemmatizing words, and 
    performing other text normalization steps.

    Parameters:
    - text (str): The input text to be preprocessed.

    Returns:
    - str: The preprocessed text.
    """
    nlp = spacy.load("de_core_news_sm")

    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('german')) 
    # Lowercase only if the entire word is not in uppercase
    if text.isupper():
        return None
    
    # Punctuation
    processed_text = text.replace('-', ' ')
    processed_text = processed_text.replace(' - ', ' ').replace('-', '')
    # special tokens
    processed_text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]+|\d+', '', processed_text)

    # Tokenize the text
    tokens = word_tokenize(processed_text)

    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a single string
    return ' '.join(tokens)

def custom_on_bad_lines(line):
    return [line[0], ' - '.join(line[1:])]

def load_df(): 
    """
    Loads data from text files stored in a specified folder and combines it 
    into a single DataFrame.

    Returns:
    - pd.DataFrame: The combined DataFrame containing the loaded data.
    """

    folder_path = '/content/Parsed_headlines'

    columns = ['date', 'headline']
    combined_df = pd.DataFrame(columns=columns)

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            name = file_name.split('.')[0].split("_")[1:]
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, delimiter=' - ', names=columns, header=None, engine='python', on_bad_lines=custom_on_bad_lines)
            df['paper'] = ' '.join(name)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    df = combined_df
    df.dropna(axis=0, how='any', inplace=True)
    df =df.drop_duplicates(subset = ['headline'], keep='first')
    return df

def encode_headlines(headlines, model):
    """
    Encodes headlines using a specified Sentence Transformer model.

    Parameters:
    - headlines (list): A list of headlines to be encoded.
    - model: A Sentence Transformer model for encoding headlines.

    Returns:
    - np.array: An array of embeddings representing the encoded headlines.
    """
    embeddings = model.encode(headlines, convert_to_numpy=True)
    return embeddings

def compute_silhouette_per_cluster_number(start,end, headline_embeddings):
    """
    Computes the silhouette score for a range of cluster numbers.

    Parameters:
    - start (int): The starting number of clusters.
    - end (int): The ending number of clusters.
    - headline_embeddings (np.array): An array of embeddings representing headlines.

    Returns:
    - None
    """
    for num_clusters in range(start,end):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(headline_embeddings)

        silhouette_avg = silhouette_score(headline_embeddings, kmeans.labels_)

        print("The average silhouette score for", num_clusters, "is:", silhouette_avg)

def compute_elbow_plot(min_clusters, max_clusters, headline_embeddings):
    """
    Computes and plots the elbow plot to determine the optimal number of clusters.

    Parameters:
    - min_clusters (int): The minimum number of clusters to consider.
    - max_clusters (int): The maximum number of clusters to consider.
    - headline_embeddings (np.array): An array of embeddings representing headlines.

    Returns:
    - None
    """
    inertia_values = []

    for k in range(min_clusters, max_clusters + 1,10):
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(headline_embeddings)

        # Compute inertia value and append to list
        inertia_values.append(kmeans.inertia_)

    # Plot the elbow plot
    plt.plot(range(min_clusters, max_clusters + 1,10), inertia_values, marker='o', label='Inertia')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Value')
    plt.title('Elbow Plot for Optimal Number of Clusters')
    plt.legend()
    plt.show()