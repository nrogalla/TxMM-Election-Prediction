import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import os
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from NewsSentiment import TargetSentimentClassifier
import numpy as np
from tika import parser
from googletrans import Translator
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

def split_headline(headline, topics):
    """
    Splits a headline into segments based on the presence of topics.

    Parameters:
    - headline (str): The headline text to be split.
    - topics (list): A list of topics to search for in the headline.

    Returns:
    - tuple: A tuple containing segments of the headline separated by topics.
    """
    headline = headline.lower()
    for topic in topics:
        topic = topic.lower()
        if topic in headline:
            parts = headline.split(topic)
            # Join the parts with the topic and include the topic separately
            segments = [part.strip() for part in parts if part.strip()]
            if headline.startswith(topic):
                return ("", topic, segments[0])
            elif headline.endswith(topic):
                return ("", segments[0], topic)
            else:
                return (segments[0], topic, segments[1])
    # If no topic is found, return the original headline as a single segment
    return (headline.strip(), None, None)

def get_score(label, prob):
  
  """  Calculates a score based on the label and probability."""
  if label == 'positive':
    return prob
  if label =='neutral':
    return 0
  return -prob


def get_topics_from_file(filename):
  """
    Reads topics from a file and returns them as a list.

    Parameters:
    - filename (str): The path to the file containing topics.

    Returns:
    - list: A list of topics read from the file.
    """
  topics = []

  # Open the text file in read mode
  with open(filename, 'r') as file:
      # Read each line of the file
      for line in file:
          # Split the line by commas to get individual topics
          line_topics = line.strip().split(', ')

          # Add the topics from the current line to the overall list of topics
          topics.extend(line_topics)
  return topics

def get_headline_sentiment(headline):
    """Computes headline sentiment based on VADER"""
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(headline)
    return sentiment_scores['compound']


def preprocess_english_text(text):
    """
    Preprocesses English text by lowercasing, removing punctuation, 
    special characters, numbers, stopwords, and lemmatizing words.

    Parameters:
    - text (str): The English text to be preprocessed.

    Returns:
    - str: The preprocessed text.
    """
    ...
    # Lowercasing
    text = text.lower()

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Removing special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]


    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def get_headline_sentiments(df): 
    """
    Computes sentiment scores for headlines in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing headlines.

    Returns:
    - None
    """
    tsc = TargetSentimentClassifier()
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    tokenizer.src_lang = "de"
    keywords_path = "/content/keywords_en/"

    # Get the list of keyword files in the keywords_path directory
    keyword_files = os.listdir(keywords_path)
    for topic_file in keyword_files:
        df[f"sentiment_{os.path.splitext(topic_file)[0]}"] = None

    df['sentiment_whole'] = None
    df['headline_en_prep'] = None
    for index, row in df.iterrows():
        # translate headline to english
        encoded_text = tokenizer(row['headline'], return_tensors="pt")
        generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
        english_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        english_text = preprocess_english_text(english_text[0])
        df.loc[index, "headline_en_prep"]= english_text

        for topic_file in keyword_files:
            topic = os.path.splitext(topic_file)[0]  # Extracting topic name without extension
            sentiments = []
            try:
                # headline splitting needed for targeted sentiment analysis
                headline_split = split_headline(english_text, get_topics_from_file(os.path.join(keywords_path, topic_file)))
            except:
                continue
            # get targeted sentment
            if not any(element is None for element in headline_split):
                data = [headline_split]

                sentiments = tsc.infer(targets=data)
                sum_score = 0
                for i, result in enumerate(sentiments):
                    sum_score += get_score(result[0]['class_label'], result[0]['class_prob'])
                df.loc[index, f"sentiment_{topic}"] = sum_score / len(sentiments)
        # vader sentiment computed for all headlines
        df.loc[index, f"sentiment_whole"] = get_headline_sentiment(english_text)


def extract_text_from_pdf(pdf_path):

    raw = parser.from_file(pdf_path)
    return raw['content']

def split_text_into_paragraphs(text):
    """
    Splits the input text into paragraphs, translates each paragraph from German to English,
    preprocesses the English text, and returns a list of English paragraphs.

    Args:
        text (str): The input text to be split into paragraphs.

    Returns:
        list of str: List of English paragraphs after preprocessing.
    """
    translator = Translator()
    paragraphs1 = text.split('\n\n')
    paragraphs2 = []
    for paragraph in paragraphs1:
        paragraphs2.extend(paragraph.split('. '))
    paragraphs = []
    for paragraph in paragraphs2:
        paragraphs.extend(re.split(r'(\d+) \n', paragraph))

    paragraphs = [string for string in paragraphs if not string.strip().replace(" ", "").isdigit()]#[string for string in paragraphs if not string == "" and not string.isdigit()]

    paragraphs = list(filter(lambda x: x.strip(), paragraphs))

    paragraphs = [item.replace('\n', '').replace('-', '') for item in paragraphs]
    english_paragraphs = []
    for paragraph in paragraphs:
        english_text = translator.translate(paragraph, src = 'de', dest='en').text
        english_paragraphs.append(preprocess_english_text(english_text))
    return english_paragraphs


def get_programme_sentiments():
  """
    Extracts sentiments from political program documents. Processes each program document by splitting
    it into paragraphs, and then for each paragraph, extracts sentiments based on predefined topics.
    The sentiment scores are averaged and returned in a DataFrame.

    Returns:
        DataFrame: DataFrame containing sentiment scores for each political program document.
"""
  tsc = TargetSentimentClassifier()
  df = pd.DataFrame(columns=["programme"])

  programmes_paths = "/content/programmes/"
  keywords_path = "/content/keywords_en/"

  # List files in the programmes directory
  programme_files = os.listdir(programmes_paths)

  for i in range(len(programme_files)):
      df.loc[i, 'programme'] = os.path.splitext(programme_files[i])[0]

  # Get the list of keyword files in the keywords_path directory
  keyword_files = os.listdir(keywords_path)

  # Create columns for each sentiment type
  for topic_file in keyword_files:
      topic = os.path.splitext(topic_file)[0]
      df[f"sentiment_{topic}"] = 0

  # Process each row (programme)
  for index, row in df.iterrows():
      # Extract text from the corresponding PDF file
      try:
        text = extract_text_from_pdf(os.path.join(programmes_paths, row['programme'] + ".pdf"))
      except:
        continue

      # Split text into paragraphs
      paragraphs = split_text_into_paragraphs(text)
      print(len(paragraphs))

      sentiments_df = {}

      # Process each paragraph
      for paragraph in paragraphs:
          # Process each topic (sentiment type)
          for topic_file in keyword_files:
              topic = os.path.splitext(topic_file)[0]

              try:
                headline_split = split_headline(paragraph, get_topics_from_file(os.path.join(keywords_path, topic_file)))
              except:
                continue
              
              if not any(element is None for element in headline_split):
                  data = [headline_split]
                  sentiments = tsc.infer(targets=data)

                  # Calculate sentiment score
                  sum_score = 0
                  for result in sentiments:
                      sum_score += get_score(result[0]['class_label'], result[0]['class_prob'])
                  avg_score = sum_score / len(sentiments)

                  if f"sentiment_{topic}" in sentiments_df:
                  # Update sentiments DataFrame
                    sentiments_df[f"sentiment_{topic}"].append(avg_score)
                  else:
                    sentiments_df[f"sentiment_{topic}"] = [avg_score]

      for key in sentiments_df:
        df.loc[index, key] = np.average(sentiments_df[key])
        
  return df
