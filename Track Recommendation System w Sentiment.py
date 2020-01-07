


import pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import gensim
from skimage import io
from gensim.test.utils import get_tmpfile
import pandas as pd
import re
import os
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import gensim
import logging
import pyLDAvis.gensim
import warnings
from os import system
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 



#uploading pickled csv that contains lyrics, lyrics with bigrams and bigrams

df = pd.read_pickle("Pickled/cleaned4.pkl")



#cleaning inputted string (lowercase, tokenize, removing punctuation, stopwords, replacing with bigrams)
from string import punctuation as punc
punc= punc + '’'

def filteringinput(input):
    filtered_sentence = []
    itwill = []
    words = word_tokenize(stringy.lower())
    baby = df['list of bigrams'].tolist()[0]
    babies = df['list of joined bigrams'].tolist()[0]
    for i in words:
        if i not in stopwords.words('english') and i not in punc:
            filtered_sentence.append(i)
            filter_string = ' '.join(filtered_sentence)
    for x,y in zip(baby, babies):
            hello = filter_string.replace(x, y)
            itwill.append(hello)
            finale = itwill[0]
    return ''.join(finale)

#function for sentiment scores


def sentiment_scores(sentence): 
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
    return sentiment_dict['compound']



# example user input
stringy = 'Didn’t mean to call my mom just go cry to her in the bathroom at work!'
data = [['1U', filteringinput(stringy), sentiment_scores(filteringinput(stringy))]] 

# Create the pandas DataFrame 

duser = pd.DataFrame(data, columns = ['trackid', 'bigram replace', 'sentiment score']) 



# import TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer


#tfidf vectorization

tfidf_vectorizer = TfidfVectorizer()
tfidf_jobid = tfidf_vectorizer.fit_transform((df['bigram replace'])) #fitting and transforming the vector
from sklearn.metrics.pairwise import cosine_similarity
user_tfidf = tfidf_vectorizer.transform(duser['bigram replace'])
cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

#create list of cosine similarity scores
output2 = list(cos_similarity_tfidf)


#recomendation dataframe
#returns dataframe of top songs sorted by cosine similarity

top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[0:30]

def get_recommendation(top, df_all, scores):
    recommendation = pd.DataFrame(columns = ['tracknames', 'artists', 'score'])
    count = 0
    for i in top:
        recommendation.at[count, 'trackid'] =df['trackid'][i]
        recommendation.at[count, 'bigram replace'] = df['bigram replace'][i]
        recommendation.at[count, 'tracknames'] = df['tracknames'][i]
        recommendation.at[count, 'artists'] = df['artists'][i]
        recommendation.at[count, 'sentiment'] =  df['sentiment'][i]
        recommendation.at[count, 'lyrics'] =  df['lower lyrics'][i]
        recommendation.at[count, 'sentiment score'] =  df['sentiment score'][i]
        recommendation.at[count, 'embed link'] =  df['embed link'][i]
        recommendation.at[count, 'original lyrics'] =  df['original lyrics'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    return recommendation




#list of cosine similarity scores

list_scores = [output2[i][0][0] for i in top]




#recomendation df without sentiment threshold
recdf = get_recommendation(top, df, list_scores)

#reccomendation df with sentiment threshold
reccell = recdf[(recdf['sentiment score'] < duser['sentiment score'][0] +.35) & (recdf['sentiment score'] > duser['sentiment score'][0] -.48)  ]







