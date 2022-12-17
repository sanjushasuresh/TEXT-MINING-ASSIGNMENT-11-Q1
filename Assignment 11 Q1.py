# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:43:57 2022

@author: SANJUSHA
"""

# pip install textblob
# pip install wordcloud
# pip install spacy
import pandas as pd
import numpy as np
import string
import re
import nltk
from textblob import TextBlob
nltk.download('punkt')
nltk.download('stopwords') 
from nltk.corpus import stopwords

# Sentimental analysis
df=pd.read_csv("Elon_musk.csv",encoding='latin-1')
df
df.head()
df["Text"]
stp_words=stopwords.words("english")
print(stp_words)

# Cleaning the tweets
one_tweet=df.iloc[4]["Text"]

def TweetCleaning(tweets):
 cleantweet=re.sub(r"@[a-zA-Z0-9]+"," ",tweets)
 cleantweet=re.sub(r"#[a-zA-Z0-9]+"," ",cleantweet)
 cleantweet=''.join(word for word in cleantweet.split() if word not in stp_words)
 return cleantweet

def calpolarity(tweets):
    return TextBlob(tweets).sentiment.polarity

def calSubjectivity(tweets):
    return TextBlob(tweets).sentiment.subjectivity

def segmentation(tweets):
    if tweets > 0:
        return "positive"
    if tweets== 0:
        return "neutral"
    else:
        return "negative"


df["cleanedtweets"]=df['Text'].apply(TweetCleaning)
df['polarity']=df["cleanedtweets"].apply(calpolarity)
df['subjectivity']=df["cleanedtweets"].apply(calSubjectivity)
df['segmentation']=df["polarity"].apply(segmentation)

df.head()

# Analysis and visualization
df.pivot_table(index=['segmentation'],aggfunc={"segmentation":'count'})
# The positive tweets are 82
# The negative tweets are 3
# The neutral tweets are 1914

# Top three positive tweets
df.sort_values(by=['polarity'],ascending=False).head(3)

# Top three negative tweets
df.sort_values(by=['polarity'],ascending=True).head(3)

# Top three neutral tweets
df['polarity']==0
df[df['polarity']==0].head(3)

df["cleanedtweets"]

# Joining the list into one string/text
text = ' '.join(df["cleanedtweets"])
text

# Punctuation
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) 
no_punc_text

# Tokenization
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])

# Removing stopwords
my_stop_words = stopwords.words('english')
no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]

# Noramalize the data
lower_words = [x.lower() for x in no_stop_tokens]
print(lower_words[0:40])

# Stemming the data
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:10])


import spacy
nlp=spacy.load("en_core_web_sm")
# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(no_stop_tokens))
print(doc[0:40])

lemmas = [token.lemma_ for token in doc]
print(lemmas[0:25])


# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stemmed_tokens)
pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=False).head(20)
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names()[50:100])
print(X.toarray()[50:100])
print(X.toarray().shape)

# Bigrams and Trigrams 
vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 100)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(df["cleanedtweets"])
bow_matrix_ngram
print(vectorizer_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())

# TFidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 10)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(df["cleanedtweets"])
print(vectorizer_n_gram_max_features.get_feature_names())
print(tf_idf_matrix_n_gram_max_features.toarray())


# Wordcloud
import matplotlib.pyplot as plt
# %matplotlib inline
from wordcloud import WordCloud

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(15, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
    
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2').generate(text)
plot_cloud(wordcloud)
plt.show()
