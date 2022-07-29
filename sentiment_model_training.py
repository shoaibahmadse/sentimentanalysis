
# coding: utf-8

# ## Plan of Action
#
#
# 1.   We are using **Amazon Alexa Reviews dataset (3150 reviews)**, that contains: **customer reviews, rating out of 5**, date of review, Alexa variant
# 2.   First we  **generate sentiment labels: positive/negative**, by marking *positive for reviews with rating >3 and negative for remaining*
# 3. Then, we **clean dataset through Ventorization Feature Engineering** (TF-IDF) - a popular technique
# 4. Post that, we use **Support Vector Classifier for Model Fitting** and check for model performance (*we are getting >90% accuracy*)
# 5. Last, we use our model to do **predictions on real Amazon reviews** using: a simple way and then a fancy way
#
#

# ## Import datasets

# In[1]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from custom_tokenizer_function import CustomTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# get_ipython().system('pip install spacy')
import spacy


# In[3]:


# In[4]:


nlp = spacy.load('en_core_web_sm')


# Set your working directory  here
#
# If you are using Google Colab, use this code snippet:
#
# ```from google.colab import drive
# drive.mount('/content/drive')```
#
# ```%cd /content/drive/My Drive/Project6_SentimentAnalysis_with_Pipeline```
#
# If you are working locally on PC, keep training data in the same directory as this code file

# In[5]:


# Loading the dataset
dump = pd.read_csv('alexa_reviews_dataset.tsv', sep='\t')

dump


# ## Data Preparation

# In[6]:


dataset = dump[['verified_reviews', 'rating']]
dataset.columns = ['Review', 'Sentiment']

dataset.head()


# In[7]:


# Creating a new column sentiment based on overall ratings
def compute_sentiments(labels):
    sentiments = []
    for label in labels:
        if label > 3.0:
            sentiment = 1
        elif label <= 3.0:
            sentiment = 0
        sentiments.append(sentiment)
    return sentiments


# In[8]:


dataset['Sentiment'] = compute_sentiments(dataset.Sentiment)


# In[9]:


dataset.head()


# In[10]:


# check distribution of sentiments

dataset['Sentiment'].value_counts()


# In[11]:


# check for null values
dataset.isnull().sum()

# no null values in the data


# ### Data Cleaning

# In[12]:


x = dataset['Review']
y = dataset['Sentiment']


# In[13]:


# Create a function to clean data
# We shall remove stopwords, punctuations & apply lemmatization


# In[14]:


# import string
# from spacy.lang.en.stop_words import STOP_WORDS


# In[15]:


# # creating a function for data cleaning

# def text_data_cleaning(sentence):
#   doc = nlp(sentence)                         # spaCy tokenize text & call doc components, in order

#   tokens = [] # list of tokens
#   for token in doc:
#     if token.lemma_ != "-PRON-":
#       temp = token.lemma_.lower().strip()
#     else:
#       temp = token.lower_
#     tokens.append(temp)

#   cleaned_tokens = []
#   for token in tokens:
#     if token not in stopwords and token not in punct:
#       cleaned_tokens.append(token)
#   return cleaned_tokens


# In[16]:


# if root form of that word is not pronoun then it is going to convert that into lower form
# and if that word is a proper noun, then we are directly taking lower form,
# because there is no lemma for proper noun


# In[17]:


# let's do a test
custom_tokenizer = CustomTokenizer()
custom_tokenizer.text_data_cleaning(
    "Hello all, It's a beautiful day outside there!")
# stopwords and punctuations removed


# ### Vectorization Feature Engineering (TF-IDF)

# In[18]:


# In[19]:


tfidf = TfidfVectorizer(tokenizer=custom_tokenizer.text_data_cleaning)
# tokenizer=text_data_cleaning, tokenization will be done according to this function


# ## Train the model

# ### Train/ Test Split

# In[20]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=dataset.Sentiment, random_state=0)


# In[21]:


x_train.shape, x_test.shape
# 2520 samples in training dataset and 630 in test dataset


# ### Fit x_train and y_train

# In[22]:


classifier = LinearSVC()


# In[23]:


pipeline = Pipeline([('tfidf', tfidf), ('clf', classifier)])
# it will first do vectorization and then it will do classification


# In[24]:


pipeline.fit(x_train, y_train)


# In[25]:


# in this we don't need to prepare the dataset for testing(x_test)


# In[26]:


joblib.dump(pipeline, 'sentiment_model.pkl')


# ## Check Model Performance

# In[27]:


# In[28]:


y_pred = pipeline.predict(x_test)


# In[29]:


# confusion_matrix
confusion_matrix(y_test, y_pred)


# In[30]:


# classification_report
print(classification_report(y_test, y_pred))
# we are getting almost 91% accuracy


# In[31]:


round(accuracy_score(y_test, y_pred)*100, 2)


# ## Predict Sentiments using Model

# ### Simple way

# In[32]:


# prediction = pipeline.predict(["Alexa is bad"])

# if prediction == 1:
#   print("Result: This review is positive")
# else:
#   print("Result: This review is negative")


# ### Fancy way

# In[33]:


# new_review = []
# pred_sentiment = []

# while True:

#   # ask for a new amazon alexa review
#   review = input("Please type an Alexa review (Type 'skip' to exit) - ")

#   if review == 'skip':
#     print("See you soon!")
#     break
#   else:
#     prediction = pipeline.predict([review])

#     if prediction == 1:
#       result = 'Positive'
#       print("Result: This review is positive\n")
#     else:
#       result = 'Negative'
#       print("Result: This review is negative\n")

#   new_review.append(review)
#   pred_sentiment.append(result)


# In[34]:


# Results_Summary = pd.DataFrame(
#     {'New Review': new_review,
#      'Sentiment': pred_sentiment,
#     })

# Results_Summary.to_csv("./predicted_sentiments.tsv", sep='\t', encoding='UTF-8', index=False)
# Results_Summary
