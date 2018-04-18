# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:57:10 2018

@author: rishabh
"""

import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup#use to read xml file as reviews are in xml

wordnet_lemmatizer = WordNetLemmatizer()#used to convert words into base form like dogs to dog, liked to like
stopwords=set(w.rstrip() for w in open('E:/RS/ML/Machine learning tuts/Target/Part4(NLP)/Code/NlP-1/stopwords.txt'))

positive_reviews=BeautifulSoup(open('E:/RS/ML/Machine learning tuts/Target/Part4(NLP)/Code/electronics/positive.review').read(),"lxml")
positive_reviews=positive_reviews.findAll('review_text')

negative_reviews=BeautifulSoup(open('E:/RS/ML/Machine learning tuts/Target/Part4(NLP)/Code/electronics/negative.review').read(),"lxml")
negative_reviews=negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews=positive_reviews[:len(negative_reviews)]

def my_tokenizer(s):
  s = s.lower() # downcase
  tokens=nltk.tokenize.word_tokenize(s)#split the string into words
  tokens=[t for t in tokens if len(t)>2]#removing short words as they are not useful
  tokens=[wordnet_lemmatizer.lemmatize(t) for t in tokens]#put words into the base form
  tokens=[t for t in tokens if t not in stopwords]#remove stopwords
  return tokens
  
word_index_map={}
current_index=0
positive_tokenized=[]
negative_tokenized=[]

for review in positive_reviews:
  tokens=my_tokenizer(review.text)
  positive_tokenized.append(tokens)
  for token in tokens:
    if token not in word_index_map:
      word_index_map[token]=current_index
      current_index+=1
  
for review in negative_reviews:
  tokens=my_tokenizer(review.text)
  negative_tokenized.append(tokens)
  for token in tokens:
    if token not in word_index_map:
      word_index_map[token]=current_index
      current_index+=1

#create input matrices
def tokens_to_vector(token,label):
  x=np.zeros(len(word_index_map)+1)
  for t in tokens:
    i=word_index_map[t]
    x[i]+=1
  x=x/x.sum()  
  x[-1]=label
  return x

N=len(positive_tokenized)+len(negative_tokenized)

data=np.zeros((N,len(word_index_map)+1))
i=0
for tokens in positive_tokenized:
  xy=tokens_to_vector(tokens,1)
  data[i,:]=xy
  i+=1
  
for tokens in negative_tokenized:
  xy=tokens_to_vector(tokens,0)
  data[i,:]=xy
  i+=1

np.random.shuffle(data)

X=data[:,:-1]
Y=data[:,-1]


Xtrain=X[:-100,]
Ytrain=Y[:-100,]
Xtest=X[-100:,]
Ytest=Y[-100:,]

logisticModel=LogisticRegression()
logisticModel.fit(Xtrain,Ytrain)
print("Classification Rate",logisticModel.score(Xtest,Ytest))

threshold=0.5
for word,index in word_index_map.items():
  weight=logisticModel.coef_[0][index]
  if weight>threshold or weight < -threshold:
    print(word,weight)