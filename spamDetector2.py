# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:34:09 2018

@author: rishabh
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import wordcloud as w

df=pd.read_csv('E:/RS/ML/Machine learning tuts/Target/Part4(NLP)/NlP-1/spam.csv', encoding='ISO-8859-1')

df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

df.columns=['labels','data']

df['b_labels']=df['labels'].map({'ham':0,'spam':1})
Y=df['b_labels'].as_matrix()

count_vectorizer=CountVectorizer(decode_error='ignore')
X=count_vectorizer.fit_transform(df['data'])

#A=count_vectorizer.get_feature_names()
#B=X.toarray()
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.33)

#create model
model=MultinomialNB()
model.fit(Xtrain,Ytrain)
print("Train Score",model.score(Xtrain,Ytrain))
print("Test Score",model.score(Xtest,Ytest))

cost=[]

#visualize the data
def visualize(label):
  words=''
  for msg in df[df['labels']==label]['data']:
     cost.append(msg)
     msg=msg.lower()
     words += msg + ' '
  wordCloud=w.WordCloud(width=600,height=400).generate(words)
  plt.imshow(wordCloud)
  plt.axis('off')
  plt.show()

visualize('spam') 
visualize('ham') 

#what we are getting wrong
df['predictions']=model.predict(X)

sneaky_spam=df[(df['predictions']==0)&(df['b_labels']==1)]['data']

for msg in sneaky_spam:
  print(msg)  
  print('\n')          
  
not_spam=df[(df['predictions']==1)&(df['b_labels']==0)]['data']
for msg in not_spam:
  print(msg)
  print('\n')
  
