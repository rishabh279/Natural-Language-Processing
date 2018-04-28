# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:09:35 2018

@author: rishabh
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd 
import numpy as np

data = pd.read_csv('').as_matrix()
np.random.shuffle(data)

X=data[:,:48]
Y=data[:,-1]

Xtrain=X[:-100]
Xtest=X[-100:]
Ytrain=Y[:-100]
Ytest=Y[-100:]

model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("Classification Rate for NB",model.score(Xtest,Ytest))

from sklearn.ensemble import AdaBoostClassifier

model=AdaBoostClassifier()
model.fit(Xtrain,Ytrain)
print("Classification Rate for Ada",model.score(Xtest,Ytest))