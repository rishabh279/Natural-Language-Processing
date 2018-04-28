# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:44:24 2018

@author: rishabh
"""

import nltk
import random
import numpy as np

from bs4 import BeautifulSoup

positive_reviews=BeautifulSoup(open('').read(),"lxml")
positive_reviews=positive_reviews.findAll('review_text')

trigrams={}
for review in positive_reviews:
  s=review.text.lower()
  tokens=nltk.tokenize.word_tokenize(s)
  for i in range(len(tokens)-2):
    k=(tokens[i],tokens[i+2])
    if k not in trigrams:
      trigrams[k]=[]
    trigrams[k].append(tokens[i+1])

middle=[]
s=positive_reviews[0].text.lower()
tokens=nltk.tokenize.word_tokenize(s)
for i in range(len(tokens)-2):
  k=(tokens[i],tokens[i+2])
  middle.append(k)
  if k not in trigrams:
    trigrams[k]=[]
  trigrams[k].append(tokens[i+1])
  
trigram_probabilities = {}

for k, words in trigrams.items():
  # create a dictionary of word -> count
  if len(set(words)) > 1:
      # only do this when there are different possibilities for a middle word
      d = {}
      n = 0
      for w in words:
          if w not in d:
              d[w] = 0
          d[w] += 1
          n += 1
      for w, c in d.items():
          d[w] = float(c) / n
      trigram_probabilities[k] = d
