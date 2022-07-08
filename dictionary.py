#!/usr/bin/env python
# coding: utf-8

# In[2]:


import io
import numpy as np


# In[3]:


value=[]
embeddings_index = {}
with io.open('glove.6B.50d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        value.append(values)
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embeddings_index[word] = coefs


# In[4]:


word_to_index={}
index_to_word={}
for i,v in enumerate(value):
    word_to_index[v[0]]=i
    index_to_word[i]=v[0]


# In[5]:


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0] 
    X_indices = np.zeros([m,max_len])
    for i in range(m): 
        sentence_words=X[i].lower().split()
        j=0
        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                j =  j+1
    return X_indices


# In[ ]:




