#!/usr/bin/env python
# coding: utf-8

# In[3]:


from dictionary import *
def my_result(str):
    import numpy as np
    import tensorflow as tf
    import emoji
   
    X=[]
    X.append(str)
    X1=np.array(X)
    X1=sentences_to_indices(X1, word_to_index, 10)
    model = tf.keras.models.load_model("./emogi_suggestor_model")
    y=np.argmax(model.predict(X1), axis=1)
    Y=[emoji.emojize(":sparkling_heart:"),emoji.emojize(":soccer_ball:"),emoji.emojize(":grinning_face:"),emoji.emojize(":disappointed_face:"),emoji.emojize(":fork_and_knife_with_plate:")]
    
    return (X[0]+Y[y[0]])


# In[4]:


r=my_result('i love you baby')


# In[5]:


r


# In[ ]:





# In[ ]:




