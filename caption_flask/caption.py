

#importing libraries
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt

import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# In[3]:


model=load_model('model_149.h5')


# In[4]:


model_ince = InceptionV3(weights='imagenet',input_shape=(299,299,3))


# In[5]:


# Creating a new model, by removing the last layer(output layer) from the inception v3
model_new = Model(model_ince.input, model_ince.layers[-2].output)


# In[ ]:






# We're converting our image size 299x299
def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


# In[7]:


# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = fea_vec.reshape(1, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec


# In[8]:


import pickle
with open("word_idx.pkl", "rb") as wi:
    wordtoix=pickle.load(wi)


# In[9]:


with open("./ixtoword.pkl", "rb") as iw:
    ixtoword=pickle.load(iw)


# In[10]:


def imageSearch(photo):
    max_length=34
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:

def captionimage(image):
    enc=encode(image)
    caption=imageSearch(enc)
    return caption

# In[12]:


