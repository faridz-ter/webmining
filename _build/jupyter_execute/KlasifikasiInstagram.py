#!/usr/bin/env python
# coding: utf-8

# # KLASIFIKASI INSTAGRAM

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Web Mining/webmining')


# In[5]:


pwd


# In[12]:


import pandas as pd
data = pd.read_csv('datascrapig.csv')
data


# In[2]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# In[3]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# In[15]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/Web Mining/webmining/stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# In[16]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# In[17]:


def preprocessing(text):
    #case folding
    text = text.lower()

    #remove non ASCII (emoticon, chinese word, .etc)
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")

    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')

    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())

    #replace weird characters
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('-', ' ')

    #tokenization and remove stopwords
    text = remove_stopwords(text)

    #remove punctuation    
    text = [''.join(c for c in s if c not in string.punctuation) for s in text]  

    #stemming
    text = stemming(text)

    #remove empty string
    text = list(filter(None, text))
    return text


# In[18]:


data['Tweet'].apply(preprocessing).to_csv('hasilpreig.csv')


# In[3]:


import pandas as pd
import numpy as np
dataPre = pd.read_csv('hasilpreig.csv')
dataPre


# In[4]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('/content/drive/MyDrive/Web Mining/webmining/hasilpreig.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['Tweet'])


# In[5]:


matrik_vsm = bag.toarray()
matrik_vsm.shape


# In[6]:


matrik_vsm[0]


# In[7]:


a = vectorizer.get_feature_names()


# In[8]:


dataTF = pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[9]:


label = pd.read_csv('/content/drive/MyDrive/Web Mining/webmining/datascrapig.csv')
dataVSM = pd.concat([dataTF.reset_index(), label["label"]], axis=1)
dataVSM


# In[10]:


dataVSM['label'].unique()


# In[11]:


dataVSM.info()


# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataVSM.drop(labels=['label'], axis=1),
    dataVSM['label'],
    test_size=0.3,
    random_state=0)


# In[13]:


X_train


# In[14]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
testing=[]
listnum=[]
for i in range(2,9):
  listnum.append(i)
  neigh = KNeighborsClassifier(n_neighbors=i)
  neigh.fit(X_train, y_train)
  Y_pred = neigh.predict(X_test) 
  testing.append(Y_pred)
testing


# In[17]:


from sklearn.metrics import make_scorer, accuracy_score,precision_score
listtest=[]
listacc=[]
for i in range(len(testing)):
  accuracy_neigh=round(accuracy_score(y_test,testing[i])* 100, 2)
  acc_neigh = round(neigh.score(X_train, y_train) * 100, 2)
  listappend=listnum[i]
  appendlist=listappend,accuracy_neigh
  listtest.append(appendlist)
  listacc.append(accuracy_neigh)
listtest


# In[18]:


from matplotlib import pyplot as plt
plt.bar(listnum, listacc)
plt.xticks(listnum)
plt.title('Nilai Akurasi Berdasarkan Input')
plt.ylabel('Persentase Akurasi')
plt.xlabel('Nilai n')


# In[19]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train)
Y_pred = neigh.predict(X_test)
from sklearn.metrics import make_scorer, accuracy_score,precision_score
testing = neigh.predict(X_test)
accuracy_neigh=round(accuracy_score(y_test,testing)* 100, 2)
accuracy_neigh

