#!/usr/bin/env python
# coding: utf-8

# # CRAWLING DATA TWITTER MENGGUNAKAN TWINT

# ## Mount Google Drive

# Moount Google Drive dengan Google Collab

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# Masuk ke direktori projek Web Mining

# In[7]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Web Mining')


# ## Intalasi Twint

# Langkah awal clone terlebih twint dari GitHub TwintProject, lalu kita masuk kedalam folder yang sudah kita clone tadi. Tinggal jalankan script dibawah untuk memasang Twint ke projek kita

# In[3]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')


# Pasang aiohttp berguna menyediakan Web-server dengan middlewares dan plugable routing 

# In[4]:


get_ipython().system('pip install aiohttp==3.7.0')


# Pasang nest-asyncio untuk runtime serentak dalam noteboook

# In[5]:


get_ipython().system('pip install nest-asyncio')


# Import nest-asyncio dan juga twint agar bisa melakukan crawling data di twitter

# In[6]:


import nest_asyncio
nest_asyncio.apply()
import twint


# ## Crawling data twitter 

# Jadi disini kita akan melakukan crawling data yang diunduh dari server twitter. Cara ini cukup simpel, cepat dan gak ribet, karena kita gak perlu punya akun twitter, gak perlu API dan tanpa limitasi juga. Kita hanya perlu sebuah tool yang bernama **twint**. 
# >**Twint** adalah sebuah tools yang digunakan untuk melakukan scrapping dari aplikasi twitter yang disetting secara khusus menggunakan bahasa pemrograman Python. Twint dapat kita gunakan dan jalankan tanpa harus menggunakan API dari Twitter itu sendiri, dengan kapasitas scrapping data maksimalnya adalah 3200 tweet. Bukan hanya digunakan pada tweet, twint juga bisa kita gunakan untuk melakukan scrapping pada user, followers, retweet dan sebagainya. Twint memanfaatkan operator pencarian twitter untuk memungkinkan proses penghapusan tweet dari user tertentu, memilih dan memilah informasi-informasi yang sensitif, termasuk email dan nomor telepon di dalamnya.

# Data yang kita ambil ialah pemberitaan terbaru mengenai data dari negara Indonesia yang sedang diretas oleh orang luar negeri berinisial "Bjorka". Kata kunci yang digunakan 'databocor' pada **c.search**, menggunakan Pandas pada **c.Pandas**, menggunakan limitasi data sebanyak 80 data pada **c.Limit**, dengan menggunakan custom data yang dimasukkan ke csx dengan label Tweet dan data yang diambil tweet-nya saja. Output atau data akan dimasukkan ke dalam file **csv**.

# In[20]:


c = twint.Config()
c.Search = 'databocor'
c.Pandas = True
c.Limit = 80
c.Store_csv = True
c.Custom["tweet"] = ["tweet"]
c.Output = "data.csv"
twint.run.Search(c)


# Membuka file **csv** yang sudah dilabeli secara manual dengan 3 kelas yaitu positif, netral, dan negatif. 

# In[9]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Web Mining/webmining')


# In[10]:


import pandas as pd
data = pd.read_csv('dataBocor.csv')
data


# ## Matrix Term Frequent

# 
# >**NLTK** adalah singkatan dari Natural Language Tool Kit, yaitu sebuah library yang digunakan untuk membantu kita dalam bekerja dengan teks. Library ini memudahkan kita untuk memproses teks seperti melakukan classification, tokenization, stemming, tagging, parsing, dan semantic reasoning.
# 
# >**Python Sastrawi** adalah pengembangan dari proyek PHP Sastrawi. Python Sastrawi merupakan library sederhana yang dapat mengubah kata berimbuhan bahasa Indonesia menjadi bentuk dasarnya. Sastrawi juga dapat diinstal melalui “pip”
# 
# 

# In[3]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# Pembuatan matriks menggunakan module pandas beserta numpy agar matriks yang dibuat sesuai dengan kebutuhan.
# 
# >**Pandas** adalah sebuah library di Python yang berlisensi BSD dan open source yang menyediakan struktur data dan analisis data yang mudah digunakan. Pandas biasa digunakan untuk membuat tabel, mengubah dimensi data, mengecek data, dan lain sebagainya. Struktur data dasar pada Pandas dinamakan DataFrame, yang memudahkan kita untuk membaca sebuah file dengan banyak jenis format seperti file .txt, .csv, dan .tsv. Fitur ini akan menjadikannya table dan juga dapat mengolah suatu data dengan menggunakan operasi seperti join, distinct, group by, agregasi, dan teknik lainnya yang terdapat pada SQL.
# 
# >**NumPy** merupakan singkatan dari Numerical Python. NumPy merupakan salah satu library Python yang berfungsi untuk proses komputasi numerik. NumPy memiliki kemampuan untuk membuat objek N-dimensi array. Array merupakan sekumpulan variabel yang memiliki tipe data yang sama. Kelebihan dari NumPy Array adalah dapat memudahkan operasi komputasi pada data, cocok untuk melakukan akses secara acak, dan elemen array merupakan sebuah nilai yang independen sehingga penyimpanannya dianggap sangat efisien.

# In[4]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# **Function Remove Stopwords** berguna menghapus kata-kata yang tidak diperlukan dalam proses nantinya,sehingga dapat mempercepat proses VSM. Kita meenggunakan kumpulan stopword dari github yang berjumlah sekitar 700 kata. 

# In[12]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/Web Mining/webmining/stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# **Stemming** merupakan proses mengubah kata dalam bahasa Indonesia ke akar katanya atau tidak ada kata yang berimbuhan pada awal maupun akhir kata serta tidak ada kata yang berulangan misalkan 'anak perempuan berjalan - jalan' menjadi 'anak perempuan jalan'

# In[13]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# **Preprocessing** terdiri dari beberapa tahapan yang terdiri dari :
# 
# 
# > * Mengubah Text menjadi huruf kecil
# * Menghilangkan Url didalam Text
# * Mengubah/menghilangkan tanda (misalkan garis miring menjadi spasi)
# * Melakukan tokenization kata dan Penghapusan Kata yang tidak digunakan
# * Memfilter kata dari tanda baca
# * Mengubah kata dalam bahasa Indonesia ke akar katanya
# * Menghapus String kosong

# In[14]:


def preprocessing(text):
    #case folding
    text = text.lower()
    #remove urls
    text = re.sub('http\S+', '', text)
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


# Membuat matriks term frekuensi yang sudah dilakukan **Preprocessing** pada data crawling. Data yang kosong diisi dengan angka 0.

# In[15]:


tf = pd.DataFrame()
for i,v in enumerate(data['tweet']):
    cols = ["Doc " + str(i+1)]    
    doc = pd.DataFrame.from_dict(nltk.FreqDist(preprocessing(v)), orient='index',columns=cols)
    tf = pd.concat([tf, doc], axis=1, sort=False)


# In[17]:


tf.index.name = 'Term'
tf = pd.concat([tf], axis=1, sort=False)
tf = tf.fillna(0)
tf


# ## Mutual Information

# In[16]:


get_ipython().system('pip install -U scikit-learn')


# In[18]:


train = tf.iloc[:,:len(data)]
test = tf.iloc[:,len(data):]


# In[19]:


cols = train.columns
df = pd.DataFrame(train[cols].gt(0).sum(axis=1), columns=['Document Frequency'])
idf = np.log10(len(cols)/df)
idf.columns = ['Inverse Document Frequency']
idf = pd.concat([df, idf], axis=1)


# In[20]:


idf


# In[21]:


df_x = data.iloc[:, 1:]
df_y = data.iloc[:, 0]


# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df.drop(labels=['Document Frequency'],axis=1),
                                               df['Document Frequency'],
                                               test_size=0.3,
                                               random_state=0)

from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(df_x,df_y)

