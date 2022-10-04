#!/usr/bin/env python
# coding: utf-8

# # CRAWLING DATA TWITTER MENGGUNAKAN TWINT

# ## Mount Google Drive

# Moount Google Drive dengan Google Collab

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# Masuk ke direktori projek Web Mining

# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Web Mining')


# ## Intalasi Twint

# Langkah awal clone terlebih twint dari GitHub TwintProject, lalu kita masuk kedalam folder yang sudah kita clone tadi. Tinggal jalankan script dibawah untuk memasang Twint ke projek kita

# In[ ]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')


# Pasang aiohttp berguna menyediakan Web-server dengan middlewares dan plugable routing 

# In[ ]:


get_ipython().system('pip install aiohttp==3.7.0')


# Pasang nest-asyncio untuk runtime serentak dalam noteboook

# In[ ]:


get_ipython().system('pip install nest-asyncio')


# Import nest-asyncio dan juga twint agar bisa melakukan crawling data di twitter

# In[ ]:


import nest_asyncio
nest_asyncio.apply()
import twint


# ## Crawling data twitter 

# Jadi disini kita akan melakukan crawling data yang diunduh dari server twitter. Cara ini cukup simpel, cepat dan gak ribet, karena kita gak perlu punya akun twitter, gak perlu API dan tanpa limitasi juga. Kita hanya perlu sebuah tool yang bernama **twint**. 
# >**Twint** adalah sebuah tools yang digunakan untuk melakukan scrapping dari aplikasi twitter yang disetting secara khusus menggunakan bahasa pemrograman Python. Twint dapat kita gunakan dan jalankan tanpa harus menggunakan API dari Twitter itu sendiri, dengan kapasitas scrapping data maksimalnya adalah 3200 tweet. Bukan hanya digunakan pada tweet, twint juga bisa kita gunakan untuk melakukan scrapping pada user, followers, retweet dan sebagainya. Twint memanfaatkan operator pencarian twitter untuk memungkinkan proses penghapusan tweet dari user tertentu, memilih dan memilah informasi-informasi yang sensitif, termasuk email dan nomor telepon di dalamnya.

# Data yang kita ambil ialah pemberitaan terbaru mengenai data dari negara Indonesia yang sedang diretas oleh orang luar negeri berinisial "Bjorka". Kata kunci yang digunakan 'databocor' pada **c.search**, menggunakan Pandas pada **c.Pandas**, menggunakan limitasi data sebanyak 80 data pada **c.Limit**, dengan menggunakan custom data yang dimasukkan ke csx dengan label Tweet dan data yang diambil tweet-nya saja. Output atau data akan dimasukkan ke dalam file **csv**.

# In[ ]:


c = twint.Config()
c.Search = 'databocor'
c.Pandas = True
c.Limit = 80
c.Store_csv = True
c.Custom["tweet"] = ["tweet"]
c.Output = "data.csv"
twint.run.Search(c)


# Membuka file **csv** yang sudah dilabeli secara manual dengan 3 kelas yaitu positif, netral, dan negatif. 

# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Web Mining/webmining')


# In[ ]:


import pandas as pd
data = pd.read_csv('dataBocor.csv')
data


# ## Preprocessing

# Preprocessing adalah proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur. Selain itu, data mining juga tidak dapat memproses data mentah, sehingga proses ini sangat penting dilakukan untuk mempermudah proses berikutnya, yakni analisis data.

# 
# >**NLTK** adalah singkatan dari Natural Language Tool Kit, yaitu sebuah library yang digunakan untuk membantu kita dalam bekerja dengan teks. Library ini memudahkan kita untuk memproses teks seperti melakukan classification, tokenization, stemming, tagging, parsing, dan semantic reasoning.
# 
# >**Python Sastrawi** adalah pengembangan dari proyek PHP Sastrawi. Python Sastrawi merupakan library sederhana yang dapat mengubah kata berimbuhan bahasa Indonesia menjadi bentuk dasarnya. Sastrawi juga dapat diinstal melalui “pip”
# 
# 

# In[9]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# Pembuatan matriks menggunakan module pandas beserta numpy agar matriks yang dibuat sesuai dengan kebutuhan.
# 
# >**Pandas** adalah sebuah library di Python yang berlisensi BSD dan open source yang menyediakan struktur data dan analisis data yang mudah digunakan. Pandas biasa digunakan untuk membuat tabel, mengubah dimensi data, mengecek data, dan lain sebagainya. Struktur data dasar pada Pandas dinamakan DataFrame, yang memudahkan kita untuk membaca sebuah file dengan banyak jenis format seperti file .txt, .csv, dan .tsv. Fitur ini akan menjadikannya table dan juga dapat mengolah suatu data dengan menggunakan operasi seperti join, distinct, group by, agregasi, dan teknik lainnya yang terdapat pada SQL.
# 
# >**NumPy** merupakan singkatan dari Numerical Python. NumPy merupakan salah satu library Python yang berfungsi untuk proses komputasi numerik. NumPy memiliki kemampuan untuk membuat objek N-dimensi array. Array merupakan sekumpulan variabel yang memiliki tipe data yang sama. Kelebihan dari NumPy Array adalah dapat memudahkan operasi komputasi pada data, cocok untuk melakukan akses secara acak, dan elemen array merupakan sebuah nilai yang independen sehingga penyimpanannya dianggap sangat efisien.

# In[10]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# **Function Remove Stopwords** berguna menghapus kata-kata yang tidak diperlukan dalam proses nantinya,sehingga dapat mempercepat proses VSM. Kita meenggunakan kumpulan stopword dari github yang berjumlah sekitar 700 kata. 

# In[ ]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/Web Mining/webmining/stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# **Stemming** merupakan proses mengubah kata dalam bahasa Indonesia ke akar katanya atau tidak ada kata yang berimbuhan pada awal maupun akhir kata serta tidak ada kata yang berulangan misalkan 'anak perempuan berjalan - jalan' menjadi 'anak perempuan jalan'

# In[ ]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# **Preprocessing** terdiri dari beberapa tahapan yang terdiri dari :
# 
# 
# * Mengubah Text menjadi huruf kecil
# * Menghilangkan non ASCII seperti emotikon, penulisan Cina, dan sebagainya.
# * Menghilangkan mention, Url didalam Text, dan hashtag.
# * Mengubah/menghilangkan tanda baca (misalkan garis miring menjadi spasi)
# * Melakukan tokenization kata dan Penghapusan Kata yang tidak digunakan
# * Memfilter kata dari tanda baca
# * Mengubah kata dalam bahasa Indonesia ke akar katanya
# * Menghapus String kosong

# In[ ]:


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


# Menyimpan data yang sudah dilakukan Preprocessing ke dalam file csv baru dan tersimpan di folder yang sama dengan file ipynb.

# In[ ]:


data['tweet'].apply(preprocessing).to_csv('hasilPreprocessing.csv')


# In[ ]:


dataPre = pd.read_csv('hasilPreprocessing.csv')
dataPre


# ## Vector Space Model

# Vector Space Model (VSM) merupakan sebuah pendekatan natural yang berbasis pada vektor dari setiap kata dalam suatu dimensi spasial. Dokumen dipandang sebagai sebuah vektor yang memiliki magnitude (jarak) dan direction (arah). Pada VSM, sebuah kata direpresentasikan dengan sebuah dimensi dari ruang vektor. Relevansi sebuah dokumen ke sebuah kueri didasarkan pada similaritas diantara vektor dokumen dan vektor kueri.

# Import modul untuk membuat Vector Space Model dari library Sklearn, serta import data hasil preprocessing 

# In[12]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('/content/drive/MyDrive/Web Mining/webmining/hasilPreprocessing.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])


# Membuat matriks menjadi matriks array dan dilakukan shape pada matriks yang sudah dibuat 

# In[13]:


matrik_vsm = bag.toarray()
matrik_vsm.shape


# In[14]:


matrik_vsm[0]


# Mengambil semua kata yang sudah di tokenizing menjadi kolom - kolom atau fitur pada matriks VSM

# In[15]:


a = vectorizer.get_feature_names()


# Menampilkan Matriks VSM yang sduah dihitung frekuensi kemunculan term pada setiap tweet atau dokumen.

# In[16]:


dataTF = pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# Menambahkan kolom label pada setiap tweet dan mengisi setiap baris pada kolom Label dengan data yang telah diisi manual.

# In[17]:


label = pd.read_csv('/content/drive/MyDrive/Web Mining/webmining/dataBocor.csv')
dataVSM = pd.concat([dataTF.reset_index(), label["label"]], axis=1)
dataVSM


# Membuat Kolom Label menjadi kolom unique

# In[18]:


dataVSM['label'].unique()


# In[19]:


dataVSM.info()


# ## Mutual Information

# Scikit-learn atau sklearn merupakan sebuah module dari bahasa pemrograman Python yang dibangun berdasarkan NumPy, SciPy, dan Matplotlib. Fungsi dari module ini adalah untuk membantu melakukan processing data ataupun melakukan training data untuk kebutuhan machine learning atau data science.

# In[20]:


get_ipython().system('pip install -U scikit-learn')


# ### Menghitung Information gain 

# Information Gain merupakan teknik seleksi fitur yang memakai metode scoring untuk nominal ataupun pembobotan atribut kontinue yang didiskretkan menggunakan maksimal entropy. Suatu entropy digunakan untuk mendefinisikan nilai Information Gain. Entropy menggambarkan banyaknya informasi yang dibutuhkan untuk mengkodekan suatu kelas. Information Gain (IG) dari suatu term diukur dengan menghitung jumlah bit informasi yang diambil dari prediksi kategori dengan ada atau tidaknya term dalam suatu dokumen.

# $$
# Entropy \ (S) \equiv \sum ^{c}_{i}P_{i}\log _{2}p_{i}
# $$
# 
# c  : jumlah nilai yang ada pada atribut target (jumlah kelas klasifikasi).
# 
# Pi : porsi sampel untuk kelas i.

# 
# $$
# Gain \ (S,A) \equiv Entropy(S) - \sum _{\nu \varepsilon \ values } \dfrac{\left| S_{i}\right| }{\left| S\right|} Entropy(S_{v})
# $$
# 
# A : atribut
# 
# V : menyatakan suatu nilai yang mungkin untuk atribut A
# 
# Values (A) : himpunan nilai-nilai yang mungkin untuk atribut A
# 
# |Sv| : jumlah Sampel untuk nilai v
# 
# |S| : jumlah seluruh sample data Entropy 
# 
# (Sv) : entropy untuk sampel sampel yang memiliki nilai v
# 

# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataVSM.drop(labels=['label'], axis=1),
    dataVSM['label'],
    test_size=0.3,
    random_state=0)


# In[22]:


X_train


# Menghitung Information gain menggunakan modul yang sudah ada di sklearn dengan mengambil data yang sudah ditrain split.

# In[23]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# Meranking setiap term mulai dari information gain terbesar sampai yang terkecil.

# In[24]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# Membuat plot berbentuk grafik batang atau bar dari data perankingan term.

# In[25]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(50, 20))


# In[26]:


from sklearn.feature_selection import SelectKBest
sel_five_cols = SelectKBest(mutual_info_classif, k=100)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]


# In[27]:


X_train=X_train.values
y_train=y_train.values
X_test=X_test.values
y_test=y_test.values


# ## Klasifikasi

# ### KNN
# Algoritma K-Nearest Neighbor (KNN) adalah sebuah metode klasifikasi terhadap sekumpulan data berdasarkan pembelajaran data yang sudah terklasifikasikan sebelumya. Termasuk dalam supervised learning, dimana hasil query instance yang baru diklasifikasikan berdasarkan mayoritas kedekatan jarak dari kategori yang ada dalam K-NN. Algoritma ini bertujuan untuk mengklasifikasikan obyek baru berdasarkan atribut dan sample-sample dari training data.

# Import algoritma KNN dari sklearn, lalu aktifkan fungsi klasifikasi KNN serta atur koefisien N, pada dataset ini N sudah diatur dengan 2 sebab akurasi terbaik terdapat pada angka N ke 2.

# In[28]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)
Y_pred = neigh.predict(X_test) 
Y_pred


# Menampilkan nilai akurasi dari algoritma KNN

# In[31]:


from sklearn.metrics import make_scorer, accuracy_score,precision_score
testing = neigh.predict(X_test) 
accuracy_neigh=round(accuracy_score(y_test,testing)* 100, 2)
accuracy_neigh


# ### Confusion Matrix
# Confusion Matrix adalah pengukuran performa untuk masalah klasifikasi machine learning dimana keluaran dapat berupa dua kelas atau lebih.  Confusion Matrix adalah tabel dengan 4 kombinasi berbeda dari nilai prediksi dan nilai aktual. 
# 

# In[32]:


import matplotlib.pyplot as plt
from sklearn import metrics


# Import pyplot untuk membuat plot matriks menjadi tidak eror jika ditampilkan, lalu import metrics dari sklearn untuk membuat matriksnya.

# In[37]:


conf_matrix =metrics.confusion_matrix(y_true=y_test, y_pred=Y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['negatif', 'netral','positif'])
cm_display.plot()
plt.show()


# ## Clustering

# ### K MEANS
# Algoritma k-means merupakan algoritma yang membutuhkan parameter input sebanyak k dan membagi sekumpulan n objek kedalam k cluster sehingga tingkat kemiripan antar anggota dalam satu cluster tinggi sedangkan tingkat kemiripan dengan anggota pada cluster lain sangat rendah. Kemiripan anggota terhadap cluster diukur dengan kedekatan objek terhadap nilai mean pada cluster atau dapat disebut sebagai centroid cluster.

# Rumus menghitung jarak terdekat digunakan formula *Ecludean* sebagai berikut :
# 
# $$
# d(i,j) = \sqrt{\sum ^{m}_{j=1}\left( x_{ij}-c_{kj}\right) ^{2}}
# $$

# In[38]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD

"""Train the Kmeans with the best n of clusters"""
modelKm = KMeans(n_clusters=3, random_state=12)
modelKm.fit(dataTF.values)
prediksi = modelKm.predict(dataTF.values)

"""Dimensionality reduction used to plot in 2d representation"""
pc=TruncatedSVD(n_components=2)
X_new=pc.fit_transform(dataTF.values)
centroids=pc.transform(modelKm.cluster_centers_)
print(centroids)
plt.scatter(X_new[:,0],X_new[:,1],c=prediksi, cmap='viridis')
plt.scatter(centroids[:,0] , centroids[:,1] , s = 50, color = 'red')

