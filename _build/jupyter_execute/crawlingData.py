#!/usr/bin/env python
# coding: utf-8

# # CRAWLING DATA

# ## Mount Google Drive

# Moount Google Drive dengan Google Collab

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# Masuk ke direktori projek Web Mining

# In[6]:


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

# In[3]:


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

# In[3]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# Pembuatan matriks menggunakan module pandas beserta numpy agar matriks yang dibuat sesuai dengan kebutuhan.
# 
# >**Pandas** adalah sebuah library di Python yang berlisensi BSD dan open source yang menyediakan struktur data dan analisis data yang mudah digunakan. Pandas biasa digunakan untuk membuat tabel, mengubah dimensi data, mengecek data, dan lain sebagainya. Struktur data dasar pada Pandas dinamakan DataFrame, yang memudahkan kita untuk membaca sebuah file dengan banyak jenis format seperti file .txt, .csv, dan .tsv. Fitur ini akan menjadikannya table dan juga dapat mengolah suatu data dengan menggunakan operasi seperti join, distinct, group by, agregasi, dan teknik lainnya yang terdapat pada SQL.
# 
# >**NumPy** merupakan singkatan dari Numerical Python. NumPy merupakan salah satu library Python yang berfungsi untuk proses komputasi numerik. NumPy memiliki kemampuan untuk membuat objek N-dimensi array. Array merupakan sekumpulan variabel yang memiliki tipe data yang sama. Kelebihan dari NumPy Array adalah dapat memudahkan operasi komputasi pada data, cocok untuk melakukan akses secara acak, dan elemen array merupakan sebuah nilai yang independen sehingga penyimpanannya dianggap sangat efisien.

# In[6]:


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


# In[4]:


import pandas as pd
import numpy as np
dataPre = pd.read_csv('hasilPreprocessing.csv')
dataPre


# ## Vector Space Model

# Vector Space Model (VSM) merupakan sebuah pendekatan natural yang berbasis pada vektor dari setiap kata dalam suatu dimensi spasial. Dokumen dipandang sebagai sebuah vektor yang memiliki magnitude (jarak) dan direction (arah). Pada VSM, sebuah kata direpresentasikan dengan sebuah dimensi dari ruang vektor. Relevansi sebuah dokumen ke sebuah kueri didasarkan pada similaritas diantara vektor dokumen dan vektor kueri.

# Import modul untuk membuat Vector Space Model dari library Sklearn, serta import data hasil preprocessing 

# In[5]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('/content/drive/MyDrive/Web Mining/webmining/hasilPreprocessing.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])


# Membuat matriks menjadi matriks array dan dilakukan shape pada matriks yang sudah dibuat 

# In[6]:


matrik_vsm = bag.toarray()
matrik_vsm.shape


# In[7]:


matrik_vsm[0]


# Mengambil semua kata yang sudah di tokenizing menjadi kolom - kolom atau fitur pada matriks VSM

# In[8]:


a = vectorizer.get_feature_names()


# Menampilkan Matriks VSM yang sduah dihitung frekuensi kemunculan term pada setiap tweet atau dokumen.

# In[9]:


dataTF = pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# Menambahkan kolom label pada setiap tweet dan mengisi setiap baris pada kolom Label dengan data yang telah diisi manual.

# In[10]:


label = pd.read_csv('/content/drive/MyDrive/Web Mining/webmining/dataBocor.csv')
dataVSM = pd.concat([dataTF.reset_index(), label["label"]], axis=1)
dataVSM


# Membuat Kolom Label menjadi kolom unique

# In[11]:


dataVSM['label'].unique()


# In[12]:


dataVSM.info()


# ## Mutual Information

# Scikit-learn atau sklearn merupakan sebuah module dari bahasa pemrograman Python yang dibangun berdasarkan NumPy, SciPy, dan Matplotlib. Fungsi dari module ini adalah untuk membantu melakukan processing data ataupun melakukan training data untuk kebutuhan machine learning atau data science.

# In[13]:


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

# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataVSM.drop(labels=['label'], axis=1),
    dataVSM['label'],
    test_size=0.3,
    random_state=0)


# In[15]:


X_train


# Menghitung Information gain menggunakan modul yang sudah ada di sklearn dengan mengambil data yang sudah ditrain split.

# In[16]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# Meranking setiap term mulai dari information gain terbesar sampai yang terkecil.

# In[17]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# Membuat plot berbentuk grafik batang atau bar dari data perankingan term.

# In[18]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(50, 20))


# Memilih K best sebanyak 75 item untuk training data

# In[19]:


from sklearn.feature_selection import SelectKBest
sel_five_cols = SelectKBest(mutual_info_classif, k=75)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]


# In[20]:


X_train=X_train.values
y_train=y_train.values
X_test=X_test.values
y_test=y_test.values


# ## Klasifikasi

# ### KNN
# Algoritma K-Nearest Neighbor (KNN) adalah sebuah metode klasifikasi terhadap sekumpulan data berdasarkan pembelajaran data yang sudah terklasifikasikan sebelumya. Termasuk dalam supervised learning, dimana hasil query instance yang baru diklasifikasikan berdasarkan mayoritas kedekatan jarak dari kategori yang ada dalam K-NN. Algoritma ini bertujuan untuk mengklasifikasikan obyek baru berdasarkan atribut dan sample-sample dari training data.

# Import algoritma KNN dari sklearn, lalu aktifkan fungsi klasifikasi KNN serta atur koefisien N, pada dataset ini kita gunakan perulangan untuk mendapatkan nilai n terbaik akurasinya 

# In[31]:


from sklearn.neighbors import KNeighborsClassifier
testing=[]
listnum=[]
for i in range(2,15):
  listnum.append(i)
  neigh = KNeighborsClassifier(n_neighbors=i)
  neigh.fit(X_train, y_train)
  Y_pred = neigh.predict(X_test) 
  testing.append(Y_pred)
testing


# Menampilkan nilai akurasi dari algoritma KNN dengan nilai n berbeda - beda

# In[32]:


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


# membuat grafik untuk melihat nilai n terbaik, dan dapat dilihat bahwa nilai n terbaik ada pada n ke 12

# In[33]:


from matplotlib import pyplot as plt
plt.bar(listnum, listacc)
plt.xticks(listnum)
plt.title('Nilai Akurasi Berdasarkan Input')
plt.ylabel('Persentase Akurasi')
plt.xlabel('Nilai n')


# membuat algoritma KNN dengan nilai n = 12 dan menampilkan nilai akurasinya

# In[34]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=12)
neigh.fit(X_train, y_train)
Y_pred = neigh.predict(X_test)
from sklearn.metrics import make_scorer, accuracy_score,precision_score
testing = neigh.predict(X_test)
accuracy_neigh=round(accuracy_score(y_test,testing)* 100, 2)
accuracy_neigh


# ### Confusion Matrix
# Confusion Matrix adalah pengukuran performa untuk masalah klasifikasi machine learning dimana keluaran dapat berupa dua kelas atau lebih.  Confusion Matrix adalah tabel dengan 4 kombinasi atau lebih berbeda dari nilai prediksi dan nilai aktual. 
# 

# In[35]:


import matplotlib.pyplot as plt
from sklearn import metrics


# Import pyplot untuk membuat plot matriks menjadi tidak eror jika ditampilkan, lalu import metrics dari sklearn untuk membuat matriksnya.

# In[36]:


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

# In[37]:


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


# ## Meringkas berita dengan Graph dan Pagerank

# ### Crawling berita dengan Scrapy

# Scrapy adalah kerangka kerja aplikasi untuk crawling web site dan mengekstraksi data terstruktur yang dapat digunakan untuk berbagai aplikasi yang bermanfaat, seperti data mining, pemrosesan informasi atau arsip sejarah. Meskipun Scrapy awalnya dirancang untuk web scraping, namu scrapy juga dapat digunakan untuk mengekstrak data menggunakan API (seperti Amazon Associates Web Services) atau sebagai web crawl

# Crochet adalah library pyhton berlisensi MIT yang memudahkan penggunaan Twisted dari regular blocking code. Beberapa kasus penggunaan meliputi:
# *   Mudah menggunakan Twisted dari blocking framework seperti Django atau Flask.
# *   Menulis library yang menyediakan blocking API, tetapi menggunakan Twisted untuk implementasinya.
# *  Port blocking code ke Twisted lebih mudah, dengan menjaga backwards compatibility layer.
# * Izinkan program Twisted normal yang menggunakan threads untuk berinteraksi dengan Twisted lebih bagus dari threaded parts. Misalnya, sangat berguna saat menggunakan Twisted sebagai WSGI container..
# 
# 
# 
# 

# In[ ]:


get_ipython().system('pip install scrapy')
get_ipython().system('pip install crochet')


# Membuat Class untuk Crawling data berita dari sebuah portal berita yaitu [Tempo](https://nasional.tempo.co). Lalu custom setting dari hasil dari crawling menjadi file CSV. Parse data crawling dari URL dengan id 'isi' dan tag p. Class ExtractFirstLine untuk ekstak data berita dari dari web dan memisahkan dari tag HTML.

# In[ ]:


import scrapy
from scrapy.crawler import CrawlerRunner
import re
from crochet import setup, wait_for
setup()

class QuotesToCsv(scrapy.Spider):
    name = "MJKQuotesToCsv"
    start_urls = [
        'https://nasional.tempo.co/read/1643157/pria-tewas-terjatuh-dari-lantai-11-hotel-yogyakarta-diduga-mahasiswa-ugm',
    ]
    custom_settings = {
        'ITEM_PIPELINES': {
            '__main__.ExtractFirstLine': 1
        },
        'FEEDS': {
            'beritaUGM.csv': {
                'format': 'csv',
                'overwrite': True
            }
        }
    }

    def parse(self, response):
        """parse data from urls"""
        for quote in response.css('#isi > p'):
            yield {'news': quote.extract()}


class ExtractFirstLine(object):
    def process_item(self, item, spider):
        """text processing"""
        lines = dict(item)["news"].splitlines()
        first_line = self.__remove_html_tags__(lines[0])

        return {'news': first_line}

    def __remove_html_tags__(self, text):
        """remove html tags from string"""
        html_tags = re.compile('<.*?>')
        return re.sub(html_tags, '', text)

@wait_for(10)
def run_spider():
    """run spider with MJKQuotesToCsv"""
    crawler = CrawlerRunner()
    d = crawler.crawl(QuotesToCsv)
    return d


# Jalankan fungsi intuk crawling data berita

# In[ ]:


run_spider()


# Menampilkan data berita yang sudah dicrawling menggunakan pandas

# In[ ]:


dataBerita = pd.read_csv('beritaUGM.csv')
dataBerita


# ### PyPDF2

# PyPDF2 adalah library Python yang memungkinkan manipulasi dokumen PDF. Ini dapat digunakan untuk membuat dokumen PDF baru, memodifikasi yang sudah ada, dan mengekstrak konten dari dokumen. PyPDF2 adalah library Python yang tidak memerlukan modul non-standar.

# In[ ]:


get_ipython().system('pip install PyPDF2')


# import library PyPDF2 dan membuat variabel untuk membaca file PDF berita yang sudah crawling.

# In[ ]:


import PyPDF2
pdfReader = PyPDF2.PdfFileReader('/content/drive/MyDrive/Web Mining/webmining/beritaUGM.pdf')
pageObj = pdfReader.getPage(0)
document = pageObj.extractText()
document


# ### PunktSentenceTokenizer

# PunktSentenceTokenizer adalah Sebuah libary untuk tokenizing atau memecah kalimat - kalimat pada sebuah paragraf.

# class tokenize terdiri dari 2 tahapan yaitu tahap prepocessing dan tahap memecah kalimat pada data beritaUGM

# In[ ]:


from nltk.tokenize.punkt import PunktSentenceTokenizer
def tokenize(document):
    # tahap prepocessing
    document = document.encode('ascii', 'replace').decode('ascii')
    document = ' '.join(re.sub("([@#?][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", document).split())
    # memecahnya menggunakan  PunktSentenceTokenizer
    doc_tokenizer = PunktSentenceTokenizer()
    # sentences_list adalah daftar masing masing kalimat dari dokumen yang ada.
    sentences_list = doc_tokenizer.tokenize(document)
    return sentences_list
sentences_list = tokenize(document)
sentences_list


# Menampilkan setiap kalimat yang sudah tokenizing

# In[ ]:


for j in range (len(sentences_list)):
    print('Kalimat {}'.format(j+1))
    print(sentences_list[j])


# ### TF IDF

# Tokenizing kata - kata pada kalimat sehingga bisa dihitung jumlah kosa kata serta menghitung TF IDF dari kata - kata tersebut.

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
vectorizer = CountVectorizer()
cv_matrix=vectorizer.fit_transform(sentences_list)


# In[ ]:


print ("Banyaknya kosa kata : ", len((vectorizer.get_feature_names_out())))
print ("Banyaknya kalimat : ", (len(sentences_list)))
print ("Kosa kata : ", (vectorizer.get_feature_names_out()))


# membuat matrix TF IDF dari kosa kata ada

# In[ ]:


normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
normal_matrix.toarray()


# ### Networkx dan Graph

# Networkx adalah salah satu package pada bahasa pemrograman Python yang berfungsi untuk mengeksplorasi dan menganalisis jaringan dan algoritma jaringan.

# Graph adalah jenis struktur data umum yang susunan datanya tidak berdekatan satu sama lain (non-linier). Graph terdiri dari kumpulan simpul berhingga untuk menyimpan data dan antara dua buah simpul terdapat hubungan saling keterkaitan. Simpul pada graph disebut dengan verteks (V), sedangkan sisi yang menghubungkan antar verteks disebut edge (E). Pasangan (x,y) disebut sebagai edge, yang menyatakan bahwa simpul x terhubung ke simpul y

# Hitung perkalian matrix TF IDF dari kosa kata ada dengan matrix tranpose

# In[ ]:


res_graph = normal_matrix * normal_matrix.T
print(res_graph)


# Membuat grafik dari res_graph yang sudah dihitung hasilnya

# In[ ]:


import networkx as nx
nx_graph = nx.from_scipy_sparse_matrix(res_graph)
nx.draw_circular(nx_graph)


# menampilkan banyak sisi dari graph yang berjumlah 151 sisi

# In[ ]:


print('Banyaknya sisi : {}'.format(nx_graph.number_of_edges()))


# ### Pagerank

# Algoritma PageRank bertujuan untuk mengukur hubungan kepentingan dalam kumpulan dokumen tersebut. Dalam algoritma PageRank dihasilkan matriks yang menghitung probabilitas bahwa pengguna akan berpindah dari satu halaman ke halaman lainnya. Algoritma PageRank dapat digunakan untuk memberikan peringkat pada
# setiap kalimat yang tersusun dalam sebuah graph. Peringkat yang dihasilkan
# oleh PageRank dapat digunakan untuk memastikan bahwa kalimat-kalimat
# yang dipilih oleh proses genetika adalah kalimat yang memiliki tingkat pentingnya tinggi. Semakin besar nilai PageRank maka semakin penting kalimat tersebut

# inisiasi algoritma Pagerank dari graph nx_graph

# In[ ]:


ranks=nx.pagerank(nx_graph)


# input hasil perhitungan PageRank ke Array untuk ditampilkan dan dihitung urutkan angka PageRank terbesar sampai terkecil.

# In[ ]:


arrRank=[]
for i in ranks:
    arrRank.append(ranks[i])


# Buat dataFrame dengan kolom data kalimat dan data nilai PageRank

# In[ ]:


dfRanks = pd.DataFrame(arrRank,columns=['PageRank'])
dfSentence = pd.DataFrame(sentences_list,columns=['News'])
dfJoin = pd.concat([dfSentence,dfRanks], axis=1)
dfJoin


# Mengurutkan dataFrame berdasarkan nilai PageRank terbesar ke yang terkecil

# In[ ]:


sortSentence=dfJoin.sort_values(by=['PageRank'],ascending=False)
sortSentence


# Dapat diketahui bahwa nilai PageRank terbesar ada pada kalimat indeks ke 8 lalu ke 5, ke 12, ke 7, dan ke 6.

# In[ ]:


sortSentence.head(5)


# ## Latent Semantic Indexing(LSI)

# > Latent Semantic Indexing adalah teknik dalam pemrosesan bahasa alami untuk menganalisis hubungan antara sekumpulan dokumen dan term yang dikandungnya dengan menghasilkan sekumpulan konsep yang berkaitan dengan dokumen dan term tersebut.

# install library nltk, Pysastrawi, dan sastrawi untuk preprocessing teks dokumen.

# In[ ]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install PySastrawi')
get_ipython().system('pip install Sastrawi')


# import data konten web yang sudah di crawling sebelumnya menggunakan library PyPDF2

# In[ ]:


import PyPDF2
pdfReader = PyPDF2.PdfFileReader('/content/drive/MyDrive/Web Mining/webmining/beritaUGM.pdf')
pageObj = pdfReader.getPage(0)
document = pageObj.extractText()
document


# import pandas, re(regular expression), stopword dari nltk.corpus, dan word_tokenize dari nltk.tokenize untuk penggunaan preprocessing pada teks dokumen.

# In[ ]:


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')


# memisah setiap kata pada teks dokumen menggunakan tanda spasi sebagai pemisahnya. 

# In[ ]:


word_tokens = word_tokenize(document)
print(word_tokens)


# stopword pada teks dokumen.

# In[ ]:


stop_words = set(stopwords.words('indonesian'))
word_tokens_no_stopwords = [w for w in word_tokens if not w in stop_words]
print(word_tokens_no_stopwords)


# import library RegexpTokenizer untuk tokenizing regular expression, TfidfVectorizer untuk membuat matrix tfidf, dan TruncatedSVD untuk algoritma LSA-nya.

# In[ ]:


import os
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[ ]:


tfidf = TfidfVectorizer(lowercase=True,
                        ngram_range = (1,1))

train_data = tfidf.fit_transform(word_tokens_no_stopwords)
train_data


# In[ ]:


num_components=10

# Create SVD object
lsa = TruncatedSVD(n_components=num_components, n_iter=100, random_state=42)

# Fit SVD model on data
lsa.fit_transform(train_data)

# Get Singular values and Components 
Sigma = lsa.singular_values_ 
V_transpose = lsa.components_.T
V_transpose


# Menampilkan hasil LSA yang dipisah menjadi 10 topik terpenting dalam teks dokumen. Sehingga dapan disimpulkan pada topik 1 menjadi topik terpenting dalam teks dokumen yaitu 'korban', 'ugm', 'yogyakarta', 'lantai', ' dan 'surat'.

# In[ ]:


terms = tfidf.get_feature_names()

for index, component in enumerate(lsa.components_):
    zipped = zip(terms, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:5]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index+1)+": ",top_terms_list)


# ## Ensemble BaggingClassifier

# Bagging juga dikenal sebagai Bootstrap aggregating, meta-algoritma ensemble machine learning yang dirancang untuk meningkatkan stabilitas dan akurasi algoritma pembelajaran mesin yang digunakan untuk analisis klasifikasi dan regresi. Ini membantu mengurangi variasi dan membantu menghindari overfitting. Contoh terbaik adalah random forest.

# ### Metode DecisionTreeClassifier

# menggunakan metode Decision Tree dengan jumlah estimator dari ensemble 500 menghasilkan nilai akurasi 0.4424

# In[38]:


from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

X = X_train
Y = y_train

# initialize the base classifier
base_cls = DecisionTreeClassifier()

# no. of base classifier
num_trees = 500

# bagging classifier
model = BaggingClassifier(base_estimator = base_cls,
						n_estimators = num_trees)

results = model_selection.cross_val_score(model, X, Y)
print("accuracy :")
print(results.mean())


# ### Metode SVC

# menggunakan metode Support Vector Classifier dengan jumlah estimator dari ensemble 500 menghasilkan nilai akurasi 0.3530

# In[39]:


from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import pandas as pd

X = X_train
Y = y_train

# initialize the base classifier
base_cls = SVC()

# no. of base classifier
num_trees = 500

# bagging classifier
model = BaggingClassifier(base_estimator = base_cls,
						n_estimators = num_trees)

results = model_selection.cross_val_score(model, X, Y)
print("accuracy :")
print(results.mean())


# ### Ensemble RandomForestClassifier dengan GridSearchCV

# GridSearchCV adalah salah satu teknik HyperParameter paling dasar yang digunakan sehingga implementasinya cukup sederhana. Semua kemungkinan permutasi dari HyperParameter untuk model tertentu digunakan untuk membangun model. Kinerja setiap model dievaluasi dan yang berkinerja terbaik dipilih. Karena GridSearchCV menggunakan berbagai kombinasi untuk membangun dan mengevaluasi kinerja model, metode ini sangat lama secara komputasi. Implementasi python dari GridSearchCV untuk algoritma Random Forest pada code berikut.

# menggunakan metode random forest classifier dengan jumlah estimator, max feature, max depth, dan criterion ditentukan oleh gridsearchCV

# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
# 'n_estimators': [i for i in range(800)],
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [50,100,200,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)


# menampilkan parameter terbaik dari random forest

# In[42]:


CV_rfc.best_params_


# hasil klasifikasi random forest mendapatkan nilai akurasi 0.4583

# In[52]:


rfc1=RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 100, max_depth=6, criterion='gini')
rfc1.fit(X_train, y_train)
pred=rfc1.predict(X_test)
print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))


# ## Ensemble StackingClassifier

# Stacking adalah metode pembelajaran ensamble yang menggabungkan beberapa algoritma machine learning melalui meta learning, Di mana algoritme tingkat dasar dilatih berdasarkan kumpulan data pelatihan lengkap, model meta mereka dilatih pada hasil akhir dari semua model tingkat dasar sebagai fitur. Kami telah berurusan dengan metode bagging dan boosting untuk menangani bias dan varians. Sekarang kita bisa belajar stacking yang meningkatkan akurasi prediksi model Anda.

# Hasil klasifikasi ensemble stacking dengan metode random forest mendapatkan nilai akurasi 0.73214

# In[21]:


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import StackingClassifier
estimators = [
    ('rf', RandomForestClassifier(random_state=42,max_features='auto', n_estimators= 100, max_depth=8, criterion='gini')),
    ('rf2', RandomForestClassifier(random_state=42,max_features='auto', n_estimators= 100, max_depth=8, criterion='entropy'))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=10, random_state=42)
)
clf.fit(X_train, y_train).score(X_train, y_train)

