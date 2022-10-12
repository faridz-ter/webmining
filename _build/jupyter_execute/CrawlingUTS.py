#!/usr/bin/env python
# coding: utf-8

# # SOAL UTS WEB MINING
# 
# 1. Lakukan analisa clustering dengan menggunakan k-mean clustering pada data twitter denga kunci pencarian " tragedi kanjuruhan"
# 
# 2. Lakukan peringkasan dokumen dari berita online ( link berita bebas) menggunakan metode pagerank
# 
# Catatan
# 
# Hasil analisa dilaporkan nenggunakan jupyter book dan diupload  di github sesuai alamat masing masing
# 
# Link  alamat diupload di schoology 

# ## Soal 1

# ### mount drive

# Mount Google Drive dengan Google Collab

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# Masuk ke direktori projek Web Mining

# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Web Mining')


# Langkah awal clone terlebih twint dari GitHub TwintProject, lalu kita masuk kedalam folder yang sudah kita clone tadi. Tinggal jalankan script dibawah untuk memasang Twint ke projek kita

# ### instal twint

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


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Web Mining/webmining')


# Jadi disini kita akan melakukan crawling data yang diunduh dari server twitter. Cara ini cukup simpel, cepat dan gak ribet, karena kita gak perlu punya akun twitter, gak perlu API dan tanpa limitasi juga. Kita hanya perlu sebuah tool yang bernama **twint**. 
# >**Twint** adalah sebuah tools yang digunakan untuk melakukan scrapping dari aplikasi twitter yang disetting secara khusus menggunakan bahasa pemrograman Python. Twint dapat kita gunakan dan jalankan tanpa harus menggunakan API dari Twitter itu sendiri, dengan kapasitas scrapping data maksimalnya adalah 3200 tweet. Bukan hanya digunakan pada tweet, twint juga bisa kita gunakan untuk melakukan scrapping pada user, followers, retweet dan sebagainya. Twint memanfaatkan operator pencarian twitter untuk memungkinkan proses penghapusan tweet dari user tertentu, memilih dan memilah informasi-informasi yang sensitif, termasuk email dan nomor telepon di dalamnya.

# Data yang kita ambil ialah pemberitaan terbaru mengenai tragedi Kanjuruhan. Kata kunci yang digunakan 'trpagedi kanjuruhan' pada **c.search**, menggunakan Pandas pada **c.Pandas**, menggunakan limitasi data sebanyak 80 data pada **c.Limit**, dengan menggunakan custom data yang dimasukkan ke csx dengan label Tweet dan data yang diambil tweet-nya saja. Output atau data akan dimasukkan ke dalam file **csv**.

# ### Crawling Tweet

# In[ ]:


c = twint.Config()
c.Search = 'tragedi kanjuruhan'
c.Pandas = True
c.Limit = 80
c.Store_csv = True
c.Custom["tweet"] = ["tweet"]
c.Output = "dataKanjuruhan.csv"
twint.run.Search(c)


# In[18]:


pwd


# Membuka file **csv** 

# In[21]:


import pandas as pd
data = pd.read_csv('dataKanjuruhan.csv')
data


# ### Preprocessing

# Preprocessing adalah proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur. Selain itu, data mining juga tidak dapat memproses data mentah, sehingga proses ini sangat penting dilakukan untuk mempermudah proses berikutnya, yakni analisis data.

# 
# >**NLTK** adalah singkatan dari Natural Language Tool Kit, yaitu sebuah library yang digunakan untuk membantu kita dalam bekerja dengan teks. Library ini memudahkan kita untuk memproses teks seperti melakukan classification, tokenization, stemming, tagging, parsing, dan semantic reasoning.
# 
# >**Python Sastrawi** adalah pengembangan dari proyek PHP Sastrawi. Python Sastrawi merupakan library sederhana yang dapat mengubah kata berimbuhan bahasa Indonesia menjadi bentuk dasarnya. Sastrawi juga dapat diinstal melalui “pip”
# 
# 

# In[22]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# Pembuatan matriks menggunakan module pandas beserta numpy agar matriks yang dibuat sesuai dengan kebutuhan.
# 
# >**Pandas** adalah sebuah library di Python yang berlisensi BSD dan open source yang menyediakan struktur data dan analisis data yang mudah digunakan. Pandas biasa digunakan untuk membuat tabel, mengubah dimensi data, mengecek data, dan lain sebagainya. Struktur data dasar pada Pandas dinamakan DataFrame, yang memudahkan kita untuk membaca sebuah file dengan banyak jenis format seperti file .txt, .csv, dan .tsv. Fitur ini akan menjadikannya table dan juga dapat mengolah suatu data dengan menggunakan operasi seperti join, distinct, group by, agregasi, dan teknik lainnya yang terdapat pada SQL.
# 
# >**NumPy** merupakan singkatan dari Numerical Python. NumPy merupakan salah satu library Python yang berfungsi untuk proses komputasi numerik. NumPy memiliki kemampuan untuk membuat objek N-dimensi array. Array merupakan sekumpulan variabel yang memiliki tipe data yang sama. Kelebihan dari NumPy Array adalah dapat memudahkan operasi komputasi pada data, cocok untuk melakukan akses secara acak, dan elemen array merupakan sebuah nilai yang independen sehingga penyimpanannya dianggap sangat efisien.

# In[23]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# **Function Remove Stopwords** berguna menghapus kata-kata yang tidak diperlukan dalam proses nantinya,sehingga dapat mempercepat proses VSM. Kita meenggunakan kumpulan stopword dari github yang berjumlah sekitar 700 kata. 

# In[24]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/Web Mining/webmining/stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# **Stemming** merupakan proses mengubah kata dalam bahasa Indonesia ke akar katanya atau tidak ada kata yang berimbuhan pada awal maupun akhir kata serta tidak ada kata yang berulangan misalkan 'anak perempuan berjalan - jalan' menjadi 'anak perempuan jalan'

# In[25]:


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

# In[26]:


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

# In[28]:


data['tweet'].apply(preprocessing).to_csv('hasilPreprocessingKanjuruhan.csv')


# In[29]:


dataPre = pd.read_csv('hasilPreprocessingKanjuruhan.csv')
dataPre


# ### Vector Space Model (VSM)

# Vector Space Model (VSM) merupakan sebuah pendekatan natural yang berbasis pada vektor dari setiap kata dalam suatu dimensi spasial. Dokumen dipandang sebagai sebuah vektor yang memiliki magnitude (jarak) dan direction (arah). Pada VSM, sebuah kata direpresentasikan dengan sebuah dimensi dari ruang vektor. Relevansi sebuah dokumen ke sebuah kueri didasarkan pada similaritas diantara vektor dokumen dan vektor kueri.

# Import modul untuk membuat Vector Space Model dari library Sklearn, serta import data hasil preprocessing

# In[30]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('/content/drive/MyDrive/Web Mining/webmining/hasilPreprocessingKanjuruhan.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])


# Membuat matriks menjadi matriks array dan dilakukan shape pada matriks yang sudah dibuat 

# In[31]:


matrik_vsm = bag.toarray()
matrik_vsm.shape


# In[32]:


matrik_vsm[0]


# Mengambil semua kata yang sudah di tokenizing menjadi kolom - kolom atau fitur pada matriks VSM

# In[33]:


a = vectorizer.get_feature_names()


# Menampilkan Matriks VSM yang sduah dihitung frekuensi kemunculan term pada setiap tweet atau dokumen.

# In[34]:


dataTF = pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# ### K MEANS
# Algoritma k-means merupakan algoritma yang membutuhkan parameter input sebanyak k dan membagi sekumpulan n objek kedalam k cluster sehingga tingkat kemiripan antar anggota dalam satu cluster tinggi sedangkan tingkat kemiripan dengan anggota pada cluster lain sangat rendah. Kemiripan anggota terhadap cluster diukur dengan kedekatan objek terhadap nilai mean pada cluster atau dapat disebut sebagai centroid cluster.

# In[44]:


import matplotlib.pyplot as plt
from sklearn import metrics
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
plt.scatter(centroids[:,0] , centroids[:,1] , s = 50, color = 'green')


# ## Soal 2

# In[45]:


pwd


# ### Crawling berita dengan Scrapy

# Scrapy adalah kerangka kerja aplikasi untuk crawling web site dan mengekstraksi data terstruktur yang dapat digunakan untuk berbagai aplikasi yang bermanfaat, seperti data mining, pemrosesan informasi atau arsip sejarah. Meskipun Scrapy awalnya dirancang untuk web scraping, namu scrapy juga dapat digunakan untuk mengekstrak data menggunakan API (seperti Amazon Associates Web Services) atau sebagai web crawl

# Crochet adalah library pyhton berlisensi MIT yang memudahkan penggunaan Twisted dari regular blocking code. Beberapa kasus penggunaan meliputi:
# *   Mudah menggunakan Twisted dari blocking framework seperti Django atau Flask.
# *   Menulis library yang menyediakan blocking API, tetapi menggunakan Twisted untuk implementasinya.
# *  Port blocking code ke Twisted lebih mudah, dengan menjaga backwards compatibility layer.
# * Izinkan program Twisted normal yang menggunakan threads untuk berinteraksi dengan Twisted lebih bagus dari threaded parts. Misalnya, sangat berguna saat menggunakan Twisted sebagai WSGI container.

# In[46]:


get_ipython().system('pip install scrapy')
get_ipython().system('pip install crochet')


# Membuat Class untuk Crawling data berita dari sebuah portal berita yaitu [Tempo](https://nasional.tempo.co). Lalu custom setting dari hasil dari crawling menjadi file CSV. Parse data crawling dari URL dengan id 'isi' dan tag p. Class ExtractFirstLine untuk ekstak data berita dari dari web dan memisahkan dari tag HTML.

# In[47]:


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


# Jalankan fungsi untuk crawling data berita

# In[48]:


run_spider()


# Menampilkan data berita yang sudah dicrawling menggunakan pandas

# In[49]:


dataBerita = pd.read_csv('beritaUGM.csv')
dataBerita


# ### PyPDF2

# PyPDF2 adalah library Python yang memungkinkan manipulasi dokumen PDF. Ini dapat digunakan untuk membuat dokumen PDF baru, memodifikasi yang sudah ada, dan mengekstrak konten dari dokumen. PyPDF2 adalah library Python yang tidak memerlukan modul non-standar.

# In[50]:


get_ipython().system('pip install PyPDF2')


# import library PyPDF2 dan membuat variabel untuk membaca file PDF berita yang sudah crawling.

# In[51]:


import PyPDF2
pdfReader = PyPDF2.PdfFileReader('/content/drive/MyDrive/Web Mining/webmining/beritaUGM.pdf')
pageObj = pdfReader.getPage(0)
document = pageObj.extractText()
document


# ### PunktSentenceTokenizer
# PunktSentenceTokenizer adalah Sebuah libary untuk tokenizing atau memecah kalimat - kalimat pada sebuah paragraf.
# class tokenize terdiri dari 2 tahapan yaitu tahap prepocessing dan tahap memecah kalimat pada data beritaUGM

# In[52]:


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

# In[53]:


for j in range (len(sentences_list)):
    print('Kalimat {}'.format(j+1))
    print(sentences_list[j])


# ### TF IDF
# Tokenizing kata - kata pada kalimat sehingga bisa dihitung jumlah kosa kata serta menghitung TF IDF dari kata - kata tersebut.

# In[54]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
vectorizer = CountVectorizer()
cv_matrix=vectorizer.fit_transform(sentences_list)


# In[55]:


print ("Banyaknya kosa kata : ", len((vectorizer.get_feature_names_out())))
print ("Banyaknya kalimat : ", (len(sentences_list)))
print ("Kosa kata : ", (vectorizer.get_feature_names_out()))


# membuat matrix TF IDF dari kosa kata ada

# In[56]:


normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
normal_matrix.toarray()


# ### Networkx dan Graph
# 
# > Networkx adalah salah satu package pada bahasa pemrograman Python yang berfungsi untuk mengeksplorasi dan menganalisis jaringan dan algoritma jaringan.
# 
# 
# > Graph adalah jenis struktur data umum yang susunan datanya tidak berdekatan satu sama lain (non-linier). Graph terdiri dari kumpulan simpul berhingga untuk menyimpan data dan antara dua buah simpul terdapat hubungan saling keterkaitan. Simpul pada graph disebut dengan verteks (V), sedangkan sisi yang menghubungkan antar verteks disebut edge (E). Pasangan (x,y) disebut sebagai edge, yang menyatakan bahwa simpul x terhubung ke simpul y
# 
# Hitung perkalian matrix TF IDF dari kosa kata ada dengan matrix tranpose

# In[57]:


res_graph = normal_matrix * normal_matrix.T
print(res_graph)


# Membuat grafik dari res_graph yang sudah dihitung hasilnya

# In[59]:


import networkx as nx
nx_graph = nx.from_scipy_sparse_matrix(res_graph)
nx.draw_circular(nx_graph)


# menampilkan banyak sisi dari graph yang berjumlah 151 sisi

# In[60]:


print('Banyaknya sisi : {}'.format(nx_graph.number_of_edges()))


# ### Pagerank
# 
# > Algoritma PageRank bertujuan untuk mengukur hubungan kepentingan dalam kumpulan dokumen tersebut. Dalam algoritma PageRank dihasilkan matriks yang menghitung probabilitas bahwa pengguna akan berpindah dari satu halaman ke halaman lainnya. Algoritma PageRank dapat digunakan untuk memberikan peringkat pada
# setiap kalimat yang tersusun dalam sebuah graph. Peringkat yang dihasilkan
# oleh PageRank dapat digunakan untuk memastikan bahwa kalimat-kalimat
# yang dipilih oleh proses genetika adalah kalimat yang memiliki tingkat pentingnya tinggi. Semakin besar nilai PageRank maka semakin penting kalimat tersebut
# 
# inisiasi algoritma Pagerank dari graph nx_graph
# 

# In[64]:


ranks=nx.pagerank(nx_graph)


# input hasil perhitungan PageRank ke Array untuk ditampilkan dan dihitung urutkan angka PageRank terbesar sampai terkecil.

# In[65]:


arrRank=[]
for i in ranks:
    arrRank.append(ranks[i])


# Buat dataFrame dengan kolom data kalimat dan data nilai PageRank

# In[66]:


dfRanks = pd.DataFrame(arrRank,columns=['PageRank'])
dfSentence = pd.DataFrame(sentences_list,columns=['News'])
dfJoin = pd.concat([dfSentence,dfRanks], axis=1)
dfJoin


# Mengurutkan dataFrame berdasarkan nilai PageRank terbesar ke yang terkecil

# In[67]:


sortSentence=dfJoin.sort_values(by=['PageRank'],ascending=False)
sortSentence


# Dapat diketahui bahwa nilai PageRank terbesar ada pada kalimat indeks ke 8 lalu ke 5, ke 12, ke 7, dan ke 6.

# In[68]:


sortSentence.head(5)

