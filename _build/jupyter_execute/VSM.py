#!/usr/bin/env python
# coding: utf-8

# # Vector Space Model

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Web Mining/content')


# In[ ]:


try : 
  import scrapy
except : 
  get_ipython().system('pip install scrapy')
  import scrapy


# In[ ]:


import pandas as pd


# In[ ]:


class LinkSpider(scrapy.Spider):
    name = 'link'
    start_urls=[]
    for i in range (1, 50+1):
        start_urls.append(f'https://pta.trunojoyo.ac.id/c_search/byprod/10/{i}')
    def parse(self, response):
        count = 0 
        link = []
        for jurnal in response.css('#content_journal > ul'):
            count += 1
            for j in range(1, 6):
                yield {
                    'link' : response.css(f'li:nth-child({j}) > div:nth-child(3) > a::attr(href)').get(),
                }


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Web Mining/webmining')


# In[ ]:


linkHasilCrawl = pd.read_csv('hasilCrawlLink.csv')
linkHasilCrawl


# In[ ]:


class Spider(scrapy.Spider):
    name = 'detail'
    data_link = pd.read_json('hasilLink.json').values
    start_urls = [link[0] for link in data_link]

    def parse(self, response):
        yield{
            'abstraksi' : response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
        }


# In[ ]:


df = pd.read_csv('hasilCrawl.csv')


# In[ ]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# In[ ]:


import pandas as pd
import re
import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# In[ ]:


data = pd.read_csv('hasilCrawl.csv')


# In[ ]:


def remove_stopwords(text):
    with open('stopword.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]

    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]

    return text


# In[ ]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    result = [stemmer.stem(word) for word in text]

    return text


# In[ ]:


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


# In[ ]:


tf = pd.DataFrame()
for i,v in enumerate(data['Abstraksi']):
    cols = ["Doc " + str(i+1)]    
    doc = pd.DataFrame.from_dict(nltk.FreqDist(preprocessing(v)), orient='index',columns=cols) 
    #doc.columns = [data['Judul'][i]]    
    tf = pd.concat([tf, doc], axis=1, sort=False)


# In[ ]:


tf.index.name = 'Term'
tf = pd.concat([tf], axis=1, sort=False)
tf = tf.fillna(0)
tf

