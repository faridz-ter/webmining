#!/usr/bin/env python
# coding: utf-8

# In[1]:


try: 
  import scrapy
except:
  get_ipython().system('pip install scrapy')
  import scrapy


# In[2]:


from scrapy.crawler import CrawlerProcess as crawling

class Spider (scrapy.Spider):
  name = 'link'
  start_urls = ['https://pta.trunojoyo.ac.id/welcome/detail/080411100001']

  def parse(self, response):
    for jurnal in response.css('#content_journal > ul > li'):
      yield {
          'judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
          'abtraksi': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
          
      }
prosesCrawling = crawling()
prosesCrawling.crawl(Spider)
prosesCrawling.start()


# Hasil Crawling Data text pada halaman [pta.trunojoyo.ac.id](https://)
# 
# 
# > *{'judul': ['SISTEM PERAMALAN PENJUALAN JANGKA PENDEK SPARE PART SEPEDA MOTOR MENGGUNAKAN NEURAL NETWORK\r\n(Studi Kasus : Suzuki Kemayoran Bangkalan)'], 'abtraksi': ['Spare part merupakan salah satu bagian penting dalam pengoperasian mesin pada sepeda motor. Peningkatan jumlah penjualan spare part yang tidak terduga saat proses tune-up menyebabkan kesulitan dalam pelayanan yang terbaik kepada konsumen. Demikian juga sebaliknya, apabila terjadi penurunan jumlah penjualan spare part, maka akan menyebabkan penumpukan spare part di gudang. Oleh karena itu diperlukan sistem peramalan yang mampu meramalkan penjualan spare part pada periode berikutnya. Sistem peramalan ini menggunakan metode Jaringan Syaraf Tiruan algoritma Propagasi Balik dengan momentum untuk meramalkan jumlah penjualan spare part  pada bulan berikutnya. Data yang telah tersimpan dihitung menggunakan epoh dan learning rate yang berbeda. Dari hasil uji coba system, maka dapat disimpulkan bahwa  dengan menggunakan semua data sebagai data training dan menggunakan learning rate 3.5 dan dengan epoh 200 akan menghasilkan tingkat kesalahan 0.0622716.']}*
# 
# 
# 
# 
# 
# 
# 
