import scrapy
from bs4 import BeautifulSoup
import re
from urllib.request import urlopen
import time

class TXTSpider(scrapy.Spider):
    name = 'mwe_spider'
    # start_urls = url_list


    def parse(self,links):

        # current_url = response.request.url
        #
        text = ''
        visit = set()
        counter = 1
        for url in links:
            print(url)
            try:
              page = urlopen(url)
              html = page.read().decode("utf-8")
            except Exception as e:
              print(e)
              continue
            # html = response.body
            split = url.split(".")
            # ind = split.index('www')
            filename = split[-2]
            soup = BeautifulSoup(html, 'lxml')

            text1 = soup.get_text()
            # print(text1)
            text1 = text1.replace('\n',' ')
            text1 = ' '.join(text1.split())
            urls = re.findall('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', html)
            text = text + '\n' +text1
            print(text1)
            # print(urls)
            for i in urls:
              if i not in visit and 'verizon' in i:
                visit.add(i)
                ind = i.find('"')
                if 'https' not in i:
                    links.append('https://'+i[:ind])
                else:

                    links.append(i[:ind])
            counter+=1
            time.sleep(10)
            # with open(f'{filename}.html', 'wb') as fp:
            #     fp.write(response.body)
            self.log(f'Saved file {filename}')
            if counter ==101:
              break
        with open(f'verizon_{counter}.txt', 'w') as fp:
              # counter+=1
              fp.write(text)
if __name__ == "__main__":
    scraper = TXTSpider()
    scraper.parse(['https://www.verizon.com/home/internet/','https://community.verizon.com/'])
