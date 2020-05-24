# -*- coding: utf-8 -*-
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from pymongo import MongoClient

BASE_URL = 'https://vpravda.ru'
NEWS_URL = 'https://vpravda.ru/articles'
ON_PAGE = 40
required_amount = 400

start_page = 0


def get_db_connection():
    client = MongoClient("mongodb+srv://parser:qB5kTo4xunLdKdl8@cluster-zufad.mongodb.net/test?retryWrites=true&w=majority")
    return client.CLSem


def write_list_news_to_db(news_list: list):
    db = get_db_connection()
    for news in news_list:
        db.News.insert_one(news)
    return


def get_html(url):
    r = requests.get(url)
    r.encoding = 'utf8'
    return r.text


def get_detail_info_about_article(url_article):
    if url_article is None or url_article == '':
        raise ValueError('ERROR: Задана пустая сслылка на статью')

    html = get_html(BASE_URL + url_article)
    soup = BeautifulSoup(html, features="html.parser")


    # Title
    try:
        title = soup.h1.text
    except Exception:
        title = None

    # Date
    try:
        date_str = soup.select_one('div.field.field-name-field-article-date.field-type-date.field-label-hidden > div > div > span').text
        article_date = datetime.strptime(date_str, '%d.%m.%Y в %H:%M')
    except Exception:
        article_date = None

    # Text
    try:
        subtitle = soup.find('div', class_='field-name-field-article-lead').text.strip().rstrip('.')

        article_content = soup.select_one('div.field-name-body')
        text = subtitle + '. ' + article_content.get_text()

        regex = r"\s+"
        text = re.sub(regex, ' ', text)

    except Exception:
        text = ''

    # Author
    try:
        copy_right = soup.select_one('div.field.field-name-field-article-author.field-type-text.field-label-hidden > div > div').text
        author = copy_right.split('.')[0]
    except Exception:
        author = None

    # Comment
    try:
        cnt_comment = len(soup.select('#comments > article'))
    except Exception:
        cnt_comment = None

    return {
        'url': url_article,
        'title': title,
        'date': article_date,
        'text': text,
        'author': author,
        'cnt_comment': cnt_comment
    }


def get_list_article_on_page(URL):
    try:
        html = get_html(URL)
        soup = BeautifulSoup(html, features="html.parser")

        articles = []
        article_list_block = soup.select('div#main div.view-display-id-page div.view-content > div')
        i = 1
        for article_block in article_list_block:
            link = ''
            try:
                link = article_block.select_one('a')['href']
                print('%d\t %s' % (i, link))

                if link:
                    article = get_detail_info_about_article(link)
                    articles.append(article)
                else:
                    print('---> Не найдена ссылка на детальную страницу нововсти')

                i += 1

            except Exception as err:
                print('>>> Ошибка обработки статьи! url: ' + link)
                print(str(err))
                continue

    except Exception as err:
        print('>>> Ошибка обработки страницы списка! url: ' + URL)
        print(str(err))
        return []

    return articles


def main():
    cur_count = 0
    cur_page = start_page
    while True:
        url = NEWS_URL + '/?page=' + str(cur_page)
        print('Обрабатывается ' + str(cur_page) + ' страница')

        articles = get_list_article_on_page(url)
        write_list_news_to_db(articles)

        cur_page += 1
        cur_count += ON_PAGE

        if cur_count >= required_amount:
            break



if __name__ == '__main__':
    main()
