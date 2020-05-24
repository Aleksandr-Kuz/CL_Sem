# -*- coding: utf-8 -*-
import re
import requests
import traceback
from pymongo import MongoClient
from bs4 import BeautifulSoup

BASE_URL = 'https://avolgograd.com/sights?obl=vgg'


def get_db_connection():
    client = MongoClient("mongodb+srv://parser:qB5kTo4xunLdKdl8@cluster-zufad.mongodb.net/test?retryWrites=true&w=majority")
    return client.CLSem


def get_html(url: str, method: str, param=None):
    if param is None:
        param = []
    if method.upper() == 'GET':
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36 OPR/68.0.3618.63 (Edition Yx)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-User': '?1',
            'Sec-Fetch-Dest': 'document',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Cookie': '_ym_uid=1589753998643935356; _ym_d=1589753998; _ym_isad=2; _ym_visorc_38657565=w'
        }
        r = requests.get(url, headers=headers)
    elif method.upper() == 'POST':
        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            # 'Content-Length': '538',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Cookie': '_ym_uid=1589753998643935356; _ym_d=1589753998; _ym_isad=2; _ym_visorc_38657565=w',
            'Host': 'avolgograd.com',
            'Origin': 'https://avolgograd.com',
            'Referer': 'https://avolgograd.com/sights?obl=vgg',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36 OPR/68.0.3618.63 (Edition Yx)',
            'X-Requested-With': 'XMLHttpRequest'
        }
        r = requests.post(url, param, headers=headers)
    else:
        return ''

    r.encoding = 'utf8'
    return r.text


def get_list_sights(html):
    if html is None or html == '':
        raise ValueError('ERROR: Пустой HTML текст страницы ')
    soup = BeautifulSoup(html, features="html.parser")

    list_sights = []
    try:
        list_block = soup.findAll('div', class_='ta-f-box ta-f-box-cat')
        for block in list_block:
            try:
                name = block.select('div.ta-200 .ta-211')[0].text
            except Exception:
                name = ''

            try:
                address = block.select('div.ta-200 .ta-212 div:nth-child(2)')[0].text
            except Exception:
                address = ''

            try:
                views = block.select('div.ta-200 .ta-221:nth-child(2) > span')[0].text
                views = int(views.strip())
            except Exception:
                views = 0

            list_sights.append({
                "name": name,
                "address": address,
                'views': views,
            })

    except Exception as e:
        print(traceback.format_exc())

    return list_sights


def write_list_sights_to_db(sights_list: list):
    db = get_db_connection()
    for sight in sights_list:
            db.Sights.insert_one(sight)
    return


def main():

    url = BASE_URL
    html = get_html(url, 'GET')
    sights_list = get_list_sights(html)

    if len(sights_list) > 0:
        write_list_sights_to_db(sights_list)

    url = 'https://avolgograd.com/wp-admin/admin-ajax.php'
    param = {
        'action': 'loadmore',
        'args': 'a:7:{s:9:"post_type";s:3:"any";s:11:"post_status";s:7:"publish";s:6:"sights";s:3:"vlg";s:14:"posts_per_page";i:30;s:7:"orderby";s:14:"modified title";s:5:"order";s:4:"DESC";s:9:"tax_query";a:1:{i:0;a:3:{s:8:"taxonomy";s:8:"a-oblast";s:5:"field";s:2:"id";s:5:"terms";i:3600;}}}',
        'page': '1',
        'tmplt': '',
        'txnm': 'sights',
        'trmid': '64'
    }
    html = get_html(url, 'POST', param)
    sights_list = get_list_sights(html)
    if len(sights_list) > 0:
        write_list_sights_to_db(sights_list)


if __name__ == '__main__':
    main()
