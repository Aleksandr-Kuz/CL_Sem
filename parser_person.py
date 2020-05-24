# -*- coding: utf-8 -*-
import sys
import re
import requests
import traceback
from pymongo import MongoClient
from bs4 import BeautifulSoup


BASE_URL = 'https://global-volgograd.ru/person'
on_page = 20


def get_db_connection():
    client = MongoClient("mongodb+srv://parser:qB5kTo4xunLdKdl8@cluster-zufad.mongodb.net/test?retryWrites=true&w=majority")
    return client.CLSem


def get_html(url):
    r = requests.get(url)
    r.encoding = 'utf8'
    return r.text


def get_list_person(html):
    if html is None or html == '':
        raise ValueError('ERROR: Пустой HTML текст страницы ')

    soup = BeautifulSoup(html, features="html.parser")

    list_person = []
    try:
        list_block = soup.findAll('div', class_='person-block')
        for block in list_block:
            try:
                FIO = block.select('div.person-text a')[0].text
                FIO = list(map(lambda x: x.title(), FIO.split(' ')))
                surname = FIO[0]
                name = FIO[1]
                if len(FIO) == 3:
                    patronymic = FIO[2]
                else:
                    patronymic = ''
            except Exception:
                name = ''
                patronymic = ''
                surname = ''

            try:
                organization = block.select('div.person-text-org a')[0].text.strip()
            except Exception:
                organization = ''

            position = ''
            block_content = block.find('div', class_='person-text').find_all(text=True, recursive=False)
            for item in block_content:
                regex = r"\s*"
                text = re.sub(regex, '', str(item), re.MULTILINE)
                text = text.strip()
                if text:
                    position = text
                    break

            try:
                rating = int(block.select('div.person-position .position')[0].text.strip())
            except Exception:
                rating = 0

            list_person.append({
                "name": name,
                "surname": surname,
                'patronymic': patronymic,
                'organization': organization,
                'position': position,
                'rating': rating
            })

    except Exception as e:
        print(traceback.format_exc())

    return list_person


def write_list_person_to_db(list_person: list):
    db = get_db_connection()
    for person in list_person:
        db.Person.insert_one(person)
    return


def main():
    cur_page = 0
    while True:
        url = BASE_URL
        if cur_page > 0:
            url = BASE_URL + '?offset=' + str(on_page * cur_page)
        print(url)
        html = get_html(url)
        person_list = get_list_person(html)

        cur_page += 1

        if len(person_list) > 0:
            write_list_person_to_db(person_list)
        else:
            break


if __name__ == '__main__':
    main()
