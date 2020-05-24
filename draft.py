from random import randint
from pymongo import MongoClient
import requests
import re

from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

BASE_URL = 'https://avolgograd.com/sights?obl=vgg'


def get_db_connection():
    client = MongoClient("mongodb+srv://parser:qB5kTo4xunLdKdl8@cluster-zufad.mongodb.net/test?retryWrites=true&w=majority")
    return client.CLSem


db = get_db_connection()
db.SentencesWithPerson.remove()


# def get_db_collection_object(collection_name: str):
#     db = get_db_connection()
#     if collection_name == 'News':
#         return db.News
#     elif collection_name == 'Person':
#         return db.Person
#     elif collection_name == 'Sights':
#         return db.Sights
#     else:
#         return None
#
#
# def find_document(collection, elements, multiple=False):
#     if multiple:
#         results = collection.find(elements)
#         return [r for r in results]
#     else:
#         return collection.find_one(elements)
#
#
# def check_person_in_db(surname: str, name=None, patronymic=None):
#     person_collection = get_db_collection_object('Person')
#     filter = {
#         'surname': surname
#     }
#
#     if name is not None:
#         # regx = re.compile('^{0}{1}'.format(name, '$' if len(name) > 1 else ''), re.IGNORECASE)
#         regx = re.compile('^{0}'.format(name), re.IGNORECASE)
#         filter.update({'name': regx})
#     if patronymic is not None:
#         # regx = re.compile('^{0}{1}'.format(patronymic, '$' if len(patronymic) > 1 else ''), re.IGNORECASE)
#         regx = re.compile('^{0}'.format(patronymic), re.IGNORECASE)
#         filter.update({'patronymic': regx})
#
#     print(filter)
#
#     return find_document(person_collection, filter, False)


# {'name': 'Андрей11111', 'surname': 'Бочаров'}
# results = check_person_in_db('Бочаров', 'Андрей', 'Петр')
# if results is None:
#     print('x3')
# else:
#     print(results)


# db = client.CLSem
# result = db.News.remove()

# names = ['Kitchen','Animal','State', 'Tastey', 'Big','City','Fish', 'Pizza','Goat', 'Salty','Sandwich','Lazy', 'Fun']
# company_type = ['LLC','Inc','Company','Corporation']
# company_cuisine = ['Pizza', 'Bar Food', 'Fast Food', 'Italian', 'Mexican', 'American', 'Sushi Bar', 'Vegetarian']
#
# person = {
#     "name": 'Андрей',
#     "surname": 'Бочаров',
#     'patronymic': 'Иванович',
#     'organization': 'Администрация (правительство) Волгоградской област',
#     'position': 'Губернатор',
#     'rating': 1
# }
#
# # result = db.Person.insert_one(person)
# result = db.Person.remove()
# print(result)


# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36 OPR/68.0.3618.63 (Edition Yx)',
#     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
#     'Sec-Fetch-Site': 'none',
#     'Sec-Fetch-Mode': 'navigate',
#     'Sec-Fetch-User': '?1',
#     'Sec-Fetch-Dest': 'document',
#     'Accept-Encoding': 'gzip, deflate, br',
#     'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
#     'Cookie': '_ym_uid=1589753998643935356; _ym_d=1589753998; _ym_isad=2; _ym_visorc_38657565=w'
# }
# r = requests.get(BASE_URL, headers=headers)
# r.encoding = 'utf8'
# print(r.text)
