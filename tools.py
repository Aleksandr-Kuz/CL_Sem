# -*- coding: utf-8 -*-
import re
import sys
import os

import pickle
import random
import pymorphy2
import requests
import subprocess
from time import time
from datetime import datetime
from bs4 import BeautifulSoup
from pymongo import MongoClient
from TomitaParser import TomitaParser

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Word2Vec

from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


necessary_part = ["NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS", "GRND"]

config_sentence_divider = '/home/vagrant/CL_Lab/Sem/sentenceDivider/config.proto'
config_extractFIO = '/home/vagrant/CL_Lab/Sem/extractFIO/config.proto'

exec_path = '/home/vagrant/tomita-parser/build/bin/tomita-parser'
descriptors = [
    ['pipe', 'r'],  # stdin
    ['pipe', 'w'],  # stdout
    ['pipe', 'w']   # stderr
]


def show_progress(value):
    sys.stderr.write('%s\r' % value)


def get_only_words(tokens):
    return list(filter(lambda x: re.match('[a-zA-ZА-Яа-я]+', x), tokens))

# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)

    if iteration == total:
        print()


def get_db_connection():
    client = MongoClient("mongodb+srv://parser:qB5kTo4xunLdKdl8@cluster-zufad.mongodb.net/test?retryWrites=true&w=majority")
    return client.CLSem


def get_db_collection_object(collection_name: str):
    db = get_db_connection()
    if collection_name == 'News':
        return db.News
    elif collection_name == 'Person':
        return db.Person
    elif collection_name == 'Sights':
        return db.Sights
    elif collection_name == 'SentencesWithPerson':
        return db.SentencesWithPerson
    else:
        return None


def find_document(collection, elements, multiple=False):
    if multiple:
        results = collection.find(elements)
        return [r for r in results]
    else:
        return collection.find_one(elements)


def check_person_in_db(surname: str, name=None, patronymic=None):
    try:
        person_collection = get_db_collection_object('Person')
        filter = {
            'surname': surname
        }

        if name is not None:
            # regx = re.compile('^{0}{1}'.format(name, '$' if len(name) > 1 else ''), re.IGNORECASE)
            regx = re.compile('^{0}'.format(name), re.IGNORECASE)
            filter.update({'name': regx})
        if patronymic is not None:
            # regx = re.compile('^{0}{1}'.format(patronymic, '$' if len(patronymic) > 1 else ''), re.IGNORECASE)
            regx = re.compile('^{0}'.format(patronymic), re.IGNORECASE)
            filter.update({'patronymic': regx})

        return find_document(person_collection, filter, False)

    except:
        return None


def get_and_save_text_news():
    news_collection = get_db_collection_object('News')
    if not news_collection:
        return

    news_list = news_collection.find().limit(10)

    print('Анализ продложений из текстов статей')

    printProgressBar(0, news_list.count(), prefix='Progress:', suffix='Complete', length=50)
    for idx, news in enumerate(news_list):
        tic = time()
        try:
            title = news['title'].rstrip('.')
            text = news['text']

            tomita = TomitaParser(exec_path, config_sentence_divider, debug=False)
            output, errors = tomita.run('{0}. {1}'.format(title, text))

            with open("./sentences/{0}.txt".format(news.get('_id')), "w", encoding='utf-8') as file_sentences:
                for lines in output.split('\n'):
                    if len(lines.strip()) > 0:
                        file_sentences.write('{1}\n'.format(news.get('_id'), lines.strip()))
        except Exception:
            pass

        toc = time()
        print(toc - tic)

        printProgressBar(idx + 1, news_list.count(), prefix='Progress:', suffix='Complete', length=50)

    print('Все предложения записаны в файл sentences.txt')


            # processing_sentences(news.get('_id'))
            # print('Записано в файл  {0} из {1}\r'.format(str(idx + 1), str(news_list.count())))


def get_analogy(s, model):
    qry = model.transform(s[0]) - model.transform(s[1]) - model.transform(s[2])
    res = model.findSynonyms((-1)*qry, 5)  # return 5 "synonyms"
    res = [x[0] for x in res]
    for k in range(0,3):
        if s[k] in res:
            res.remove(s[k])
    return res[0]


def processing_sentence(sentence: str):
    print(sentence)
    if len(sentence) < 3:
        return sentence, []

    try:
        tmp = sentence.split('||')
        news_id = ''
        text_sentence = ''
        if len(tmp) == 2:
            news_id = tmp[0]
            text_sentence = tmp[1]
        else:
            text_sentence = sentence


        tomita = TomitaParser(exec_path, config_extractFIO, debug=False)
        facts, leads = tomita.run(text_sentence)

        if len(facts) > 0:
            mas_sentences_with_person = []

            for fact in facts:
                if 'surname' not in fact:
                    continue
                name = None if 'name' not in fact else fact['name']
                patronymic = None if 'patrn' not in fact else fact['patrn']

                person_obj = check_person_in_db(fact['surname'], name, patronymic)

                if person_obj is not None:
                    sentences_with_person_collection = get_db_collection_object('SentencesWithPerson')
                    sentences_with_person = {
                        'news_id': news_id,
                        'person_id': person_obj.get('_id'),
                        'sentence': text_sentence,
                        'person_FIO': '{0} {1} {2}'.format(person_obj['surname'], person_obj['name'], person_obj['patronymic'])
                    }
                    res = sentences_with_person_collection.insert_one(sentences_with_person)
                    mas_sentences_with_person.append(str(res.inserted_id))

            if len(mas_sentences_with_person) > 0:
                return text_sentence, mas_sentences_with_person
            else:
                return text_sentence, []
        else:
            return text_sentence, []
    except Exception:
        return sentence, []


def normalization_sentence(tokens):
    normal_words = []
    try:
        morf = pymorphy2.MorphAnalyzer()

        for word in tokens:
            p = morf.parse(word.lower())[0]
            part = p.tag.POS
            if part in necessary_part:
                normal_words.append(p.normal_form)
        return normal_words
    except:
        return []


def processing_extract_person():
    files = os.listdir('./sentences/')
    print(files)
    for file_name in files:
        with open('./sentences/{0}'.format(file_name), "r", encoding='utf-8') as file_sentences:
            for line in file_sentences.readlines():
                processing_sentence(line.strip())



def train_model_sentences_with_person():
    sentences_with_person_collection = get_db_collection_object('SentencesWithPerson')

    with open("sentences_with_person.txt", "w", encoding='utf-8') as file_sentences_with_person:
        for sen in sentences_with_person_collection.find():
            file_sentences_with_person.write('{0}\n'.format(sen['sentence']))

    spark = SparkSession \
        .builder \
        .appName("SentenceProcessor") \
        .getOrCreate()

    input_data = spark.sparkContext.textFile('./sentences_with_person.txt')
    prepared_data = input_data.map(lambda x: (x, len(x)))
    prepared_data = prepared_data.filter(lambda x: x[1] > 0)

    prepared_df = prepared_data.toDF().selectExpr('_1 as sentence', '_2 as length')
    # prepared_df.show(truncate=False)

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    words_data = tokenizer.transform(prepared_df)
    # words_data.show(truncate=False)

    # Отфильтровать токены, оставив только слова
    filtered_words_data = words_data.rdd.map(lambda x: (x[0], x[1], get_only_words(x[2])))
    filtered_df = filtered_words_data.toDF().selectExpr('_1 as sentence', '_2 as length', '_3 as words')
    # filtered_df.show()

    # Удалить стоп-слова (союзы, предлоги, местоимения и т.д.)
    stop_words = stopwords.words('russian')
    remover = StopWordsRemover(inputCol='words', outputCol='filtered', stopWords=stop_words)
    filtered = remover.transform(filtered_df)

    #
    normalize_words_data = filtered.rdd.map(lambda x: (x[0], x[1], x[2], normalization_sentence(x[3])))
    normalized_df = normalize_words_data.toDF().selectExpr('_1 as sentence', '_2 as length', '_3 as words', '_4 as normalize_words')
    # normalized_df.show()

    #
    vectorizer = CountVectorizer(inputCol='normalize_words', outputCol='raw_features').fit(normalized_df)
    featurized_data = vectorizer.transform(normalized_df)
    featurized_data.cache()

    #
    idf = IDF(inputCol='raw_features', outputCol='features')
    idf_model = idf.fit(featurized_data)
    rescaled_data = idf_model.transform(featurized_data)

    # Построить модель Word2Vec
    word2Vec = Word2Vec(vectorSize=300, minCount=0, inputCol='normalize_words', outputCol='result')
    doc2vec_pipeline = Pipeline(stages=[tokenizer, word2Vec])
    model = word2Vec.fit(rescaled_data)
    w2v_df = model.transform(rescaled_data)
    # w2v_df.show(truncate=False)

    # print(model.findSynonyms('бочаров', 2).show())

    # sc = spark.sparkContext
    path = './models/model_person'
    #
    # print(sc, path)
    model.write().overwrite().save(path)

    #m = Word2Vec.load('./models/model_person/')
    # pickle.dump(model, './models/model_person/mp.model')

    spark.stop()

    # s = ('губернатор', 'андрей', 'высокий')
    # print(get_analogy(s, model))
    # print(model.getVectors().show())

    # w2v_df.select(['sentence', 'normalize_words', 'result']).show(truncate=False)



if __name__ == '__main__':
    get_and_save_text_news()
    # train_model_sentences_with_person()
    #
    # spark = SparkSession \
    #     .builder \
    #     .appName("SentenceProcessor_1") \
    #     .getOrCreate()
    #
    # w2v_model = Word2Vec.load('./models/model_person/')
    #
    # spark.stop()
    #
    # train_model_sentences_with_person()
    # spark_processing()
    # get_and_save_text_news()





# mas_sentence = output.split('\n')
# for s in mas_sentence:
#     file_sentences.write('{0}\n'.format(s.strip()))

# subproc = subprocess.Popen([exec_path, config_sentence_divider], stdin=sys.stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# output, errors = subproc.communicate(input='{0}. {1}'.format(title, text).encode('utf-8'))

# subproc = subprocess.Popen([exec_path, config_sentence_divider], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                            stderr=subprocess.PIPE)
# subproc.stdin.write(b"input")
# data = subproc.communicate() # output, errors
# subproc.wait()