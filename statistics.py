from random import randint
from pymongo import MongoClient
import requests
import re
import csv
import tools


def count_number_of_references():
    persons_collection = tools.get_db_collection_object('Person')
    persons = persons_collection.find()

    statistics = []
    for person in persons:
        sentences_with_person_collection = tools.get_db_collection_object('SentencesWithPerson')
        result = sentences_with_person_collection.find({"person_id": person.get('_id')})

        count = 0
        for sentence in result:
            count += 1

        statistics.append(('{0} {1} {2}'.format(person['surname'], person['name'], person['patronymic']), count))

    statistics.sort(key=lambda x: x[1], reverse=True)
    for item in statistics:
        print('{0}: {1}'.format(item[0], item[1]))


def calculate_the_overall_rating_of_sentences():
    persons_collection = tools.get_db_collection_object('Person')
    persons = persons_collection.find()

    statistics = []
    for person in persons:
        tonality_sentences_with_person_collection = tools.get_db_collection_object('TonalitySentencesWithPerson')
        result = tonality_sentences_with_person_collection.find({"person_id": person.get('_id')})

        positive = 0
        negative = 0
        for rating in result:
            if rating['tonality'] > 0:
                positive += 1
            else:
                negative += 1

        negative = abs(negative)

        statistics.append((positive, negative, person['surname'] + " " + person['name'] + " " + person['patronymic']))
    statistics.sort(key=lambda x: x[0] + x[1], reverse=True)

    for item in statistics:
        total = item[0] + item[1]
        if total > 0:
            print(
                '{0}\t{1}({2}%)\t{3}({4}%)\t{5}'.format(total, item[0], round((item[0] / total) * 100, 2), item[1], round((item[1] / total) * 100, 2), item[2]))
        else:
            print('0\t0(0.0%)\t0(0.0%)\t{0}'.format(item[2]))


if __name__ == '__main__':
    calculate_the_overall_rating_of_sentences()

