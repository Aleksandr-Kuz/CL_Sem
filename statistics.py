from random import randint
from pymongo import MongoClient
import requests
import re
import csv
import tools

if __name__ == '__main__':

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
