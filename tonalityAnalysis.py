# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
import _pickle as cPickle
import pymorphy2
import csv
import re

filenamePositive = "./csv/positive.csv"
filenameNegative = "./csv/negative.csv"

filenameNormFormPos = "./csv/normFormPos.csv"
filenameNormFormNeg = "./csv/normFormNeg.csv"
filenameNormFormPos_test = "./csv/normFormPos_test.csv"
filenameNormFormNeg_test = "./csv/normFormNeg_test.csv"

filenameVecPos = "./csv/vecPos.csv"
filenameVecNeg = "./csv/vecNeg.csv"
filenameVecPos_test = "./csv/vecPos_test.csv"
filenameVecNeg_test = "./csv/vecNeg_test.csv"

filenameWVModel = "./models/W2V/analyzer.model"
filenameClassifier = "./models/Classifier/classifier.pkl"

dataPositive = []
dataNegative = []
dataPositiveTest = []
dataNegativeTest = []

necessary_part = ["NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS", "GRND"]
morf = pymorphy2.MorphAnalyzer()
vector_size = 300

WVmodel = 0
forest = 0


def quantityRowInCSV(filename):
    num = 0
    with open(filename, "r", encoding='utf8', newline="") as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
        num = sum(1 for line in reader)
    return num


def normalizationOfSentence(sentence):
    normal_words = []
    s = sentence.lower()
    # s = s.translate(
    #     str.maketrans("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789", " " * 42))
    tokens = s.split()
    for word in tokens:
        p = morf.parse(word)[0]
        part = p.tag.POS
        if part in necessary_part:
            normal_words.append(p.normal_form)
    return normal_words


def create_w2v_model(dt_Positive, dt_Negative, dt_PositiveTest, dt_NegativeTest):
    # Нормализация
    ldt_Positive = len(dt_Positive)
    ldt_Negative = len(dt_Negative)
    ldt_PositiveTest = len(dt_PositiveTest)
    ldt_NegativeTest = len(dt_NegativeTest)
    sumL = ldt_Positive + ldt_Negative + ldt_PositiveTest + ldt_NegativeTest
    print(ldt_Positive + ldt_Negative + ldt_PositiveTest + ldt_NegativeTest)

    for i in range(0, ldt_Positive):
        if i < len(dt_Positive):
            norm = normalizationOfSentence(dt_Positive[i])
            if len(norm) == 0:
                dt_Positive.remove(dt_Positive[i])
            else:
                dt_Positive[i] = norm
        print("Нормализовано ", (i + 1) / sumL * 100, "%")
    for i in range(0, ldt_Negative):
        if i < len(dt_Negative):
            norm = normalizationOfSentence(dt_Negative[i])
            if len(norm) == 0:
                dt_Negative.remove(dt_Negative[i])
            else:
                dt_Negative[i] = norm
        print("Нормализовано ", (ldt_Positive + i + 1) / sumL * 100, "%")

    LP_N = ldt_Positive + ldt_Negative

    for i in range(0, ldt_PositiveTest):
        if i < len(dt_PositiveTest):
            norm = normalizationOfSentence(dt_PositiveTest[i])
            if len(norm) == 0:
                dt_PositiveTest.remove(dt_PositiveTest[i])
            else:
                dt_PositiveTest[i] = norm
        print("Нормализовано ", (LP_N + i + 1) / sumL * 100, "%")
    LP_N += ldt_PositiveTest
    for i in range(0, ldt_NegativeTest):
        if i < len(dt_NegativeTest):
            norm = normalizationOfSentence(dt_NegativeTest[i])
            if len(norm) == 0:
                dt_NegativeTest.remove(dt_NegativeTest[i])
            else:
                dt_NegativeTest[i] = norm
        print("Нормализовано ", (LP_N + i + 1) / sumL * 100, "%")

    print("\nНачато создание модели")
    mdl = Word2Vec(dataPositive + dataNegative + dt_PositiveTest + dt_NegativeTest, size=vector_size, window=7,
                   min_count = 0, workers=8, sg=1)
    mdl.init_sims(replace=True)
    mdl.save(filenameWVModel)
    print("     Создание модели закончено")
    return mdl


def load_w2v_Model():
    return Word2Vec.load(filenameWVModel)