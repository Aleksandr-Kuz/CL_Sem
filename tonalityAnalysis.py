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

necessary_part = ["NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS", "GRND"]
morf = pymorphy2.MorphAnalyzer()
vector_size = 300

# W2V_model = 0
# classifier = 0


def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)

    if iteration == total:
        print()


def read_twit(file_path):
    training_sample = []
    test_sample = []
    with open(file_path, "r", encoding='utf8', newline="") as file:
        counter = 1
        reader = csv.reader(file, delimiter=';', quotechar='"')
        num = quantityRowInCSV(file_path)
        for row in reader:
            string = re.sub(r"[^А-Яа-я\s]+", "", row[3]).strip()
            string = re.sub(r"[_A-Za-z0-9]+", "", string).strip()
            string = re.sub(r"[\s]{2,}", " ", string)
            if counter / num <= 0.9:
                training_sample.append(string)
            else:
                test_sample.append(string)
            counter += 1

    return training_sample, test_sample


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


def load_w2v_model():
    return Word2Vec.load(filenameWVModel)


def recordNormalForms(dt, file_path):
    with open(file_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        for line in dt:
            writer.writerow(line)


def normalization_mas_sentence(mas_sentence, file_path_save=None):
    result = []
    progress_bar(0, len(mas_sentence), prefix='Progress:', suffix='Complete', length=50)
    for i in range(0, len(mas_sentence)):
        norm = normalizationOfSentence(mas_sentence[i])
        if len(norm) != 0:
            result.append(norm)
        progress_bar(i, len(mas_sentence), prefix='Progress:', suffix='Complete', length=50)

    if file_path_save is not None:
        recordNormalForms(result, file_path_save)

    if len(result) > 0:
        return result


def create_w2v_model(dt_Positive, dt_Negative, dt_PositiveTest, dt_NegativeTest):
    mdl = Word2Vec(dt_Positive + dt_Negative + dt_PositiveTest + dt_NegativeTest, size=vector_size, window=7,
                   min_count=0, workers=8, sg=1)
    mdl.init_sims(replace=True)
    mdl.save(filenameWVModel)
    return mdl


def test_model(model, data_positive_test, data_negative_test):
    arr = model.predict(data_positive_test)
    number_positive = 0
    for i in arr:
        if i == 1:
            number_positive += 1

    arr = model.predict(data_negative_test)
    number_negative = 0
    for i in arr:
        if i == -1:
            number_negative += 1

    print("ACC = ", (number_positive + number_negative) / (len(data_positive_test) + len(data_negative_test)))


def featurize_w2v(model, sentences):
    v = []
    for word in sentences:
        n = 0
        count = 0
        for i in range(0, vector_size):
            print(word)
            try:
                vec = model[word]
                n += vec[i]
                count += 1
            except KeyError:
                continue
        v.append(n / count)

    return v


def calc_vector(WVmodel, mas_sentence, file_path=None):
    for i in range(0, len(mas_sentence)):
        mas_sentence[i] = featurize_w2v(WVmodel, mas_sentence[i])

    if file_path is not None:
        with open(file_path, "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            for line in mas_sentence:
                writer.writerow(line)


def read_normalize_form(file_path):
    result = []
    with open(file_path, "r", encoding='utf8', newline="") as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
        for row in reader:
            result.append(row)
    return result


def init(read_source=False, read_normalize=False, load_models=False, load_classifier=False):
    if read_source:
        # Чтение позитивных
        print("Чтение позитивных твитов")
        data_positive, data_positive_test = read_twit(filenamePositive)
        print("     Готово")

        # Чтение негативных
        print("Чтение негативных твитов")
        data_negative, data_negative_test = read_twit(filenameNegative)
        print("     Готово")

        # Нормализация
        print("Нормализация предложений")
        data_positive = normalization_mas_sentence(data_positive, file_path_save=filenameNormFormPos)
        data_negative = normalization_mas_sentence(data_negative, file_path_save=filenameNormFormNeg)
        data_positive_test = normalization_mas_sentence(data_positive_test, file_path_save=filenameNormFormPos_test)
        data_negative_test = normalization_mas_sentence(data_negative_test, file_path_save=filenameNormFormNeg_test)
        print("     Готово")

    if read_normalize:
        data_positive = read_normalize_form(filenameNormFormPos)
        data_negative = read_normalize_form(filenameNormFormNeg)
        data_positive_test = read_normalize_form(filenameNormFormPos_test)
        data_negative_test = read_normalize_form(filenameNormFormNeg_test)

    if not load_models:
        print("Создание WV модели")
        model_w2v = create_w2v_model(data_positive, data_negative, data_positive_test, data_negative_test)
        print("     Готовов")
    else:
        print("Загрузка WV модели")
        model_w2v = load_w2v_model()
        print("     Модель загружена")

    if not load_classifier:
        print("Считаем векора")
        calc_vector(model_w2v, data_positive, filenameVecPos)
        calc_vector(model_w2v, data_negative, filenameVecNeg)
        calc_vector(model_w2v, data_positive_test, filenameVecPos_test)
        calc_vector(model_w2v, data_negative_test, filenameVecNeg_test)
        print("     Готовов")

        print("\nНачато создание леса")
        Y_pos = [1 for _ in range(len(data_positive))]
        Y_neg = [-1 for _ in range(len(data_negative))]

        forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        classifier.fit(data_positive + data_negative, Y_pos + Y_neg)
        print("     Лес построен")

        with open(filenameClassifier, 'wb') as fid:
            cPickle.dump(classifier, fid)
    else:
        print("\nЗагрузка классификатора")
        with open(filenameClassifier, 'rb') as fid:
            forest = cPickle.load(fid)
        print("     Классификатор загружен")

    return model_w2v, forest


if __name__ == '__main__':
    model, classifier = init(read_source=False, read_normalize=False, load_models=True, load_classifier=True)

    sentence = 'Напомним , разбить понизить, потерять, заболеть, разочаровательный, грубо губернатор Волгоградской области Андрей Бочаров.'
    mas_normalized_word = normalizationOfSentence(sentence)

    print(mas_normalized_word)

    mas_collapsed_vectors = featurize_w2v(model, mas_normalized_word)
    # $arr = classifier.predict(mas_collapsed_vectors)
    # print(arr)