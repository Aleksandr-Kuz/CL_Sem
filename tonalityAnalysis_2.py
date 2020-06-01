# -*- coding: utf-8 -*-
import re
import string
import random
import json
import pickle
import pymorphy2
from pymongo import MongoClient
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

morf = pymorphy2.MorphAnalyzer()
necessary_part = ["NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS", "GRND"]

filenamePositiveJSON_ru = "positive_tweets_ru.json"
filenameNegativeJSON_ru = "negative_tweets_ru.json"

positive_tag = "Positive"
negative_tag = "Negative"

classifier_path = 'my_classifier_1.pickle'


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
    elif collection_name == 'TonalitySentencesWithPerson':
        return db.TonalitySentencesWithPerson
    else:
        return None


def normalizationWord(word):
    w = word.lower()
    if w.isalpha():
        p = morf.parse(w)[0]
        part = p.tag.POS
        if part in necessary_part:
            return p.normal_form
        else:
            return w
    else:
        return w


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens, lang='rus'):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            # cleaned_tokens.append(token.lower())
            cleaned_tokens.append(normalizationWord(token))
    return cleaned_tokens


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens, lang='rus'):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict((token, True) for token in tweet_tokens)


def save_classifier(classifier, classifier_path):
    f = open(classifier_path, 'wb')
    pickle.dump(classifier, f)
    f.close()


def get_classifier(path):
    file = open(path, 'rb')
    classifier = pickle.load(file)
    file.close()

    return classifier


def create_classifier():
    stop_words = stopwords.words('russian')

    positive_tweet_tokens = twitter_samples.tokenized(filenamePositiveJSON_ru)
    negative_tweet_tokens = twitter_samples.tokenized(filenameNegativeJSON_ru)

    min_len = min(len(positive_tweet_tokens), len(negative_tweet_tokens))
    if len(positive_tweet_tokens) > min_len:
        positive_tweet_tokens = positive_tweet_tokens[:min_len]
    if len(positive_tweet_tokens) > min_len:
        negative_tweet_tokens = negative_tweet_tokens[:min_len]

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    freq_dist_pos = FreqDist(all_pos_words)

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, positive_tag) for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, negative_tag) for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)
    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))

    save_classifier(classifier, classifier_path)

    return classifier


def get_tonality(classifier, text):
    stop_words = stopwords.words('russian')
    tokens = word_tokenize(text, 'russian')

    cleaned_tokens = remove_noise(tokens, stop_words)
    tokens_for_model = dict((token, True) for token in cleaned_tokens)
    return classifier.classify(tokens_for_model)


def analyze_all_sentences_with_person():
    classifier = get_classifier(classifier_path)

    sentences_with_person_collection = get_db_collection_object('SentencesWithPerson')
    tonality_sentences_with_person_collection = get_db_collection_object('TonalitySentencesWithPerson')

    collect = sentences_with_person_collection.find()

    print('Анализ тональности предложений:')
    printProgressBar(0, collect.count(), prefix='Progress:', suffix='Complete', length=50)
    for idx, item in enumerate(collect):
        tmp = {
            'news_id': item['news_id'],
            'person_id': item['person_id'],
            'sentence': item['sentence'],
            'tonality': 1 if get_tonality(classifier, item['sentence']) == positive_tag else -1,
            'person_FIO': item['person_FIO']
        }
        tonality_sentences_with_person_collection.insert_one(tmp)

        printProgressBar(idx + 1, collect.count(), prefix='Progress:', suffix='Complete', length=50)


if __name__ == '__main__':
    analyze_all_sentences_with_person()








    # # create_classifier()
    # classifier = get_classifier(classifier_path)
    # sentences_with_person_collection = get_db_collection_object('SentencesWithPerson')
    # tonality_sentences_with_person_collection = get_db_collection_object('TonalitySentencesWithPerson')
    # collect = sentences_with_person_collection.find().limit(10)
    #
    # result = []
    # for item in collect:
    #     tmp = {
    #         'news_id': item['news_id'],
    #         'person_id': item['person_id'],
    #         'sentence': item['sentence'],
    #         'tonality': 1 if get_tonality(classifier, item['sentence']) == positive_tag else -1,
    #         'person_FIO': item['person_FIO']
    #     }
    #     print(tmp['tonality'], tmp['sentence'])
    #     result.append((tmp['tonality'], tmp['sentence']))
    #     tonality_sentences_with_person_collection.insert_one(tmp)

    # print(classifier.prob_classify(tokens_for_model))




    # model, classifier = init(read_source=False, read_normalize=False, load_models=True, load_classifier=True)
    #
    # sentence = 'Напомним , разбить понизить, потерять, заболеть, разочаровательный, грубо губернатор Волгоградской области Андрей Бочаров.'
    # mas_normalized_word = normalizationOfSentence(sentence)
    #
    # print(mas_normalized_word)
    #
    # mas_collapsed_vectors = featurize_w2v(model, mas_normalized_word)
    # # $arr = classifier.predict(mas_collapsed_vectors)
    # # print(arr)