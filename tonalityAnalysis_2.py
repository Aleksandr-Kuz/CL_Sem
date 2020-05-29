# -*- coding: utf-8 -*-
import re
import string
import random
import json
import pickle
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

filenamePositiveJSON_ru = "positive_tweets_ru.json"
filenameNegativeJSON_ru = "negative_tweets_ru.json"


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
            cleaned_tokens.append(token.lower())
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


if __name__ == '__main__':
    load_model = False

    if load_model:
        stop_words = stopwords.words('russian')

        positive_tweet_tokens = twitter_samples.tokenized(filenamePositiveJSON_ru)
        negative_tweet_tokens = twitter_samples.tokenized(filenameNegativeJSON_ru)

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

        positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
        negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

        dataset = positive_dataset + negative_dataset
        random.shuffle(dataset)
        train_data = dataset[:7000]
        test_data = dataset[7000:]

        classifier = NaiveBayesClassifier.train(train_data)
        print("Accuracy is:", classify.accuracy(classifier, test_data))
        print(classifier.show_most_informative_features(10))

        f = open('my_classifier.pickle', 'wb')
        pickle.dump(classifier, f)
        f.close()

    else:
        f = open('my_classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()




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