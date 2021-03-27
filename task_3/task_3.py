import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

from nltk.stem import WordNetLemmatizer
import re

import gensim.downloader
from gensim.models import KeyedVectors

data = pd.read_csv("spam.csv", encoding='ISO-8859-1')
data = data[['v1', 'v2']]
data['v1'] = (data['v1'] == 'spam').astype(int)

texts = data['v2'].values
labels = data['v1'].values

texts_train, texts_test, y_train, y_test = train_test_split(texts, labels, test_size=0.5, random_state=42)


def preprocess(text):
    return re.findall(re.compile(r'[A-ZА-Яa-zа-яёË]+'), text.lower())


lemmatizer = WordNetLemmatizer()


def make_lemmas(tokens):
    return ' '.join(list(map(lambda t: lemmatizer.lemmatize(t), tokens)))


texts_train = [make_lemmas(preprocess(text=text)) for text in texts_train]
texts_test = [make_lemmas(preprocess(text=text)) for text in texts_test]


def make_bag_of_words(comments, number_of_features):
    bag = dict()
    for comment in comments:
        for word in comment.split(' '):
            if word not in bag.keys():
                bag[word] = 1
            else:
                bag[word] += 1

    return dict(sorted(bag.items(), key=lambda item: item[1], reverse=True)[:number_of_features])


k = 10000
bow_vocabulary = make_bag_of_words(comments=texts_train, number_of_features=k)


def text_to_bow(text):
    bow = [0] * len(bow_vocabulary)
    voc_map = dict()

    for num, w in enumerate(bow_vocabulary, start=0):
        voc_map[w] = num

    for token in text.split(' '):
        if token in voc_map:
            bow[voc_map[token]] += 1

    return np.array(bow, 'float32')


X_train_bow = np.stack(list(map(text_to_bow, texts_train)))
X_test_bow = np.stack(list(map(text_to_bow, texts_test)))

k_max = len(set(' '.join(texts_train).split()))


class BinaryNaiveBayes:
    delta = 1.0  # add this to all word counts to smoothe probabilities

    def fit(self, X, y):
        """
        Fit a NaiveBayes classifier for two classes
        :param X: [batch_size, vocab_size] of bag-of-words features
        :param y: [batch_size] of binary targets {0, 1}
        """
        # first, compute marginal probabilities of every class, p(y=k) for k = 0,1
        self.p_y = np.array([sum(c == 0 for c in y) / len(y), sum(c == 1 for c in y) / len(y)], 'float32')

        # count occurrences of each word in texts with label 1 and label 0 separately
        word_counts_positive = [0] * len(X[0])
        word_counts_negative = [0] * len(X[0])

        for t in range(len(y)):
            if y[t] == 1:
                for x in range(len(X[t])):
                    word_counts_positive[x] += X[t][x]
            else:
                for x in range(len(X[t])):
                    word_counts_negative[x] += X[t][x]

        # finally, lets use those counts to estimate p(x | y = k) for k = 0, 1
        self.p_x_given_positive = [0.0] * len(word_counts_positive)
        self.p_x_given_negative = [0.0] * len(word_counts_negative)

        for i in range(len(word_counts_positive)):
            self.p_x_given_positive[i] = (word_counts_positive[i] + self.delta) / \
                                         (word_counts_positive[i] + word_counts_negative[i])
            self.p_x_given_negative[i] = (word_counts_negative[i] + self.delta) / \
                                         (word_counts_positive[i] + word_counts_negative[i])
        # # both must be of shape [vocab_size]; and don't forget to add self.delta!

        return self

    def predict_scores(self, X):
        """
        :param X: [batch_size, vocab_size] of bag-of-words features
        :returns: a matrix of scores [batch_size, k] of scores for k-th class
        """

        # compute scores for positive and negative classes separately.
        # these scores should be proportional to log-probabilities of the respective target {0, 1}
        # note: if you apply logarithm to p_x_given_*, the total log-probability can be written
        # as a dot-product with X
        p_x_given_negative_lg = list(map(lambda p: np.log(p), self.p_x_given_negative))
        p_x_given_positive_lg = list(map(lambda p: np.log(p), self.p_x_given_positive))
        score_negative = [np.log(self.p_y[0]) + np.dot(p_x_given_negative_lg, x) for x in X]
        score_positive = [np.log(self.p_y[1]) + np.dot(p_x_given_positive_lg, x) for x in X]

        # you can compute total p(x | y=k) with a dot product
        return np.stack([score_negative, score_positive], axis=-1)

    def predict(self, X):
        return self.predict_scores(X).argmax(axis=-1)


naive_model = BinaryNaiveBayes().fit(X_train_bow, y_train)

for name, X, y, model in [
    ('train', X_train_bow, y_train, naive_model),
    ('test ', X_test_bow, y_test, naive_model)
]:
    print('Here 3')
    proba = model.predict_scores(X)[:, 1] - model.predict_scores(X)[:, 0]
    auc = roc_auc_score(y, proba)
    plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))

plt.plot([0, 1], [0, 1], '--', color='black', )
plt.legend(fontsize='large')
plt.grid()

test_accuracy = np.mean(naive_model.predict(X_test_bow) == y_test)
print(f"Model accuracy: {test_accuracy:.3f}")
assert test_accuracy > 0.75, "Accuracy too low. There's likely a mistake in the code."
print("Well done!")


bow_vocabulary_2 = list(bow_vocabulary.keys())

probability_ratio = sorted([(naive_model.p_x_given_positive[i]/naive_model.p_x_given_negative[i], i) for i in range(len(naive_model.p_x_given_positive))], key=lambda tup: tup[0], reverse=True)
top_negative_words = list(map(lambda p: bow_vocabulary_2[p[1]], probability_ratio))[:25]

for i, word in enumerate(top_negative_words):
    print(f"#{i}\t{word.rjust(10, ' ')}\t(ratio={probability_ratio[bow_vocabulary_2.index(word)][0]})")


bow_model = LogisticRegression().fit(X_train_bow, y_train)

for name, X, y, model in [
    ('train', X_train_bow, y_train, bow_model),
    ('test ', X_test_bow, y_test, bow_model)
]:
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))

plt.plot([0, 1], [0, 1], '--', color='black',)
plt.legend(fontsize='large')
plt.grid()

test_accuracy = np.mean(bow_model.predict(X_test_bow) == y_test)
print(f"Model accuracy: {test_accuracy:.3f}")
assert test_accuracy > 0.77, "Hint: tune the parameter C to improve performance"
print("Well done!")


def tf(tokens):

    voc = dict()

    for token in tokens:
        if token not in voc.keys():
            voc[token] = 1
        else:
            voc[token] += 1

    for key in voc.keys():
        voc[key] = voc[key] / float(len(tokens))

    return voc


def idf(token, corpus):
    return np.log10(len(corpus) / sum([1.0 for t in corpus if token in t]))


def make_tf_idf_for_tokens(corpus):

    corpus = list(map(lambda cmnt: cmnt.split(' '), corpus))

    tf_idf_voc = dict()

    for comment in corpus:

        computed_tf = tf(comment)

        for word in computed_tf:

            tf_idf_voc[word] = computed_tf[word] * idf(word, corpus)

    return dict(sorted(tf_idf_voc.items(), key=lambda item: item[1], reverse=True)[:10000])


tf_idf_dict = make_tf_idf_for_tokens(corpus=texts_train)


def text_to_tf_idf(text):

    tf_idf = [0.0] * len(tf_idf_dict)
    tf_idf_map = dict()

    for num, word in enumerate(tf_idf_dict, start=0):
        tf_idf_map[word] = num

    for token in text.split(' '):
        if token in tf_idf_map:
            tf_idf[tf_idf_map[token]] = tf_idf_dict[token]

    return np.array(tf_idf, 'float32')


X_train_bow_2 = np.stack(list(map(text_to_tf_idf, texts_train)))
X_test_bow_2 = np.stack(list(map(text_to_tf_idf, texts_test)))


tf_model = LogisticRegression().fit(X_train_bow_2, y_train)

for name, X, y, model in [
    ('train', X_train_bow, y_train, tf_model),
    ('test ', X_test_bow, y_test, tf_model)
]:
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))

plt.plot([0, 1], [0, 1], '--', color='black',)
plt.legend(fontsize='large')
plt.grid()

test_accuracy = np.mean(tf_model.predict(X_test_bow) == y_test)
print(f"Model accuracy: {test_accuracy:.3f}")
assert test_accuracy > 0.77, "Hint: tune the parameter C to improve performance"
print("Well done!")

embeddings = gensim.downloader.load('glove-wiki-gigaword-100')
print(gensim.downloader.info()['models'].keys())


# def vectorize_sum(comment):
#     """
#     implement a function that converts preprocessed comment to a sum of token vectors
#     """
#     embedding_dim = embeddings.wv.vectors.shape[1]
#     features = np.zeros([embedding_dim], dtype='float32')
#
#     for token in comment.split(' '):
#
#
#     return features
#
#
# X_train_wv = np.stack([vectorize_sum(text) for text in texts_train])
# X_test_wv = np.stack([vectorize_sum(text) for text in texts_test])
#
# wv_model = LogisticRegression().fit(X_train_wv, y_train)
#
# for name, X, y, model in [
#     ('bow train', X_train_bow, y_train, bow_model),
#     ('bow test ', X_test_bow, y_test, bow_model),
#     ('vec train', X_train_wv, y_train, wv_model),
#     ('vec test ', X_test_wv, y_test, wv_model)
# ]:
#     proba = model.predict_proba(X)[:, 1]
#     auc = roc_auc_score(y, proba)
#     plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))
#
# plt.plot([0, 1], [0, 1], '--', color='black',)
# plt.legend(fontsize='large')
# plt.grid()
#
# assert roc_auc_score(y_test, wv_model.predict_proba(X_test_wv)[:, 1]) > 0.92, "something's wrong with your features"