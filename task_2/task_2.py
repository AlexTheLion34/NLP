import re
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk import word_tokenize
from association import evaluate_association
from nltk.collocations import TrigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk import Text


def tokenize(source, regexp):
    return regexp.findall(source)


word = r'[A-ZА-Яa-zа-яёË]+'

# Великий Гэтсби
text_1 = open('Text.txt', 'r') \
    .read() \
    .lower()

# Задание 1, 2, 3
tokens_1 = tokenize(source=text_1, regexp=re.compile(word))

# Задание 4
stop_words = stopwords.words('russian')

tokens_without_stop_words = [token for token in tokens_1 if token not in stop_words]

# Задание 5
tri_grams = list(ngrams(sequence=tokens_without_stop_words, n=3))


def count_frequency(trigrams):
    result = dict()

    for tri_gram in trigrams:
        if tri_gram not in result.keys():
            result[tri_gram] = 1
        else:
            result[tri_gram] += 1

    return result.items()


file_1 = open('res/task_1-5.tsv', 'a')
file_1.write('3-грамма\tчастота\n')

for trigram, freq in count_frequency(trigrams=tri_grams):
    file_1.write(' '.join(trigram) + '\t' + str(freq) + '\n')

file_1.close()

# Задание 6
my_evaluation = evaluate_association(trigrams=list(tri_grams), num_of_tokens=len(tokens_without_stop_words))[:30]

# Задание 7
trigram_measures = TrigramAssocMeasures()

tokens_2 = word_tokenize(open('Text.txt', 'r').read(), 'russian', True)

text_2 = Text(tokens_2)
finder_thr_1 = TrigramCollocationFinder.from_words(text_2)
evaluation_with_punctuation = finder_thr_1.nbest(TrigramAssocMeasures().student_t, 30)

file_2 = open('res/task_7_with_p.tsv', 'a')
file_2.write('Мои 3-граммы\tNLTK 3-граммы\n')

for i in range(30):
    file_2.write(' '.join(my_evaluation[i][0]) + '\t' + ' '.join(evaluation_with_punctuation[i][0]) + '\n')

file_2.close()

text_3 = Text(tokens_without_stop_words)
finder_thr_2 = TrigramCollocationFinder.from_words(text_3)
evaluation_without_punctuation = finder_thr_2.nbest(TrigramAssocMeasures().student_t, 30)

file_3 = open('res/task_7_without_p.tsv', 'a')
file_3.write('Мои 3-граммы\tNLTK 3-граммы\n')

for i in range(30):
    file_3.write(' '.join(my_evaluation[i][0]) + '\t' + ' '.join(evaluation_without_punctuation[i][0]) + '\n')

file_3.close()

file_4 = open('res/task_7_final.tsv', 'a')
file_4.write('Мои 3-граммы\tT-score 1\tNLTK 3-граммы\tT-Score 2\n')

for i in range(30):
    file_4.write(' '.join(my_evaluation[i][0])
                 + '\t'
                 + str(my_evaluation[i][1])
                 + '\t'
                 + ' '.join(evaluation_without_punctuation[i][0])
                 + '\t'
                 + str(evaluation_without_punctuation[i][1])
                 + '\n')

file_4.close()
