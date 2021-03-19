import re
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
import sys
import csv


def tokenize(source, regexp):
    if regexp == generic:
        return list(map(lambda item: item[0], regexp.findall(source)))
    else:
        return regexp.findall(source)


# Задание 1
word = r'[A-ZА-Яa-zа-яёË]+'
word_and_separator = r'[A-ZА-Яa-zа-яёË]+|[.,!?;:]+'
phone_number = r'\+?[1-9][\-( ]?(\d{3})[\-) ]?(\d{3})[\- ]?(\d{2})[\- ]?(\d{2})'
emoticons = r'[\:][\)\(]'

generic = re.compile('(%s|%s|%s)' % (emoticons, word_and_separator, phone_number))

# Чехов
main_text = '''
    Жизнь пренеприятная штука, но сделать ее прекрасной очень нетрудно. Для того, чтобы ощущать 
    в себе счастье без перерыва, 
    даже в минуты скорби и печали, нужно: уметь довольствоваться настоящим и радоваться сознанию, что “могло бы быть 
    и хуже”. А это нетрудно:
    Когда у тебя в кармане загораются спички, то радуйся и благодари небо, что у тебя в кармане не пороховой погреб.
    Когда в твой палец попадает заноза, радуйся: “Хорошо, что не в глаз”!
    Если твоя жена… играет гаммы, то не выходи из себя, а не находи себе место от радости, что ты слушаешь игру, 
    а не вой шакалов…
    Радуйся, что ты не лошадь, не свинья, не осел, не медведь, которого водят цыгане, не клоп… Радуйся, что ты не 
    хромой, не слепой, не глухой, не немой, не холерный…
    Если у тебя болит один зуб, то ликуй, что у тебя болят не все зубы.
    Если жена тебе изменила, то радуйся, что она изменила тебе, а не отечеству.
    И так далее…
    Вот несколько мобильных телефонов, для тестирования: +79114567890, 89114567890, 
    +7-911–456-78-90, 8(911)456 78 90, +7(911)4567890. Грусть:(. Радость:):).
'''

file_1 = open('res/task_1.txt', 'a')

for token in tokenize(source=main_text, regexp=re.compile(pattern=generic)):
    file_1.write(str(token) + '\n')

file_1.close()

# Задание 2
stemmer = SnowballStemmer("russian")

file_2 = open('res/task_2.txt', 'a')

# Тескт, разбитый на слова
tokens = tokenize(source=main_text, regexp=re.compile(pattern=word))

for token in tokens:
    file_2.write(str(token) + ' ' + str(stemmer.stem(token)) + '\n')

file_2.close()

# Задание 3
morph = pymorphy2.MorphAnalyzer(lang='ru')

dataset = open('dict.tsv')

read_tsv = csv.reader(dataset, delimiter="\t")
csv.field_size_limit(sys.maxsize)

# Мапа токенов и лемм из словаря
map_of_dict_lemmas = dict()

for row in read_tsv:
    map_of_dict_lemmas[row[0]] = (row[1], row[2])

dataset.close()

# Мапа токенов и лемм из текста
map_of_my_lemmas = dict()

for token in tokens:
    map_of_my_lemmas[str(token)] = str(morph.parse(token)[0].normal_form), str(morph.parse(token)[0].tag)

file_3_1 = open('res/task_3_1.txt', 'a')
file_3_1.write('Dict, My\n')

# Сопоставление токенов из словаря и текста
for key in map_of_my_lemmas.keys():
    if key in map_of_dict_lemmas.keys():
        file_3_1.write(str(key)
                       + ' : '
                       + str(map_of_dict_lemmas[key])
                       + ', '
                       + str(key)
                       + ' : '
                       + str(map_of_my_lemmas[key]) + '\n')
# 10 строчка
# 77 строчка
file_3_1.close()

file_3_2 = open('res/task_3_2.txt', 'a')

# Выявление токенов, для которых было найдено более 1 леммы
for token in tokens:
    res = morph.parse(token)
    if str(token) in map_of_dict_lemmas.keys() and len(res) > 1:
        lemmas = []
        for parse in res:
            lemmas.append((parse.normal_form, parse.tag))
            file_3_2.write(str(token) + ' : ' + str(lemmas) + '\n')

file_3_2.close()

# Задание 4
file_4 = open('res/Petrenko.tsv', 'a')

for token in tokens:
    file_4 \
        .write(str(token) + '\t' + str(stemmer.stem(token)) + '\t' + str(morph.parse(token)[0].normal_form) + '\n')

file_4.close()
