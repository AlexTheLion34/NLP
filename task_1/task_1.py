import re
from nltk.stem.snowball import SnowballStemmer
import pymorphy2


def tokenize(source, regexp):
    if regexp == generic:
        return list(map(lambda item: item[0], regexp.findall(source)))
    else:
        return regexp.findall(source)


# Task 1
word = r'[A-ZА-Яa-zа-яёË]+'
word_and_separator = r'([A-ZА-Яa-zа-яёË]+|[.,!?;:]+)'
phone_number = r'((\+7|8)[\-( ]?(\d{3})[\-) ]?(\d{3})[\- ]?(\d{2})[\- ]?(\d{2}))'
emoticons = r'(:[\(\)])+'

generic = re.compile('(%s|%s|%s)' % (word_and_separator, phone_number, emoticons))

main_text = 'Этот текст предназначен для теста регулярных выражений. ' \
            'Same, but written in English: This text is for testing ' \
            'regular expressions. Вот несколько мобильных телефонов, ' \
            'для тестирования: +79114567890, 89114567890, ' \
            '+7-911–456-78-90, 8 911 456 78 90, 8(911)456 78 90, ' \
            '+7(911)4567890. Должно работать. Must work. Точно работает?' \
            'Вот тут проверяются еще нектороые знаки препинания ; !!!' \
            'Теперь проверим смайлы :). Несколько смайлов :):). И Грустный тоже (:.'

file_1 = open('./res/task_1.txt', 'a')

for token in tokenize(source=main_text, regexp=re.compile(pattern=generic)):
    file_1.write(str(token) + '\n')

file_1.close()

# Task 2
stemmer = SnowballStemmer("russian")

file_2 = open('./res/task_2.txt', 'a')

for token in tokenize(source=main_text, regexp=re.compile(pattern=word)):
    file_2.write(str(token) + ' ' + str(stemmer.stem(token)) + '\n')

file_2.close()

# Task 3
morph = pymorphy2.MorphAnalyzer()

morph_text = 'С камушка на камушек порхал воробышек, а на террасе, искусно задрапированной гобеленами с дефензивой ' \
             'кронштадтского инфантерийского батальона, под искусственным абажуром, закамуфлированным под ' \
             'марокканский минарет, веснушчатая свояченица вдовствующего протоиерея Агриппина Саввична потчевала ' \
             'коллежского асессора, околоточного надзирателя и индифферентного ловеласа Фаддея Аполлинарьевича ' \
             'винегретом со снетками.'

file_3 = open('./res/task_3.txt', 'a')

for token in tokenize(source=morph_text, regexp=re.compile(pattern=word)):
    file_3.write(str(token) + ' ' + str(morph.parse(token)) + '\n')

file_3.close()

# Task 4
file_4 = open('./res/Petrenko.tsv', 'a')

for token in tokenize(source=main_text, regexp=re.compile(pattern=word)):
    file_4 \
        .write(str(token) + '\t' + str(stemmer.stem(token)) + '\t' + str(morph.parse(token)[0].normal_form) + '\n')

file_4.close()
