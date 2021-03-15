import re
from nltk.stem.snowball import SnowballStemmer
import pymorphy2


def tokenize(source, pattern):
    return pattern.findall(source)


word_and_separator = re.compile(r'[\w]+|[.,!?;]')
word = re.compile(r'[\w]+')
phone_number = re.compile(r'((\+7|8)[\-( ]?(\d{3})[\-) ]?(\d{3})[\- ]?(\d{2})[\- ]?(\d{2}))')
address = re.compile(r'')
emoticons = re.compile(r'')
formula = re.compile(r'')

src_text = 'Эмоции — это то, что ты получаешь, в не зависимости, хочешь ты этого или нет. И это не просто эмоции. ' \
           'Я не испытывал такого никогда: с одной стороны, тазик над головой со зловонной жидкостью так и плещется, ' \
           'опрокидываясь на тебя иногда после \'доброго\' комментария и сразу же наполняясь опять, ожидая ' \
           'своей минуты. С другой — это гормоны радости, которые начинают быстро выделяться, когда звездочки и ' \
           'лайки растут и ты понимаешь, что кому-то твой труд пришелся по вкусу. Так вот, состояние в котором ты ' \
           'находишься после первой статьи — это термоядерная смесь положительного и отрицательного, которая ' \
           'уживается в тебе одновременно. Я не любитель такого каламбура эмоций, но точно знаю, что многим эти ' \
           'состояния просто необходимы, как воздух. Кому-то полезно встряхнуться и посмотреть на некоторые вещи' \
           ' другими глазами. Но во всех случаях, надо быть психологически готовым к совершенно неожиданному ' \
           'развитию событий.'

stemmer = SnowballStemmer("russian")
morph = pymorphy2.MorphAnalyzer()

tokens = tokenize(source=src_text, pattern=word)

file = open("Petrenko.tsv", "a")

for token in tokens:
    stem = stemmer.stem(token)
    mrph = morph.parse(token)[0].normal_form
    file.write(str(token) + '    ' + str(stem) + '    ' + str(mrph) + '\n')

file.close()
