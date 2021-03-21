from functools import reduce


# Расчет меры ассоциации
def evaluate_association(trigrams, num_of_tokens):
    one_g = dict()
    tri_g = dict()

    result = dict()

    for tri_gram in trigrams:
        if tri_gram not in tri_g.keys():
            tri_g[tri_gram] = 1
        else:
            tri_g[tri_gram] += 1

        for t in tri_gram:
            if t not in one_g.keys():
                one_g[t] = 1
            else:
                one_g[t] += 1

    for tri_gram in trigrams:
        # Подсчет частот для трриграммы, ключевого слова и коллокатов
        o_iii = tri_g[tri_gram]
        o_ixx = one_g[tri_gram[0]]
        o_xix = one_g[tri_gram[1]]
        o_xxi = one_g[tri_gram[2]]

        result[tri_gram] = (t_score(metrics=(o_iii, (o_ixx, o_xix, o_xxi), num_of_tokens)))

    return sorted(list(result.items()), key=lambda tup: tup[1], reverse=True)


# Подсчет метрики t-score
def t_score(metrics):
    score = (metrics[0] - f_n_c(metrics[1]) / metrics[2] ** 2) / metrics[0] ** 0.5
    return score


# Подсчет абсолютных частот для ключевого слова и коллаката
def f_n_c(values):
    return reduce(lambda a, b: a * b, values)
