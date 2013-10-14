#!/usr/bin/python
# -*- coding: utf-8 -*-

# Morphologic tools for Russian (Python 2.7) using pymorphy

import itertools
import re
from commontools import *
from pymorphy.contrib import lastnames_ru

def extract_words(sentences):
    """
    Выборка всех слов из наборов токенов
    """
    words = {ind: info for sentence in sentences for (ind, info) in sentence if len(info[1]) > 1}

def is_synable():
    """
    Проверка вхождения во множество синонимизируемых частей речи
    """
    return {u"С", u"Г", u"ИНФИНИТИВ", u"П", u"Н", u"КР_ПРИЛ", u"ПРИЧАСТИЕ", u"КР_ПРИЧАСТИЕ", u"МС-П"}

def good_pos():
    """
    Множество полезных для синонимизации частей речи
    """
    return {u"С", u"МС", u"Г", u"ИНФИНИТИВ", u"П", u"Н", u"КР_ПРИЛ", u"ПРИЧАСТИЕ", u"КР_ПРИЧАСТИЕ", u"МС-П", u"ПРЕДЛ"}

def non_nominal(all_vars):
    """
    Выбор варианта нормальной формы с граммемами без именительного падежа
    """
    return [info for info in all_vars if not u"им" in info["info"].split(",")]

def suffix(word, i):
    """
    Суффикс длины i
    """
    if i == 0:
        return u"_"
    if len(word) > i:
        return word[-i:]
    return word

def cut_suffix(word, suff):
    """
    Отрезание от слова суффикса, если он есть
    """
    if word.endswith(suff):
        return word[:word.find(suff)]
    return word

def get_suffixes(lemmtoken):
    """
    Извлечение набора суффиксов для списка лемм
    """
    word = lemmtoken[0].upper()
    lemms = [x["norm"] for x in lemmtoken[1:]]
    stem = longest_common([word] + lemms) # longest common prefix          
    return sorted(list(set([suffix(lemma, len(lemma) - len(stem)) for lemma in lemms])))

def longest_common(words):
    """
    Наибольший общий префикс нескольких слов
    """
    char_tuples = itertools.izip(*words)
    prefix_tuples = itertools.takewhile(lambda x: all(x[0] == y for y in x), char_tuples)
    return "".join(x[0] for x in prefix_tuples)

def smart_dict_slice(sentence, num, radius):
    """
    Обертка среза словаря (левый и правый контексты слова)
    """
    ind_list = [ind for (ind, info) in sentence if "class" in info[1].keys()]
    num_index = ind_list.index(num)
    sliced = set(smart_slice(ind_list, num_index - radius, num_index) + smart_slice(ind_list, num_index + 1, num_index + radius))
    context = set([word_info[1]["norm"] for (ind, word_info) in sentence if ind in sliced])        
    return context

def ngram_slice(sentence, num, N):
    """
    Обертка среза словаря (левый контекст слова)
    """
    inds = [ind for (ind, info) in sentence if "class" in info[1].keys()] # Индексы всех слов в предложении
    num_ind = inds.index(num)
    ngram = smart_slice(inds, num_ind - N + 1, num_ind + 1)
    return [info[1]["norm"] for (ind, info) in sentence if ind in ngram]

def restore_lemm(sent_words, suffitem, word_ind):
    """
    Восстановление леммы по суффиксу
    """
    word = sent_words[word_ind][0]
    lemms = [x["norm"] for x in sent_words[word_ind][1:]]      
    best_lemma = longest_common([word.upper()] + lemms) + suffitem # Лучшая лемма
    return tuple([word] + [info for info in sent_words[word_ind][1:] if info["norm"] == best_lemma])

def good_lemm(norm, case):
    """
    Проверяем, подходит ли данный вариант нормальной формы при восстановлении по падежу
    """
    if not norm.has_key("info"):
        return True
    if case in norm["info"].split(","):
        return True
    return False

def xrestore_lemm(sent_words, case, ind):
    """
    Восстановление леммы по падежу
    """
    good_lemms = [info for info in sent_words[ind][1:] if good_lemm(info, case)]
    if good_lemms:
        return tuple([sent_words[ind][0]] + good_lemms)
    return sent_words[ind]

def concord_verb(verb_grams, noun_grams):
    """
    Проверка согласования существительного с глаголом
    """
    diff = set(noun_grams.split(",")).difference(set(verb_grams.split(",")))
    same_amount = not diff.intersection({u"ед", u"мн"})
    if same_amount:
        if u"мн" in noun_grams.split(","):
            return True
        elif u"прш" in verb_grams.split(","):
            return not diff.intersection({u"мр", u"жр", u"мр-жр", u"ср"})
        else:
            return True
    return False

def nom_case_disamb(sent_words, word_ind):
    """
    Снятие омонимии между именительным и винительным падежами
    """

    inds = list(sent_words.keys())
    targets = [info for info in sent_words[word_ind][1:] if info.has_key("norm")]
   
    if word_ind == inds[0]: # Если это первое слово в предложении
        return sent_words[word_ind]    
    
    prev_verbs = {ind: info for (ind, info) in sent_words.iteritems()
                  if ind < word_ind and info[1]["class"] in {u"Г", u"ИНФИНИТИВ"}} # Все предшествующие глаголы в предложении

    if prev_verbs == {}:
        return sent_words[word_ind]
    
    min_verb_ind = min(prev_verbs.keys())
    max_verb_ind = max(prev_verbs.keys())
    last_verb = [x for ind, x in prev_verbs.iteritems() if ind == max_verb_ind][0]

    # Если существительное, предположительно, является дополнением, оно не может быть в именительном падеже

    min_noun_inds = [ind for ind in inds if ind < max_verb_ind and sent_words[ind][1]["class"] in {u"С", u"МС"}]
    nom_noun_inds = [ind for ind in min_noun_inds if u"им" in sent_words[ind][1]["info"].split(",")]

    if not nom_noun_inds:   # Впереди нет существительных в именительном падеже
        # В одном из вариантов существительное согласовано с глаголом слева?
        good_targets = any(target for target in targets if concord_verb(last_verb[1]["info"], target["info"]))
        if good_targets:    # Да
            return tuple([sent_words[word_ind][0]] + good_targets)
        reslist = non_nominal(targets)  # Нет - значит, падеж винительный
        if reslist:
            return tuple([sent_words[word_ind][0]] + reslist)

    if nom_noun_inds:   # Впереди есть существительные в именительном падеже
        if any(concord_verb(last_verb[1]["info"], sent_words[ind][1]["info"]) for ind in nom_noun_inds):
            reslist = non_nominal(targets)  # Где-то впереди было подлежащее - значит, падеж винительный
            if reslist:
                return tuple([sent_words[word_ind][0]] + reslist)
    return sent_words[word_ind]
    
def case_disambig(sent_words, word_ind, best_lexeme):
    """
    Снятие падежной омонимии существительного
    """
    cur_paradigm = sent_words[word_ind][1]   
    all_vars = [info for info in sent_words[word_ind][1:] if info["norm"] == best_lexeme and info["class"] in {u"С", u"МС"}]
    if len(all_vars) < 1:
        raise ValueError("Error: a noun not found!")
    if len(all_vars) == 1:
        return all_vars[0]
    
    inds = list(sent_words.keys())
   
    if word_ind == inds[0]: # Если это первое слово в предложении
        return all_vars[0]

    # Если перед существительным стоит предлог, оно не может быть в именительном падеже    
    ind_prev = inds[inds.index(word_ind) - 1] # Индекс предыдущего слова
    prev_paradigm = sent_words[ind_prev][1]
    
    if prev_paradigm["class"] == u"ПРЕДЛОГ" and u"им" in cur_paradigm["info"].split(","):
        reslist = non_nominal(all_vars)
        if reslist:
            return reslist[0]
        
    prev_verbs = {ind: info for (ind, info) in sent_words.iteritems() if ind < word_ind and info[1]["class"] in {u"Г", u"ИНФИНИТИВ"}} # Все предшествующие глаголы в предложении
    if prev_verbs == {} and not prev_paradigm["class"] in {u"П", u"МС-П", u"ЧИСЛ-П", u"ПРИЧАСТИЕ"}:
        return all_vars[0]

    # Если существительное, предположительно, является дополнением, оно не может быть в именительном падеже
    if prev_verbs != {}:
        min_verb_ind = min(prev_verbs.keys())
        min_noun_inds = [ind for ind in inds if ind < min_verb_ind and cur_paradigm["class"] in {u"С", u"МС"}]
        if any(ind for ind in min_noun_inds if u"им" in cur_paradigm["info"].split(",")):
            reslist = non_nominal(all_vars)
            if reslist:
                return reslist[0]
            
    # Пытаемся снять неоднозначность по предшествующему прилагательному
    if prev_paradigm["class"] in {u"П", u"МС-П", u"ЧИСЛ-П", u"ПРИЧАСТИЕ"}:
        if not prev_paradigm.has_key("info"):
            return all_vars[0]
        features = set(prev_paradigm["info"].split(","))
        feat_list = [(info, set(info["info"].split(","))) for info in all_vars]
        best_vars = argmax([(info, len(feature_set.intersection(features))) for info, feature_set in feat_list])
        if best_vars == []:
            return all_vars[0]
        return best_vars[0]
    
    return all_vars[0]

def get_same_caps(pattern, word):
    """
    Соответствие прописных и строчных букв (возвращает слово такого же типа, что и заданное)
    """
    if pattern.istitle():
        items = [part.lower() for part in word.strip().split()]
        if len(items) == 1:
            return word.title()
        items[0] = items[0].title()
        return " ".join(items)
    if pattern.islower():
        return word.lower()
    if pattern.isupper():
        return word.upper()
    return word.lower()

def inflect_comb(morph, words, gram_form, gram_class):
    """
    Приведение словосочетания к нужной форме (для синонимизации).
    morph - морфологический словарь pymorphy
    """
    items = words.split()
    if len(items) == 1:
        return inflect(morph, words, gram_form, gram_class)
    result = []
    poses = []
    matched = False

    # Поиск совпадающих частей речи
    for part in items:
        info = morph.get_graminfo(part)
        if not info:
            continue
        pos = info[0]["class"]
        if pos == u"ИНФИНИТИВ" and gram_class == u"Г":
            pos = u"Г"
        poses.append(pos)
        if pos != gram_class:
            result.append(part)
            continue
        inflected = inflect(morph, part, gram_form, gram_class)
        if not inflected:
            return None
        result.append(inflected)
        matched = True

    if not matched:
        return None

    # Фраза-синоним не должна заканчиваться предлогом или предлогом с местоимением
    string = " ".join(poses)
    if string.endswith(u"ПРЕДЛ") or string.endswith(u"ПРЕДЛ МС"):
        string = string[:string.find(u"ПРЕДЛ")]
        poses = string.split()
        while len(poses) < len(result):
            result.pop()

    # Согласование прилагательных с существительным
    if u"П" in set(poses) and u"С" in set(poses):
        info = morph.get_graminfo(items[poses.index(u"С")])
        gr_info = info[0]["info"]
        for pos in poses:
            if pos != u"П":
                continue
            inflection = inflect(morph, result[poses.index(pos)], gr_info, u"П")
            if inflection:
                result[poses.index(pos)] = inflection

    return " ".join(result)            

def inflect(morph, word, gram_form, gram_class):
    """
    Приведение слова к нужной форме.
    morph - морфологический словарь pymorphy
    """
    inflected = morph.inflect_ru(word, gram_form, gram_class)
    if inflected != word:
        return inflected
    forms = morph.decline(word, gram_form, gram_class)
    if forms:
        return forms[0]["word"]        
    return None

def get_name_vars(info):
    """
    Нахождение вариантов леммы, в которых слово является именем
    """
    lem_vars = []
    for var in info[1:]:
        if u"имя" in var["info"].split(","):
            lem_vars.append(var)
    return lem_vars

def is_name(info):
    """
    Проверка того, является ли слово именем хотя бы в одном из вариантов лемм
    """
    return any(u"имя" in var["info"].split(",") for var in info[1:])

def concord(var_name, *var_adjs):
    """
    Проверка согласования существительного и одного или нескольких прилагательных в роде/числе/падеже
    """
    grams_name = set(var_name["info"].split(","))
    good = []
    for var_adj in var_adjs:
        grams_adj = set(var_adj["info"].split(","))
        diff = grams_adj.difference(grams_name)
        if diff == set([]) or diff == {"мн"}:
            good.append(var_adj)
        else:
            break
    return good

def nearest_left_noun(sentence, words, ind):
    """
    Нахождение ближайшего существительного слева, согласованного с данным
    """
    inds = list(words.keys())
    sent = dict(sentence)
    if ind == inds[0]:
        return None
    prev_ind = inds[inds.index(ind) - 1]
    if words[prev_ind][0][0].islower():
        return None
    if any(sent[i][0].strip() != "" for i in range(prev_ind + 1, ind)):
        return None
    nouns = [var for var in words[prev_ind][1:] if var["class"] == u"С"]  # Варианты соседа слева
    if not nouns:
        return None
    names = [var for var in words[ind][1:] if u"имя" in var["info"].split(",")] # Варианты данного слова
    combinations = [(var_adj, var_name) for var_adj, var_name in itertools.product(nouns, names) if concord(var_name, var_adj)]
    return (prev_ind, combinations)

def nearest_right_noun(sentence, words, ind):
    """
    Нахождение ближайших двух существительных справа, согласованных с данным
    """
    inds = list(words.keys())
    sent = dict(sentence)
    if ind == inds[-1]:
        return None
    next_ind = inds[inds.index(ind) + 1]
    if words[next_ind][0][0].islower():
        return None
    if any(sent[i][0].strip() != "" for i in range(ind + 1, next_ind)):
        return None
    nouns = [var for var in words[next_ind][1:] if var["class"] == u"С"] # Варианты соседа справа
    if not nouns:
        return None
    names = [var for var in words[ind][1:] if u"имя" in var["info"].split(",")] # Варианты данного слова
    combinations = [(var_name, var_adj)
                    for var_name, var_adj
                    in itertools.product(names, nouns)
                    if concord(var_name, var_adj)]
    if inds[-1] == next_ind:
        return (next_ind, combinations)

    next_ind2 = inds[inds.index(ind) + 2]
    if words[next_ind2][0][0].islower():
        return (next_ind, combinations)
    if any(sent[i][0].strip() != "" for i in range(next_ind + 1, next_ind2)):
        return (next_ind, combinations)
    nouns2 = [var for var in words[next_ind2][1:] if var["class"] == u"С"]
    combinations = [(var_name, concord(var_name, var_adj, var_adj2))
                    for var_name, var_adj, var_adj2 in
                    itertools.product(nouns2, sent[next_ind][1:], sent[next_ind2][1:])
                    if concord(var_name, var_adj, var_adj2)]
    return (next_ind, next_ind2, combinations)

def inflect_female(morph, lemma, grams):
    """
    Приводит существительное (фамилию/отчество) к женскому роду, если необходимо
    """
    if u"жр" in grams:
        result = lastnames_ru.normalize(morph, lemma, u"жр")
        if result:
            return result
    return lemma
            
def get_name_inds(morph, sentence):
    """
    Собирает имена из предложения
    """
    marks = {} # Результат в формате {(index1, index2, ..., indexn): имя в канонической форме}
    words = {ind: info for (ind, info) in sentence if "info" in info[1].keys()}
    name_words = {ind: info for ind, info in words.iteritems() if any(u"имя" in var["info"].split(",") for var in info[1:]) and info[0][0].isupper()}
    
    for ind, info in name_words.iteritems():
        left = nearest_left_noun(sentence, words, ind)
        right = nearest_right_noun(sentence, words, ind)
        if not left and not right:
            marks[(ind)] = [[var["norm"]] for var in info[1:] if u"имя" in var["info"].split(",")]
            continue
        if not right:
            combs = left[1]
            marks[(left[0], ind)] = [(inflect_female(morph, comb[0]["norm"], comb[0]["info"]), comb[1]["norm"])
                                     for comb in combs]
            continue

        combs = right[-1]
        if len(right) == 2:
            marks[(ind, right[0])] = [(comb[0]["norm"], inflect_female(morph, comb[1]["norm"], comb[1]["info"]))
                         for comb in combs]

        if len(right) == 3:
            for comb in combs:
               
                if len(comb[1]) == 2:
                    marks[(ind, right[0], right[1])] = [(comb[0]["norm"],
                          inflect_female(morph, comb[1][0]["norm"], comb[1][0]["info"]),
                        inflect_female(morph, comb[1][1]["norm"], comb[1][1]["info"]))
                         for comb in combs]           
    return marks
