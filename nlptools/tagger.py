#!/usr/bin/python2.7
# -*- encoding: utf-8 -*-

"""
Снятие морфологической омонимии русского текста.
Статистический тэггер-лемматизатор для русского языка на основе библиотеки pymorphy.
Использует статистику совместного употребления окончаний слов в тексте,
а также статистику зависимости падежей от управляющих предлогов.
Для Python 2.7
"""

import sys
import os
import time
import codecs
import itertools
import re
import math
from random import choice
from datetime import datetime
from collections import defaultdict, OrderedDict
import struct

import dawg
from pymorphy import get_morph
from pickling import *
from tokenizer import Tokenizer
from dater import Dater
from commontools import *
from morphotools import *

class Tagger(object):
    """
    Статистический тэггер-лемматизатор для русского языка на основе pymorphy
    """

    def __init__(self, morph=None, morph_simple=None, dater=None):
        """
        Инициализация тэггера. Создание регулярных выражений для лематизации.
        Подгрузка словарей аббревиатур, исключений.
        Создание словарей месяцев, падежей и пр.

        morph - морфологический словарь pymorphy с предсказанием по префиксам.
        morph_simple - морфологический словарь pymorphy без предсказания по префиксам.

        Примечание:
        В pymorphy с включенным предсказанием по префиксам (check_prefixes=True) появляется много несуществующих слов.
        Т.к. это свойство задается сразу при инициализации объекта, приходится создавать 2 объекта:
        с предсказанием и без него.
        Во время лемматизации вначале применяется объект морфологии
        без предсказания по префиксу (morph_simple), если он подключен.
        Если он не знает какого-то слова, для этого слова
        используется объект морфологии с предсказанием про префиксу (morph), если он подключен.

        При этом Tagger не требует обязательного указания обоих объектов морфологии:
        достаточно указать только один (он и будет использоваться при лемматизации).
        """
        if not morph and not morph_simple:
            raise ValueError("No morphoanalyzer found!")
        # Рег. выражения для лемматизации
        self.digit = re.compile("^\d+$")
        self.eng = re.compile(u"^\d*[a-zA-Z]+(?:-[a-zA-Z])?$", re.UNICODE)
        self.short = re.compile(u"^[A-ZА-ЯЁ][a-zа-яё]?$")

        # Рег. выражения для разбиения текста на предложения
        self.splitter = re.compile("[.?!]+")
        self.starter = re.compile(u"[А-ЯЁA-Z\d\"\'\(\)\[\]~`«s-]")
        self.bad_ender = re.compile(u"^[А-ЯЁа-яёA-Za-z][а-яёa-z]?$")
        # Морфология
        self.morph = morph
        self.morph_simple = morph_simple
        # Обработка дат
        self.dater = dater
        # Аббревиатуры
        self.abbrs = unpkl_2layered_s(os.path.join(os.path.dirname(sys.argv[0]), "dicts/abbr.pkl"))
        # Доп. правила
        self.excepts = unpkl_2layered_s(os.path.join(os.path.dirname(sys.argv[0]), "dicts/exceptions.pkl"))
        # Падежи
        self.cases = {u"им", u"рд", u"дт", u"вн", u"тв", u"пр"}
        self.caseable = {u"С", u"П", u"ПРИЧАСТИЕ", u"МС-П", u"ЧИСЛ-П"}
        # Суффиксные частоты
        self.freqs = dawg.BytesDAWG()
        self.weights = defaultdict(float)
        self.small = 0.0

    def gram_bad(self, word):
        """
        Возвращает грамматические признаки для слова, которого нет в словаре pymorphy.
        
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U
        """
        if self.dater:
            if self.dater.is_date(word):
                date = self.dater.check_date(word)
                if date:
                    return {"norm": date[0], "class": date[1], "info": date[1] + "-U"}
                return {"norm": word}
        if re.match(self.digit, word):
            return {"norm": word.upper(), "class": u"ЧИСЛ"}      
        if re.match(self.eng, word):
            if word.endswith(u"s'") or word.endswith("'s"):
                return {"norm": word.upper(), "class": u"П"}
            if word.endswith(u"'a") or word.endswith(u"'а"):
                word = word[:-2]
            return {"norm": word.upper(), "class": u"С"}
        if self.abbrs.has_key(word):
            return self.abbrs[word]
        return {"norm": word.upper()}

    def check_lemma(self, norm, word):
        """
        Проверка того, что возвращается непустая лемма (если лемма пуста, вместо нее возвращается само слово)
        """
        lexeme = norm["norm"] if norm["norm"] != "" else word.upper()
        if "-" in lexeme:
            parts = lexeme.split("-")
            lemmas = []
            for part in parts:
                part_norm = self.morph_simple.get_graminfo(part)
                if part_norm:
                    part_lemma = part_norm[0]["norm"]
                else:
                    part_lemma = part
                lemmas.append(part_lemma)
            lexeme = "-".join(lemmas)
        return {"norm": lexeme, "class": norm["class"], "info": norm["info"]}
        
    def gram_first(self, word):
        """
        Возвращает для слова его ПЕРВУЮ лемму, часть речи и грамматические признаки в виде словаря
        
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U
        """
        if self.excepts.has_key(word):
            return [self.excepts[word]]
        data = []
        try:
            data = self.morph_simple.get_graminfo(word.upper())
        except Exception:
            data = self.morph.get_graminfo(word.upper())
        if data:   
            return [self.check_lemma(data[0], word)]
        return [self.gram_bad(word)]

    def gram_all(self, word):
        """
        Возвращает для слова ВСЕ его леммы, части речи и грамматические признаки в виде кортежа словарей
        
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U
        """
        if self.excepts.has_key(word):
            return [self.excepts[word]]
        data = []
        try:
            data = self.morph_simple.get_graminfo(word.upper())
        except Exception:
            data = self.morph.get_graminfo(word.upper())
        if data:
            return [self.check_lemma(info, word) for info in data]
        return [self.gram_bad(word)]

    def lemm_list(self, word, true_lemma):
        """
        Возвращает список лемм слова, вначале - правильную лемму
        """
        norms = []
        try:
            norms = self.morph_simple.get_graminfo(word.upper())
        except Exception:
            norms = self.morph.get_graminfo(word.upper())
        if norms:
            lemms = set([info["norm"] for info in norms])
            if len(lemm) == 1:
                if true_lemma in lemms:
                    return [true_lemma]    
            if true_lemma in norms:
                norms.remove(true_lemma)
            return [true_lemma] + sorted(list(norms))           
        return [word.upper()]

    @staticmethod
    def prepare_corpus(trainfile, suff_len):
        """
        Обработка тренировочного корпуса: убираем все, кроме суффиксов
        """
        with codecs.open(filename, "r", encoding="UTF8") as fin, codecs.open(filename + ".suffs", "w", encoding="UTF8") as fout:
            for line in fin:
                line = line.strip()
                if line == "<S>" or line == "</S>": # Если это метка начала или конца предложения
                    fout.write(line)
                    continue
                items = line.split("\t")
                if len(items) <= 2:
                    continue # Это знак препинания
                lemms = lemm_list(*items[:2]) # Список возможных лемм, первая - правильная
                word = items[0].upper()
                suff = suffix(word, suff_len) # Трехбуквенный суффикс слова
                stem = longest_common([word] + lemms) # Наибольший общий префикс (стем?)
                lem_flexes = [suffix(lemma, len(lemma) - len(stem)) for lemma in lemms] # Берем только суффиксы от всех лемм
                fout.write("{0}\t{1}\n".format(suff, "\t".join(lem_flexes)))
        return True

    @staticmethod
    def count_sentence_suffs(sentence, freqs, cfreqs, radius):
        """
        Сбор статистики на основе суффиксов: обработка одного предложения
        """
        if not sentence:
            return True
        pairs = dict(enumerate(sentence))
        hom_nums = [num for (num, info) in pairs.items() if len(info) > 2] # Номера омонимов в предложении
        for hom_num in hom_nums:
            for num in smart_range(pairs.keys(), hom_num, radius):
                freqs[(num - hom_num, pairs[num][0], tuple(sorted(pairs[hom_num][1:])))][pairs[hom_num][1]] += 1
                cfreqs[(num - hom_num, tuple(sorted(pairs[hom_num][1:])))][pairs[hom_num][1]] += 1        
        return True

    def find_prep(self, sentence, ind):
        """
        Нахождение ближайшего предлога слева, который управляет данным словом
        
        ind - номер данного слова в предложении sentence
        """
        sent = dict(enumerate(sentence))
        cur = ind - 1
        for cur in list(xrange(ind))[::-1]:
            if len(sent[cur]) < 3 and not re.match(self.splitter, sent[cur][0]):
                continue
            if not sent[cur][2] in {u"П", u"ПРИЧАСТИЕ", u"ЧАСТ", u"СОЮЗ"}:
                break
        if sent[cur][2] == u"ПРЕДЛ":
            return sent[cur][1]
        return "_"

    def count_sentence_cases(self, sentence, freqs):
        """
        Сбор статистики на основе падежей: обработка одного предложения (sentence)
        
        freqs - словарь для наполнения статистикой
        """
        if not sentence:
            return True
        for (info, ind) in zip(sentence, xrange(len(sentence))):
            if len(info) < 3:
                continue
            if not info[2].split("|")[0] in self.caseable: # Работаем только со словами, которые могут иметь падеж
                continue            
            norms = self.gram_all(info[0])  # Все возможные варианты лемм текущего слова
            try:
                true_cases = set(info[2].split("|")[1].split(",")).intersection(self.cases)
                if len(true_cases) > 1:
                    continue
                true_case = true_cases.pop()
                all_vars = [norm for norm in norms if norm.has_key("info")]
                all_cases = set([x for y in [set(norm["info"].split(",")).intersection(self.cases) for norm in all_vars] for x in y])
                if not true_case in all_cases or len(all_cases) == 1:
                    continue
                prep = self.find_prep(sentence, ind)
                freqs[(prep, tuple(sorted(all_cases)))][true_case] += 1
            except Exception:
                continue
                      
        return True
   
    def train(self, trainfile, radius=4, suff_len=3):
        """
        Сбор статистики на основе суффиксов: обработка всего корпуса
        
        trainfile - размеченный корпус,
        radius - это радиус контекста, который учитывается при выбора правильной леммы,
        suff_len - длина суффиксов, на которых основано обучение 
        """
        # Если тренировочный корпус еще не подготовлен, делаем это прямо сейчас
        if trainfile.endswith(".lemma"):
            Tagger.prepare_corpus(trainfile, suff_len)
            trainfile += ".suffs"

        freqs = defaultdict(lambda: defaultdict(int))
        cfreqs = defaultdict(lambda: defaultdict(int))
        ranks = defaultdict(float)
        caseranks = defaultdict(float)

        # Структура словаря: {<Номер в контексте>, <Контекст>, <Список омонимов> : <Выбранный омоним> : <Вероятность>}
        normfreqs = defaultdict(lambda: defaultdict(float))
        # Структура словаря: {<Номер в контексте>, <Контекст>: <Ранг>}
        normweights = defaultdict(float)

        # Собираем частоты из корпуса
        with codecs.open(trainfile, "r", encoding="UTF8") as fin:
            sentence = []
            for line in fin:
                line = line.strip()
                if line == "<S>":
                    continue
                if line == "</S>":
                    Tagger.count_sentence_suffs(sentence, freqs, cfreqs, radius)
                    del sentence[:]
                    sentence = []
                    continue
                sentence.append(line.split("\t"))

        # Нормализуем частоты
        for k, v in freqs.items():
            total = sum([freq for freq in v.values()])
            for hom, freq in v.items():
                normfreqs[k][hom] = float(freq) / total
                
        # Вычисляем ранги контекстов
        for k, v in cfreqs.items():
            total = sum(v.values())
            entropy = - float(sum([freq * math.log(freq) for freq in v.values()]) - total * math.log(total)) / total
            ranks[k] = 1.0 / math.exp(entropy)

        # Вычисляем веса контекстов
        for k, v in ranks.items():
            normweights[k[0]] += v

        v_sum = sum([v for v in normweights.values()])
        for k, v in normweights.items():
            normweights[k] = v / v_sum

        # Сериализуем частоты и веса (ранги) контекстов (чем больше энтропия распределения по омонимам, тем меньше ранг (вес))

        dfreqs = dawg.BytesDAWG([(u"{0:d}\t{1}\t{2}\t{3}".format(k[0], k[1], " ".join(k[2]), hom), struct.pack("f", freq))
                                     for k, v in normfreqs.iteritems() for hom, freq in v.iteritems()])
        dfreqs.save(trainfile + ".freqs.dawg")
        dump_data(trainfile + ".weights.pkl", normweights)

        # Сериализуем small-значение (для тех случаев, которых нет в словаре)
        small = 1.0 / (2 * sum([freq for k, v in normfreqs.iteritems() for v1, freq in v.iteritems()]))
        dump_data(trainfile + ".small", small)  
        return True

    def train_cases(self, trainfile, threshold=4, small_diff=0.01):
        """
        Обучение снятию падежной омонимии (автоматическое извлечение правил)
        
        trainfile - размеченный корпус,
        threshold - минимальная абсолютная частота вхождения правила в корпус,
        small_diff - максимальная допустимая разность между двумя вариантами правила с наибольшей вероятностью.
        """
        freqs = defaultdict(lambda: defaultdict(int))
        self.caserules = defaultdict(str)

        # Собираем частоты из корпуса
        with codecs.open(trainfile, "r", encoding="UTF8") as fin:
            sentence = []
            for line in fin:
                line = line.strip()
                if line == "<S>":
                    continue
                if line == "</S>":
                    self.count_sentence_cases(sentence, freqs)
                    del sentence[:]
                    sentence = []
                    continue
                sentence.append(line.split("\t"))

        # Извлекаем правила
        for k, v in freqs.iteritems():
            good_values = {case: freq for case, freq in v.iteritems() if freq >= threshold}
            total = sum(good_values.values())
            for case, freq in good_values.iteritems():
                freqs[k][case] = float(freq) / total
            chosen = argmax([(case, freq) for case, freq in good_values.iteritems()])
            if not chosen:
                continue
            if len(chosen) > 1:
                continue
            if len(v.keys()) == 1:
                self.caserules[k] = sorted(chosen)[0]
                continue
            second = argmax([(case, freq) for case, freq in good_values.iteritems() if case != chosen[0]])
            if second:
                if freqs[k][chosen[0]] - freqs[k][second[0]] < small_diff:
                    continue
            self.caserules[k] = sorted(chosen)[0]
        
        # Тестовый вывод в файл
        #with codecs.open("prep_stat2.txt", "w", encoding="UTF8") as fout:
        #    for k, v in sorted(freqs.items()):
                #total = sum([freq for freq in v.values()])
                #entropy = - sum([float(freq) * math.log(float(freq) / total) / total  for freq in v.values()])
        #        entropy = - sum([freq * math.log(freq) for freq in v.values()])
        #        for case, freq in sorted(v.iteritems()):
        #            fout.write(u"{0}\t{1}\t{2}\t{3:.3f}\t{4:.3f}\n".format(k[0], "|".join(k[1]), case, freq, entropy))
                
        # Сериализуем правила
        # Структура: <Предлог>, <Список падежей> : <Правильный падеж>
        dump_data(trainfile + ".caserules.pkl", self.caserules)      
        return True

    def dump_preps(self, filename):
        """
        Запись статистики по предлогам и падежам в текстовый файл
        """
        with codecs.open(filename, "w", encoding="UTF8") as fout:
            for k, v in sorted(self.caserules.iteritems()):
                fout.write(u"{0}\t{1}\t{2}\n".format(k[0], "|".join(k[1]), v))
        return True
    
    def load_statistics(self, trainfile):
        """
        Загрузка суффиксной и падежной статистики
        """
        try:
            self.caserules = unpkl_1layered_s(trainfile + ".caserules.pkl")
            #self.freqs = unpkl_2layered_f(trainfilesuff + ".freqs.pkl")
            self.weights = unpkl_1layered_f(trainfile + ".suffs.weights.pkl")
            self.freqs = dawg.BytesDAWG()
            self.freqs.load(trainfile + ".suffs.freqs.dawg")
            #self.weights = dawg.BytesDAWG()
            #self.weights.load(trainfile + ".suffs.weights.dawg")
            with open(trainfile + ".suffs.small", "rb") as fin:
                self.small = pickle.load(fin)
        except Exception as e:
            print "Tagger statistics not found!\n", e
            sys.exit()

    def lemmatize(self, tokens, make_all=True):
        """
        Получение словаря нумерованных лемматизированных токенов по простому списку токенов
        
        make_all - подгружать все варианты нормальных форм (а не только первую),
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U    
        """
        action = self.gram_all if make_all else self.gram_first
        return dict(enumerate([tuple([token] + action(token)) for token in tokens]))

    def make_sents(self, lemmtokens):
        """
        Разбиение текста на предложения
        
        lemmtokens - словарь лемматизированных токенов текста вида {номер: (лемматизированный токен)}
        """
        bound = False
        sentences = []
        cur_sent = []
        for ind, info in lemmtokens.iteritems():
            if re.match(self.splitter, info[0]): # Возможная граница предложения
                cur_sent.append((ind, info))
                if len(cur_sent) == 1:
                    bound = False
                    continue
                if not re.match(self.bad_ender, cur_sent[-2][1][0]):    # Последний токен предложения не может быть одной буквой
                    bound = True
                continue
            if bound and info[0].strip() == "": # Пробельные символы между предложениями
                cur_sent.append((ind, info))
                continue
            if bound and not re.match(self.starter, info[0]):
                bound = False
                cur_sent.append((ind, info))
                continue
            if bound and re.match(self.starter, info[0]):# and cur_sent[-1][1][0].strip() == "": # Возможное начало предложения
                sentences.append(cur_sent)
                cur_sent = []
                cur_sent.append((ind, info))
                bound = False
                continue
            cur_sent.append((ind, info))
        if cur_sent:
            sentences.append(cur_sent)
        return tuple(sentences)

    def parse_simple(self, sent_tokens, sent_words):
        """
        Снятие частеречной омонимии для однобуквенных и двухбуквенных слов предложения
        """
        short_ambigs = [ind for ind in sent_words.keys() if re.match(self.short, sent_words[ind][0])]
        for ind in short_ambigs:
            try:
                if re.match(self.splitter, sent_tokens[ind + 1][0]) and sent_words[ind][1]["class"] != u"С":
                    sent_words[ind][1]["class"] = u"С"
                    sent_words[ind][1]["info"] = u"аббр"
            except Exception:
                continue
        return sent_words

    def parse_cases(self, sent_tokens, sent_words):
        """
        Снятие падежной омонимии слов предложения
        """
        caseambigs = [ind for ind in sent_words.keys()
                if len(sent_words[ind]) > 2
                and all(info["class"] in self.caseable for info in sent_words[ind][1:])]

        for ind in caseambigs:
            all_vars = [info for info in sent_words[ind][1:] if info.has_key("info")]
            all_cases = set([x for y in [set(info["info"].split(",")).intersection(self.cases) for info in all_vars] for x in y])
            for cur in list(xrange(min(sent_tokens.keys()), ind))[::-1]:
                if re.match(self.splitter, sent_tokens[cur][0]):
                    break
                if not sent_tokens[cur][1].has_key("class"):
                    continue
                if not sent_tokens[cur][1]["class"] in {u"П", u"ПРИЧАСТИЕ", u"ЧАСТ", u"СОЮЗ"}:
                    break
            try:
                if sent_tokens[cur][1]["class"] == u"ПРЕДЛ":
                    prep = sent_tokens[cur][1]["norm"]
                else:
                    prep = "_"
                if all_cases != {u"им", u"вн"} or prep != "_":
                    case = self.caserules[(prep, tuple(sorted(all_cases)))]
                    if case:
                        sent_words[ind] = xrestore_lemm(sent_words, case, ind)
                else:
                    sent_words[ind] = nom_case_disamb(sent_words, ind)
            except Exception:
                continue

        return True
    
    def parse_sent(self, sentence, radius, suff_len, small_diff, process_cases):
        """
        Снятие морфологической омонимии предложения
        
        sentence - предложение (список нумерованных токенов),
        radius - радиус контекста, который учитывается при выбора правильной леммы,
        suff_len - длина суффиксов, на которых основано обучение,
        small_diff - максимальная допустимая разность между двумя вариантами правила с наибольшей вероятностью,
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U
        process_cases=True -> снимаем падежную омонимию
        """
        if len(sentence) == 1:
            return sentence
        # Словарь токенов данного предложения
        sent_tokens = dict(sentence)
        # Словарь слов данного предложения
        sent_words = {ind: info for (ind, info) in sentence if len(info[1]) > 1}
        # Список суффиксов словоформ
        suffs = [suffix((info)[0], suff_len).upper() for (ind, info) in sorted(sent_words.iteritems(), key=lambda x: x[0])]
        # Словарь формата {(номер_абс, номер_отн): отсортированный список суффиксов}
        suffixes = OrderedDict([((ind, rel_num), get_suffixes(lemmtoken))
                    for (rel_num, (ind, lemmtoken)) in itertools.izip(range(len(sent_words.keys())), sorted(sent_words.iteritems(), key=lambda x: x[0]))])
        # Номера неоднозначностей (абс. и отн.)
        ambigs = [(ind, rel_num) for ((ind, rel_num), suff_list) in sorted(suffixes.iteritems(), key=lambda x: x[0][0]) if len(suff_list) > 1]
        # Снятие частеречной омонимии для однобуквенных и двухбуквенных слов
        sent_words = self.parse_simple(sent_tokens, sent_words)
            
        # Снятие омонимии во всех остальных случаях

        # Набор контекстов для данного предложения
        contexts = {(num, rel_num):
            [(-i, suff) for (i, suff) in itertools.izip(range(1, radius + 1), smart_slice(suffs, rel_num - radius, rel_num)[::-1])] +
            [(i, suff) for (i, suff) in itertools.izip(range(1, radius + 1), smart_slice(suffs, rel_num + 1, rel_num + radius + 1))]
                    for (num, rel_num) in ambigs}

        # Снятие омонимии на уровне лемм
        for (ind, rel_num) in ambigs:
            suff_list = suffixes[(ind, rel_num)]
            pairs = contexts[(ind, rel_num)]
            probs = [(var, sum([get_floatDAWG(self.freqs, u"{0:d}\t{1}\t{2}\t{3}".format(rel_ind, sf, " ".join(suff_list), var), self.small) * self.weights[rel_ind]
                         for (rel_ind, sf) in pairs])) for var in suff_list]
            arg_max = argmaxx(probs) # Список наиболее вероятных суффиксов

            if arg_max:
                if len(arg_max) == len(suff_list): # Если все варианты одинаковые, берем тот, который предлагает pymorphy
                    continue
                
                second_prob = max([prob for (var, prob) in probs if prob < arg_max[0][1]])
                if arg_max[0][1] - second_prob < small_diff: # Ограничение на разницу между двумя макс. вероятностями
                    continue
                
                suffitem = sorted(arg_max)[0][0].replace("_", "") # Лучший суффикс

                # Восстановление леммы по найденному суффиксу
                sent_words[ind] = restore_lemm(sent_words, suffitem, ind)

        if self.dater:  # Обработка дат, если необходимо
            self.dater.parse_dates(sent_words, sent_tokens)

        if process_cases:   # Снятие падежной омонимии, если необходимо
            self.parse_cases(sent_tokens, sent_words)
                                 
        new_sentence = []   # Предложение со снятой омонимией
        for ind, info in sentence:        

            if sent_words.has_key(ind):
                new_sentence.append((ind, sent_words[ind]))
            else:
                new_sentence.append((ind, info))
        return tuple(new_sentence)

    def write_stream(self, lemmtokens, fout, radius, suff_len, sent_marks, process_cases, small_diff):
        """
        Снятие морфологической омонимии текста и запись в поток вывода по предложениям
        
        lemmtokens - нумерованные лемматизированные токены,
        fout - поток вывода,
        radius - радиус контекста, который учитывается при выбора правильной леммы,
        suff_len - длина суффиксов, на которых основано обучение,
        sent_marks=True -> вставляем метки начала и конца предложения: <S>, </S>
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U,
        process_cases=True -> снимаем падежную омонимию,
        small_diff - максимальная допустимая разность между двумя вариантами правила с наибольшей вероятностью,
        """
        for sentence in self.make_sents(lemmtokens):
            Tagger.write_sentence(self.parse_sent(sentence, radius, suff_len, small_diff, process_cases), fout, sent_marks)
        return True

    @staticmethod
    def write_sentence(sentence, fout, sent_marks):
        """
        Запись лемматизированного предложения в файл
        
        fout - поток вывода,
        sent_marks=True -> вставляем метки начала и конца предложения: <S>, </S>        
        """  
        if sent_marks:
            fout.write("<S>\n")
        for (ind, info) in sentence:
            if info[0].strip() != "":
                fout.write(u"{0}\t{1}".format(info[0], info[1]["norm"]))
                if info[1].has_key("class"):
                    fout.write(u"\t" + info[1]["class"])
                if info[1].has_key("info"):
                    fout.write(u"\t" + info[1]["info"])
                fout.write(u"\n")
        if sent_marks:
            fout.write(u"</S>\n")
        return True

    def parse_all(self, lemmtokens, outfile, radius=4, suff_len=3, sent_marks=False, process_cases=True, small_diff=0.001):
        """
        Обработка всего текста сразу (с записью результата в файл)

        lemmtokens - нумерованные лемматизированные токены,
        outfile - файл, в который будет записан обработанный текст,
        radius - радиус контекста, который учитывается при выбора правильной леммы,
        suff_len - длина суффиксов, на которых основано обучение,
        sent_marks=True -> вставляем метки начала и конца предложения: <S>, </S>
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U,
        process_cases=True -> снимаем падежную омонимию,
        small_diff - максимальная допустимая разность между двумя вариантами правила с наибольшей вероятностью,
        """
        with codecs.open(outfile, "w", encoding="UTF8") as fout:
            self.write_stream(lemmtokens, fout, radius, suff_len, sent_marks, process_cases, small_diff)
        return True

    def parse_chunks(self, filename, radius=4, suff_len=3, chunks=2000, sent_marks=False, process_cases=True, small_diff=0.001):
        """
        Обработка текста частями и запись результата в файл

        filename - исходный текстовый файл,
        radius - радиус контекста, который учитывается при выбора правильной леммы,
        suff_len - длина суффиксов, на которых основано обучение,
        chunks - примерная длина одного чанка (в строках),
        sent_marks=True -> вставляем метки начала и конца предложения: <S>, </S>
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U,
        process_cases=True -> снимаем падежную омонимию,
        small_diff - максимальная допустимая разность между двумя вариантами правила с наибольшей вероятностью,
        """
        buff = []
        counter = 0
        tokens = {}
        tok_r = Tokenizer()
        # Читаем тестовый файл
        with codecs.open(filename, "r", encoding = "UTF8") as fin, codecs.open(filename + ".lemma", "w", encoding = "UTF8") as fout:
            for line in fin:
                if len(buff) >= chunks and re.search(self.splitter, buff[-1]):
                    part_1 = re.split(self.splitter, buff[-1])[0] + re.findall(self.splitter, buff[-1])[0]
                    part_rest = buff[-1][len(part_1) + 1:]
                    self.parse_chunk(buff[:-1] + [part_1], fout, tok_r, radius, suff_len, sent_marks, process_cases, small_diff)
                    del buff[:]
                    buff = [part_rest]
                    counter += 1
                    print "chunk", counter, "done!"
                buff.append(line)
            if buff != []:
                self.parse_chunk(buff, fout, tok_r, radius, suff_len, sent_marks, process_cases, small_diff)

    def parse_chunk(self, buff, fout, tok_r, radius, suff_len, sent_marks, process_cases, small_diff):
        """
        Снятие морфологической омонимии текстового фрагмента и запись результата в открытый поток вывода

        buff - текущий текстовый фрагмент для обработки,
        fout - поток вывода,
        tok_r - используемый токенизатор,
        radius - радиус контекста, который учитывается при выбора правильной леммы,
        suff_len - длина суффиксов, на которых основано обучение,
        sent_marks=True -> вставляем метки начала и конца предложения: <S>, </S>
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U,
        process_cases=True -> снимаем падежную омонимию,
        small_diff - максимальная допустимая разность между двумя вариантами правила с наибольшей вероятностью,   
        """
        lemmtokens = self.lemmatize(tok_r.tokenize("".join(buff))) # Словарь токенов
        self.write_stream(lemmtokens, fout, radius, suff_len, sent_marks, process_cases, small_diff)
        return True

    def get_parsed_sents(self, tokens, radius=4, suff_len=3, process_cases=True, small_diff=0.001):
        """
        Получение списка предложений со снятой морфологической омонимией

        tokens - список токенов исходного текста,
        radius - радиус контекста, который учитывается при выбора правильной леммы,
        suff_len - длина суффиксов, на которых основано обучение,
        Если self.dater != None, для всех дат в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
        а в качестве граммем - формат YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD с суффиксом -B, -L, -I или -U,
        process_cases=True -> снимаем падежную омонимию,
        small_diff - максимальная допустимая разность между двумя вариантами правила с наибольшей вероятностью,          
        """
        return [self.parse_sent(sentence, radius, suff_len, small_diff, process_cases) for sentence in self.make_sents(self.lemmatize(tokens))]

if __name__ == "__main__":

    filename = os.path.join(os.path.dirname(sys.argv[0]), "test/freview.txt")
    trainfile = os.path.join(os.path.dirname(sys.argv[0]),"dicts/ruscorpora.txt.lemma")
    #prepsfile = os.path.join(os.path.dirname(sys.argv[0]),"corpora/preps_stat.txt")
    
    print "STARTED:", str(datetime.now())
    start = time.time()

    morph = get_morph(os.path.join(os.path.dirname(sys.argv[0]),"pydicts").decode("UTF8"))  # Подгружаем русский словарь
    morph_simple = get_morph(os.path.join(os.path.dirname(sys.argv[0]),"pydicts").decode("UTF8"), check_prefixes=False) # Подгружаем русский словарь - 2
    tok = Tokenizer()   # Подгружаем токенизатор
    dater = Dater() # Подгружаем обработчик дат
    tagger = Tagger(morph, morph_simple, dater)  # Подгружаем тэггер
    #tagger.train_cases(trainfile) # Обучаем тэггер падежам
    #tagger.train(trainfile + ".suffs") # Обучаем тэггер суффиксам
    tagger.load_statistics(trainfile)   # Загружаем суффиксную статистику  
    #tagger.dump_preps(prepsfile)   # Выписываем правила падежей в зависимости от предлогов в текстовый файл

    print "Statistics loaded! It took", time.time() - start, "\nParsing file..."

    tokens = []  
    with codecs.open(filename, "r", encoding="UTF8") as fin:    # Читаем тестовый файл
        tokens = tok.tokenize(fin.read()) # Список токенов
    # Записываем результат в файл
    tagger.parse_all(tagger.lemmatize(tokens), filename + ".lemma", sent_marks=True)
        
    print "FINISHED:", str(datetime.now())
    print "Time elapsed: ", time.time() - start
