#!/usr/bin/python
# -*- coding: utf-8 -*-

# Синонимизатор текста со снятием морфологической омонимии

import sys
import os
import codecs
import string
import re
import itertools
import math
import time
from operator import mul
from random import choice
from datetime import datetime
from collections import deque, defaultdict
import cPickle as pickle

from pymorphy import get_morph
import dawg

from tokenizer import Tokenizer
from tagger import Tagger
from commontools import *
from morphotools import *
from pickling import *

def html(text):
    """
    HTML-форматирование текста
    """
    return "<p>" + re.sub(r"\n+", "</p><p>", text) + "</p>"

def pl(filename):
    """
    Обертка адреса для подгружаемых словарей
    """
    return os.path.join(os.path.dirname(sys.argv[0]), "dicts/" + filename)

def error(text, show_details, details=''):
    """
    Обертка вывода ошибки в браузере
    """
    print text
    if show_details:
        print "Details:", details
    return False

class Synonimizer(object):
    """
    Синонимизатор русского текста со снятием морфологической омонимии
    """

    def __init__(self, morph, params):
        """
        Инициализация синонимизатора
        """
        args = set(params)
        self.morph = morph    #  Морфология pymorphy
        self.UseIdioms = True if "-i" in args else False
        self.UseBayes = True if "-b" in args else False
        self.UseDisambig = True if "-disamb" in args else False
        self.UseNGram = True if "-n" in args else False
        self.UseDetails = True if "-d" in args else False
        self.UseOnlySyn = True if "-os" in args else False
        self.UseViterbi = True if "-v" in args else False
        self.UseCollocs = True if "-col" in args else False
        
        # Выводим параметры, с которыми запущен синонимизатор (если необходимо)
        if self.UseDetails:
            print "<font color='blue'>" + " ".join(params) + "</font><br><br>"

        # "Уровень" подбора синонимов: 1 - Точный, 2 - Средний, 3 - Низкий, с макс. синонимизацией
        Level = 3
        if "-level" in args:
            try:
                Level = int(sys.argv[sys.argv.index("-level") + 1])
            except Exception:
                Level = 3

        # Имя файла со словарем синонимов
        if "-dict" in args:
            try:
                synfile = sys.argv[sys.argv.index("-dict") + 1]
            except Exception:
                synfile = "base.pkl"
        else:
            raise IOError("Synonims dictionary not found!")

        # Пороги отсечения вероятностей в зависимости от "уровня" - для N-граммной модели
        SMALL_PROBS = {1: 0.5, 2: 0.2, 3: 0}
        # Максимальное допустимое число равноценных вариантов синонимов в зависимости от "уровня"
        VARS_COUNT = {1: 3, 2: 5, 3: 10}

        self.vars_count = VARS_COUNT[Level]
        self.small_prob = SMALL_PROBS[Level]

        (self.samplefile, corpusfile) = params[1:3]
        
        # Проверка параметров: метод синонимизации
        if sum(map(bool, (self.UseBayes, self.UseNGram, self.UseViterbi, self.UseCollocs))) != 1:
            raise ValueError("Choose only one of the three methods: Bayesian, Ngram or Viterbi!!!")

        self.actions = {self.UseNGram: self.calc_ngram_sent, self.UseBayes: self.calc_bayes_sent, self.UseViterbi: self.calc_viterbi_sent,
                        self.UseCollocs: self.calc_colloc_sent} 

        # Подгружаем сериализованный словарь синонимов. Формат словаря: {лексема: множество лексем-синонимов}.
        self.syns = unpkl_1layered_sets(pl(synfile)) 

        # Подгружаем словарь идиом, если необходимо. Формат: множество.
        if self.UseIdioms:   
            self.idioms = set()
            with codecs.open(pl("idioms.txt.lemma"), "r", encoding = "UTF8") as fin:
                for line in fin:
                    self.idioms.add(line.strip())

        # Подгружаем коллокации, если необходимо
        if self.UseCollocs:
            self.collocs = dawg.IntDAWG()
            self.collocs.load(pl(corpusfile + ".collocs.dawg"))
            self.posfreqs = unpkl_3layered_f(pl(corpusfile + ".pos.pkl"))

        # Подгружаем частоты униграмм и общее число слов в корпусе
        if not self.UseCollocs:
            self.freqs = dawg.IntDAWG()
            self.freqs.load(pl(corpusfile + "_freqs_1.dawg"))
            self.f_sum = 0
            with open(pl(corpusfile + "_1_sum.pkl"), "rb") as fin:
                self.f_sum = pickle.load(fin)

        # Подгружаем список контекстов, если необходимо
        if self.UseBayes:
            self.N = 5
            self.contexts = dawg.IntDAWG()
            self.contexts.load(pl(corpusfile + "_contexts_5.dawg"))
                
        # Подгружаем частоты n-грамм, если необходимо
        if self.UseNGram or self.UseViterbi:
            self.N = 3
            (self.freqs2, self.freqs3) = (dawg.IntDAWG(), dawg.IntDAWG())
            self.freqs2.load(pl(corpusfile + "_freqs_2.dawg"))
            self.freqs3.load(pl(corpusfile + "_freqs_3.dawg"))

        if self.UseCollocs:
            self.indexed_wsyns = defaultdict(str) # Словарь типа {номер токена: синоним}
        else:
            self.indexed_syns = defaultdict(str) # Словарь типа {номер токена: синоним}

    @staticmethod
    def add_contexts(all_syns, contexts, sentence, n):
        """
        Добавление в словарь слов из контекста (для контекстной модели)
        """
        for i in xrange(len(sentence)):
            word = sentence[i]
            context_set = set(smart_slice(sentence, i - n, i) + smart_slice(sentence, i + 1, i + n))
            if word in all_syns:
                for word in context_set:
                    contexts[sentence[i]][word] += 1
        return True

    @staticmethod
    def train_bayes(corpus, n):
        """
        Обучение контекстной модели для наивного байесовского классификатора
        """
        syns = {}
        with open(pl("total.pkl"), "rb") as fin:    # Подгружаем словарь синонимов
            syns = pickle.load(fin)
        all_syns = set(syns.keys())
        all_syns = all_syns.union(set(itertools.chain.from_iterable(syns.values())))  # Все синонимы
        contexts = defaultdict(lambda: defaultdict(int))  # Словарь контекстов
        sentence = []
        # Чтение из файла корпуса и запись в выходной файл
        with codecs.open(pl(corpus), "r", encoding="UTF8") as fin:
            for line in fin:
                line = line.strip()
                if line == u"<S>":
                    sentence = [u"*START*"]
                    continue
                if line == u"</S>":
                    sentence.append(u"*STOP*")
                    Synonimizer.add_contexts(all_syns, contexts, sentence, n)
                    sentence = []
                if line == "" or len(line.split("\t")) <= 2:
                    continue
                sentence.append(line.split("\t")[1])            
        
        d = dawg.IntDAWG([("|".join((word1, word2)), freq) for word1, v1 in contexts.iteritems() for word2, freq in v1.iteritems()])
        d.save(pl(corpus + "_contexts_" + str(n) + ".dawg"))  # Сериализация словарей

        print corpus, "bayes saved"
        return True

    @staticmethod
    def add_counts(freqs, buff, n):
        """
        Добавление N-граммы в словарь (при обучении)
        """
        for i in xrange(len(buff) - n + 1):
            if n == 3:
                freqs[buff[i]][buff[i + 1]][buff[i + 2]] += 1
            if n == 2:
                freqs[buff[i]][buff[i + 1]] += 1       
        return True

    @staticmethod
    def train_ngram(corpus, n):
        """
        Обучение N-граммной модели (2- или 3-)
        """
        if n == 3:
            freqs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # Словарь частот
        elif n == 2:
            freqs = defaultdict(lambda: defaultdict(int))
        else:
            raise ValuError("Wrong n parameter for N-gram model training!")
        
        # Чтение из файла корпуса
        with codecs.open(pl(corpus), "r", encoding="UTF8") as fin:             
            for line in fin:
                line = line.strip()
                if line == u"<S>":
                    buff = ["*START*"] * n
                    continue
                if line == u"</S>": # Конец предложения
                    buff.append(u"*STOP*")
                    Synonimizer.add_counts(freqs, buff, n)       
                if line == "" or len(line.split("\t")) <= 2:    # Знак препинания
                    continue    
                # Накапливаем частоты лемм
                lem = line.split("\t")[1]
                if lem != "":
                    buff.append(lem)

        if n == 3:
            d = dawg.IntDAWG([("|".join((word1, word2, word3)), freq) for word1, v1 in freqs.iteritems() for word2, v2 in v1.iteritems() for word3, freq in v2.iteritems()])      
        elif n == 2:
            d = dawg.IntDAWG([("|".join((word1, word2)), freq) for word1, v1 in freqs.iteritems() for word2, freq in v1.iteritems()])

        d.save(pl(corpus + "_freqs_" + str(n) + ".dawg"))    # Сериализация
        print corpus, n, "gram saved"
        return True

    @staticmethod
    def add_counts_distant(freqs, pos, valid_pos, all_lexemes, buff, n):
        """
        Добавление несмежных биграмм в словарь
        """
        for i in range(0, len(buff) - n):
            for j in range(i + 1, i + n + 1):
                if not (buff[i][2] in valid_pos and buff[j][2] in valid_pos and (buff[i][1] in all_lexemes or buff[j][1] in all_lexemes)):
                    continue
                freqs[(buff[i][0].lower().replace(u"ё", u"е"), buff[j][0].lower().replace(u"ё", u"е"))] += 1
                pos[(buff[i][2], buff[j][2])] += 1
        return True

    @staticmethod
    def xadd_counts_distant(trie, pos, valid_pos, all_lexemes, buff, n):
        """
        Добавление несмежных биграмм в словарь
        """
        for i in range(0, len(buff) - n):
            for j in range(i + 1, i + n + 1):
                if not (buff[i][2] in valid_pos and buff[j][2] in valid_pos and (buff[i][1] in all_lexemes or buff[j][1] in all_lexemes)):
                    continue
                result = "|".join((buff[i][0].lower().replace(u"ё", u"е"), buff[j][0].lower().replace(u"ё", u"е")))
                if result in trie:
                    trie[result] += 1
                else:
                    trie[result] = 1
                pos[(buff[i][2], buff[j][2])] += 1
        return True

    @staticmethod
    def count_distant(corpus, step, min_freq):
        """
        Сбор частот несмежных биграмм (с шагом step, минимальной частотой min_freq)
        """
        freqs = defaultdict(int) # Словарь частот несмежных биграмм (словоформы)
        pos = defaultdict(int) # Словарь частот несмежных биграмм (части речи)
        buff = [] # Временный буфер для хранения текущей N-граммы
        valid_pos = good_pos()

        syns = unpkl_1layered_sets(pl("total.pkl")) # Самый большой словарь синонимов
        all_lexemes = set([x for word, w_set in syns.iteritems() for x in w_set]).union(set(syns.keys()))    # Все слова из словаря синонимов
        
        with codecs.open(pl(corpus), "r", encoding = "UTF8") as fin:                     
            for line in fin:
                line = line.strip(string.whitespace)
                if line == "<S>":
                    buff = []
                    continue
                if line == "</S>": # Если это не слово, идем дальше
                    if len(buff) > step:
                        Synonimizer.add_counts_distant(freqs, pos, valid_pos, all_lexemes, buff, step)
                    del list(buff)[:]
                    continue
                if len(line.split("\t")) <= 2:
                    if not line.split("\t")[0] == "-":
                        if len(buff) > step:
                            Synonimizer.add_counts_distant(freqs, pos, valid_pos, all_lexemes, buff, step)
                        del list(buff)[:]
                        buff = []
                    continue
                buff.append(line.split("\t"))

        # Тестовый вывод частеречной статистики в файл
        f_sum = sum(pos.values())
        with codecs.open(pl(corpus + ".pos"), "w", encoding="UTF8") as fout:
            for pos_items, freq in sorted(pos.iteritems(), key=lambda x: x[1], reverse=True):
                fout.write(u"{0}\t{1:.6f}\n".format("\t".join(pos_items), float(freq) / f_sum))

        pos_nested = inner_func_float2()
        for pos_pair, freq in pos.iteritems():
            pos_nested[pos_pair[0]]["R"][pos_pair[1]] = float(freq) / f_sum
            pos_nested[pos_pair[1]]["L"][pos_pair[0]] = float(freq) / f_sum            

        dump_data(pl(corpus + ".bigrams.pkl"), {bigram: freq for bigram, freq in freqs.iteritems() if freq >= min_freq})
        dump_data(pl(corpus + ".pos.pkl"), pos_nested)
        print "Distant bigrams and pos counts saved"
        return True

    @staticmethod
    def count_words(corpus):
        """
        Составление частотного словаря слов по корпусу
        """
        freqs = defaultdict(int)
        valid_pos = good_pos()
        
        with codecs.open(pl(corpus), "r", encoding="UTF8") as fin:
            for line in fin:
                line = line.strip()
                if line in {"<S>", "</S>"}:
                    continue
                items = line.split("\t")
                if len(items) <= 2:
                    continue
                if items[2] in valid_pos:
                    freqs[items[0].lower().replace(u"ё", u"е")] += 1
                
        dump_data(pl(corpus + ".unigrams.pkl"), freqs)
        print "Word counts saved"
        return True

    @staticmethod
    def train_collocations(corpus):
        """
        Сбор коллокаций по корпусу (по несмежным биграммам слов, не разделенных никакими знаками препинания, кроме тире).
        (Обучение коллокационной модели)
        """
        bigrams = unpkl_1layered_i(pl(corpus + ".bigrams.pkl"))
        unigrams = unpkl_1layered_i(pl(corpus + ".unigrams.pkl"))
        collocs = defaultdict(int)
        N = sum(unigrams.values())

        for bigram, freq in bigrams.iteritems():
            try:
                collocs[bigram] = MI(freq, unigrams[bigram[0]], unigrams[bigram[1]], N)
            except Exception:
                print freq, unigrams[bigram[0]], unigrams[bigram[1]], N
                continue

        # Тестовый вывод в файл
        with codecs.open(pl(corpus + ".collocs"), "w", encoding="UTF8") as fout:
            fout.write("Collocate_1\tCollocate_2\tFreqs\tMI\n")
            for bigram, mi in sorted(collocs.iteritems(), key=lambda x: x[1], reverse=True):
                fout.write(u"{0}\t{1:d}\t{2:.4f}\n".format("\t".join(bigram), bigrams[bigram], mi))

        dcollocs = dawg.IntDAWG([("|".join(bigram), int(freq * 100)) for bigram, freq in collocs.iteritems()])
        dcollocs.save(pl(corpus + ".collocs.dawg"))
        print "Collocations saved"
        return True

    @staticmethod
    def xtrain_collocations(corpus, step):
        """
        Сбор коллокаций по корпусу (по несмежным биграммам слов, не разделенных никакими знаками препинания, кроме тире).
        (Обучение коллокационной модели)
        """
        alphabet = u"-|абвгдеёжзийклмнопрстуфхцчшщъыьэюя"    # Алфавит для trie
        trie = datrie.BaseTrie(alphabet)    # Trie с частотами несмежных биграмм (словоформы)
        pos = defaultdict(int) # Словарь частот несмежных биграмм (части речи)
        buff = [] # Временный буфер для хранения текущей N-граммы
        valid_pos = good_pos()

        syns = unpkl_1layered_sets(pl("total.pkl")) # Самый большой словарь синонимов
        all_lexemes = set([x for word, w_set in syns.iteritems() for x in w_set]).union(set(syns.keys()))    # Все слова из словаря синонимов
        
        with codecs.open(pl(corpus), "r", encoding = "UTF8") as fin:                     
            for line in fin:
                line = line.strip(string.whitespace)
                if line == "<S>":
                    buff = []
                    continue
                if line == "</S>": # Если это не слово, идем дальше
                    if len(buff) > step:
                        Synonimizer.xadd_counts_distant(trie, pos, valid_pos, all_lexemes, buff, step)
                    del list(buff)[:]
                    continue
                if len(line.split("\t")) <= 2:
                    if not line.split("\t")[0] == "-":
                        if len(buff) > step:
                            Synonimizer.xadd_counts_distant(trie, pos, valid_pos, all_lexemes, buff, step)
                        del list(buff)[:]
                        buff = []
                    continue
                buff.append(line.split("\t"))

        # Тестовый вывод частеречной статистики в файл
        f_sum = sum(pos.values())
        with codecs.open(pl(corpus + ".pos"), "w", encoding="UTF8") as fout:
            for pos_items, freq in sorted(pos.iteritems(), key=lambda x: x[1], reverse=True):
                fout.write(u"{0}\t{1:.6f}\n".format("\t".join(pos_items), float(freq) / f_sum))

        pos_nested = inner_func_float2()
        for pos_pair, freq in pos.iteritems():
            pos_nested[pos_pair[0]]["R"][pos_pair[1]] = float(freq) / f_sum
            pos_nested[pos_pair[1]]["L"][pos_pair[0]] = float(freq) / f_sum            

        dump_data(pl(corpus + ".pos.pkl"), pos_nested)
        trie.save(pl(corpus + ".bigrams.trie"))
        
        print "Distant bigrams and pos counts saved"
        unigrams = unpkl_1layered_i(pl(corpus + ".unigrams.pkl"))
        N = sum(unigrams.values())

        for bigram, freq in trie.items(u""):
            try:
                trie[bigram] = MI(freq, unigrams[bigram[0]], unigrams[bigram[1]], N)
            except Exception:
                print freq, unigrams[bigram[0]], unigrams[bigram[1]], N
                continue

        # Тестовый вывод в файл
        with codecs.open(pl(corpus + ".collocs"), "w", encoding="UTF8") as fout:
            fout.write("Collocate_1\tCollocate_2\tFreqs\tMI\n")
            for bigram, mi in sorted(trie.items(u""), key=lambda x: x[1], reverse=True):
                fout.write(u"{0}\t{1:d}\t{2:.4f}\n".format("\t".join(bigram), bigrams[bigram], mi))

        trie.save(pl(corpus + ".collocs.trie"))
        print "Collocations saved"
        return True  

    @staticmethod
    def train_unigram(corpus):
        """
        Обучение униграммной модели
        """
        freqs = defaultdict(int)    # Словарь частот
        # Чтение из файла корпуса
        with codecs.open(pl(corpus), "r", encoding="UTF8") as fin:
            for line in fin:
                line = line.strip()
                if line == u"<S>":
                    freqs[u"*START*"] += 1
                    continue
                if line == u"</S>":
                    freqs[u"*STOP*"] += 1
                    continue 
                if line.strip() == "" or len(line.split("\t")) <= 2:
                    continue
                lem = line.split("\t")[1]
                if lem != "":
                    freqs[line.split("\t")[1]] += 1

        d = dawg.IntDAWG([(word, freq) for word, freq in freqs.iteritems()]) 
        d.save(pl(corpus + "_freqs_1.dawg"))    # Сериализация
        
        with open(pl(corpus + "_1_sum.pkl"), "wb") as fout: # Запоминаем количество слов в корпусе
            pickle.dump(sum(freqs.values()), fout, pickle.HIGHEST_PROTOCOL)
        print corpus, "unigram saved"
        return True

    def extract_synable(self, sentences):
        """
        Извлечение номеров только тех слов, которые подлежат замене синонимами
        """
        self.target_nums = set([ind for sentence in sentences
                                for (ind, info) in sentence
                                if info[1]["norm"] in set(self.syns.keys())
                                and info[1]["class"] in is_synable()])
        
        if not self.UseIdioms:
            return True

        pattern = re.compile("(?:" + "|".join([" ".join(["\d+[:]" + word for word in idiom.split()])
                                               for idiom in self.idioms]) + u")(?![А-ЯЁ])")
        idioms_found = re.findall(pattern, " ".join([str(ind) + ":" + x[1]["norm"]
                                                     for sentence in sentences for (ind, x) in sentence if "class" in x[1].keys()]))
                
        nums = re.compile("\d+(?=[:])") 
        self.target_nums -= set([int(x) for x in re.findall(nums, " ".join(idioms_found))])
        return True

    def colloc_probs(self, sentence, ind, step):
        """
        Вычисление оценок вероятностей синонимов с помощью коллокаций для одного слова
        """
        sent_words = {num: info for (num, info) in sentence if "class" in info[1].keys()}
        inds = list(sent_words.keys())
        
        main_ind = inds.index(ind)
        wform = sent_words[ind][0]
        main_lexeme = sent_words[ind][1]["norm"]  # Нормальная форма слова, которое мы хотим заменить синонимом
        main_pos = sent_words[ind][1]["class"]  # Часть речи слова, которое мы хотим заменить синонимом
        main_grams = ""  # Граммемы слова, которое мы хотим заменить синонимом
        if "info" in sent_words[ind][1].keys():
            main_grams = sent_words[ind][1]["info"]
        
        lefts = smart_slice(inds, main_ind - step, main_ind)
        rights = smart_slice(inds, main_ind + 1, main_ind + step + 1)
        pos_l = set([sent_words[x][1]["class"] for x in lefts])
        pos_r = set([sent_words[x][1]["class"] for x in rights])

        pos_l_valid = pos_l.intersection(set(self.posfreqs[main_pos]["L"].keys()))
        pos_r_valid = pos_l.intersection(set(self.posfreqs[main_pos]["R"].keys()))

        valid_left = [x for x in lefts if sent_words[x][1]["class"] in pos_l_valid] # Номера слов слева, которые будем проверять на биграммы
        valid_right = [x for x in rights if sent_words[x][1]["class"] in pos_r_valid] # Номера слов справа, которые будем проверять на биграммы

        infl_syns = set([inflect_comb(self.morph, syn, main_grams, main_pos) for syn in self.syns[main_lexeme]])    # Ставим синонимы в нужную форму
        if None in infl_syns:
            infl_syns.remove(None)
        infl_syns = set([get_same_caps(wform, x) for x in infl_syns])   # Синонимы данного слова в нужной форме

        probs = [(syn, sum([get_DAWG(self.collocs,
                                           sent_words[x_l][0].lower().replace(u"ё", u"е") +
                                           "|" +
                                           syn.lower()) * self.posfreqs[main_pos]["L"][sent_words[x_l][1]["class"]]
                           for x_l in valid_left]) +
                 sum([get_DAWG(self.collocs,
                                           sent_words[x_r][0].lower().replace(u"ё", u"е") +
                                           "|" +
                                           syn.lower()) * self.posfreqs[main_pos]["R"][sent_words[x_r][1]["class"]]
                           for x_r in valid_right]))
                 for syn in infl_syns]
        arg_value = argmaxx(probs)
        if arg_value:
            if arg_value[0][1] > 0:
                return choice(arg_value)[0]
        return None
    
    def calc_colloc_sent(self, sentence, step=3):
        """
        Вычисление оценок вероятностей синонимов с помощью коллокаций для одного предложения
        """
        for (num, info) in [(num, info) for (num, info) in sentence if num in self.target_nums]:
            if not self.syns[info[1]["norm"]]:
                continue
            result = self.colloc_probs(sentence, num, step)
            if not result:
                continue
            self.indexed_wsyns[num] = result            
        return True

    def smooth_ngram(self, ngram_parts, alpha=0.4):
        """
        Сглаживание оценки вероятности n-граммы методом Stupid Backoff
        """
        result = 1
        trigram = "|".join(ngram_parts)
        bigram = "|".join(ngram_parts[1:])
        if trigram in self.freqs3:
            return float(self.freqs3[trigram]) / self.freqs2["|".join(ngram_parts[:-1])]
        unigram = ngram_parts[-1]
        if bigram in self.freqs2:
            return alpha * float(self.freqs2[bigram]) / self.freqs[ngram_parts[1]]
        if unigram in self.freqs:
            return math.pow(alpha, 2) * float(self.freqs[unigram]) / self.f_sum
        return math.pow(alpha, 3) / self.f_sum

    def calc_ngram_sent(self, sentence):
        """
        Вычисление оценок n-граммной вероятности (со сглаживанием Stupid Backoff))
        """
        for (num, info) in [(num, info) for (num, info) in sentence if num in self.target_nums]:
            ngram_parts = ngram_slice(sentence, num, self.N)
            true_prob = self.smooth_ngram(ngram_parts)
            lexeme = info[1]["norm"]
            if not self.syns[lexeme]:
                continue
            probs = [(var, self.smooth_ngram(ngram_parts[:-1] + [var]) / true_prob) for var in self.syns[lexeme]]
            arg_max = argmax([(var, prob) for (var, prob) in probs if prob > self.small_prob])
            if arg_max and len(arg_max) <= self.vars_count:
                self.indexed_syns[num] = choice(arg_max)
        return True

    def K(self, word, k):
        """
        Возвращает множество допустимых синонимов для слова на k-ом месте в предложении
        """
        if k <= 0:
            return set(["*START*"])
        if word in set(self.syns.keys()):
            return self.syns[word]
        return set([word])

    def calc_viterbi_sent(self, sentence):
        """
        Вычисление по алгоритму Витерби (со сглаживанием Stupid Backoff)
        """
        # Список лемм предложения
        x = [""] + [info[1]["norm"] for (ind, info) in sentence if "class" in info[1].keys()]
        n = len(x) - 1
        y = [""] * (n + 1)

        # Алгоритм Витерби
        pi = defaultdict(float)
        bp = defaultdict(float)
        pi[0, "*START*", "*START*"] = 1.0

        for k in xrange(1, n + 1):
            for u in self.K(x[k - 1], k - 1):
                for v in self.K(x[k], k):
                    bp[k, u, v], pi[k, u, v] = choice(argmaxx(
                    [(w, pi[k - 1, w, u] / len(self.K(x[k], k)) * self.smooth_ngram((w, u, v)))
                      for w in self.K(x[k - 2], k - 2)]))
                                                                                               
        y[n - 1], y[n] = choice(argmax([((u, v), pi[n, u, v] * self.smooth_ngram(("*STOP*", u, v)))
                                           for u in self.K(x[n - 1], n - 1) for v in self.K(x[n], n)]))

        for k in range(n - 2, 0, -1):
            y[k] = bp[k + 2, y[k + 1], y[k + 2]]

        y = y[1: n + 1]

        sent_words = {ind: info for (ind, info) in sentence if "class" in info[1].keys()}
        self.indexed_syns.update({ind: y[k] for ((ind, info), k) in zip(sorted(sent_words.iteritems()), xrange(n)) if y[k] != x[k + 1]})
            
        return True
        
    def calc_bayes_sent(self, sentence):
        """
        Вычисление оценок байесовской вероятности со сглаживанием Лапласа
        с учетом только тех слов, которые встретились в контексте
        """
        sent = dict(sentence)
        indexed_contexts = dict([(num, smart_dict_slice(sentence, num, self.N)) for (num, info) in sentence if num in self.target_nums]) # Словарь типа {номер токена: контекстное множество слов}      
        # Перебираем контексты слов, которые нужно заменить синонимами
        for num, context in indexed_contexts.iteritems(): 
            lexeme = sent[num][1]["norm"]
            if not self.syns[lexeme]:
                continue
            probs = [(var, reduce(mul, [float(self.contexts["|".join((var, word))] + 1) / (get_DAWG(self.freqs, var) + 2) \
                                        for word in context \
                                        if "|".join((var, word)) in self.contexts], 0) * float(get_DAWG(self.freqs, var)) / self.f_sum) \
                      for var in self.syns[lexeme]]
            arg_max = argmax(probs)
            if arg_max and len(arg_max) <= self.vars_count:
                self.indexed_syns[num] = choice(arg_max)      
        return True

    def choose_syns(self, sentences):
        """
        Подбор наиболее вероятных синонимов выбранным методом
        """
        for sentence in sentences:
            self.actions[True](sentence)
        return True

    def insert_wsyns(self, sentences):
        """
        Запись выходного файла с замененными синонимами: упрощенная версия для метода коллокаций
        """
        output = []
        for sentence in sentences:
            for num, token in sentence: # Перебираем все токены тестового файла          
                wform = token[0]
                syn = self.indexed_wsyns[num] # Проверяем, найден ли для токена синоним (по номеру токена)

                if syn == "" or syn == wform:
                    output.append(wform)
                    continue

                synonim = "<font color='blue'>" + syn + "</font>"
                if not self.UseOnlySyn:
                    synonim = u"%s (%s)" % (wform, synonim)
                output.append(synonim)
        return output
    
    def insert_syns(self, sentences):
        """
        Запись выходного файла с замененными синонимами
        """
        output = []
        for sentence in sentences:
            for num, token in sentence: # Перебираем все токены тестового файла          
                wform = token[0]
                syn = self.indexed_syns[num] # Проверяем, найден ли для токена синоним (по номеру токена)
                    
                if syn == "":
                    output.append(wform)
                    continue
                
                gram_class = token[1]["class"]
                gram_info = ""
                
                if "info" in token[1].keys():
                    gram_info = token[1]["info"]

                inflected = inflect_comb(self.morph, syn, gram_info, gram_class) # Ставим синоним в нужную форму
                
                if not inflected:
                    output.append(wform)
                    continue

                synonim = "<font color='blue'>" + get_same_caps(wform, inflected) + "</font>"
                if not self.UseOnlySyn:
                    synonim = u"%s (%s)" % (wform, synonim)
                output.append(synonim)

        return output

    def synonimize(self, sentences):
        """
        Возвращает текст с синонимами
        """
        if self.UseCollocs:
            return self.insert_wsyns(sentences)
        else:
            return self.insert_syns(sentences)

if __name__ == "__main__":

    try:

        morphcorpus = "ruscorpora.txt.lemma" # Обучающий корпус для снятия морфологической омонимии
        traincorpus = "corpus_mix.txt.lemma"    # Обучающий корпус для синонимизации (для метода коллокаций)
        syncorpus = "corp.txt.lemma"  # Обучающий корпус для синонимизации (для всех остальных методов)

        #Synonimizer.train_unigram(syncorpus)
        #Synonimizer.train_ngram(syncorpus, 2)
        #Synonimizer.train_ngram(syncorpus, 3)
        #Synonimizer.train_bayes(syncorpus, 5)

        #Synonimizer.count_words(traincorpus)
        #Synonimizer.count_distant(traincorpus, 3, 4)
        #Synonimizer.xtrain_collocations(traincorpus, 3)
        
        #params = sys.argv
        filename = os.path.join(os.path.dirname(sys.argv[0]), "test/telenok.txt") # Файл, который будем синонимизировать
        
        params = ["synonimizer.py", filename, traincorpus,  "-i",  "-disamb", "-col", "-dict", "base.pkl", "-level", 3]

        UseDetails = True if "-d" in params else False

        print "STARTED:", str(datetime.now())
        start = time.time()

        morph = get_morph(os.path.join(os.path.dirname(sys.argv[0]),"pydicts").decode("UTF8")) # Подгружаем русский словарь
        morph_simple = get_morph(os.path.join(os.path.dirname(sys.argv[0]),"pydicts").decode("UTF8"), check_prefixes=False) # Подгружаем русский словарь
        tok = Tokenizer()   # Подгружаем токенизатор
        tagger = Tagger(morph, morph_simple)  # Подгружаем тэггер    
        syner = Synonimizer(morph_simple, params) # Подгружаем синонимизатор
        
        print "Synonimizer statistics loaded! It took", time.time() - start

        # Чтение файла (2 попытки: cp1251 и utf-8)
        try:
            text = read_file(filename)
        except Exception as e:
            error("Encoding detection failed! Windows-1251 or UTF-8 without BOM expected.", syner.UseDetails, str(e))
            sys.exit()

        tokens = tok.tokenize(text)
        if syner.UseDisambig:
            ttime = time.time()
            tagger.load_statistics(pl(morphcorpus))
            print "Tagger statistics loaded! It took", time.time() - ttime, "\nReading file..."
            sentences = tagger.get_parsed_sents(tokens)         # Снимаем морфологическую омонимию
        else:
            print "Reading file..."
            sentences = tagger.make_sents(tagger.lemmatize(tokens, make_all=False)) # Берем первый вариант леммы
        syner.extract_synable(sentences)    # Находим слова, которые можно синонимизировать
        syner.choose_syns(sentences)    # Подбираем синонимы для этих слов

        # Вставляем выбранные синонимы в текст и записываем все в файл
        with open(filename + ".html", "w") as fout:
            fout.write('<html><meta http-equiv="content-type" content="text/html; charset=utf-8" />')
            fout.write(html(u"".join(syner.synonimize(sentences)).encode("UTF8")))
            fout.write('</html>')

        print "FINISHED:", str(datetime.now())
        print "Time elapsed: ", time.time() - start

    except IOError as e:
        error("Error while opening/reading file!", UseDetails, str(e))
    except ValueError as e:
        error("Value error! Load UTF-8 (without BOM) text file, please.", UseDetails, str(e))
    except ImportError as e:
        error("Import error!", UseDetails, str(e))
    except KeyError as e:
        error("Key error!", UseDetails, str(e))
    except Exception as e:
        error("Unknown error!", UseDetails, str(e))
