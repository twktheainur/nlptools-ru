#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import time
from datetime import datetime
from pymorphy import get_morph
from dater import Dater
from tagger import Tagger


filename = os.path.join(os.path.dirname(sys.argv[0]), "corpora/somecorpus.txt")
traincorpus = os.path.join(os.path.dirname(sys.argv[0]),"dicts/ruscorpora.txt.lemma")
trainsuffcorpus = traincorpus + ".suffs"

print "STARTED:", str(datetime.now())
start = time.time()

morph = get_morph(os.path.join(os.path.dirname(sys.argv[0]),"pydicts").decode("UTF8")) # Подгружаем русский словарь
morph_simple = get_morph(os.path.join(os.path.dirname(sys.argv[0]),"pydicts").decode("UTF8"), check_prefixes=False) # Подгружаем русский словарь
# Подгружаем обработчик дат
dater = Dater()
# Подгружаем тэггер
tagger = Tagger(morph, morph_simple, dater)
# Подгружаем суффиксную статистику для тэггера
tagger.load_statistics(traincorpus, trainsuffcorpus)
# Лемматизируем частями
tagger.parse_chunks(filename, sent_marks=True)

print "FINISHED:", str(datetime.now())
print "Time elapsed: ", time.time() - start

