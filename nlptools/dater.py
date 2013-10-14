#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
Обработка дат для лемматизации (Python 2.7)
"""

import re

class Dater(object):
    """
    Обработчик дат (для русского языка).
    Если слово является датой или ее частью, в качестве части речи указывается "DD" "MM", "YY" или "YYYY",
    а в качестве граммем - формат (YYYY-MM-DD, YY-MM-DD, YYYY-MM или MM-DD) с суффиксом -B, -L, -I или -U.
    
    Даты в тексте размечаются в формате BILOU (begin-in-last-out-unit).
    Например:
    04    04    DD    YYYY-MM-DD-B
    июня    06    MM    YYYY-MM-DD-I
    2013    2013    YYYY    YYYY-MM-DD-L
    """

    def __init__(self):
        """
        Инициализация.
        """

        # Рег. выражения для лемматизации
        self.day = re.compile(u"^\d{1,2}(?:-?[оегмуы]{2,3})?$")
        self.dig = re.compile("^\d+")
        self.date = re.compile("^\d+(?:[.:/-]\d+)+$")
        self.datesplitter = re.compile("[.:/-]")
        self.year = re.compile(u"^\d{4}Г?$")
        # Длина месяца (в днях)
        self.months = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        # Названия месяцев (с учетом сокращений и возможных ошибок лемматизации)
        self.monthnames = {u"ЯНВАРЬ": 1, u"ЯНВ": 1, u"ФЕВРАЛЬ": 2, u"ФЕВ": 2,
                           u"МАРТ": 3, u"МАРТА": 3, u"АПРЕЛЬ": 4, u"АПР": 4,
                           u"МАЙ": 5, u"МАЯ": 5, u"ИЮНЬ": 6, u"ИЮЛЬ": 7,
                           u"АВГУСТ": 8, u"АВГ": 8, u"СЕНТЯБРЬ": 9, u"СЕН": 9,
                           u"СЕНТ": 9, u"ОКТЯБРЬ": 10, u"ОКТ": 10, u"НОЯБРЬ": 11,
                           u"НОЯБРЯ": 11, u"ДЕКАБРЬ": 12, u"ДЕК": 12}
        # Максимально допустимый год
        self.maxyear = 2050

    def is_date(self, word):
        """
        Проверка слова на соответствие рег. выражению даты.
        """
        return re.match(self.date, word)

    def correct_date(self, month, day, year=None):
        """
        Проверка того, что данная тройка (месяц, день, год) может являться датой (год может не указываться).
        Возвращает дату в формате ISO или None.
        """
        if year:
            if year >= self.maxyear or year <= 0:
                return None
        try:
            if day > self.months[month] or day <= 0:
                return None
            date = "{0:0>2d}-{1:0>2d}".format(month, day)
            if year:
                date = str(year) + "-" + date
            return date
        except Exception:
            return None

    def check_double_date(self, parts):
        """
        Приведение даты к формату MM-DD или YYYY-MM.
        Возвращает дату в формате ISO или None.
        """
        (p1, p2) = parts       
        if len(p1) <= 2 and len(p2) <= 2:  
            date1 = self.correct_date(int(p2), int(p1)) 
            if date1:
                return (date1, "MM-DD")          
            date2 = self.correct_date(int(p1), int(p2))
            if date2:
                return (date2, "MM-DD")
            return None
        if len(p1) == 4 and int(p1) < self.maxyear and len(p2) <= 2 and int(p2) <= 12:
            return (".".join((p2.zfill(2), p1)), "YYYY-MM")
        if len(p2) == 4 and int(p2) < self.maxyear and len(p1) <= 2 and int(p1) <= 12:
            return (".".join((p1.zfill(2), p2)), "YYYY-MM")
        return None

    def check_triple_date(self, parts):
        """
        Приведение даты к формату YY_MM-DD или YYYY-MM-DD.
        Возвращает дату в формате ISO или None.
        """
        (p1, p2, p3) = parts
        if len(p1) <= 2 and len(p2) <= 2 and (len(p3) == 2 or len(p3) == 4) and 0 < int(p3) < self.maxyear:           
            date1 = self.correct_date(int(p2), int(p1), int(p3))
            dtype = "YYYY-MM-DD" if len(p3) == 4 else "YY-MM-DD"
            if date1:      
                return (date1, dtype)
            date2 = self.correct_date(int(p1), int(p2), int(p3))    
            if date2:
                return (date2, dtype)
            return None
        if (len(p1) == 2 or len(p1) == 4) and len(p2) <= 2 and len(p3) <= 2 and 0 < int(p1) < self.maxyear:            
            date1 = self.correct_date(int(p2), int(p3), int(p1))
            dtype = "YYYY-MM-DD" if len(p1) == 4 else "YY-MM-DD"
            if date1:    
                return (date1, dtype) 
            date2 = self.correct_date(int(p3), int(p2), int(p1))
            if date2:
                return (date2, dtype)
        return None

    def check_date(self, word):
        """
        Проверка даты и приведение к формату ISO (YYYY-MM-DD).      
        word - предполагаемая дата в произвольном формате (числа, разделенные точкой/двоеточием/слешем).
        Возвращает дату в формате ISO или None.
        """
        parts = re.split(self.datesplitter, word)
        if len(parts) < 2 or len(parts) > 3:
            return None
        if any(not len(x) in {1, 2, 4} for x in parts):
            return None      
        if len(parts) == 2: # Количество частей = 2
            return self.check_double_date(parts)
        # Количество частей = 3
        return self.check_triple_date(self, parts)

    def parse_days(self, inds, sent_words, sent_tokens):
        """
        Обработка дней.
        """
        digits = [ind for ind, info in sent_tokens.iteritems() if re.match(self.day, info[0])]
        for ind in digits:
            word = sent_words[ind][0]
            try:
                day = int(word)
            except Exception:
                try:
                    day = int(re.findall(self.dig, word)[0])
                except Exception:
                    print word
                    sys.exit()
            try:
                next_ind = inds[inds.index(ind) + 1]
                next_lexeme = sent_words[next_ind][1]["norm"]
                month_id = self.monthnames[next_lexeme]
                
                if day <= self.months[month_id]:
                    sent_words[ind] = (sent_words[ind][0], {"norm": sent_words[ind][0].zfill(2), "class": u"DD", "info": "MM-DD-B"}) 
                    sent_words[next_ind] = (sent_words[next_ind][0], {"norm": "{0:0>2d}".format(month_id), "class": u"MM", "info": "MM-DD-L"})
                    next_ind2 = inds[inds.index(ind) + 2]
                if not re.match(self.year, sent_words[next_ind2][1]["norm"]):
                    continue             
                if sent_words[next_ind2][1].has_key("info"):
                    if sent_words[next_ind2][1]["info"] != "YYYY-U":
                        continue
                elif int(sent_words[next_ind2][0]) >= self.maxyear:
                    continue
                sent_words[ind] = (sent_words[ind][0], {"norm": sent_words[ind][0].zfill(2), "class": u"DD", "info": "YYYY-MM-DD-B"})
                sent_words[next_ind] = (sent_words[next_ind][0], {"norm": "{0:0>2d}".format(month_id), "class": u"MM", "info": "YYYY-MM-DD-I"})
                sent_words[next_ind2] = (sent_words[next_ind2][0], {"norm": sent_words[next_ind2][1]["norm"], "class": u"YYYY", "info": "YYYY-MM-DD-L"})
            except Exception:
                continue
        return True

    def parse_months(self, inds, sent_words, sent_tokens):
        """
        Обработка месяцев.       
        """
        months = [ind for ind, info in sent_words.iteritems() if info[1]["norm"] in set(self.monthnames.keys())]
        for ind in months:
            try:
                next_ind = inds[inds.index(ind) + 1]
                if not re.match(self.year, sent_words[next_ind][1]["norm"]):
                    continue
                if sent_words[next_ind][1].has_key("info"):
                    if sent_words[next_ind][1]["info"] != "YYYY-U":
                        continue
                elif int(sent_words[next_ind][0]) >= self.maxyear:
                    continue
                month_id = self.monthnames[sent_words[ind][1]["norm"]]
                sent_words[ind] = (sent_words[ind][0], {"norm": "{0:0>2d}".format(month_id), "class": u"MM", "info": "YYYY-MM-B"})
                sent_words[next_ind] = (sent_words[next_ind][0], {"norm": sent_words[next_ind][1]["norm"], "class": u"YYYY", "info": "YYYY-MM-L"})
            except Exception:
                continue
        return True

    def parse_years(self, inds, sent_words, sent_tokens):
        """
        Обработка лет.
        """
        years = [ind for ind, info in sent_tokens.iteritems() if re.match(self.year, info[1]["norm"])]  
        for ind in years:
            year_str = sent_words[ind][0]
            try:
                year = int(year_str)
            except Exception:
                year = int(year_str[:4])
                
            if year >= self.maxyear:
                continue
            
            if len(year_str) == 5:  # Запись вида 2011г
                sent_words[ind] = (year_str, {"norm": year_str[:-1], "class": u"YYYY", "info": "YYYY-U"})
                continue         
            try:    # Запись вида 2011 - проверяем, что следующя лемма - ГОД или Г 
                next_ind = inds[inds.index(ind) + 1]
                if sent_words[next_ind][1]["norm"] in {u"Г", u"ГОД"}:      
                    sent_words[ind] = (year_str, {"norm": year_str, "class": u"YYYY", "info": "YYYY-U"})
                    if sent_words[next_ind][1]["class"] != u"С":
                        sent_words[next_ind] = (sent_words[next_ind][0], {"norm": u"ГОД", "class": u"С", "info": u"мр"})
            except Exception:
                continue
        return True
    
    def parse_dates(self, sent_words, sent_tokens):
        """
        Обработка дат в предложении.
        
        sent_words - словарь (нумерованный список) лемматизированных слов предложения, sent_tokens - токенов
        Каждое слово имеет формат [словоформа, {парадигма_1}, {парадигма_2}, ...]
        Парадигмы имеют вид словаря {"norm": norm, "class": class, "info": info}
        """
        inds = sorted(sent_words.keys())
        self.parse_years(inds, sent_words, sent_tokens) # Обработка лет
        self.parse_days(inds, sent_words, sent_tokens)  # Обработка дней
        self.parse_months(inds, sent_words, sent_tokens)  # Обработка месяцев
        return True

