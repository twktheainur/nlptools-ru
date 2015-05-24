# Tools for tokenization, POS-tagging and paraphrasing of a Russian text #

### Python 2.7 version: nlptools (uses pymorphy) ###
### Python 3 version: nlptools2 (uses pymorphy2, recommended) ###

## Dependencies ##
| **nlptools** | <a href='http://pythonhosted.org/pymorphy/'>pymorphy</a> , <a href='http://pypi.python.org/pypi/DAWG'>DAWG</a> |
|:-------------|:---------------------------------------------------------------------------------------------------------------|
| **nlptools2**| <a href='http://pymorphy2.readthedocs.org'>pymorphy2</a>, <a href='http://pypi.python.org/pypi/DAWG'>DAWG</a>  |

## Main components ##
### nlptools ###
  * <a href='https://code.google.com/p/nlptools-ru/source/browse/nlptools/tokenizer.py'>Tokenizer</a> (regex-based) for UTF-8
  * <a href='https://code.google.com/p/nlptools-ru/source/browse/nlptools/tagger.py'>Tagger-lemmatizer</a> (based on pymorphy, trained on the disambiguated part of <a href='http://ruscorpora.ru'>Russian National Corpus</a>)
  * <a href='https://code.google.com/p/nlptools-ru/source/browse/nlptools/dater.py'>Dater</a> (dates recognition and annotation tool)
  * <a href='https://code.google.com/p/nlptools-ru/source/browse/nlptools/synonimizer.py'>Synonymizer</a> (a word-to-word text paraphraser)

**Synonymizer on-line version**: http://pronoza.ru

### nlptools2 ###
  * <a href='https://code.google.com/p/nlptools-ru/source/browse/nlptools2/tokenizer.py'>Tokenizer</a> (regex-based) for UTF-8
  * <a href='https://code.google.com/p/nlptools-ru/source/browse/nlptools2/segmentizer.py'>Segmentizer</a> (splitting a list of tokens into separate sentences)
  * <a href='https://code.google.com/p/nlptools-ru/source/browse/nlptools2/tagger.py'>Tagger-lemmatizer</a> (based on pymorphy2, trained on the disambiguated part of <a href='http://ruscorpora.ru'>Russian National Corpus</a>)
  * <a href='https://code.google.com/p/nlptools-ru/source/browse/nlptools2/dater.py'>Dater</a> (dates recognition and annotation tool)
  * Synonymizer coming soon...

