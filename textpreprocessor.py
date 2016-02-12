import nltk
import re

def tokenize(s, lower=True):
	'''
	:s : pd.Series; each element as a document
	:lower : boolean; transform words in lower case if True

	:return : pd.Series; each element as a list of words after tokenization, every word in lower case
	'''
	from nltk.tokenize import word_tokenize
	if lower: return s.map(str.lower).map(word_tokenize)
	return s.map(word_tokenize)

def get_corpus(s):
	'''
	:s : pd.Series; each element as a list of words from tokenization

	:return : list; corpus from s
	'''
	l = []
	s.map(lambda x: l.extend(x))
	return l

def get_stop_words_and_freq(s, n):
	from collections import Counter
	l = get_corpus(s)
	l = [x for x in Counter(l).most_common(n)]
	return l

def get_stop_words(s, n):
	'''
	:s : pd.Series; each element as a list of words from tokenization
	:n : int; n most frequent words are judged as stop words 

	:return : list; a list of stop words
	'''
	from collections import Counter
	l = get_corpus(s)
	l = [x[0] for x in Counter(l).most_common(n)]
	return l

def remove_stop_words(s, n, stop_words = None, nltk = False):
	'''
	:s : pd.Series; each element as a list of words from tokenization
	:n : int; n most frequent words are judged as stop words 

	:return : pd.Series; stop words removed
	'''
	if nltk: 
		from nltk.corpus import stopwords
		stop_words = stopwords.words('english')
	elif not stop_words: stop_words = get_stop_words(s, n)
	return s.map(lambda x: [e for e in x if e not in stop_words])

def to_tfidf(s, cv=None, tfidf=None, stop_words=None):
	'''
	:s : pd.Series; each element as a list of words after pre-processing
	:cv : CountVectorizer; if None, s will be used to create cv
	:tfidf : TfidfTransformer; if None, s will be used to create tfidf

	:return : if cv is not None: sparse matrix; Tf-idf-weighted document-term matrix
			  if cv is None: CountVectorizer; cv created by s. TfidfTransformer; tfidf created by s. sparse matrix; Tf-idf-weighted document-term matrix
	'''
	l = []
	s.map(lambda x: l.extend([x]))
	if not cv:
		from sklearn.feature_extraction.text import CountVectorizer
		from sklearn.feature_extraction.text import TfidfTransformer
		cv = CountVectorizer(stop_words=stop_words)
		tfidf = TfidfTransformer()
		return cv, tfidf, tfidf.fit_transform(cv.fit_transform(l))
	return tfidf.fit_transform(cv.fit_transform(l))



def extract_nltk_entities(sentences):
	'''
	:sentences : list; each element a sentence in the document.

	:return : dict; named entities as keys and type of the entities as values. e.g., {'Tom':__PERSON__}. The types of the entities are the same to NE Type of NLTK (http://www.nltk.org/book/ch07.html)
	'''

	ne = {}
	for sent in sentences:
		words, subs = [], []
		for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)), binary=False):
			if hasattr(chunk, 'label'):
				val = '__{0}__'.format(chunk.label())
				key = ' '.join(c[0] for c in chunk.leaves())
				ne[key] = val
	return ne

def extra_entity_patterns(names):
	patterns = {
	'__EMAIL__' : re.compile(r'[\w\.-]+@[\w\.-]+'),
	'__DATE__' : re.compile(r'\b([0-9]{1,2})[-/:.]([0-9]{1,2})[-/:.]([0-9]{4})\b|\b([0-9]{4})[-/:.]([0-9]{1,2})[-/:.]([0-9]{1,2})\b'),
	'__TIME__' : re.compile(r'\b[0-1][0-9][-:][0-5][0-9][-:][0-5][0-9]\b|\b[2][0-3][-:][0-5][0-9][-:][0-5][0-9]\b'),
	'__URL__' : re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
	'__AT__': re.compile(r'@\w+')
	}

	keys = ["__{0}__".format(n.upper()) for n in names]
	return dict((k,patterns[k]) for k in keys if k in patterns)


def replace_entities(sentences, extra_patterns=['email','date','time','url']):
	'''
	:sentences : list; each element a sentence in the document.

	:return : list; the document whose entities are replaced.
			  list; names of entities' types.

	Besides NE Type from NLTK, '__EMAIL__', '__TIME__' and '__URL__' are implemented. '__DATE__' can handle more situations.
	'''
	ne = extract_nltk_entities(sentences)
	if ne:
		pattern = re.compile(r'\b(' + '|'.join(ne.keys()) + r')\b')
		sentences = [pattern.sub(lambda x: ne[x.group()], s) for s in sentences]
	# print ne
	patterns = extra_entity_patterns(extra_patterns)

	for k, v in patterns.iteritems():
		sentences = [v.sub(k, s) for s in sentences]
	entities = list(set(ne.values()))
	entities.extend(patterns.keys())
	return sentences, entities

def remove_entities(sentences, extra_patterns=['email','date','time','url']):
	'''
	:sentences : list; each element a sentence in the document.

	:return : list; the document whose entities are removed.
	'''
	ne = extract_nltk_entities(sentences)
	if ne:
		ne = {k:'' for k, _ in ne.iteritems()}

	patterns = extra_entity_patterns(extra_patterns)

	for k, v in patterns.iteritems():
		sentences = [v.sub('', s) for s in sentences]
	entities = list(set(ne.values()))
	entities.extend(patterns.keys())
	return sentences, entities

def remove_gibberish(words, exceptions):
	'''
	:words : list of lists; each outer list represents a sentence, each inner list a word in the document.
	:exceptions : list; words that are not regarded as gibberish besides words in nltk.corpus.words.words().

	:return : list of lists; gibberish is removed.
	'''
	english_vocab = set(w.lower() for w in nltk.corpus.words.words())
	vocabs = english_vocab.union(set(exceptions))
	return [[e for e in w if e in vocabs] for w in words]

