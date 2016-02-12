#untest

import importHelper
import pandas as pd
import random

class lm(object):
	"""
	statistical language model based on MLE method. Both jelinek-mercer and dirichlet smoothing methods are implemented
	"""
	def __init__(self, a=0.1, smooth_method='jelinek_mercer'):
		super(lm, self).__init__()
		'''
		:a : float; discount parameter; should be tuned via cross validation
		:smooth_method: function; method selected to discount the probabilities	
		'''
		self.a = a
		smooth_method = getattr(self, smooth_method)
		self.smooth_method = smooth_method

		self.counter = 0

	def df_to_ct(self, df):
		from collections import Counter	
		l = []
		df.map(lambda x: l.extend(x))
		return pd.Series(dict(Counter(l)))

	def ct_to_prob(self, d):
		total_occur = d.sum()
		return d/float(total_occur)

	def df_to_prob(self, df):
		'''
		df: list of lists; each containing a document of words, like [[a],[b,c],...]
		out: pd.Series; the probabilities of each word, like ({a:0.3,b:0.3,...})
		'''
		return self.ct_to_prob(self.df_to_ct(df))	


	def fit(self, X, Y):
		'''
		:X : pd.Series; features; features are actually a list of words, standing for the document.
		:Y : pd.Series; labels

		:return : pd.DataFrame; language model
		'''
		if len(Y) != 0  and len(X) != 0:
			from math import log
			cats = Y.unique()	
			p_ref = self.df_to_prob(X)
			model = pd.DataFrame()
			model['unseen'] = (p_ref*self.a).map(log)
			for c in cats:
				idx = Y[Y == c].index
				ct = self.df_to_ct(X.loc[idx])
				p_ml = self.ct_to_prob(ct)
				model[c] = self.smooth_method(ct, p_ml,p_ref)
				model[c].fillna(model['unseen'],inplace=True)
			model.drop(['unseen'],axis=1,inplace=True)
			self.model = model
		else: print 'input is empty'

	def jelinek_mercer(self, ct, p_ml,p_ref,a=0.1):
		from math import log
		log_p_s = (p_ml*(1-a)+p_ref.loc[p_ml.index]*a).map(log)
		return log_p_s

	def dirichlet(self, ct, p_ml,p_ref,a=0.1):
		from math import log
		d = len(p_ml)
		u = a / (1+a)*d
		log_p_s = ((ct+u*p_ref.loc[ct.index])/(d+u)).map(log)
		return log_p_s

	
	def predict_item(self, l, N):
		model = self.model
		# self.counter += 1
		# if self.counter % 200 == 0: print self.counter/float(N)
		in_list = [e for e in l if e in model.index]
		if not in_list: 
			return model.columns[random.randint(0,len(model.columns)-1)]
		selected_model =  model.loc[in_list,:]
		s = selected_model.sum(axis=0)

		label = s.loc[s==s.max()].index[0]
		# print label
		word = selected_model.loc[selected_model[label] == selected_model[label].max(),:].index[0]
		# print word
		self.predwords[label].append(word)
		# print self.predwords
		return label


	def predict(self, df):
		self.predwords = dict(zip(self.model.columns,[[] for _ in xrange(len(self.model.columns))])) #tricky
		return df.map(lambda x: self.predict_item(x,len(df)))

	def get_params(self):
		# print self.a
		# print self.smooth_method.__name__
		return (self.a, self.smooth_method.__name__)

	def get_predictive_words(self, n=3):
		from collections import Counter
		total_words = {k:len(v) for k,v in self.predwords.iteritems()}
		most_predictive_words = {k:Counter(v).most_common(n) for k, v in self.predwords.iteritems()} 
		most_predictive_words = {label:{w:v/float(length) for w, v in words} for label, length, words in zip(total_words.keys(), total_words.values(), most_predictive_words.values())}
		return most_predictive_words

def main():
	pass

if __name__ == '__main__':
	main()