#untest

import importHelper
import pandas as pd
import random

everything = importHelper.load('everything')
static_vars = everything.static_vars

class lm(object):
	"""
	statistical language model based on MLE method. Both jelinek-mercer and dirichlet smoothing methods are implemented
	"""
	def __init__(self, X, Y, a=0.1, smooth_method=jelinek_mercer):
		super(lm, self).__init__()
		'''
		:a : float; discount parameter; should be tuned via cross validation
		:smooth_method: function; method selected to discount the probabilities	
		'''
		self.a = a
		self.smooth_method = smooth_method

	def fit(self, X, Y):
		'''
		:X : pd.DataFrame; features; features are actually a list of words, standing for the document.
		:Y : pd.Series or pd.DataFrame; labels

		:return : pd.DataFrame; language model
		'''
		cats = Y.unique()	
		p_ref = df_to_prob(X)
		model = pd.DataFrame()
		model['unseen'] = p_ref*self.a
		for c in cats:
			idx = Y[Y == c].index
			ct = df_to_ct(X.loc[idx,:])
			p_ml = ct_to_prob(ct)
			model[c] = self.smooth_method(ct, p_ml,p_ref)
			model[c].fillna(model['unseen'],inplace=True)
		model.drop(['unseen'],axis=1,inplace=True)
		self.model = model

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

	def df_to_prob(self, df):
		'''
		df: list of lists; each containing a document of words, like [[a],[b,c],...]
		out: pd.Series; the probabilities of each word, like ({a:0.3,b:0.3,...})
		'''
		return self.ct_to_prob(self.df_to_ct(df))	

	def df_to_ct(self, df):
		from collections import Counter	
		l = []
		df.map(lambda x: l.extend(x))
		return pd.Series(dict(Counter(l)))

	def ct_to_prob(self, d):
		total_occur = d.sum()
		return d/float(total_occur)


	def predict(self, df):
		return df.map(lambda x: predict_item(x,len(df)))

	@static_vars(counter=0)
	def predict_item(l, N):
		model = self.model
		predict_item.counter += 1
		if predict_item.counter % 20 == 0: print predict_item.counter/float(N)
		in_list = [e for e in l if e in model.index]
		if not in_list: 
			return model.columns[random.randint(0,len(model.columns)-1)]
		s = model.loc[in_list,:].sum(axis=0)
		return s.loc[s==s.max()].index[0]

def main():
	pass

if __name__ == '__main__':
	main()