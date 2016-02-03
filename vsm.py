from textpreprocessor import to_tfidf
import pandas as pd

class vsm(object):
	"""docstring for vsm"""
	def __init__(self, x, y, stop_words=None):
		super(vsm, self).__init__()
		'''
		:x : pd.Series; trainset, each element as a list of words after pre-processing
		:y : pd.Series; labels
		:stop_words : list; stop words
		'''
		self.x, self.cv, self.tfidf = to_tfidf(x, stop_words)
		self.y = y
			
	def predict(self, x):
		'''
		:x : pd.Series; testset, each element as a list of words after pre-processing
		
		:return : np.array; predicted labels 
		'''
		tfidf_matrix = to_tfidf(x, self.cv, self.tfidf)
		scores = pd.DataFrame(tfidf_matrix.dot(self.x.transpose()))
		max_scores_idx = scores.idmax(axis=0)
		label_idx = dict(zip(range(len(self.y)),list(self.y)))
		prediction = max_scores_idx.map(label_idx)
		return prediction


