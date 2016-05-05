
<<<<<<< HEAD
def cvScore(clf, X, Y, cv=5):
=======
import numpy as np
def sklearn_cross_validation(clf, X, Y, cv=5):
>>>>>>> preprocessor
	from sklearn import cross_validation
	return cross_validation.cross_val_score(clf, X, Y, cv=cv)

def param_selector(clf, params, X, Y):
	from sklearn.grid_search import GridSearchCV
	# params = {"n_estimators": [10, 50, 100], "min_samples_leaf": [5, 15, 30]}
	clf = GridSearchCV(clf, params)
	clf.fit(X, Y)
<<<<<<< HEAD
	return clf.best_estimator_, clf.grid_scores_

def cross_validation(clf, X, Y, cv=5):
	df['row'] = range(len(df))
	df.set_index(['row'],inplace=True)
	k = [(len(df)-1)/cv*j for j in range(cv+1)]
	score = [0.0]*cv
	for i in range(cv):		
		train = pd.concat([df.loc[:k[i],:],df.loc[k[i+1]:,:]])
		test = df.loc[k[i]:k[i+1],:]
		model = lm.fit(train,x_name,y_name,par)
		pred = lm.predict(test[x_name],model)
		score[i] = (pred == test[y_name]).sum()/float(len(test))
	return sum(score)/float(cv)
=======
	return clf.best_estimator_

# rewrite by using from sklearn.cross_validation import train_test_split
def cross_validation(clf, X, Y, cv=5, avg=False):
	'''
	:clf : classifier with fit() and predict() method
	:X : pd.DataFrame; features
	:Y : pd.DataFrame(1 column) or pd.Series; labels
	:cv : int; cross validation folders

	:return : list of float; cross validation scores
	'''
	import pandas as pd

	k = [(len(X))/cv*j for j in range(cv+1)]
	# print len(X)
	score = [0.0]*cv
	for i in range(cv):	
		train_x, train_y = pd.concat([X[:k[i]],X[k[i+1]:]]), pd.concat([Y[:k[i]],Y[k[i+1]:]])
		test_x, test_y = X[k[i]:k[i+1]], Y[k[i]:k[i+1]]

		# print train_y
		# print len(test_x)
		clf.fit(X,Y)
		pred = clf.predict(test_x)
		# pred = [0]*len(test_y)
		score[i] = (pred == test_y).sum()/float(len(test_y))
	if avg: return sum(score)/float(len(score))
	return score

def train_test_split(X, y, test_size):
	from sklearn.cross_validation import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	return X_train, X_test, y_train, y_test

# 	np.set_printoptions(suppress=True)
def confusion_matrix_from_cv(clf, X, Y, cv=5):
	from sklearn.metrics import confusion_matrix

	classes = len(Y.unique())
	cm = np.zeros([classes+1,classes])

	for i in range(cv):
		X_train, X_test, y_train, y_test = train_test_split(X, Y, 1/float(cv))

		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		cm[:classes,:classes] += confusion_matrix(y_test, y_pred)
	cm[classes] = np.array([cm[j,j]/float(cm.sum(1)[j]) for j in range(classes)])
	return cm
>>>>>>> preprocessor
