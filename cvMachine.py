
def cvScore(clf, X, Y, cv=5):
	from sklearn import cross_validation
	return cross_validation.cross_val_score(clf, X, Y, cv=cv)

def paramSelector(clf, params, X, Y):
	from sklearn.grid_search import GridSearchCV
	# params = {"n_estimators": [10, 50, 100], "min_samples_leaf": [5, 15, 30]}
	clf = GridSearchCV(clf, params)
	clf.fit(X, Y)
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