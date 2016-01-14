class preprocessor(object):
	"""docstring for preprocessor"""
	
	def __init__(self, df, test=False, yName = None):
		super(preprocessor, self).__init__()
		varCols = [e for e in df.columns if e != yName]
		self.X = df[varCols]
		self.ref = {}
		self.test = test
		if not test:
			self.Y = df[yName]	


	def shuffle(self, seed=1337):
		np.random.seed(seed)
		shuffle = np.arange(len(self.Y))
		np.random.shuffle(shuffle)
		# print shuffle
		# print len(shuffle)
		self.X = self.X.reindex(shuffle)
		self.Y = self.Y[shuffle]

	def binarize(self):
		pass

	def _binarize(self, colNames):
		for colName in colNames:
			ref = self.ref[colName]
			self.ref[colName] = []
			for k,v in ref.iteritems():
				l = self.X[colName].map(lambda x: int(x == v))
				if sum(l.unique()) == 0: continue
				self.X[k] = l
				self.ref[colName].extend([k])
			dropKey = self.ref[colName][0]
			# print dropKey
			self.ref[colName] = []
			self.drop([colName,dropKey], axis=1, inplace=True) #not needed!

	# def delClass(self, classCol, threshold = 5):
	# 	import everything
	# 	counts = self.Y.value_counts(sort=False)
	# 	lessCounts = counts[counts < threshold]
	# 	exclude = everything.Contain(lessCounts.index)
	# 	self.Y = self.Y[self.Y != exclude]
	# 	self.X = self.X[self.Y[classCol] != exclude]

		# self.df = self.df.drop()
	def cleanOutliers(self):
		if self.test: self.outliers_to_avg()
		else: self.delOutliers()

	def delOutliers(self):
		pass


	def outliers_to_avg(self):
		pass


	def numerizeDf(self, colNames=None):
		if colNames == None: 
			if not self.test: colNames = list(self.X.columns)+[self.Y.name]
			else: colNames = self.X.columns
		for col in colNames:
			try:
				if self.X[col].dtype.name == 'object': self.X[col], self.ref[col] = self.__numerizeSeries(self.X[col])
			except KeyError:
				try:
					self.Y, self.ref[col] = self.__numerizeSeries(self.Y)
				except KeyError: print "no keys exist!"

	def _numerizeSeries(self,l):
		# print l		
		dist = l.unique()
		ref = dict(zip(dist,range(len(dist))))
		# print ref
		return l.map(ref), ref

	def scaleCols(self, colNames=None, scale=1.0):
		if colNames == None: 
			if not self.test: colNames = list(self.X.columns)+[self.Y.name]
			else: colNames = self.X.columns			
		for col in colNames:
			if self.X[col].dtype.name != 'object': 
				self.X[col] = (self.X[col]-self.X[col].min())/(self.X[col].max()-self.X[col].min())*scale
				self.X[col] = self.X[col].round(decimals=3)

	def drop(self,colnames,inplace=False, axis=0):
		self.X.drop(colnames,inplace=inplace, axis=axis)

	def getProcessedData(self):
		if not self.test: return self.X, self.Y
		return self.X

	def getReference(self):
		return self.ref