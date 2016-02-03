import imp

def load(name):
	path = '../Module/' + name +'.py'
	return imp.load_source(name, path)
