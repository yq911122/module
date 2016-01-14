class Contain:
	def __init__(self, items):
		self.items = set(items)
	def __ne__(self, other):
		return other not in self.items
	def __eq__(self, other):
		return other in self.items

def merge_dicts(*dict_args):
	'''
	Given any number of dicts, shallow copy and merge into a new dict,
	precedence goes to key value pairs in latter dicts.
	'''
	result = {}
	for dictionary in dict_args:
		result.update(dictionary)
	return result

def dictFromList(keys, values=None):
	if values == None: values = range(len(keys))
	return dict(zip(keys,values))

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate