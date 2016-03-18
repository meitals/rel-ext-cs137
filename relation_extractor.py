import corpus_reader

class RelInstance(object):
	def __init__(self, entity1, entity2, relType):
		self.tokens = '_'.join([entity1, entity2])
		self.relType = relType
		self.features = [] #list of feature names to be written

class RelExtractor(object):

	def __init__(self):
		self.train_instances = [] #list of RelInstance objects
		self.test_instances = [] #list of RelInstance objects

	def featurize(self, corpus, rel_inst_list):
		"""creates RelInstance objects and add to their features

		reads from corpus and adds to either train or test instance list"""
		pass

	def train(self):
		"""writes training file and runs Mallet"""
		with open('featurized_training', 'w') as training_file:
			for instance in self.train_instances:
				pass

		os.system()

	def test(self):
		"""writes test file and runs Mallet

		infile: featurized_test, outfile: tagged_test"""
		with open('featurized_test', 'w') as test_file:
			for instance in self.test_instances:
				pass

		os.system()

	def write_gold(self):
		"""creates a gold_file called gold_test"""
		with open('gold_test', 'w') as gold_file:
			pass

	def evaluate(self):
		os.system('python relation-evaluator.py gold_test tagged_test'.format(gold_fpath, output_fpath))