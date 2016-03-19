"""
Workflow:
1. Initiate RelExtractor object
2. featurize training corpus
3. train model 
4. featurize test corpus
5. label test using model
6. evaluate accuracy

"""


import corpus_reader
import relation_feature_extractor
import itertools

class RelInstance(object):
	def __init__(self, entity1, entity2, relType):
		self.tokens = '_'.join([entity1, entity2])
		self.relType = relType
		self.features = [] #list of feature names to be written
	def __str__(self):
		return (entity1, entity2)

class RelExtractor(object):

	def __init__(self):
		self.train_instances = [] #list of RelInstance objects
		self.test_instances = [] #list of RelInstance objects

	def featurize(self, corpus_file, rel_inst_list, reading_gold_file):
		"""creates RelInstance objects and add to their features
		reads from corpus and adds to either train or test instance list"""
		instances = list(itertools.chain.from_iterable(Feature_Extractor(corpus_file, reading_gold_file))
		return instances

	def train(self):
		"""writes training file and runs Mallet"""
		with open('featurized_training', 'w') as training_file:
			for instance in self.train_instances:
				feature_str = ' '.join(instance.features)
				training_file.write('{} {} {}\n'.format(instance.tokens, instance.relType, instance.feature_str))

		os.system('bin/mallet train-classifier --input featurized_training --output-classifier relext_model \
			--trainer MaxEnt')

	def test(self):
		"""writes test file and runs Mallet

		infile: featurized_test, outfile: labeled_test"""

		with open('featurized_test', 'w') as test_file:
			for instance in self.test_instances:
				feature_str = ' '.join(instance.features)
				test_file.write('{} {}\n'.format(instance.tokens, instance.feature_str))
		
		os.system('bin/mallet classify-file --input featurized_test --output labeled_test --classifier relext_model')

	def evaluate(self):
		"""creates gold file and compares to labled test"""

		with open('gold_test', 'w') as gold_file:
			for instance in self.test_instances:
				feature_str = ' '.join(instance.features)
				gold_file.write('{} {} {}\n'.format(instance.tokens, instance.relType, instance.feature_str))
		
		os.system('python relation-evaluator.py gold_test labeled_test'.format(gold_fpath, output_fpath))