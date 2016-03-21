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
from relation_feature_extractor import *
import itertools, os

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
		"""Creates RelInstance objects and add to their features
		reads from corpus and adds to either train or test instance list"""
		fe = FeatureExtractor(corpus_file, reading_gold_file)
		fe.featurize()
		rel_inst_list = list(itertools.chain.from_iterable(fe.rel_inst_list))
		return rel_inst_list

	def train(self,training_file):
		"""writes training file and runs Mallet"""
		self.train_instances = self.featurize(training_file,self.train_instances,True)
		with open('featurized_training', 'w') as training_file:
			for instance in self.train_instances:
				feature_str = ' '.join(instance.features)
				rel_type = instance.relType.split('.')[0]
				#print 'rel_type=', rel_type
				training_file.write('{} {} {}\n'.format(instance.tokens, rel_type, feature_str))

		os.system('Mallet1/bin/mallet import-file --input featurized_training --line-regex ^(\S*)[\s]*(\S*)[\s]*(.*)$ --output featurized_training.mallet')
		os.system('Mallet1/bin/mallet train-classifier --input featurized_training.mallet --output-classifier relext_model \
			--trainer MaxEnt')

	def test(self,test_file):
		"""writes test file and runs Mallet
		infile: featurized_test, outfile: labeled_test"""
		self.test_instances = self.featurize(test_file,self.test_instances,True)
		with open('featurized_test', 'w') as test_file:
			for instance in self.test_instances:
				feature_str = ' '.join(instance.features)
				test_file.write('{} {}\n'.format(instance.tokens, feature_str))
		
		os.system('Mallet1/bin/mallet classify-file --input featurized_test --output labeled_test --classifier relext_model')

	def evaluate(self):
		"""creates gold file and compares to labled test"""

		with open('gold_test', 'w') as gold_file:
			for instance in self.test_instances:
				rel_type = instance.relType.split('.')[0]
				gold_file.write('{}\n'.format(rel_type))

		with open('labeled_test') as labeled_file:
			with open('output_test', 'w') as output_file:
				for line in labeled_file.readlines():
					label = self.get_highest_probability_label(line)
					output_file.write('{}\n'.format(label))
		
		os.system('python relation-evaluator.py gold_test output_test')

	def get_highest_probability_label(self,line):
		line = line.split()
		name = line[0]
		max_prob = 0
		max_label = ""
		for i in range(2,len(line),2):
			p = float(line[i])
			if p > max_prob:
				max_prob = p
				max_label = line[i-1]
		#print name,max_label,max_prob
		return max_label

if __name__ == "__main__":
	rel_ext = RelExtractor()
	rel_ext.train('rel-trainset.gold')
	#print len(rel_ext.train_instances)
	rel_ext.test('rel-devset.gold')
	#print len(rel_ext.test_instances)
	rel_ext.evaluate()


