"""Adds methods for finding features for relation extraction
	Possible features:
	1. Get relation lists from training data, if entity pairs match, 
		add a relation.



"""

from corpus_reader import *
from relation_extractor import *

class FeatureExtractor:

	def __init__(self, corpus_file, reading_gold_file):
		c = CorpusReader(corpus_file, reading_gold_file)
		self.reading_gold_file = reading_gold_file
		self.docs = c.corpus.values()
		self.rel_inst_list = self.create_rel_inst_list()

	def featurize(self):
		"""Call all featurizing functions here"""
		self.get_in_between_words()


	def get_relations_list_from_gold_files(self):
		# Create pairs where the key is the word pair and the value is the relation
		relation_pairs = {}
		for doc in self.docs:
			ttl = doc.two_tokens #List of related entities
			for tt in ttl:
				if tt.tag != "no_rel":
					relation_pairs[tt.token1+"_"+tt.token2] = tt.tag
		return relation_pairs


	def create_rel_inst_list(self):
		rel_inst_list = []
		for doc in self.docs:
			doc_list = []
			for tt in doc.two_tokens:
				if self.reading_gold_file:
					doc_list.append(RelInstance(tt.token1,tt.token2,tt.tag))
				else:
					doc_list.append(RelInstance(tt.token1,tt.token2,""))
			rel_inst_list.append(doc_list)
		return rel_inst_list


	def get_in_between_words(self):
		"""Gets words in between related words"""
		for i in range(len(self.docs)):
			doc = self.docs[i]
			pos_tagged_sents = doc.pos_tagged_sents
			for j in range(len(doc.two_tokens)):
				tt = doc.two_tokens[j]
				sent = pos_tagged_sents[int(tt.sent_offset1)]
				# We assume entity1 comes before entity2
				start = int(tt.end_token1)
				end = int(tt.begin_token2)
				in_between_words = []
				in_between_pos = []
				for k in range(start,end-1):
					in_between_words.append(sent[k][0])
					in_between_pos.append(sent[k][1])
				self.rel_inst_list[i][j].features.append('_'.join(in_between_pos))
				self.rel_inst_list[i][j].features.append('_'.join(in_between_words))
		return


if __name__ == "__main__":
	fe = FeatureExtractor('rel-trainset.gold',True)
	rl = fe.get_relations_list_from_gold_files()
	print len(rl)
	print rl.keys()[0],rl.get(rl.keys()[0])