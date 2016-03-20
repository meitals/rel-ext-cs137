"""Adds methods for finding features for relation extraction
	Possible features:
	1. Get relation lists from training data, if entity pairs match, 
		add a relation.



"""

from corpus_reader import *
from relation_extractor import *
import re

class FeatureExtractor:

	def __init__(self, corpus_file, reading_gold_file):
		c = CorpusReader(corpus_file, reading_gold_file)
		self.reading_gold_file = reading_gold_file
		self.docs = c.corpus.values()
		self.rel_inst_list = self.create_rel_inst_list()

	def featurize(self):
		"""Call all featurizing functions here"""
		self.featurize_get_in_between_words()
		self.featurize_get_nearest_common_ancestor()


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


	def featurize_get_in_between_words(self):
		"""Gets words in between related words"""
		for doc_i, doc in enumerate(self.docs):
			pos_tagged_sents = doc.pos_tagged_sents
			for tt_i, tt in enumerate(doc.two_tokens):
				words, pos = self.get_in_between_words_and_pos(doc, tt)
				self.rel_inst_list[doc_i][tt_i].features.append('_'.join(pos))
				self.rel_inst_list[doc_i][tt_i].features.append('_'.join(words))
		return


	def get_in_between_words_and_pos(self, document, two_tokens):
		sent = document.pos_tagged_sents[int(two_tokens.sent_offset1)]
		# We assume entity1 comes before entity2
		start = int(two_tokens.end_token1)
		end = int(two_tokens.begin_token2)
		in_between_words = []
		in_between_pos = []
		for k in range(start,end):
			in_between_words.append(sent[k][0])
			in_between_pos.append(sent[k][1])
		return in_between_words, in_between_pos


	def featurize_get_nearest_common_ancestor(self):
		for doc_i, doc in enumerate(self.docs):
			for tt_i, tt in enumerate(doc.two_tokens):
				# split clears up instances like Arizona_Rattlers, which 
				# are tow words in the parsed sentences
				token1 = tt.token1.split("_")
				token2 = tt.token2.split("_")
				in_between_words = self.get_in_between_words_and_pos(doc, tt)[0]
				subtree_string = token1 + in_between_words + token2
				tt_sent_tree = doc.parses[int(tt.sent_offset1)]
				tt_subtree = self.get_subtree_between_words(tt_sent_tree, subtree_string)
				if isinstance(tt_subtree, nltk.tree.Tree):
					label = tt_subtree.label()
				else:
					label = 'no_comm_subtree'
				self.rel_inst_list[doc_i][tt_i].features.append('comm._ancestor__'+label)


	def get_subtree_between_words(self, tree, token_sequence, smallest=[]):
		"""
            Smallest subtree from tree that contains token_sequence
            Used to find the parse of a particular entity
        """
		if self.words_in_tree(tree, token_sequence):
			smallest = tree
			for child in tree:
				if isinstance(child, nltk.tree.Tree):
					smallest = self.get_subtree_between_words(child, token_sequence, smallest)
		else: 
			return smallest
		return smallest


	def words_in_tree(self, tree, token_sequence):
		"""
			checks to see if the token sequence appears in the tree
		"""
		tree_words = " ".join(tree.leaves())
		target = " ".join([token for token in token_sequence])
		return target in tree_words


if __name__ == "__main__":
	fe = FeatureExtractor('rel-trainset.gold',True)
	rl = fe.get_relations_list_from_gold_files()
	print len(rl)
	print rl.keys()[0],rl.get(rl.keys()[0])
	fe.featurize()
	for x in fe.rel_inst_list[1]:
		print(x.features)
		print('\n\n')

