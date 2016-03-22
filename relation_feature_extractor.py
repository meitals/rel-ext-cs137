"""Adds methods for finding features for relation extraction
	Possible features:
	1. Get relation lists from training data, if entity pairs match, 
		add a relation.
	2. Get words and POS in between entities
	3. Get parse tree in between entities
	4. Get entity types



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
		self.featurize_get_tokens_v1()
		self.featurize_get_tokens_v2()
		self.featurize_get_entity_types()
		self.featurize_add_minimal_tree_nodes()
		self.featurize_get_bigrams()
		self.featurize_in_dependency_relation()
		self.featurize_target_pos()
		self.featurize_border_words()

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

	def featurize_in_dependency_relation(self):
		for doc_i, doc in enumerate(self.docs):
			for tt_i, tt in enumerate(doc.two_tokens):
				self.rel_inst_list[doc_i][tt_i].features.append("d_relation__"+tt.in_dependency_relation)
				if tt.in_dependency_relation == "":
					self.rel_inst_list[doc_i][tt_i].features.append("in_d_relation__False")
				else:
					self.rel_inst_list[doc_i][tt_i].features.append("in_d_relation__True")


	def featurize_get_entity_types(self):
		for doc_i, doc in enumerate(self.docs):
			for tt_i, tt in enumerate(doc.two_tokens):
				types = tt.entity_type1+'_'+tt.entity_type2
				self.rel_inst_list[doc_i][tt_i].features.append(types)
	
	def featurize_get_in_between_words(self):
		"""Gets words in between related words"""
		for doc_i, doc in enumerate(self.docs):
			pos_tagged_sents = doc.pos_tagged_sents
			for tt_i, tt in enumerate(doc.two_tokens):
				words, pos = self.get_in_between_words_and_pos(doc, tt)
				words = ["inbetweenwords__"+word for word in words]
				pos = ["inbetweenpos__"+p for p in pos]
				self.rel_inst_list[doc_i][tt_i].features.extend(pos)
				self.rel_inst_list[doc_i][tt_i].features.extend(words)
		return

	def featurize_get_in_between_words(self):
		"""Gets words in between related words"""
		for doc_i, doc in enumerate(self.docs):
			pos_tagged_sents = doc.pos_tagged_sents
			for tt_i, tt in enumerate(doc.two_tokens):
				words, pos = self.get_in_between_words_and_pos(doc, tt)
				words = ["inbetweenwords__"+word for word in words]
				pos = ["inbetweenpos__"+p for p in pos]
				self.rel_inst_list[doc_i][tt_i].features.extend(pos)
				self.rel_inst_list[doc_i][tt_i].features.extend(words)
		return


	def featurize_get_tokens_v1(self):
		for doc_i, doc in enumerate(self.docs):
			for tt_i, tt in enumerate(doc.two_tokens):
				# split clears up instances like Arizona_Rattlers, which 
				# are tow words in the parsed sentences
				self.rel_inst_list[doc_i][tt_i].features.append('token__'+tt.token1)
				self.rel_inst_list[doc_i][tt_i].features.append('token__'+tt.token2)


	def featurize_get_tokens_v2(self):
		for doc_i, doc in enumerate(self.docs):
			for tt_i, tt in enumerate(doc.two_tokens):
				# split clears up instances like Arizona_Rattlers, which 
				# are tow words in the parsed sentences
				self.rel_inst_list[doc_i][tt_i].features.append('both_token__'+tt.token1+'_'+tt.token2)


	def featurize_get_bigrams(self):
		for doc_i, doc in enumerate(self.docs):
			for tt_i, tt in enumerate(doc.two_tokens):
				# split clears up instances like Arizona_Rattlers, which 
				# are tow words in the parsed sentences
				bigrams = self.get_bigrams(doc, tt)
				for bg in bigrams:
					self.rel_inst_list[doc_i][tt_i].features.append('bigram__'+bg)


	def get_in_between_words_and_pos(self, document, two_tokens):
		sent = document.pos_tagged_sents[two_tokens.sent_offset1]
		# We assume entity1 comes before entity2
		start = two_tokens.end_token1
		end = two_tokens.begin_token2
		in_between_words = []
		in_between_pos = []
		for k in range(start,end):
			in_between_words.append(sent[k][0])
			in_between_pos.append(sent[k][1])
		return in_between_words, in_between_pos


	def get_bigrams(self, document, two_tokens):
		bigrams = []
		in_between_words = self.get_in_between_words_and_pos(document, two_tokens)[0]
		all_words = [two_tokens.token1] + in_between_words + [two_tokens.token2]
		for index, word in enumerate(all_words):
			if index != len(all_words)-1:
				bigrams.append(word+'_'+all_words[index+1])
		return bigrams

	def featurize_get_nearest_common_ancestor(self):
		for doc_i, doc in enumerate(self.docs):
			for tt_i, tt in enumerate(doc.two_tokens):
				# split clears up instances like Arizona_Rattlers, which 
				# are tow words in the parsed sentences
				token1 = tt.token1.split("_")
				token2 = tt.token2.split("_")
				in_between_words = self.get_in_between_words_and_pos(doc, tt)[0]
				subtree_string = token1 + in_between_words + token2
				tt_sent_tree = doc.parses[tt.sent_offset1]
				tt_subtree = self.get_subtree_between_words(tt_sent_tree, subtree_string)
				if isinstance(tt_subtree, nltk.tree.Tree):
					label = tt_subtree.label()
				else:
					label = 'no_comm_subtree'
				self.rel_inst_list[doc_i][tt_i].features.append('comm._ancestor__'+label)


	def featurize_add_minimal_tree_nodes(self):
		""" adds nodes (not leaves) that are between the target words"""
		for doc_i, doc in enumerate(self.docs):
			for tt_i, tt in enumerate(doc.two_tokens):
				# split clears up instances like Arizona_Rattlers, which 
				# are tow words in the parsed sentences
				token1 = tt.token1.split("_")
				token2 = tt.token2.split("_")
				in_between_words = self.get_in_between_words_and_pos(doc, tt)[0]
				subtree_string = token1 + in_between_words + token2
				tt_sent_tree = doc.parses[tt.sent_offset1]
				tt_subtree = self.get_subtree_between_words(tt_sent_tree, subtree_string)

				if isinstance(tt_subtree, nltk.tree.Tree):
					tt_subtree_labels = self.get_tree_labels(tt_subtree)
					tt_subtree_labels = '_'.join(tt_subtree_labels)
				else:
					tt_subtree_labels = 'no_comm_subtree'
				self.rel_inst_list[doc_i][tt_i].features.append('subtree_node_labels__'+tt_subtree_labels)


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


	def get_tree_labels(self, tree):
		labels = [tree.label()]
		for child in tree:
			if isinstance(child, nltk.tree.Tree):
				labels.extend(self.get_tree_labels(child))
		return labels

	def featurize_target_pos(self):
		"""add POS tag for second token 

		note: for some reason this works better than including both pos tags"""
		for doc_i, doc in enumerate(self.docs):
			pos_tagged_sents = doc.pos_tagged_sents
			for tt_i, tt in enumerate(doc.two_tokens):
				pos1, pos2 = self.get_target_pos(doc, tt)
				self.rel_inst_list[doc_i][tt_i].features.append('targetpos_{}'.format(pos1))
				self.rel_inst_list[doc_i][tt_i].features.append('targetpos_{}'.format(pos2))

	def get_target_pos(self, document, two_tokens):
		sent = document.pos_tagged_sents[two_tokens.sent_offset1]
		token1_pos = sent[two_tokens.begin_token1][1] 
		token2_pos = sent[two_tokens.begin_token2][1]
		return token1_pos, token2_pos

	def featurize_border_words(self):
		"""word before token1, word after token2

		using bigrams hurt performance"""
		for doc_i, doc in enumerate(self.docs):
			for tt_i, tt in enumerate(doc.two_tokens):
				before_word, after_word, before_bigram, after_bigram = self.get_border_words(doc, tt)
				self.rel_inst_list[doc_i][tt_i].features.append('BEFOREWORD__{}'.format(before_word))
				self.rel_inst_list[doc_i][tt_i].features.append('AFTERWORD__{}'.format(after_word))
				#self.rel_inst_list[doc_i][tt_i].features.append('BEFOREBIGRAM__{}'.format(before_bigram))
				#self.rel_inst_list[doc_i][tt_i].features.append('AFTERBIGRAM__{}'.format(after_bigram))


	def get_border_words(self, document, two_tokens):
		sent = document.pos_tagged_sents[two_tokens.sent_offset1]
		before_ind = two_tokens.begin_token1-1
		after_ind = two_tokens.end_token2
		if before_ind == -1:
			before_word = '<start>'
		else:
			before_word = sent[before_ind][0]
		if after_ind == len(sent):
			after_word = '<end>'
		else:
			after_word = sent[after_ind][0]
		
		word1 = sent[two_tokens.begin_token1][0]
		word2 = sent[two_tokens.begin_token2][0]
		before_bigram = '_'.join([before_word, word1])
		after_bigram = '_'.join([word2, after_word])

		return before_word, after_word, before_bigram, after_bigram

if __name__ == "__main__":
	fe = FeatureExtractor('rel-trainset.gold',True)
	rl = fe.get_relations_list_from_gold_files()
	print len(rl)
	print rl.keys()[0],rl.get(rl.keys()[0])
	fe.featurize()
	for x in fe.rel_inst_list[1]:
		print(x.features)
		print('\n\n')

