import nltk.tree
import os,re

class CorpusReader:
    def __init__(self, filename, reading_gold_file=False):
        # filename is the name of the rel-[whatever]set.[postfix] file
        self.filename = 'data/'+filename
        self.reading_gold_file = reading_gold_file
        self.corpus = self.read_corpus(reading_gold_file)

    def read_corpus(self, reading_gold_file=False):
        print("Creating corpus from " + self.filename + "...")
        if reading_gold_file:
            doc_filename_index = 1
        else:
            doc_filename_index = 0

        # hashes {document_id : document object}
        # documents contain lists of parses and 'twotoken's
        corpus = {}
        relation_lines = self.get_file_lines(self.filename)
        for line in relation_lines:
            line_split = line.split()
            doc_filename = line_split[doc_filename_index]
            # Create document as necessary
            if doc_filename not in corpus:
                corpus[doc_filename] = Document(
                                            title=doc_filename,
                                            parses=self.get_document_parses(doc_filename),
                                            pos_tagged_sents=self.get_pos_tagged_sents(doc_filename),
                                            dparses=self.get_dependency_relations(doc_filename),
                                            reading_gold_file=reading_gold_file
                                        )
            # now we are sure document is in corpus. Add the two_tokens line
            corpus[doc_filename].two_tokens.append(TwoTokens(
                                                    split_line=line_split,
                                                    reading_gold_file=reading_gold_file,
                                                    dp=corpus[doc_filename].dparses
                                                    ))
        return corpus


    def get_document_parses(self, doc_filename):
        lines = self.get_file_lines('data/parsed-files/'+doc_filename+'.head.rel.tokenized.raw.parse')
        no_empty_lines = [line for line in lines if line not in [' ', '', '\n']]
        tree_lines = []
        for line in no_empty_lines:
            tree_lines.append(nltk.tree.Tree.fromstring(line))
        return tree_lines

    def get_pos_tagged_sents(self, doc_filename):
        lines = self.get_file_lines('data/postagged-files/'+doc_filename+'.head.rel.tokenized.raw.tag')
        no_empty_lines = [line for line in lines if line not in [' ', '', '\n']]
        pos_tagged_sents = []
        for line in no_empty_lines:
            sent = [tok.split("_") for tok in line.split()]
            pos_tagged_sents.append(sent)
        return pos_tagged_sents

    def get_dependency_relations(self, doc_filename):
        lines = self.get_file_lines('data/dependency-parsed-files/'+doc_filename+'.head.rel.tokenized.raw.dparse')
        ind = 0
        dependency_relation_sents = []
        pattern = re.compile(r"\((.+)\)")
        sent = {}
        for line in lines:
            if len(line.strip()) == 0:
                dependency_relation_sents.append(sent)
                ind += 1
                sent = {}
                continue
            groups = re.split(pattern,line) #In format relation(word1, word2)
            sent[groups[1]] = groups[0]
        return dependency_relation_sents

    def get_plaintext(self, doc_filename):
        lines = self.get_file_lines('data/text-files/'+doc_filename+'.head.rel.tokenized.raw')
        no_empty_lines = [line for line in lines if line not in [' ', '', '\n']]
        sents = []
        for line in no_empty_lines:
            sent = line.split()
            sents.append(sent)
        return sents

    def get_file_lines(self, filepath):
        #filepath is relative to being in the project directory
        f = open(filepath, 'r')
        lines = f.readlines()
        f.close
        return lines

def check_dependency_relation(dependency_relation_sents,sent,token1,token2,offset1,offset2):
    if int(sent) < len(dependency_relation_sents):
        d_r_sent = dependency_relation_sents[int(sent)]
        key = "{}-{}, {}-{}".format(token1,offset1,token2,offset2)
        if key in d_r_sent.keys():
            return True
    return False

class Document:
    def __init__(self, title, parses, pos_tagged_sents, dparses, reading_gold_file=False):
        self.title = title
        self.reading_gold_file = reading_gold_file
        self.parses = parses
        self.pos_tagged_sents = pos_tagged_sents
        self.dparses = dparses
        self.two_tokens = []
        if len(dparses) != len(pos_tagged_sents):
            # print title,len(dparses),len(pos_tagged_sents)
            dparses = []


class TwoTokens:
    def __init__(self, split_line, reading_gold_file=False, dp=[]):
        self.split_line = split_line
        #gold files have the tag at the beginning--grab and remove it
        if reading_gold_file:
            self.tag = split_line[0].split('.')[0]
            split_line = self.split_line[1:]
        self.doc_num = split_line[0]
        self.sent_offset1 = int(split_line[1])
        self.begin_token1 = int(split_line[2])
        self.end_token1 = int(split_line[3])
        self.entity_type1 = split_line[4]
        self.entity_id1 = split_line[5]
        self.token1 = split_line[6]
        self.sent_offset2 = int(split_line[7])
        self.begin_token2 = int(split_line[8])
        self.end_token2 = int(split_line[9])
        self.entity_type2 = split_line[10]
        self.entity_id2 = split_line[11]
        self.token2 = split_line[12]
        self.in_dependency_relation = check_dependency_relation(dp,self.sent_offset1,
            self.token1,self.token2,self.begin_token1,self.begin_token2)

if __name__ == '__main__':
    # Example Usage
    # Corpus is a dict from {doc_name : document_object}
    # c = CorpusReader('rel-trainset.gold', reading_gold_file=True)
    # c = CorpusReader('rel-devset.gold', reading_gold_file=True)
    c = CorpusReader('rel-testset.gold', reading_gold_file=True)
    # corpus = c.corpus

    # print (corpus[c.corpus.keys()[0]].title)
    # for i in corpus[c.corpus.keys()[0]].pos_tagged_sents:
    #     print(i)
    #print (corpus[c.corpus.keys()[0]].pos_tagged_sents)
