import nltk.tree

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
                                            reading_gold_file=reading_gold_file
                                        )
            # now we are sure document is in corpus. Add the two_tokens line
            corpus[doc_filename].two_tokens.append(TwoTokens(
                                                    split_line=line_split,
                                                    reading_gold_file=reading_gold_file
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
        for line in no_empty_lines:
            pos_tagged_sents = [tok.split("_") for tok in line.split()]
        return pos_tagged_sents


    def get_file_lines(self, filepath):
        #filepath is relative to being in the project directory
        f = open(filepath, 'r')
        lines = f.readlines()
        f.close
        return lines

class Document:
    def __init__(self, title, parses, pos_tagged_sents, reading_gold_file=False):
        self.title = title
        self.reading_gold_file = reading_gold_file
        self.parses = parses
        self.pos_tagged_sents = pos_tagged_sents
        self.two_tokens = []


class TwoTokens:
    def __init__(self, split_line, reading_gold_file=False):
        self.split_line = split_line
        #gold files have the tag at the beginning--grab and remove it
        if reading_gold_file:
            self.tag = split_line[0]
            split_line = self.split_line[1:]
        self.doc_num = split_line[0]
        self.column3 = split_line[1]
        self.column4 = split_line[2]
        self.column5 = split_line[3]
        self.entity_type1 = split_line[4]
        self.column7 = split_line[5]
        self.token1 = split_line[6]
        self.column9 = split_line[7]
        self.column10 = split_line[8]
        self.column11 = split_line[9]
        self.entity_type2 = split_line[10]
        self.column13 = split_line[11]
        self.token2 = split_line[12]

if __name__ == '__main__':
    # Example Usage
    # Corpus is a dict from {doc_name : document_object}
    c = CorpusReader('rel-trainset.gold', reading_gold_file=True)
    corpus = c.corpus
    print (corpus[c.corpus.keys()[0]].pos_tagged_sents)
