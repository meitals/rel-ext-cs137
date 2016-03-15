class CorpusReader:
    def __init__(self, filename, reading_gold_file=False):
        # filename is the name of the rel-[whatever]set.[postfix] file
        self.filename = filename
        self.reading_gold_file = reading_gold_file
        self.read_corpus
        self.corpus = self.read_corpus()

    def read_corpus(self):
        # hashes {document_id : document object}
        # documents contain lists of parses and 'twotoken's
        corpus = {}
        relation_lines = self.get_file_lines(self.filename)
        for line in relation_lines:
            line_split = line.split()
            # Keep working here

    def get_file_lines(self, filepath):
        #filepath is relative to being in the project directory
        f = open(filepath, 'r')
        lines = f.readlines()
        f.close
        return lines

class Document:
    def __init__(self, title, reading_gold_file=False):
        self.title = title
        self.reading_gold_file = reading_gold_file
        # create two_tokens 


class TwoTokens:
    def __init__(self, split_line, reading_gold_file=False):
        self.split_line = split_line
        #gold files have the tag at the beginning--grab and remove it
        if reading_gold_file:
            self.tag = split_line[0]
            self.split_line = self.split_line[1:]
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
