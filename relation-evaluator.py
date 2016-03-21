#!/usr/bin/python
#compute the accuracy of an NE tagger

#usage: evaluate-head.py [gold_file][output_file]

import sys, re

if len(sys.argv) != 3:
    sys.exit("usage: evaluate-head.py [gold_file][output_file]")

#gold standard file
goldfh = open(sys.argv[1], 'r')
#system output
testfh = open(sys.argv[2], 'r')

gold_tag_list = []
#gold_word_list = []
test_tag_list = []

emptyline_pattern = re.compile(r'^\s*$')

for gline in goldfh.readlines():
    if not emptyline_pattern.match(gline):
        parts = gline.split()
        #print parts
        gold_tag_list.append(parts[0])


for tline in testfh.readlines():
    if not emptyline_pattern.match(tline):
        parts = tline.split()
        #print parts
        test_tag_list.append(parts[0])

test_total = 0
gold_total = 0
correct = 0

test_totals_type = {}
gold_totals_type = {}
correct_type = {}

#print gold_tag_list
#print test_tag_list

for i in range(len(gold_tag_list)):
    if gold_tag_list[i] != 'no_rel':
        gold_total += 1
        gold_totals_type[gold_tag_list[i]] = gold_totals_type.get(gold_tag_list[i], 0) + 1
    if test_tag_list[i] != 'no_rel':
        test_total += 1
        test_totals_type[test_tag_list[i]] = test_totals_type.get(test_tag_list[i], 0) + 1
    if gold_tag_list[i] != 'no_rel' and gold_tag_list[i] == test_tag_list[i]:
        correct_type[gold_tag_list[i]] = correct_type.get(gold_tag_list[i], 0) + 1
        correct += 1
    # if test_tag_list[i] != 'no_rel' and gold_tag_list[i] == 'no_rel':
    #     print("fp",i,test_tag_list[i])
    # if gold_tag_list[i] != 'no_rel' and test_tag_list[i] == 'no_rel':
    #     print('fn',i,gold_tag_list[i])


print correct, gold_total, test_total

precision = 0
if test_total > 0:
    precision = float(correct) / test_total
recall = 0
if gold_total > 0:
    recall = float(correct) / gold_total
f = 0
if precision+recall > 0:
    f = precision * recall * 2 / (precision + recall)

print correct, gold_total, test_total
print 'precision =', precision, 'recall =', recall, 'f1 =', f

error_dict = {} 
for ind, gold_type in enumerate(gold_tag_list):
    test_type = test_tag_list[ind]
    if gold_type != test_type:
        error_name = 'GOLD_{}\tTEST_{}'.format(gold_type, test_type)
        error_dict[error_name] = error_dict.get(error_name, 0) + 1

with open('confusion_matrix.txt', 'w') as confusion_file:
    for error in error_dict.keys():
        confusion_file.write('{}:\t{}\n'.format(error, error_dict[error]))

with open('prf_by_reltype.txt', 'w') as prf_file:
    for reltype in sorted(correct_type.keys()):
        num_correct = float(correct_type[reltype])
        prec_denom = test_totals_type[reltype]
        rec_denom = gold_totals_type[reltype]
        try:
            type_prec = num_correct/prec_denom
            type_rec = num_correct/rec_denom
            type_f1 = type_prec * type_rec * 2 / (type_prec + type_rec)
        except ZeroDivisionError:
            type_prec = 0.0
            type_rec = 0.0
            type_f1 = 0.0
        prf_file.write('Reltype: {} P: {} R: {} F1: {}\n'.format(
            reltype, type_prec, type_rec, type_f1))
    prf_file.write('Overall P: {} R: {} F1: {}'.format(precision, recall, f))


    
