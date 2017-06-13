__author__ = 'Mohammad'

import json
import operator

train_questions = json.load(open('data/OpenEnded_mscoco_train2014_questions.json'))['questions']
train_annotations = json.load(open('data/mscoco_train2014_annotations.json'))['annotations']
related_answers = dict()
for question, annotation in zip(train_questions, train_annotations):
	if question['question_id'] != annotation['question_id']:
		raise AssertionError("question id's are not equal")
	q = question['question'].lower()
	if q not in related_answers:
		related_answers[q] = list()
	for ans in annotation['answers']:
		related_answers[q].append(ans['answer'])
	related_answers[q].append(annotation['multiple_choice_answer'])

lens = dict()
for q in related_answers:
	tmp = len(related_answers[q])
	if tmp not in lens:
		lens[tmp] = 0
	lens[tmp] += 1

sorted_lens = sorted(lens.items(), key=operator.itemgetter(1))
print(sorted_lens)