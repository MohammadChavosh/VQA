__author__ = 'Mohammad'

import json

train_questions = json.load(open('data/OpenEnded_mscoco_train2014_questions.json'))['questions']
train_questions_file = open('data/train_questions.txt', 'w')
for question in train_questions:
	train_questions_file.write(question['question'])
	train_questions_file.write('\n')
train_questions_file.close()