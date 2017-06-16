__author__ = 'Mohammad'

import json


def get_related_answers():
	train_annotations = json.load(open('data/mscoco_train2014_annotations.json'))['annotations']
	train_questions = json.load(open('data/OpenEnded_mscoco_train2014_questions.json'))['questions']
	related_answers = dict()
	for question, annotation in zip(train_questions, train_annotations):
		if question['question_id'] != annotation['question_id']:
			raise AssertionError("question id's are not equal")
		q = question['question']
		if q not in related_answers:
			related_answers[q] = set()
		for ans in annotation['answers']:
			related_answers[q].add(ans['answer'])
		related_answers[q].add(annotation['multiple_choice_answer'])
	return related_answers
