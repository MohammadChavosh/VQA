__author__ = 'Mohammad'

import random
import json
import skimage.io
import skimage.transform
import skimage.color


def load_image(path, size=224):
	img = skimage.io.imread(path)
	if len(img.shape) == 2:
		img = skimage.color.gray2rgb(img)
	short_edge = min(img.shape[:2])
	yy = int((img.shape[0] - short_edge) / 2)
	xx = int((img.shape[1] - short_edge) / 2)
	crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
	resized_img = skimage.transform.resize(crop_img, (size, size))
	return resized_img


def get_related_answers(is_train):
	if is_train:
		annotations = json.load(open('data/mscoco_train2014_annotations.json'))['annotations']
		questions = json.load(open('data/OpenEnded_mscoco_train2014_questions.json'))['questions']
	else:
		annotations = json.load(open('data/mscoco_val2014_annotations.json'))['annotations']
		questions = json.load(open('data/OpenEnded_mscoco_val2014_questions.json'))['questions']
	related_answers = dict()
	for question, annotation in zip(questions, annotations):
		if question['question_id'] != annotation['question_id']:
			raise AssertionError("question id's are not equal")
		q = question['question']
		if q not in related_answers:
			related_answers[q] = set()
		for ans in annotation['answers']:
			related_answers[q].add(ans['answer'])
		related_answers[q].add(annotation['multiple_choice_answer'])
	return related_answers


def get_vqa_data(is_train, sampling_ratio):
	if is_train:
		annotations = json.load(open('data/mscoco_train2014_annotations.json'))['annotations']
		questions = json.load(open('data/OpenEnded_mscoco_train2014_questions.json'))['questions']
		images_path = 'data/train2014/COCO_train2014_'
	else:
		annotations = json.load(open('data/mscoco_val2014_annotations.json'))['annotations']
		questions = json.load(open('data/OpenEnded_mscoco_val2014_questions.json'))['questions']
		images_path = 'data/val2014/COCO_val2014_'
	vqa_triplets = list()
	for question, annotation in zip(questions, annotations):
		if question['question_id'] != annotation['question_id']:
			raise AssertionError("question id's are not equal")
		q = question['question']
		img_num = str(question['image_id'])
		img_path = images_path
		for i in range(12 - len(img_num)):
			img_path += '0'
		img_path += img_num + '.jpg'
		vqa_triplets.append((q, annotation['multiple_choice_answer'], img_path))
	if sampling_ratio < 1:
		vqa_triplets = random.sample(vqa_triplets, int(round(len(vqa_triplets) * sampling_ratio)))
	return vqa_triplets
