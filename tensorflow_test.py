from train import get_batch_for_test, load_data, load_related_train_data

questions_vocab_processor, answers_vocab_processor, max_question_length = load_related_train_data()
questions, answers, images_paths = load_data(questions_vocab_processor, answers_vocab_processor, True)
output_len = len(answers_vocab_processor.vocabulary_) - 1
for step in range(10665, 10680):
	batch_in_questions, batch_in_images, batch_out, size = get_batch_for_test(step, questions, answers, images_paths, output_len)
