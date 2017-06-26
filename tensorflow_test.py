from train import get_batch, load_related_train_data, load_train_data

step = 520
questions_vocab_processor, answers_vocab_processor, max_question_length = load_related_train_data()
questions, answers, images_paths = load_train_data(questions_vocab_processor, answers_vocab_processor)
batch_in_questions, batch_in_images, batch_out = get_batch(step, questions, answers, images_paths, len(answers_vocab_processor.vocabulary_))

