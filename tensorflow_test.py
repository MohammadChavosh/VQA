from tensorflow.contrib import learn

question_texts = ['salam azizam', 'salam salam azizam khobi', 'salam golabi azizam pesaram havij bastani boz khar']
max_question_length = max([len(question.split(" ")) for question in question_texts])
questions_vocab_processor = learn.preprocessing.VocabularyProcessor(max_question_length, min_frequency=1)
questions_vocab_processor.fit(question_texts)
print questions_vocab_processor.vocabulary_, len(questions_vocab_processor.vocabulary_)
print 'salam' in questions_vocab_processor.vocabulary_._mapping
print(list(questions_vocab_processor.transform(question_texts)))
print(list(questions_vocab_processor.transform(['salam baghali', 'khodafez aziz'])))
# questions = np.array(list(questions_vocab_proc