__author__ = 'Mohammad'

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn, rnn
from data_loader import get_related_answers

# Parameters
embedding_dim = 300
word2vec_file = 'GoogleNews-vectors-negative300.bin'
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 100
n_hidden = 512
n_classes = 2

# Load data
related_answers = get_related_answers()
question_texts = related_answers.keys()
answers_vocab = list()
for q in related_answers:
    for ans in related_answers[q]:
        answers_vocab.append(ans)

# Build vocabulary
max_question_length = max([len(questions.split(" ")) for questions in question_texts])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_question_length)
vocab_processor.fit(question_texts + answers_vocab)
questions = np.array(list(vocab_processor.transform(question_texts)))


def load_word2vec():
    initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), embedding_dim))
    with open(word2vec_file, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        counter = 0
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab_processor.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                print(word)
            else:
                f.read(binary_len)
            counter += 1
            if counter % 100000 == 0:
                print(counter)
    return initW


def train():
    with tf.Graph().as_default():
        embedding_W = tf.Variable(tf.random_uniform([len(vocab_processor.vocabulary_), embedding_dim], -1.0, 1.0), name="embedding_W")
        input_questions = tf.placeholder(tf.int32, [None, questions.shape[1]], name="input_questions")
        embedded_chars = tf.nn.embedding_lookup(embedding_W, input_questions)
        unstacked_embedded_chars = tf.unstack(embedded_chars, max_question_length, 1)

        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, unstacked_embedded_chars, dtype=tf.float32)

        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.initialize_all_variables())
            initW = load_word2vec()
            sess.run(embedding_W.assign(initW))

