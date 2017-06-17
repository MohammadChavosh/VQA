__author__ = 'Mohammad'

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn, rnn
from data_loader import get_related_answers

# Parameters
embedding_dim = 300
word2vec_file = 'data/GoogleNews-vectors-negative300.bin'
learning_rate = 0.001
training_iters = 5000
batch_size = 128
display_step = 100
n_hidden = 512
n_classes = 2


def load_train_data():
    related_answers = get_related_answers(True)
    question_texts = related_answers.keys()
    answers_vocab = list()
    ans_question_num = list()
    counter = 0
    for q in question_texts:
        for ans in related_answers[q]:
            answers_vocab.append(ans)
            ans_question_num.append(counter)
        counter += 1

    max_question_length = max([len(question.split(" ")) for question in question_texts])
    questions_vocab_processor = learn.preprocessing.VocabularyProcessor(max_question_length)
    questions = np.array(list(questions_vocab_processor.fit_transform(question_texts)))

    answers_vocab_processor = learn.preprocessing.VocabularyProcessor(1)
    answers_list = np.array(list(answers_vocab_processor.fit_transform(answers_vocab)))
    answers = dict()
    for i in range(len(ans_question_num)):
        if ans_question_num[i] not in answers:
            answers[ans_question_num[i]] = list()
        answers[ans_question_num[i]].append(answers_list[i][0])
    return questions, answers, questions_vocab_processor, answers_vocab_processor, max_question_length


def load_validation_data(questions_vocab_processor, answers_vocab_processor):
    related_answers = get_related_answers(False)
    question_texts = related_answers.keys()
    questions = np.array(list(questions_vocab_processor.transform(question_texts)))

    answers_vocab = list()
    ans_question_num = list()
    counter = 0
    for q in question_texts:
        for ans in related_answers[q]:
            answers_vocab.append(ans)
            ans_question_num.append(counter)
        counter += 1
    answers_list = np.array(list(answers_vocab_processor.transform(answers_vocab)))
    answers = dict()
    for i in range(len(ans_question_num)):
        if ans_question_num[i] not in answers:
            answers[ans_question_num[i]] = list()
        answers[ans_question_num[i]].append(answers_list[i][0])
    return questions, answers


def load_word2vec(questions_vocab_processor):
    init_embedding_w = np.random.uniform(-0.25, 0.25, (len(questions_vocab_processor.vocabulary_), embedding_dim))
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
            idx = questions_vocab_processor.vocabulary_.get(word)
            if idx != 0:
                init_embedding_w[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
            counter += 1
            if counter % 100000 == 0:
                print counter
    print 'loading word2vec file is complete'
    return init_embedding_w


def get_batch(step, questions, answers, answers_vocab_len):
    batch_start = (step * batch_size) % len(questions)
    batch_in = questions[batch_start:batch_start + batch_size]
    batch_out = np.zeros((batch_size, answers_vocab_len))
    for i in range(batch_start, batch_start + len(batch_in)):
        for ans in answers[i]:
            batch_out[i - batch_start, ans - 1] = 1

    tmp = batch_size - len(batch_in)
    if tmp > 0:
        for i in range(0, tmp):
            for ans in answers[i]:
                batch_out[i + len(batch_in), ans - 1] = 1
        batch_in = np.concatenate((batch_in, questions[0:tmp]), axis=0)
    return batch_in, batch_out


def get_batch_for_test(step, questions, answers, answers_vocab_len):
    batch_start = (step * batch_size) % len(questions)
    batch_in = questions[batch_start:batch_start + batch_size]
    batch_out = np.zeros((len(batch_in), answers_vocab_len))
    for i in range(batch_start, batch_start + len(batch_in)):
        for ans in answers[i]:
            batch_out[i - batch_start, ans - 1] = 1
    return batch_in, batch_out, len(batch_in)


def run():
    questions, answers, questions_vocab_processor, answers_vocab_processor, max_question_length = load_train_data()
    with tf.Graph().as_default():
        embedding_w = tf.Variable(tf.random_uniform([len(questions_vocab_processor.vocabulary_), embedding_dim], -1.0, 1.0), name="embedding_w")
        input_questions = tf.placeholder(tf.int32, [None, questions.shape[1]], name="input_questions")
        output_answers = tf.placeholder(tf.float32, [None, len(answers_vocab_processor.vocabulary_)], name="output_answers")
        embedded_chars = tf.nn.embedding_lookup(embedding_w, input_questions)
        unstacked_embedded_chars = tf.unstack(embedded_chars, max_question_length, 1)

        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, _ = rnn.static_rnn(lstm_cell, unstacked_embedded_chars, dtype=tf.float32)
        out_w = tf.Variable(tf.random_normal([n_hidden, len(answers_vocab_processor.vocabulary_)]), name="out_w")
        out_bias = tf.Variable(tf.random_normal([len(answers_vocab_processor.vocabulary_)]), name="out_bias")
        pred = tf.matmul(outputs[-1], out_w) + out_bias
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=output_answers))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            init_embedding_w = load_word2vec()
            sess.run(embedding_w.assign(init_embedding_w))
            step = 0
            while step < training_iters:
                batch_in, batch_out = get_batch(step, questions, answers, len(answers_vocab_processor.vocabulary_))
                sess.run(optimizer, feed_dict={input_questions: batch_in, output_answers: batch_out})
                if step % display_step == 0:
                    loss = sess.run(cost, feed_dict={input_questions: batch_in, output_answers: batch_out})
                    print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss))
                step += 1
            print("Optimization Finished!")

            step = 0
            losses = []
            while step * batch_size < len(questions):
                batch_in, batch_out, size = get_batch_for_test(step, questions, answers, len(answers_vocab_processor.vocabulary_))
                loss = sess.run(cost, feed_dict={input_questions: batch_in, output_answers: batch_out})
                losses.append(loss * size)
                if step % display_step == 0:
                    print("Iter " + str(step))
                step += 1
            total_train_loss = sum(losses) / len(questions)
            print("Total Training Loss= " + "{:.6f}".format(total_train_loss))

            questions, answers = load_validation_data(questions_vocab_processor, answers_vocab_processor)
            step = 0
            losses = []
            while step * batch_size < len(questions):
                batch_in, batch_out, size = get_batch_for_test(step, questions, answers, len(answers_vocab_processor.vocabulary_))
                loss = sess.run(cost, feed_dict={input_questions: batch_in, output_answers: batch_out})
                losses.append(loss * size)
                if step % display_step == 0:
                    print("Iter " + str(step))
                step += 1
            total_train_loss = sum(losses) / len(questions)
            print("Total Validation Loss= " + "{:.6f}".format(total_train_loss))


run()
