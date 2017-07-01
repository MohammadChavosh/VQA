__author__ = 'Mohammad'

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn, rnn
from data_loader import get_related_answers, get_vqa_data, load_image

# Parameters
embedding_dim = 300
word2vec_file = 'data/GoogleNews-vectors-negative300.bin'
learning_rate = 0.001
batch_size = 8
display_step = 10
save_step = 10
n_hidden = 256
pre_output_len = 256
img_features_len = 512


def load_related_train_data():
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
    questions_vocab_processor.fit(question_texts)
    # questions = np.array(list(questions_vocab_processor.fit_transform(question_texts)))

    answers_vocab_processor = learn.preprocessing.VocabularyProcessor(1, min_frequency=20)
    answers_vocab_processor.fit(answers_vocab)
    print "answers size={}".format(len(answers_vocab_processor.vocabulary_) - 1)
    return questions_vocab_processor, answers_vocab_processor, max_question_length


def load_data(questions_vocab_processor, answers_vocab_processor, is_train):
    vqa_triplets = get_vqa_data(is_train)
    question_texts = list()
    answers_vocab = list()
    images = list()
    for (q, a, v) in vqa_triplets:
        if a in answers_vocab_processor.vocabulary_._mapping:
            question_texts.append(q)
            answers_vocab.append(a)
            images.append(v)

    questions = np.array(list(questions_vocab_processor.transform(question_texts)))
    answers = np.array(list(answers_vocab_processor.transform(answers_vocab)))

    return questions, answers, images


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


def get_batch(step, questions, answers, images_paths, answers_vocab_len):
    batch_start = (step * batch_size) % len(questions)
    batch_in_questions = questions[batch_start:batch_start + batch_size]
    batch_in_images = list()
    batch_out = np.zeros((batch_size, answers_vocab_len))
    for i in range(batch_start, batch_start + len(batch_in_questions)):
        batch_in_images.append(load_image(images_paths[i]))
        batch_out[i - batch_start, answers[i] - 1] = 1

    tmp = batch_size - len(batch_in_questions)
    if tmp > 0:
        for i in range(0, tmp):
            batch_out[i + len(batch_in_questions), answers[i] - 1] = 1
            batch_in_images.append(load_image(images_paths[i]))
        batch_in_questions = np.concatenate((batch_in_questions, questions[0:tmp]), axis=0)
    return batch_in_questions, np.asarray(batch_in_images), batch_out


def get_batch_for_test(step, questions, answers, images_paths, answers_vocab_len):
    batch_start = (step * batch_size) % len(questions)
    batch_in_questions = questions[batch_start:batch_start + batch_size]
    batch_in_images = list()
    batch_out = np.zeros((len(batch_in_questions), answers_vocab_len))
    for i in range(batch_start, batch_start + len(batch_in_questions)):
        batch_in_images.append(load_image(images_paths[i]))
        batch_out[i - batch_start, answers[i] - 1] = 1

    return batch_in_questions, np.asarray(batch_in_images), batch_out, len(batch_in_questions)


def run():
    questions_vocab_processor, answers_vocab_processor, max_question_length = load_related_train_data()
    questions, answers, images_paths = load_data(questions_vocab_processor, answers_vocab_processor, True)

    sess = tf.Session()

    res_net_loader = tf.train.import_meta_graph('data/tensorflow-resnet-pretrained-20160509/ResNet-L152.meta')
    res_net_loader.restore(sess, 'data/tensorflow-resnet-pretrained-20160509/ResNet-L152.ckpt')

    graph = tf.get_default_graph()
    images = graph.get_tensor_by_name("images:0")
    raw_img_features = graph.get_tensor_by_name("avg_pool:0")
    raw_to_img_features_w = tf.Variable(tf.random_normal([raw_img_features.shape.as_list()[1], img_features_len]),
                                        name="raw_to_img_w")
    raw_to_img_features_bias = tf.Variable(tf.random_normal([img_features_len]), name="raw_to_img_bias")
    img_features = tf.nn.relu(tf.matmul(raw_img_features, raw_to_img_features_w) + raw_to_img_features_bias)

    embedding_w = tf.Variable(tf.random_uniform([len(questions_vocab_processor.vocabulary_), embedding_dim], -1.0, 1.0), name="embedding_w")
    input_questions = tf.placeholder(tf.int32, [None, questions.shape[1]], name="input_questions")
    embedded_chars = tf.nn.embedding_lookup(embedding_w, input_questions)
    unstacked_embedded_chars = tf.unstack(embedded_chars, max_question_length, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    encoded_questions, _ = rnn.static_rnn(lstm_cell, unstacked_embedded_chars, dtype=tf.float32)
    q_w = tf.Variable(tf.random_normal([n_hidden, n_hidden]), name="q_w")
    q_bias = tf.Variable(tf.random_normal([n_hidden]), name="q_bias")
    questions_features = tf.nn.relu(tf.matmul(encoded_questions[-1], q_w) + q_bias)

    output_len = len(answers_vocab_processor.vocabulary_) - 1
    output_answers = tf.placeholder(tf.float32, [None, output_len], name="output_answers")

    # tmp_len = img_features_len * pre_output_len
    # q_to_img_w = tf.Variable(tf.random_normal([n_hidden, tmp_len]), name="q_to_img_w")
    # q_to_img_bias = tf.Variable(tf.random_normal([tmp_len]), name="q_to_img_bias")
    # img_out_w = tf.matmul(questions_features, q_to_img_w) + q_to_img_bias
    # img_out_w = tf.reshape(img_out_w, [-1, img_features_len, pre_output_len])
    img_out_w = tf.Variable(tf.random_normal([img_features_len, pre_output_len]), name="img_w")
    q_out_w = tf.Variable(tf.random_normal([n_hidden, pre_output_len]), name="q_out_w")
    out_bias = tf.Variable(tf.random_normal([pre_output_len]), name="out_bias")

    pre_output = tf.nn.relu(tf.matmul(img_features, img_out_w) + tf.matmul(questions_features, q_out_w) + out_bias)
    pre_output_w = tf.Variable(tf.random_normal([pre_output_len, output_len]), name="pre_out_w")
    pre_output_bias = tf.Variable(tf.random_normal([output_len]), name="pre_out_bias")

    prediction = tf.matmul(pre_output, pre_output_w) + pre_output_bias
    prediction = tf.identity(prediction, name="prediction")
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=output_answers), name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    step = tf.Variable(0, name="step")
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        init_embedding_w = load_word2vec(questions_vocab_processor)
        sess.run(embedding_w.assign(init_embedding_w))

        saver = tf.train.Saver()
        if os.path.isfile('data/trained_models/vqa_model.meta'):
            saver = tf.train.import_meta_graph('data/trained_models/vqa_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('data/trained_models/'))
            print "Restored step={}".format(sess.run(step))

        while sess.run(step) * batch_size < len(questions):
            pythonic_step = sess.run(step)
            batch_in_questions, batch_in_images, batch_out, _ = get_batch_for_test(pythonic_step, questions, answers, images_paths, output_len)
            sess.run(optimizer, feed_dict={input_questions: batch_in_questions, images: batch_in_images, output_answers: batch_out})
            sess.run(tf.assign_add(step, 1))
            if pythonic_step % display_step == 0:
                loss = sess.run(cost, feed_dict={input_questions: batch_in_questions, images: batch_in_images, output_answers: batch_out})
                print("Iter " + str(pythonic_step) + ", Minibatch Loss= " + "{:.6f}".format(loss))
            if pythonic_step % save_step == 0:
                saver.save(sess, 'data/trained_models/vqa_model')
                print("Saving...")
        print("Optimization Finished!")
        saver.save(sess, 'data/trained_models/vqa_model')

        sess.run(tf.assign(step, 0))
        total_size = 0
        losses = []
        while sess.run(step) * batch_size < len(questions):
            pythonic_step = sess.run(step)
            batch_in_questions, batch_in_images, batch_out, size = get_batch_for_test(pythonic_step, questions, answers, images_paths, output_len)
            loss = sess.run(cost, feed_dict={input_questions: batch_in_questions, images: batch_in_images, output_answers: batch_out})
            losses.append(loss * size)
            total_size += size
            if pythonic_step % display_step == 0:
                print("Training samples {} out of {}".format(pythonic_step * batch_size, len(questions)))
                print("Till now training loss= " + "{:.6f}".format(sum(losses) / total_size))
            sess.run(tf.assign_add(step, 1))
        total_train_loss = sum(losses) / total_size
        print("Total Training Loss= " + "{:.6f}".format(total_train_loss))

        if total_size != len(questions):
            print("BUG!!!!")
            print(total_size)
            print(len(questions))
            return

        questions, answers = load_data(questions_vocab_processor, answers_vocab_processor, False)
        sess.run(tf.assign(step, 0))
        total_size = 0
        losses = []
        while sess.run(step) * batch_size < len(questions):
            pythonic_step = sess.run(step)
            batch_in_questions, batch_in_images, batch_out, size = get_batch_for_test(pythonic_step, questions, answers, images_paths, output_len)
            loss = sess.run(cost, feed_dict={input_questions: batch_in_questions, images: batch_in_images, output_answers: batch_out})
            losses.append(loss * size)
            total_size += size
            if pythonic_step % display_step == 0:
                print("Validation samples {} out of {}".format(pythonic_step * batch_size, len(questions)))
                print("Till now validation loss= " + "{:.6f}".format(sum(losses) / total_size))
                print("Total Training Loss= " + "{:.6f}".format(total_train_loss))
            sess.run(tf.assign_add(step, 1))
        total_validation_loss = sum(losses) / len(questions)
        print("Total Validation Loss= " + "{:.6f}".format(total_validation_loss))

        if total_size != len(questions):
            print("BUG!!!!")
            print(total_size)
            print(len(questions))
            return

if __name__ == "__main__":
    run()
