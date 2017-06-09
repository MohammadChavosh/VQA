import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

# Parameters
embedding_dim = 300
word2vec_file = 'GoogleNews-vectors-negative300.bin'

# Load data
print("Loading data...")
x_text = ['How many skateboards in the photo?']

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print(x)

# Training

with tf.Graph().as_default():
    W = tf.Variable(tf.random_uniform([len(vocab_processor.vocabulary_), embedding_dim], -1.0, 1.0), name="W")
    input_x = tf.placeholder(tf.int32, [None, x.shape[1]], name="input_x")
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), embedding_dim))
        # load any vectors from the word2vec
        print("Load word2vec file {}\n".format(word2vec_file))
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

        sess.run(W.assign(initW))
    print(sess.run(embedded_chars, feed_dict={input_x: x}))