__author__ = 'Mohammad'

import tensorflow as tf
from train import load_related_train_data, load_data, batch_size, get_batch_for_test, display_step


def run():
    questions_vocab_processor, answers_vocab_processor, max_question_length = load_related_train_data()
    questions, answers, images_paths = load_data(questions_vocab_processor, answers_vocab_processor, True)

    sess = tf.Session()
    saver = tf.train.import_meta_graph('vqa_model-5000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    input_questions = graph.get_tensor_by_name('input_questions:0')
    images = graph.get_tensor_by_name("images:0")
    output_answers = graph.get_tensor_by_name('output_answers:0')
    cost = graph.get_tensor_by_name('cost:0')

    with sess.as_default():

        step = 0
        losses = []
        while step * batch_size < len(questions):
            batch_in_questions, batch_in_images, batch_out, size = get_batch_for_test(step, questions, answers, images_paths, len(answers_vocab_processor.vocabulary_))
            loss = sess.run(cost, feed_dict={input_questions: batch_in_questions, images: batch_in_images, output_answers: batch_out})
            losses.append(loss * size)
            if step % display_step == 0:
                print("Training samples {} out of {}".format(step * batch_size, len(questions)))
            step += 1
        total_train_loss = sum(losses) / len(questions)
        print("Total Training Loss= " + "{:.6f}".format(total_train_loss))

        questions, answers = load_data(questions_vocab_processor, answers_vocab_processor, False)
        step = 0
        losses = []
        while step * batch_size < len(questions):
            batch_in_questions, batch_in_images, batch_out, size = get_batch_for_test(step, questions, answers, images_paths, len(answers_vocab_processor.vocabulary_))
            loss = sess.run(cost, feed_dict={input_questions: batch_in_questions, images: batch_in_images, output_answers: batch_out})
            losses.append(loss * size)
            if step % display_step == 0:
                print("Validation samples {} out of {}".format(step * batch_size, len(questions)))
                print("Total Training Loss= " + "{:.6f}".format(total_train_loss))
            step += 1
        total_validation_loss = sum(losses) / len(questions)
        print("Total Validation Loss= " + "{:.6f}".format(total_validation_loss))

if __name__ == "__main__":
    run()
