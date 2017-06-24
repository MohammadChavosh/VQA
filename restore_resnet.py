__author__ = 'Mohammad'

import tensorflow as tf

sess = tf.Session()

saver = tf.train.import_meta_graph('data/tensorflow-resnet-pretrained-20160509/ResNet-L152.meta')
saver.restore(sess, 'data/tensorflow-resnet-pretrained-20160509/ResNet-L152.ckpt')

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("avg_pool")
# w2 = graph.get_tensor_by_name("w2:0")
# feed_dict ={w1:13.0,w2:17.0}
#
# #Now, access the op that you want to run.
op_to_restore = graph.get_tensor_by_name("avg_pool:0")
#
# print sess.run(op_to_restore,feed_dict)
# #This will print 60 which is calculated