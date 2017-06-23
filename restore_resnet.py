__author__ = 'Mohammad'

import tensorflow as tf

sess = tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('data/tensorflow-resnet-pretrained-20160509/ResNet-L152.meta')
saver.restore(sess, 'data/tensorflow-resnet-pretrained-20160509/ResNet-L152.ckpt')

for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale5'):
    print i.name   # i.name if you want just a name

# # Access saved Variables directly
# print(sess.run('bias:0'))
# # This will print 2, which is the value of bias that we saved
#
# # Now, let's access and create placeholders variables and
# # create feed-dict to feed new data
#
# graph = tf.get_default_graph()
# w1 = graph.get_tensor_by_name("w1:0")
# w2 = graph.get_tensor_by_name("w2:0")
# feed_dict ={w1:13.0,w2:17.0}
#
# #Now, access the op that you want to run.
# op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
#
# print sess.run(op_to_restore,feed_dict)
# #This will print 60 which is calculated