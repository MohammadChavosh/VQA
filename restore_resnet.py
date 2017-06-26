__author__ = 'Mohammad'

import tensorflow as tf
import numpy as np
from data_loader import load_image

sess = tf.Session()

saver = tf.train.import_meta_graph('data/tensorflow-resnet-pretrained-20160509/ResNet-L152.meta')
saver.restore(sess, 'data/tensorflow-resnet-pretrained-20160509/ResNet-L152.ckpt')

graph = tf.get_default_graph()
images = graph.get_tensor_by_name("images:0")
img = load_image("data/train2014/COCO_train2014_000000000009.jpg")
img_features = graph.get_tensor_by_name("avg_pool:0")
features = sess.run(img_features, {images: img[np.newaxis, :]})
print features[0], features[0].shape
