__author__ = 'Mohammad'

import tensorflow as tf
import skimage.io
import skimage.transform
import numpy as np


def load_image(path, size=224):
	img = skimage.io.imread(path)
	short_edge = min(img.shape[:2])
	yy = int((img.shape[0] - short_edge) / 2)
	xx = int((img.shape[1] - short_edge) / 2)
	crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
	resized_img = skimage.transform.resize(crop_img, (size, size))
	return resized_img


sess = tf.Session()

saver = tf.train.import_meta_graph('data/tensorflow-resnet-pretrained-20160509/ResNet-L152.meta')
saver.restore(sess, 'data/tensorflow-resnet-pretrained-20160509/ResNet-L152.ckpt')

graph = tf.get_default_graph()
images = graph.get_tensor_by_name("images:0")
img = load_image("data/train2014/COCO_train2014_000000000009.jpg")
before_fc_pool = graph.get_tensor_by_name("avg_pool:0"),
img_features = sess.run(before_fc_pool, {images: img[np.newaxis, :]})
print img_features[0], img_features[0].shape
