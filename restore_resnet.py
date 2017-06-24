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
i = [
	# graph.get_tensor_by_name("scale1/Relu:0"),
	# graph.get_tensor_by_name("scale2/MaxPool:0"),
	# graph.get_tensor_by_name("scale2/block1/Relu:0"),
	# graph.get_tensor_by_name("scale2/block2/Relu:0"),
	# graph.get_tensor_by_name("scale2/block3/Relu:0"),
	# graph.get_tensor_by_name("scale3/block1/Relu:0"),
	# graph.get_tensor_by_name("scale5/block3/Relu:0"),
	graph.get_tensor_by_name("avg_pool:0"),
	# graph.get_tensor_by_name("prob:0"),
]
o = sess.run(i, {images: img[np.newaxis, :]})
print o[0], o[0].shape
