from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from tf_nn_skeleton import ModelSkeleton

class SqueezeNet(ModelSkeleton):
	def __init__(self, mc):
		ModelSkeleton.__init__(self, mc)
	    self._add_forward_graph()
	    self._add_interpretation_graph()
	    self._add_loss_graph()
	    self._add_train_graph()
	    self._add_viz_graph()

	def _avgpooling_layer(self, layer_name, inputs, 
		size, stride, padding='SAME'):
	"""
	Average pooling layer constructor.

	Args:
		layer_name: layer name
		inputs: input tensor
		size: kernel size
		stride: stride
		padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
	Returns:
		An average pooling layer operation.
	"""
	with tf.variable_scope(layer_name) as scope:
    	out =  tf.nn.avg_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, stride, stride, 1],
                            padding=padding)
		activation_size = np.prod(out.get_shape().as_list()[1:])
		self.activation_counter.append((layer_name, activation_size))
		return out

	def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
		freeze=False):
		"""
		Fire layer constructor. Modified from Bichen Wu's
		squeezeDet constructor.

		Args:
			layer_name: layer name
			inputs: input tensor
			s1x1: number of 1x1 filters in squeeze layer.
			e1x1: number of 1x1 filters in expand layer.
			e3x3: number of 3x3 filters in expand layer.
			freeze: if true, do not train parameters in this layer.
		Returns:
			fire layer operation.
		"""
		assert sq1x1 < ex1x1+ex3x3, "Too many squeeze layer filters \
			in {}".format(layer_name)
		sq1x1 = self._conv_layer(
			layer_name+'/s1x1', inputs, filters=s1x1, size=1, stride=1,
			padding='SAME', stddev=stddev, freeze=freeze)
		ex1x1 = self._conv_layer(
			layer_name+'/e1x1', sq1x1, filters=e1x1, size=1, stride=1,
			padding='SAME', stddev=stddev, freeze=freeze)
		ex3x3 = self._conv_layer(
			layer_name+'/e3x3', sq1x1, filters=e3x3, size=3, stride=1,
			padding='SAME', stddev=stddev, freeze=freeze)

		return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

	def _add_forward_graph(self):
		"""
		NN architecture specification.
		"""
		mc = self.mc

		conv1 = self._conv_layer(layer_name='conv1', inputs=self.input_image,
			filters=64, size=3, stride=2, padding='SAME', freeze=True)
		pool1 = self._pooling_layer(layer_name='pool1', inputs=conv1,
			size=3, stride=2, padding='SAME')

		fire2 = self._fire_layer(layer_name='fire2', inputs=pool1,
			s1x1=16, e1x1=64, e3x3=64)
		fire3 = self._fire_layer(layer_name='fire3', inputs=fire2,
			s1x1=16, e1x1=64, e3x3=64)
		fire4 = self._fire_layer(layer_name='fire4', inputs=fire3,
			s1x1=16, e1x1=64, e3x3=64)
		pool4 = self._pooling_layer( layer_name='pool4', inputs=fire4,
			size=3, stride=2, padding='SAME')

		fire5 = self._fire_layer(layer_name='fire5', inputs=pool4,
			s1x1=32, e1x1=128, e3x3=128)
		fire6 = self._fire_layer(layer_name='fire6', inputs=fire5,
			s1x1=32, e1x1=128, e3x3=128)
		fire7 = self._fire_layer(layer_name='fire7', inputs=fire6,
			s1x1=32, e1x1=128, e3x3=128)
		fire8 = self._fire_layer(layer_name='fire8', inputs=fire7,
			s1x1=32, e1x1=128, e3x3=128)
		pool8 = self._pooling_layer(layer_name='pool8', inputs=fire8,
			size=3, stride=2, padding='SAME')

		fire9 = self._fire_layer(layer_name='fire8', inputs=fire7,
			s1x1=48, e1x1=192, e3x3=192)
		dropout9 = tf.nn.dropout(fire9, self.keep_prob, name='dropout9')
		
		num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)

		conv10 = self._conv_layer( layer_name='conv10', inputs=dropout9,
			filters=num_output, size=3, stride=1, padding='SAME',
			xavier=False, relu=False, stddev=0.0001)
		pool10 = self._avgpooling_layer( layer_name='pool10', inputs=conv10,
			size=3, stride=2, padding='SAME')

		logits = tf.squeeze(inputs=pool10, size=, name='logits')
		self.preds = logits