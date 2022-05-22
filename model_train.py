"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import tensorflow as tf

class Model(object):
  def __init__(self):

    with open('config.json') as config_file:
      config = json.load(config_file)

    lamda = config['lamda']
    
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    # first fully connected layer
    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # #batch norm
    # batch_mean2, batch_var2 = tf.nn.moments(h_fc1, [0])
    # scale2 = tf.Variable(tf.ones([1024]))
    # beta2 = tf.Variable(tf.zeros([1024]))
    # batch_norm = tf.nn.batch_normalization(h_fc1,batch_mean2,batch_var2,beta2,scale2,0.3)
    # batch_norm_sig  = tf.nn.sigmoid(batch_norm)

    # output layer
    W_fc2 = self._weight_variable([1024,10])
    b_fc2 = self._bias_variable([10])

    # self.pre_softmax = tf.matmul(batch_norm_sig, W_fc2) + b_fc2
    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

    self.y_xent = tf.math.exp(lamda * self.pre_softmax)

    self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_input, logits=self.y_xent)



    self.xent = tf.reduce_sum(self.y_xent)


    self.y_pred = tf.argmax(self.pre_softmax, 1)
    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def multi_class_hinge_loss(self, logits, label, batch_size, n_classes):
    # get the correct logit
    flat_logits = tf.reshape(logits, (-1,))
    correct_id = tf.range(0, batch_size) * n_classes + label
    correct_logit = tf.gather(flat_logits, correct_id)

    # get the wrong maximum logit
    max_label = tf.argmax(logits, 1)
    top2, _ = tf.nn.top_k(logits, k=2, sorted=True)
    top2 = tf.split(1, 2, top2)
    for i in xrange(2):
        top2[i] = tf.reshape(top2[i], (batch_size, ))
    wrong_max_logit = tf.select(tf.equal(max_label, label), top2[1], top2[0])

    # calculate multi-class hinge loss
    return tf.math.reduce_mean(tf.maximum(0., 1. + wrong_max_logit - correct_logit))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
