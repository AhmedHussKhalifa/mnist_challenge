"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
import pickle
import tensorflow as tf
# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model_train import Model
from pgd_attack import LinfPGDAttack, YangAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

attack_yang = YangAttack(config['epsilon'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)
large_num_of_attacks = config['large_num_of_attacks']


# if tf.test.gpu_device_name():
#   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# exit(0)
with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  min_softmax = []
  max_softmax = []
  avg_softmax = []
  # for ii in range(max_num_training_steps):
  for ii in range(20):
    x_batch, y_batch = mnist.train.next_batch(batch_size)

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    
    # Compute Adversarial Perturbations
    
    start = timer()
    # for rep in range(large_num_of_attacks):
    # x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    x_batch_adv, y_batch = attack_yang.perturb(x_batch, y_batch, large_num_of_attacks)
    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}
    
    # Actual training step
    sess.run(train_step, feed_dict=adv_dict)
    
    end = timer()
    training_time += end - start
    
    y_xent = sess.run(model.y_xent, feed_dict=adv_dict)
    # softmax = sess.run(model.softmax, feed_dict=adv_dict)

    print("#"*20)
    # print(type(y_xent))
    # tf.math.reduce_max(
    print("After Expo (min, max, avg) :", np.min(y_xent) , np.max(y_xent), np.mean(y_xent))
    # print("After", np.min(softmax) , np.max(softmax), np.mean(softmax))



    avg_softmax.append(np.mean(y_xent))
    min_softmax.append(np.min(y_xent))
    min_softmax.append(np.max(y_xent))


    print("#"*20)


    # correct_prediction = sess.run(model.correct_prediction, feed_dict=adv_dict)
    # y_xent = sess.run(model.y_xent, feed_dict=adv_dict)
    # print(y_pred)
    # print(y_batch)
    # print(correct_prediction)


    adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0

    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

  # pkl_filename_dir = os.path.join(str(config['lamda'])+".pkl")
  pkl_filename_dir = str(config['lamda'])+".pkl"
  print(pkl_filename_dir)
  all_data = [avg_softmax, max_softmax,  min_softmax]
  with open(pkl_filename_dir, 'wb') as handle:
    pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # # Actual training step
    # start = timer()
    # sess.run(train_step, feed_dict=adv_dict)
    # end = timer()
    # training_time += end - start
