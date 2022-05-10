from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

from model_sagar import model
from pgd_attack_sagar import Attack
with open('config.json') as config_file:
    config = json.load(config_file)


# Setting up training parameters
tf.random.set_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

model = model()

attack = Attack(model,
               config['epsilon'],
               config['k'],
               config['a'],
               config['random_start'],
               config['loss_func'])