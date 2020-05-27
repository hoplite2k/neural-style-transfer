import tensorflow as tf
from layers import *

def content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))