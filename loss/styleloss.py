import tensorflow as tf
from loss.grammatrix import *
def style_loss(base_style, gram_target):
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  return tf.reduce_mean(tf.square(gram_style - gram_target)) / (4. * (channels ** 2) * (width * height) ** 2)