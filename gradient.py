import tensorflow as tf
from loss.loss import *

def compute_grads(model, loss_weights, init_image, gram_style_features, content_features):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(model, loss_weights, init_image, gram_style_features, content_features)
  total_loss = all_loss[0]
  return tape.gradient(total_loss, init_image), all_loss