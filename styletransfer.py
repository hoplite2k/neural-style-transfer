from model import *
import tensorflow as tf
from layerfeatures import *
from loss.grammatrix import *
from imgmanipulation import deprocess_img
from gradient import *
import numpy as np

def style_transfer(content_path, 
                       style_path,
                       iterations=1000,
                       content_weight=1e-6, 
                       style_weight=1e12): 
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False

  style_features, content_features = feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

  init_image = load_and_process_img(content_path)
  init_image = tf.Variable(init_image, dtype=tf.float32)

  opt = tf.optimizers.Adam(learning_rate=5, epsilon=1e-1)

  best_loss, best_img = float('inf'), None

  loss_weights = (style_weight, content_weight)
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  for i in range(iterations):
    grads, all_loss = compute_grads(model, loss_weights, init_image, gram_style_features, content_features)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    
    if loss < best_loss:
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

    if i % 100 == 0:
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            .format(loss, style_score, content_score))
  return best_img, best_loss 