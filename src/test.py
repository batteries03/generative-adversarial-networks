import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

TRAINING_DIR = './output/model/'

with tf.Session() as session:
    saver = tf.train.import_meta_graph(os.path.join(TRAINING_DIR, 'convnet.ckpt.meta'))
    saver.restore(session, os.path.join(TRAINING_DIR, 'convnet.ckpt'))
    
    inputs = tf.get_default_graph().get_tensor_by_name('convnet/inputs:0')
    outputs = tf.get_default_graph().get_tensor_by_name('convnet/conv-3/Relu:0')
    for i in range(10):
        plt.figure()
        plt.title(str(i))
        
        image = session.run(outputs, {inputs.name: [[i]]})
        
        image = image.squeeze()
        
        plt.imshow(image, cmap='gray')
        
    plt.show()
