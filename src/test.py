import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TRAINING_DIR = './model/'

def plot(samples, grid):
    fig = plt.figure(figsize=grid)
    gs = gridspec.GridSpec(*grid)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.squeeze(), cmap='gray')

    return fig

def sample_seed_inputs(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

GENERATOR_SEED_SIZE = 100

if not os.path.exists('./samples/'):
    os.makedirs('./samples/')

with tf.Session() as session:
    saver = tf.train.import_meta_graph(os.path.join(TRAINING_DIR, 'GAN.ckpt.meta'))
    saver.restore(session, os.path.join(TRAINING_DIR, 'GAN.ckpt'))

    noise_inputs = tf.get_default_graph().get_tensor_by_name('GAN/generator_seed_inputs:0')
    labels_inputs = tf.get_default_graph().get_tensor_by_name('GAN/labels_inputs:0')
    training_mode = tf.get_default_graph().get_tensor_by_name('GAN/training_mode:0')

    outputs = tf.get_default_graph().get_tensor_by_name('GAN/generator/generator/conv-3/Sigmoid:0')

    noise = sample_seed_inputs(100, GENERATOR_SEED_SIZE)

    for i in range(10):
        samples = session.run(outputs, {noise_inputs.name: noise, labels_inputs.name: [[i]] * noise.shape[0], training_mode.name: False})
        samples = samples[:16]

        fig = plot(samples, (4, 4))
        plt.title(str(i))
        plt.savefig('samples/{}.png'.format(str(i)), bbox_inches='tight')
        plt.close(fig)
