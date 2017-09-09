import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os
import tensorflow as tf
import layers


with open('../datasets/mnist/mnist.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin')


(train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = data
#разбивка на тренировочные картинки, проверочные, тестовые.
train_images = np.reshape(train_images, [-1, 28, 28, 1])
train_labels = np.reshape(train_labels, [-1, 1])
valid_images = np.reshape(valid_images, [-1, 28, 28, 1])
valid_labels = np.reshape(valid_labels, [-1, 1])
test_images = np.reshape(test_images, [-1, 28, 28, 1])
test_labels = np.reshape(test_labels, [-1, 1])

#функция рисования результатов работы генератора
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
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

def generator(inputs):
    with tf.name_scope('generator'):
        net = layers.fully_connected_layer(1, inputs, 7 * 7 * 8)
        net = tf.reshape(net, [-1, 7, 7, 8])
        net = layers.unpool(net)
        net = layers.conv2d_layer(1, net, [5, 5, 8])
        net = layers.unpool(net)
        net = layers.conv2d_layer(2, net, [5, 5, 1], tf.nn.sigmoid, zero_biases=True)

        return net

def discriminator(inputs):
    with tf.name_scope('discriminator'):
        net = layers.fully_connected_layer(1, inputs, 128)
        net = layers.fully_connected_layer(2, net, 1, tf.nn.sigmoid, zero_biases=True, zero_weights=True)
        return net

#обнуление графа
tf.reset_default_graph()

#создание сети в графе
with tf.name_scope('GAN'):
    generator_seed_inputs = tf.placeholder(tf.float32, [None, GENERATOR_SEED_SIZE], name='generator_seed_inputs')

    _generator_inputs = generator_seed_inputs
    with tf.variable_scope('generator'):
        generator_outputs = generator(_generator_inputs)

    discriminator_inputs = tf.placeholder(tf.float32, [None] + list(train_images.shape[1:]), name='inputs')
    with tf.variable_scope('discriminator') as vs:
        with tf.name_scope('real'):
            discriminator_outputs_real_prob = discriminator(discriminator_inputs)
        vs.reuse_variables()
        with tf.name_scope('fake'):
            discriminator_outputs_fake_prob = discriminator(generator_outputs)

#элементы графа для обучения сети
with tf.name_scope('training'):
    with tf.name_scope('discriminator'):
        discriminator_targets_real = tf.ones_like(discriminator_outputs_real_prob, name='discriminator_targets_real')
        discriminator_targets_fake = tf.zeros_like(discriminator_outputs_fake_prob, name='discriminator_targets_fake')

        _loss_real = tf.reduce_mean(tf.log(tf.clip_by_value(discriminator_outputs_real_prob, 1e-9, 1)))
        _loss_fake = tf.reduce_mean(tf.log(tf.clip_by_value(1 - discriminator_outputs_fake_prob, 1e-9, 1)))
        discriminator_loss = _loss_real + _loss_fake

        #минимизация функции потерь по весовым коэффициентам
        discriminator_lr_var = tf.Variable(1e-3, trainable=False)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        optimizer = tf.train.AdamOptimizer(discriminator_lr_var)
        discriminator_updates = optimizer.minimize(-discriminator_loss, var_list=params) # maximization

    with tf.name_scope('generator'):
        #целевые значения, к которым должна придти сеть в результате обучения
        generator_targets = tf.ones_like(discriminator_outputs_fake_prob, name='generator_targets')
        #функция потерь (ошибки)
        generator_loss = tf.reduce_mean(tf.log(tf.clip_by_value(discriminator_outputs_fake_prob, 1e-9, 1)))

        #минимизация функции потерь по весовым коэффициентам
        generator_lr_var = tf.Variable(1e-3, trainable=False)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        optimizer = tf.train.AdamOptimizer(generator_lr_var)
        generator_updates = optimizer.minimize(-generator_loss, var_list=params) # maximization

# сохранение параметров для графа
save_vars = tf.global_variables()
saver = tf.train.Saver(save_vars)

# шаг тренировки. заполняем узлы графа, картинками для обучения, которые используются для входа, и вычисляем обновления весов
def train_discriminator_step(session, images, labels, seed):
    input_feed = {}

    input_feed[discriminator_inputs.name] = images
    #input_feed[generator_inputs.name] = labels
    input_feed[generator_seed_inputs.name] = seed

    output_feed = [discriminator_updates]

    _ = session.run(output_feed, input_feed)

def train_generator_step(session, labels, seed):
    input_feed = {}

    #input_feed[generator_inputs.name] = labels
    input_feed[generator_seed_inputs.name] = seed

    output_feed = [generator_updates]

    _ = session.run(output_feed, input_feed)

def generator_step(session, labels, seed):
    input_feed = {}

    #input_feed[generator_inputs.name] = labels
    input_feed[generator_seed_inputs.name] = seed

    return session.run(generator_outputs, input_feed)

# заполнение узлов графа картинками и целевыми значениями.расчет функции потерь
def valid_step(session, images, labels, seed, summary):
    input_feed = {}

    #input_feed[generator_inputs.name] = labels
    input_feed[discriminator_inputs.name] = images
    input_feed[generator_seed_inputs.name] = seed

    output_feed = [generator_loss, discriminator_loss, summary]

    return session.run(output_feed, input_feed)


# цикл обучения

EPOCHS = 100
STEPS_PER_CHECKPOINT = 5
BATCH_SIZE = 250

TRAINING_DIR = './model/'

if not os.path.exists(TRAINING_DIR):
    os.makedirs(TRAINING_DIR)

if not os.path.exists('./output/'):
    os.makedirs('./output/')

checkpoint_path = os.path.join(TRAINING_DIR, 'GAN.ckpt')

tf.summary.scalar('geneartor loss', generator_loss)
tf.summary.scalar('generator learning rate', generator_lr_var)

tf.summary.scalar('discriminator loss', discriminator_loss)
tf.summary.scalar('discriminator learning rate', discriminator_lr_var)

summary_op = tf.summary.merge_all()

nbatches = len(train_images) // BATCH_SIZE

with tf.Session() as session:
    train_summary_writer = tf.summary.FileWriter(os.path.join(TRAINING_DIR, 'summary', 'train'), session.graph)
    valid_summary_writer = tf.summary.FileWriter(os.path.join(TRAINING_DIR, 'summary', 'valid'), session.graph)

    print('Initializing parameters ', flush=True, end='')
    session.run(tf.global_variables_initializer())
    print('[OK]', flush=True)

    ckpt = tf.train.get_checkpoint_state(TRAINING_DIR)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(session, ckpt.model_checkpoint_path)

    tf.train.write_graph(session.graph_def, TRAINING_DIR, 'GAN.pb', as_text=False)

    print('Start training.', flush=True)
    try:
        for epoch in range(0, EPOCHS):
            samples = generator_step(session, np.random.randint(0, 9, [16, 1]), sample_seed_inputs(16, GENERATOR_SEED_SIZE))
            fig = plot(samples)
            plt.savefig('output/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

            start = time.time()

            print('Epoch #%i: ' % (epoch+1), end='', flush=True)

            for b in range(nbatches):
                batch = np.arange(b*BATCH_SIZE, (b+1)*BATCH_SIZE)

                for _ in range(3):
                    train_discriminator_step(session, train_images[batch], train_labels[batch], sample_seed_inputs(BATCH_SIZE, GENERATOR_SEED_SIZE))
                train_generator_step(session, train_labels[batch], sample_seed_inputs(BATCH_SIZE, GENERATOR_SEED_SIZE))

            batch = np.random.choice(len(train_images), BATCH_SIZE, replace=False)
            train_gen_loss, train_dis_loss, summary = valid_step(session, train_images[batch], train_labels[batch], sample_seed_inputs(BATCH_SIZE, GENERATOR_SEED_SIZE), summary_op)

            train_summary_writer.add_summary(summary, epoch)

            batch = np.random.choice(len(valid_images), BATCH_SIZE, replace=False)
            valid_gen_loss, valid_dis_loss, summary = valid_step(session, valid_images[batch], valid_labels[batch], sample_seed_inputs(BATCH_SIZE, GENERATOR_SEED_SIZE), summary_op)
            valid_summary_writer.add_summary(summary, epoch)

            elapsed = time.time() - start
            print('train generator loss = %.6f, train discriminator loss = %.6f, valid generator loss = %.6f, valid discriminator loss = %.6f, elapsed %.3f sec.' % (train_gen_loss, train_dis_loss, valid_gen_loss, valid_dis_loss, elapsed), flush=True)

            if (epoch+1) % STEPS_PER_CHECKPOINT == 0:
                saver.save(session, checkpoint_path)

        print('Training process is finished.', flush=True)

        samples = generator_step(session, np.random.randint(0, 9, [16, 1]), sample_seed_inputs(16, GENERATOR_SEED_SIZE))
        fig = plot(samples)
        plt.savefig('output/{}.png'.format(str(EPOCHS).zfill(3)), bbox_inches='tight')
        plt.close(fig)
    finally:
        saver.save(session, checkpoint_path)
        tf.train.write_graph(session.graph_def, TRAINING_DIR, 'GAN.pb', as_text=False)
