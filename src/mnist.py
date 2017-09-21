import numpy as np
import pickle
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os
import tensorflow as tf
import layers


EPOCHS = 100
STEPS_PER_CHECKPOINT = 5
BATCH_SIZE = 100

TRAINING_DIR = './model/'


with open('../dataset/mnist/mnist.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin')


(train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = data
#разбивка на тренировочные картинки, проверочные, тестовые.
train_images = np.reshape(train_images, [-1, 28, 28, 1])
train_labels = np.reshape(train_labels, [-1, 1])
valid_images = np.reshape(valid_images, [-1, 28, 28, 1])
valid_labels = np.reshape(valid_labels, [-1, 1])
test_images = np.reshape(test_images, [-1, 28, 28, 1])
test_labels = np.reshape(test_labels, [-1, 1])

# дополнение до размера 32x32
train_images = np.pad(train_images, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='edge')
valid_images = np.pad(valid_images, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='edge')
test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='edge')

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

def generator(inputs, batch_size, training):
    with tf.name_scope('generator'):
        net = layers.fully_connected_layer(1, inputs, 4 * 4 * 512, None)
        net = tf.reshape(net, [batch_size, 4, 4, 512])
        net = layers.batch_norm(net, training, name='bn1')
        net = layers.conv2d_transpose_layer(1, net, [5, 5, 256], batch_size, stride=2)
        net = layers.batch_norm(net, training, name='bn2')
        net = layers.conv2d_transpose_layer(2, net, [5, 5, 128], batch_size, stride=2)
        net = layers.batch_norm(net, training, name='bn3')
        net = layers.conv2d_transpose_layer(3, net, [5, 5, 1], batch_size, tf.nn.sigmoid, stride=2, zero_biases=True)

        return net

def discriminator(inputs, labels, training):
    with tf.name_scope('discriminator'):
        #net = layers.batch_norm(inputs, training, name='bn1')
        net = layers.conv2d_layer(1, inputs, [5, 5, 16], lambda x: layers.lrelu(x, 0.2), stride=2)
        #net = layers.batch_norm(net, training, name='bn2')
        net = layers.conv2d_layer(2, net, [5, 5, 32], lambda x: layers.lrelu(x, 0.2), stride=2)
        #net = layers.batch_norm(net, training, name='bn3')
        net = layers.conv2d_layer(3, net, [5, 5, 64], lambda x: layers.lrelu(x, 0.2), stride=2)
        #net = layers.batch_norm(net, training, name='bn4')
        net = layers.conv2d_layer(4, net, [5, 5, 128], lambda x: layers.lrelu(x, 0.2), stride=2)
        net = layers.max_pool2d(net, [2, 2])
        #net = layers.batch_norm(net, training, name='bn5')
        net = layers.conv2d_layer(5, net, [1, 1, 1], tf.nn.sigmoid)
        net = tf.reshape(net, [-1, 1])

        return net

#обнуление графа
tf.reset_default_graph()

#создание сети в графе
with tf.name_scope('GAN'):
    training_mode = tf.placeholder(tf.bool, name='training_mode')
    labels_inputs = tf.placeholder(tf.int32, [BATCH_SIZE, 1], name='labels_inputs')
    _labels_inputs = tf.cast(tf.one_hot(tf.squeeze(labels_inputs), 10), tf.float32)
    _labels_inputs = tf.reshape(_labels_inputs, [-1, 10])

    generator_seed_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, GENERATOR_SEED_SIZE], name='generator_seed_inputs')

    _generator_inputs = generator_seed_inputs
    with tf.variable_scope('generator'):
        _inputs = _generator_inputs# tf.concat([_generator_inputs, _labels_inputs], axis=1)
        generator_outputs = generator(_inputs, BATCH_SIZE, training_mode)

    discriminator_inputs = tf.placeholder(tf.float32, [BATCH_SIZE] + list(train_images.shape[1:]), name='inputs')
    with tf.variable_scope('discriminator') as vs:
        with tf.name_scope('real'):
            discriminator_outputs_real_prob = discriminator(discriminator_inputs, _labels_inputs, training_mode)
        vs.reuse_variables()
        with tf.name_scope('fake'):
            discriminator_outputs_fake_prob = discriminator(generator_outputs, _labels_inputs, training_mode)

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
    input_feed[labels_inputs.name] = labels
    input_feed[generator_seed_inputs.name] = seed
    input_feed[training_mode.name] = True

    output_feed = [discriminator_updates]

    _ = session.run(output_feed, input_feed)

def train_generator_step(session, labels, seed):
    input_feed = {}

    input_feed[labels_inputs.name] = labels
    input_feed[generator_seed_inputs.name] = seed
    input_feed[training_mode.name] = True

    output_feed = [generator_updates]

    _ = session.run(output_feed, input_feed)

def generator_step(session, labels, seed):
    input_feed = {}

    input_feed[labels_inputs.name] = labels
    input_feed[generator_seed_inputs.name] = seed
    input_feed[training_mode.name] = False

    return session.run(generator_outputs, input_feed)

# заполнение узлов графа картинками и целевыми значениями.расчет функции потерь
def valid_step(session, images, labels, seed, summary):
    input_feed = {}

    input_feed[labels_inputs.name] = labels
    input_feed[discriminator_inputs.name] = images
    input_feed[generator_seed_inputs.name] = seed
    input_feed[training_mode.name] = False

    output_feed = [generator_loss, discriminator_loss, summary]

    return session.run(output_feed, input_feed)


# цикл обучения

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
            samples = generator_step(session, np.random.randint(0, 9, [BATCH_SIZE, 1]), sample_seed_inputs(BATCH_SIZE, GENERATOR_SEED_SIZE))[:16]
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

        samples = generator_step(session, np.random.randint(0, 9, [BATCH_SIZE, 1]), sample_seed_inputs(BATCH_SIZE, GENERATOR_SEED_SIZE))[:16]
        fig = plot(samples)
        plt.savefig('output/{}.png'.format(str(EPOCHS).zfill(3)), bbox_inches='tight')
        plt.close(fig)
    finally:
        saver.save(session, checkpoint_path)
        tf.train.write_graph(session.graph_def, TRAINING_DIR, 'GAN.pb', as_text=False)
