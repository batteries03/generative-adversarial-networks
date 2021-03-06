import numpy as np
import tensorflow as tf


#сверточный слой
def conv2d_layer(i, tensor, filter_shape, activation = tf.nn.relu, stride=1, zero_biases=False, zero_weights=False):
    with tf.variable_scope('conv-%d' % i):
        #размер весов [число строк, число столбцов, число входных каналов, число выходных каналов]
        shape = [filter_shape[0], filter_shape[1], tensor.shape[3], filter_shape[2]]

        #веса сверток
        if (zero_weights):
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.zeros_initializer())
        else:
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        #веса смещения
        if (zero_biases):
            biases = tf.get_variable('biases', [shape[-1]], tf.float32, initializer=tf.zeros_initializer())
        else:
            biases = tf.get_variable('biases', [shape[-1]], tf.float32, initializer=tf.ones_initializer())

        #операция свертки. приходит входной тензор с весами
        conv = tf.nn.conv2d(tensor,
                            weights,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        #функция активации нейрнов сверточного слоя поэлементно
        if activation:
            act = activation(tf.nn.bias_add(conv, biases))
        else:
            act = tf.nn.bias_add(conv, biases)
    return act

def conv2d_transpose_layer(i, tensor, filter_shape, batch_size, activation = tf.nn.relu, stride=1, zero_biases=False, zero_weights=False):
    with tf.variable_scope('conv-%d' % i):
        #размер весов [число строк, число столбцов, число входных каналов, число выходных каналов]
        shape = [filter_shape[0], filter_shape[1], filter_shape[2], tensor.shape[3]]
        out_shape = [int(batch_size), int(tensor.shape[1]*stride), int(tensor.shape[2]*stride), filter_shape[2]]

        #веса сверток
        if (zero_weights):
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.zeros_initializer())
        else:
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        #веса смещения
        if (zero_biases):
            biases = tf.get_variable('biases', [filter_shape[2]], tf.float32, initializer=tf.zeros_initializer())
        else:
            biases = tf.get_variable('biases', [filter_shape[2]], tf.float32, initializer=tf.ones_initializer())

        #операция свертки. приходит входной тензор с весами
        conv = tf.nn.conv2d_transpose(tensor,
                                      weights,
                                      out_shape,
                                      strides=[1, stride, stride, 1],
                                      padding='SAME')
        #функция активации нейрнов сверточного слоя поэлементно
        if activation:
            act = activation(tf.nn.bias_add(conv, biases))
        else:
            act = tf.nn.bias_add(conv, biases)
    return act

#слой объединения. подается результат свертки и функция выбирает максимальное значение, проходя по блокам
def max_pool2d(tensor, pool_size):
    return tf.nn.max_pool(tensor,
                         ksize=[1, pool_size[0], pool_size[1], 1],
                         strides=[1, pool_size[0], pool_size[1], 1],
                         padding='SAME')

#обычный персептрон. функция активации. матричное умножение с целью преобразования признаков из одного пространства в другое
def fully_connected_layer(i, tensor, n_neurons, activation = tf.nn.relu, zero_weights=False, zero_biases=False):
    with tf.variable_scope('fc-%d' % i):
        if(len(tensor.shape) > 2):
            tensor = tf.reshape(tensor, [-1, int(np.prod(tensor.shape[1:]))])

        shape = [int(tensor.shape[1]), n_neurons]

        if (zero_weights):
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.zeros_initializer())
        else:
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        if (zero_biases):
            biases = tf.get_variable('biases', [shape[-1]], tf.float32, initializer=tf.zeros_initializer())
        else:
            biases = tf.get_variable('biases', [shape[-1]], tf.float32, initializer=tf.ones_initializer())

        matmul = tf.matmul(tensor, weights)
        if activation:
            act = activation(tf.nn.bias_add(matmul, biases))
        else:
            act = tf.nn.bias_add(matmul, biases)
    return act

def unpool(value):
    with tf.name_scope('unpool') as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, out], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def batch_norm(input_batch, training, name='batch_norm'):
    with tf.variable_scope(name or 'batch_norm'):
        offset = tf.Variable(tf.zeros(input_batch.shape[-1]), trainable=True, name='offset')
        scale = tf.Variable(tf.ones(input_batch.shape[-1]), trainable=True, name='scale')

        batch_mean, batch_std = tf.nn.moments(input_batch, axes=list(range(len(input_batch.shape)-1)), keep_dims=True)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_std])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_std)

        def mean_var_without_update():
            return ema.average(batch_mean), ema.average(batch_std)

        mean, std = tf.cond(training, mean_var_with_update, mean_var_without_update)

        return tf.nn.batch_normalization(input_batch, mean, std, offset, scale, variance_epsilon=1e-5)
