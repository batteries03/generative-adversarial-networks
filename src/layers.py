import numpy as np
import tensorflow as tf


#сверточный слой
def conv2d_layer(i, tensor, filter_shape, activation = tf.nn.relu):
    with tf.variable_scope('conv-%d' % i):
        #размер весов [число строк, число столбцов, число входных каналов, число выходных каналов]
        shape = [filter_shape[0], filter_shape[1], tensor.shape[3], filter_shape[2]]
        
        #веса сверток
        weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        #веса смещения
        biases = tf.get_variable('biases', [shape[-1]], tf.float32, initializer=tf.ones_initializer())
        
        #операция свертки. приходит входной тензор с весами
        conv = tf.nn.conv2d(tensor, 
                            weights,
                            strides=[1, 1, 1, 1],
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
def fully_connected_layer(i, tensor, n_neurons, activation = tf.nn.relu):
    with tf.variable_scope('fc-%d' % i):
        if(len(tensor.shape) > 2):
            tensor = tf.reshape(tensor, [-1, int(np.prod(tensor.shape[1:]))])
            
        shape = [int(tensor.shape[1]), n_neurons]
        
        weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.contrib.layers.xavier_initializer())
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
