import tensorflow as tf

def conv2d(_input,
           output_dim,
           kernel_size=(3,3),
           strides=1,
           padding='same',
           use_bias=True,
           name=None):
    initializer = tf.truncated_normal_initializer(0., 0.02)
    _output = tf.layers.conv2d(_input,
                            filters=output_dim,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            use_bias=use_bias,
                            kernel_initializer=initializer,
                            name=name)
    return _output

def batch_norm(_input, is_training, name=None):
    return tf.layers.batch_normalization(_input,
                                         momentum=0.99,
                                         epsilon=1e-3,
                                         training=is_training,
                                         fused=None,
                                         virtual_batch_size=None,
                                         name=name)

def maxpool(_input,
            pool_size=(2,2),
            strides=2,
            padding='same',
            name=None):
    return tf.layers.MaxPooling2D(pool_size=pool_size,
                               strides=strides,
                               padding=padding,
                               data_format='channels_last',
                               name=name)(_input)

def avgpool(_input,
            pool_size=(2,2),
            strides=2,
            padding='same',
            name=None):
    return tf.layers.AveragePooling2D(pool_size=pool_size,
                               strides=strides,
                               padding=padding,
                               data_format='channels_last',
                               name=name)(_input)

def fc(_input, units, name=None):
    initializer = tf.truncated_normal_initializer(0., 0.02)
    return tf.layers.Dense(units=units,
                           use_bias=True,
                           kernel_initializer=initializer,
                           name=name)(_input)

def f(_input, output_dim, is_training, name=None):
    with tf.variable_scope(name):
        x = batch_norm(_input, is_training=is_training, name='00_batchnorm')
        x = conv2d(x, 32, (5,5), name='00_conv')
        x = tf.nn.relu(x, name='00_relu')
        x = maxpool(x, (3,3), 2, name='00_maxpool')
        x = conv2d(x, 32, (3,3), name='01_conv')
        feat_out = tf.nn.relu(x, name='01_relu')
        return feat_out

def g(_input, output_dim, is_training, name=None):
    with tf.variable_scope(name):
        x = conv2d(_input, 64, (3,3), strides=2, name='00_conv')
        x = tf.nn.relu(x, name='00_relu')
        x = conv2d(x, 64, (3,3), name='01_conv')
        feat_low = tf.nn.relu(x, name='01_relu')
        x = avgpool(feat_low, name='02_avgpool')
        x = tf.layers.flatten(x, name='02_flatten')
        x = fc(x, output_dim, name='03_dense')
        y_low = tf.nn.softmax(x, axis=-1, name='03_softmax')
        return y_low, x

def h(_input, output_dim, is_training, name=None):
    with tf.variable_scope(name):
        x = conv2d(_input, 32, (3,3), name='00_conv')
        x = batch_norm(x, is_training, name='00_batchnorm')
        x = tf.nn.relu(x, name='00_relu')
        x = conv2d(x, output_dim, (3,3), name='01_conv')
        # x = avgpool(x, (14,14), output_dim, 'valid', name='01_avgpool')
        x = tf.layers.flatten(x, name='01_flatten')
        x = fc(x, output_dim, name='01_dense')
        px = tf.nn.softmax(x, axis=-1, name='01_softmax')
        return px, x