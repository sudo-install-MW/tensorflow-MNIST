import tensorflow as tf

def cnn_layer(img_input):
    # summary = []
    ##################################################################################################
    # conv layer 1
    print("Input shape :", img_input.get_shape())
    print("Initializing weights for layer 1 filter")
    filter_1 = tf.truncated_normal([3, 3, 1, 32])
    filter_1 = tf.Variable(filter_1)
    print("Initialized weights for layer 1 filter :", filter_1.get_shape())
    variable_summaries(filter_1)
    # filter_1_summary = tf.summary.image(tensor=filter_1, name='filter_1')
    # summary.append(filter_1_summary)

    layer_1 = tf.nn.conv2d(input=img_input, filter=filter_1, strides=[1, 1, 1, 1], padding="SAME")

    # create bias variable
    print("Initializing bias for layer 1")
    layer_1_bias = tf.truncated_normal([int(filter_1.get_shape()[3])])
    layer_1_bias = tf.Variable(layer_1_bias)
    print("Initialized bias for layer 1 :", layer_1_bias.get_shape())
    layer_1_out = tf.nn.relu(layer_1 + layer_1_bias)
    print("Layer_1 out shape :", layer_1_out.get_shape())

    layer_1_p = tf.nn.max_pool(value=layer_1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    print("layer_1 shape after max pooling :", layer_1_p.get_shape())

    # layer 1 out --> 14x14x32

    ##################################################################################################
    # conv layer 2
    # create filter weights
    print("Initializing weights for layer 2 filter")
    filter_2 = tf.truncated_normal([3, 3, 32, 64])
    filter_2 = tf.Variable(filter_2)
    print("Initialized weights for layer 2 filter :", filter_2.get_shape())
    layer_2 = tf.nn.conv2d(input=layer_1_p, filter=filter_2, strides=[1, 1, 1, 1], padding="SAME")
    # filter_2_summary = tf.summary.image(tensor=filter_2, name='filter_2')
    # summary.append(filter_2_summary)

    # create bias variable
    print("Initializing bias for layer 2")
    layer_2_bias = tf.truncated_normal([64])
    layer_2_bias = tf.Variable(layer_2_bias)
    print("Initialized bias for layer 2 :", layer_2_bias.get_shape())
    layer_2_out = tf.nn.relu(layer_2 + layer_2_bias)
    print("Layer_2 out shape :", layer_2_out.get_shape())
    layer_2_p = tf.nn.max_pool(value=layer_2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    print("layer_2 shape after max pooling :", layer_2_p.get_shape())

    # layer 2 out --> 7x7x64
    ##################################################################################################

    # fc layer 3
    # possible error
    print("flattening layer 2")
    layer_2_flat = tf.layers.flatten(inputs=layer_2_p)
    print("layer_2_flat shape :", layer_2_flat.get_shape())
    # possible error
    print("Initializing weights for fc layer 1")
    fc_1_weight = tf.truncated_normal(shape=(int(layer_2_flat.get_shape()[1]), 512), stddev=0.1)
    fc_1_weight = tf.Variable(fc_1_weight)
    print("Initialized weights for fc layer 1 :", fc_1_weight.get_shape())
    print("Initializing bias for fc layer 1")
    fc_1_bias = tf.truncated_normal([512])
    print("fc layer 1 bias shape is", fc_1_bias.get_shape())
    fc_1_out = tf.matmul(layer_2_flat, fc_1_weight) + fc_1_bias
    print("fc_1_out shape is :", fc_1_out.get_shape())

    ##################################################################################################
    print("Initializing weight for fc layer 2")
    fc_2_weight = tf.truncated_normal(shape=(int(fc_1_out.get_shape()[1]), 10), stddev=0.1)
    fc_2_weight = tf.Variable(fc_2_weight)
    print("Initialized weight for fc layer 2 :", fc_2_weight.get_shape())
    print("Initializing bias for fc layer 2")
    fc_2_bias = tf.truncated_normal([10])
    fc_2_bias = tf.Variable(fc_2_bias)
    print("Initialized bias for fc layer 2 :", fc_2_bias.get_shape())
    out = tf.matmul(fc_1_out, fc_2_weight) + fc_2_bias
    print("shape of output is :", out.get_shape())

    ##################################################################################################

    return out#, summary


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
