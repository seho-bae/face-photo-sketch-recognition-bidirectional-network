import numpy as np
import tensorflow as tf

def linear(input_, output_size, scope="linear", use_bias=True, bias_start=0.0, spectral_norm=False, update_collection=None, reuse=False):#, weight_norm=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        matrix = tf.get_variable("weights", [shape[1], output_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
        # weight normalization (Xiang & Li, 2017)
        #if weight_norm == True:
        #    matrix = tf.nn.l2_normalize(matrix, [0])

        if spectral_norm:
            matrix = weights_spectral_norm(matrix, update_collection=update_collection)

        if use_bias == True:
            bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
            return tf.matmul(input_, matrix, name=scope) + bias
        else:
            return tf.matmul(input_, matrix, name=scope)


def conv2d(input_, output_channels, ksize=3, stride=1, padding='SAME', scope="conv2d", use_bias=True, spectral_norm=False, update_collection=None, reuse=False):#, weight_norm=False):
    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        w = tf.get_variable('weights', [ksize, ksize, input_.get_shape()[-1], output_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # weight normalization (Xiang & Li, 2017)
        #if weight_norm == True:
        #    w = tf.nn.l2_normalize(w, [0, 1, 2])

        if spectral_norm:
            w = weights_spectral_norm(w, update_collection=update_collection)

        conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding, name=scope)

        if use_bias == True:
            biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
            
        return conv


def dilated_conv2d(input_, output_channels, dilation, ksize=3, stride=1, padding='SAME', scope="conv2d", use_bias=True, spectral_norm=False, update_collection=None, reuse=False):#, weight_norm=False):
    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        w = tf.get_variable('weights', [ksize, ksize, input_.get_shape()[-1], output_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # weight normalization (Xiang & Li, 2017)
        #if weight_norm == True:
        #    w = tf.nn.l2_normalize(w, [0, 1, 2])

        if spectral_norm:
            w = weights_spectral_norm(w, update_collection=update_collection)

        conv = tf.nn.atrous_conv2d(input_, w, rate=dilation, padding=padding, name=scope)
        
        if use_bias == True:
            biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
            
        return conv

# upscale followed by conv2d
def deconv2d(input_, output_channels, ksize=3, stride=2, padding='SAME', scope="deconv2d", use_bias=True, spectral_norm=False, update_collection=None, reuse=False):#, weight_norm=False):
    shape = input_.get_shape().as_list()
    output_shape = tf.stack([tf.shape(input_)[0], shape[1]*stride, shape[2]*stride, output_channels])

    with tf.variable_scope(scope):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [ksize, ksize, output_channels, input_.get_shape()[-1]], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # weight normalization (Xiang & Li, 2017)
        #if weight_norm == True:
        #    w = tf.nn.l2_normalize(w, [0, 1, 2])

        if spectral_norm:
            w = weights_spectral_norm(w, update_collection=update_collection)
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding, name=scope)

        if use_bias == True:
            biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)
        else:
            deconv = tf.reshape(deconv, output_shape)

        return deconv

def batch_normalization(x, train_mode, epsilon=1e-6, decay = 0.9, name="batch_norm", use_vars=True, reuse=False):
    #bn = tf.keras.layers.BatchNormalization(name=name)(x,training=train_mode)
    #return bn
    return tf.layers.batch_normalization(x, training=train_mode, name=name, reuse=reuse)
    #
    # shape = x.get_shape().as_list()
    # with tf.variable_scope(name) as scope:
    #     if reuse:
    #         tf.get_variable_scope().reuse_variables()
    #     if use_vars:
    #         beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
    #         gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.constant_initializer(1.))
    #     else:
    #         beta = None
    #         gamma = None
    #     mean = tf.get_variable("mean", [shape[-1]], initializer=tf.constant_initializer(0.), trainable=False)
    #     var = tf.get_variable("var", [shape[-1]], initializer=tf.constant_initializer(1.), trainable=False)
    #     try:
    #         batch_mean, batch_var = tf.nn.moments(x,[0, 1, 2])
    #     except:
    #         batch_mean, batch_var = tf.nn.moments(x,[0])
    #
    # train_mean = tf.assign(mean, mean * decay + batch_mean * (1 - decay))
    # train_var = tf.assign(var, var * decay + batch_var * (1 - decay))
    # with tf.control_dependencies([train_mean, train_var]):
    #     train_bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon, name=name)
    # inference_bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name=name)
    #
    # return tf.cond(train_mode, lambda: train_bn, lambda: inference_bn)



def affine(x, reuse=False, name='affine'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.constant_initializer(1.))
    return tf.add(tf.multiply(x, gamma),beta, name=name+'_affine')


def tprelu(x, init=0.2, reuse=False, name='tprelu'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        t = tf.get_variable("tangent", [1], tf.float32, initializer=tf.constant_initializer(init))
        a = tf.get_variable("translation", shape[-1], tf.float32, initializer=tf.constant_initializer(0.))
    x = x - a
    p = tf.add(tf.nn.relu(x), tf.multiply(t,tf.minimum(x,0)))
    return tf.add(p,a,name=name+'_tprelu')


def tlrelu(x, leak=0.2, name="tlrelu", reuse=False):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        a = tf.get_variable("translation", shape[-1], tf.float32, initializer=tf.constant_initializer(0.))
    x = x - a
    return tf.add(tf.maximum(x, leak*x), a, name=name+'_tlrelu')


def avg_pool(x, ksize=2, strides=2, padding='SAME', name="avg_pool"):
    return tf.nn.avg_pool(x, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding=padding, name=name)


def instance_normalization(x, train_mode=True, epsilon=1e-6, decay = 0.9, name="instance_norm", use_vars=True, reuse=False):
    shape = x.get_shape().as_list()
    if len(shape) == 2:
        return x
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if use_vars:
            beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
            gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.constant_initializer(1.))
        else:
            beta = None
            gamma = None
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)

    normalized = tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(variance, epsilon)))
    if use_vars:
        normalized = tf.add(tf.multiply(gamma, normalized), beta)

    return normalized


def adaptive_instance_normalization(x, y, epsilon=1e-6, name="AdaIN", reuse=False):
    shape = x.get_shape().as_list()
    y_shape = y.get_shape().as_list()
    if len(shape) == 2:
        return x
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)

        if len(y_shape) == len(shape):
            beta, gamma = tf.nn.moments(y, axes=[1, 2], keep_dims=True)
        elif len(y_shape) == 2:
            y = tf.reshape(y, [-1, 1, 1, y.get_shape().as_list()[-1]])
            beta, gamma = tf.split(y, 2, axis=3)
        else:
            print('AdaIN error')
            assert False

        normalized = tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(variance, epsilon)))
        normalized = tf.add(tf.multiply(gamma, normalized), beta)

    return normalized


def residual_block(x, f_size=128, ksize=3, stride=1, rectifier='relu', norm='batch', padding='VALID', name="residual", train_mode=True, reuse=False, use_se=False):
    use_bias=True
    padding='SAME'

    # activation function
    if (rectifier == 'lrelu') or (rectifier == 'LReLU'):
        activation = lrelu
    elif (rectifier == 'prelu') or (rectifier == 'PReLU'):
        activation = prelu
    elif (rectifier == 'relu') or (rectifier == 'ReLU'):
        activation = relu
    elif (rectifier == 'sigmoid') or (rectifier == 'Sigmoid'):
        activation = sigmoid
    elif (rectifier == 'tanh') or (rectifier == 'tanh'):
        activation = tanh
    else:
        activation = pass_activation

    # normalization function
    if norm == 'instance':
        batch_norm = instance_normalization
    elif (norm == 'batch') or (norm == 'batchnorm'):
        batch_norm = batch_normalization
        use_bias = False
    else:
        batch_norm = pass_normalization

    h0 = conv2d(x, f_size, ksize=ksize, stride=stride, padding=padding, scope=name+'_conv0', use_bias=use_bias, reuse=reuse)
    h0 = batch_norm(h0, name=name+'_bn0', train_mode=train_mode, reuse=reuse)
    if use_se:
        h0 = se_block(h0, reduction_ratio=16, scope=name+'_SE0', rect=rectifier, reuse=reuse)
    h0 = activation(h0, name=name+'_conv0')

    h1 = conv2d(h0, f_size, ksize=ksize, stride=1, padding=padding, scope=name+'_conv1', use_bias=use_bias, reuse=reuse)
    h1 = batch_norm(h1, name=name+'_bn1', train_mode=train_mode, reuse=reuse)
    if use_se:
        h1 = se_block(h1, reduction_ratio=16, scope=name+'_SE1', rect=rectifier, reuse=reuse)

    # weighted skip connection
    shape = x.get_shape().as_list()
    in_size = shape[-1]
    if (stride > 1) or (in_size != f_size):
        x_tmp = conv2d(x, f_size, ksize=1, stride=stride, padding=padding, scope=name+'_skip', use_bias=use_bias, reuse=reuse)
    else:
        x_tmp = x

    output = x_tmp + h1
    output = activation(output, name=name+'_conv1')

    return output

def residual_bottleneck(x, f_size=256, ksize=3, stride=1, rectifier='relu', bottleneck=4., norm='batch', padding='VALID', name="residual_b", train_mode=True, reuse=False, use_se=False):
    use_bias=True
    padding='SAME'

    # activation function
    if (rectifier == 'lrelu') or (rectifier == 'LReLU'):
        activation = lrelu
    elif (rectifier == 'prelu') or (rectifier == 'PReLU'):
        activation = prelu
    elif (rectifier == 'relu') or (rectifier == 'ReLU'):
        activation = relu
    elif (rectifier == 'sigmoid') or (rectifier == 'Sigmoid'):
        activation = sigmoid
    elif (rectifier == 'tanh') or (rectifier == 'tanh'):
        activation = tanh
    else:
        activation = pass_activation

    # normalization function
    if norm == 'instance':
        batch_norm = instance_normalization
    elif (norm == 'batch') or (norm == 'batchnorm'):
        batch_norm = batch_normalization
        use_bias = False
    else:
        batch_norm = pass_normalization

    f_size_tmp = int(f_size/bottleneck)

    h0 = conv2d(x, f_size_tmp, ksize=1, stride=stride, padding=padding, scope=name+'_conv0', use_bias=use_bias, reuse=reuse)
    h0 = batch_norm(h0, name=name+'_bn0', train_mode=train_mode, reuse=reuse)
    if use_se:
        h0 = se_block(h0, reduction_ratio=16, scope=name+'_SE0', rect=rectifier, reuse=reuse)
    h0 = activation(h0, name=name+'_conv0')

    h1 = conv2d(h0, f_size_tmp, ksize=ksize, stride=1, padding=padding, scope=name+'_conv1', use_bias=use_bias, reuse=reuse)
    h1 = batch_norm(h1, name=name+'_bn1', train_mode=train_mode, reuse=reuse)
    if use_se:
        h1 = se_block(h1, reduction_ratio=16, scope=name+'_SE1', rect=rectifier, reuse=reuse)
    h1 = activation(h1, name=name+'_conv1')

    h2 = conv2d(h1, f_size, ksize=1, stride=1, padding=padding, scope=name+'_conv2', use_bias=use_bias, reuse=reuse)
    h2 = batch_norm(h2, name=name+'_bn2', train_mode=train_mode, reuse=reuse)
    if use_se:
        h2 = se_block(h2, reduction_ratio=16, scope=name+'_SE2', rect=rectifier, reuse=reuse)

    # weighted skip connection
    shape = x.get_shape().as_list()
    in_size = shape[-1]
    if (stride > 1) or (in_size != f_size):
        x_tmp = conv2d(x, f_size, ksize=1, stride=stride, padding=padding, scope=name+'_skip', use_bias=use_bias, reuse=reuse)
    else:
        x_tmp = x

    output = x_tmp + h2
    output = activation(output, name=name+'_conv2')

    return output


def style_block(x, A, B=None, f_size=512, ksize=3, name='style', use_bias=False, padding='SAME', reuse=False, batch_size=8, rectifier='lrelu', use_noise=False):
    # activation function
    if (rectifier == 'lrelu') or (rectifier == 'LReLU'):
        activation = lrelu
    elif (rectifier == 'prelu') or (rectifier == 'PReLU'):
        activation = prelu
    elif (rectifier == 'relu') or (rectifier == 'ReLU'):
        activation = relu
    elif (rectifier == 'sigmoid') or (rectifier == 'Sigmoid'):
        activation = sigmoid
    elif (rectifier == 'tanh') or (rectifier == 'tanh'):
        activation = tanh
    else:
        activation = pass_activation

    A_shape = A.get_shape().as_list()

    with tf.variable_scope(name):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if x is not None:
            h = conv2d(x, f_size, ksize, stride=1, padding=padding, scope='conv',  use_bias=use_bias)
            h = activation(h, 0.2, reuse=reuse)
        else:
            h = tf.get_variable("const", [1, ksize, ksize, f_size], tf.float32, initializer=tf.random_normal_initializer())
            if batch_size > 1:
                h = tf.tile(h, [batch_size, 1, 1, 1])

        shape = h.get_shape().as_list()

        # B is noise vector
        if use_noise:
            if B is None:
                B = tf.random_normal([batch_size, f_size], name='noise')
                scale = tf.get_variable("B_scale", [1, f_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                Bh = scale*B
                Bh = tf.reshape(Bh, [-1, 1, 1, f_size])
                h = h+Bh
            else:
                assert False

        if len(A_shape) == 2:
            Ah = linear(A, 2*shape[-1], scope='A_conv', use_bias=use_bias, reuse=reuse)
        else:
            # reshape
            # fc
            print('StyleGAN: A is not vector')
            assert False

        output = adaptive_instance_normalization(h, Ah, name='AdaIN', reuse=reuse)

    return output


'''
def dense_block(x, growth_rate=4, ksize=3, n_layers=4, bottleneck=False, norm='instance', rect='lrelu', name="dense", train_mode=True, reuse=False):
    layers = []
    if norm == 'instance':
        batch_norm = instance_normalization
    elif (norm == 'batch') or (norm == 'batchnorm'):
        batch_norm = batch_normalization
    else:
        batch_norm = pass_normalization

    if rect == 'relu':
        rectifier = tf.nn.relu
    else:
        rectifier = lrelu

    layers.append(x)
    for i in range(n_layers):
        with tf.variable_scope(name+str(i)):
            normalized = batch_norm(layers[-1], train_mode=train_mode, reuse=reuse)
            rectified = rectifier(normalized)
            if bottleneck:
                rectified = conv2d(rectified, 4*growth_rate, ksize=1, scope='bottelneck', reuse=reuse)
            convolved = conv2d(rectified, growth_rate, ksize=ksize, reuse=reuse)
            concated = tf.concat([layers[-1], convolved], axis=3)
            layers.append(concated)

    return layers[-1]
'''

def se_block(x, reduction_ratio=16, scope='SE', rect='relu', reuse=False, spectral_norm=False):
    shape = x.get_shape().as_list()
    channels = shape[3]
    reducted = int(channels/reduction_ratio)

    if rect == 'lrelu':
        rectifier = lrelu
    else:
        rectifier = tf.nn.relu
    
    with tf.variable_scope(scope) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # Sqeeze (using average)
        h = tf.nn.avg_pool(x, ksize=[1, shape[1], shape[2], 1], strides=[1,1,1,1], padding='VALID')
        h = tf.reshape(h, [-1, channels])
        # Excitation 1
        h = linear(h, reducted, scope="excitation1", reuse=reuse, spectral_norm=spectral_norm)
        h = rectifier(h)
        # Excitation 2
        h = linear(h, channels, scope="excitation2", reuse=reuse, spectral_norm=spectral_norm)
        h = tf.nn.sigmoid(h)
        # Scale
        h = tf.reshape(h, [-1, 1, 1, channels])
        y = tf.multiply(h, x, name='scale')

    return y


# For convinience
def pass_normalization(x, train_mode=True, epsilon=1e-6, decay = 0.9, name="instance_norm", use_vars=True, reuse=False):
    return x


# spectral_norm
def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm

def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite+1
        
        u_hat, v_hat,_ = power_iteration(u,iteration)
        
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        
        w_mat = w_mat/sigma
        
        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not(update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))
            
            w_norm = tf.reshape(w_mat, w_shape)
    return w_norm


def relu(x, leak=0.2, reuse=False, name="_relu"):
    return tf.nn.relu(x, name=name+"_relu")


def lrelu(x, leak=0.2, reuse=False, name="lrelu"):
    return tf.maximum(x, leak*x, name=name+'_lrelu')


def prelu(x, init=0.2, reuse=False, name='_prelu'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        t = tf.get_variable("tangent", [1], tf.float32, initializer=tf.constant_initializer(init))
    return tf.add(tf.nn.relu(x), tf.multiply(t,tf.minimum(x,0)),name=name+'_prelu')


def sigmoid(x, leak=0.2, reuse=False, name="_sigmoid"):
    return tf.nn.sigmoid(x, name=name+"_sigmoid")


def tanh(x, leak=0.2, reuse=False, name="_tanh"):
    return tf.math.tanh(x, name=name+"_tanh")


def pass_activation(x, leak=0.2, reuse=False, name="_pass"):
    return x

def upsample_2d(x, factor=2):
    with tf.variable_scope('NaiveUpsample'):
        _N, H, W, C = x.shape.as_list()
        x = tf.reshape(x, [-1, H, 1, W, 1, C])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        return tf.reshape(x, [-1, H * factor, W * factor, C])

def downsample_2d(x, factor=2):
    with tf.variable_scope('NaiveDownsample'):
        _N, H, W, C = x.shape.as_list()
        x = tf.reshape(x, [-1, H // factor, factor, W // factor, factor, C])
        return tf.reduce_mean(x, axis=[2, 4])

def style2_block(x, A, B=None, f_size=512, ksize=3, name='style2', use_bias=True, padding='SAME', reuse=False,
                 rectifier='lrelu', use_noise=False, demodulate=True, fused_modconv=True):
    # activation function
    if (rectifier == 'lrelu') or (rectifier == 'LReLU'):
        activation = lrelu
    elif (rectifier == 'prelu') or (rectifier == 'PReLU'):
        activation = prelu
    elif (rectifier == 'relu') or (rectifier == 'ReLU'):
        activation = relu
    elif (rectifier == 'sigmoid') or (rectifier == 'Sigmoid'):
        activation = sigmoid
    elif (rectifier == 'tanh') or (rectifier == 'tanh'):
        activation = tanh
    else:
        activation = pass_activation

    A_shape = A.get_shape().as_list()

    with tf.variable_scope(name):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # initial layer
        if x is None:
            x = tf.get_variable("const", [1, ksize, ksize, f_size], tf.float32, initializer=tf.random_normal_initializer())
            if A_shape[0] > 1:
                x = tf.tile(x, [A_shape[0], 1, 1, 1])
            ksize=3

        # get weight
        w = tf.get_variable('weights', [ksize, ksize, x.get_shape()[-1], f_size], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        ww = w[np.newaxis]  # [BkkIO] Introduce minibatch dimension.

        # modulate
        if len(A_shape) == 2:
            s = linear(A, x.shape[-1].value, scope='A_linear', use_bias=use_bias, reuse=reuse)
        else:
            # reshape
            # fc
            print('StyleGAN: A is not vector')
            assert False
        ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype)  # [BkkIO] Scale input feature maps.

        # demodulate
        if demodulate:
            d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1, 2, 3]) + 1e-8)  # [BO] Scaling factor.
            ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :]  # [BkkIO] Scale output feature maps.

        # reshape/scale input
        x = tf.transpose(x, [0, 3, 1, 2])  # NHWC->NCHW
        if fused_modconv:
            x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])  # Fused => reshape minibatch to convolution groups.
            w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
        else:
            x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype)  # [BIhw] Not fused => scale input activations.

        # convolution with optional up/downsampling
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1, 1, 1, 1], padding=padding)

        # reshape/scale output
        if fused_modconv:
            x = tf.reshape(x, [-1, f_size, x.shape[2], x.shape[3]])  # Fused => reshape convolution groups back to minibatch.
        elif demodulate:
            x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype)  # [BOhw] Not fused => scale output activations.

        x = tf.transpose(x, [0, 2, 3, 1])  # NCHW->NHWC

        # add bias
        if use_bias == True:
            biases = tf.get_variable('biases', [f_size], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, biases)
        # activation
        x = activation(x, 0.2, reuse=reuse)

        # B is noise vector
        if use_noise:
            if B is None:
                B = tf.random_normal([A_shape[0], f_size], name='noise')
                scale = tf.get_variable("B_scale", [1, f_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                Bh = scale*B
                Bh = tf.reshape(Bh, [-1, 1, 1, f_size])
                x = x+Bh
            else:
                assert False

    return x


def torgb(x, A, f_size=512, padding='SAME', reuse=False, fused_modconv=True, name='tRGB'):
    return style2_block(x, A, f_size=f_size, ksize=1, name=name, use_bias=True, padding=padding, reuse=reuse,
                        rectifier='pass', use_noise=False, demodulate=False, fused_modconv=fused_modconv)


def fromrgb(x, f_size=512, rectifier='lrelu', padding='SAME', reuse=False, name='fRGB'):
    # activation function
    if (rectifier == 'lrelu') or (rectifier == 'LReLU'):
        activation = lrelu
    elif (rectifier == 'prelu') or (rectifier == 'PReLU'):
        activation = prelu
    elif (rectifier == 'relu') or (rectifier == 'ReLU'):
        activation = relu
    elif (rectifier == 'sigmoid') or (rectifier == 'Sigmoid'):
        activation = sigmoid
    elif (rectifier == 'tanh') or (rectifier == 'tanh'):
        activation = tanh
    else:
        activation = pass_activation

    x = conv2d(x, f_size, ksize=1, stride=1, padding=padding, scope=name, use_bias=True, spectral_norm=False, reuse=reuse)

    return activation(x, 0.2, reuse=reuse)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Minibatch standard deviation layer.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
    y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
    y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
    y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
