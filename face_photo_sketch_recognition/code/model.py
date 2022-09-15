import numpy as np
import tensorflow as tf
from ops import *
import math

# Will add AdaIN
def build_layer(x, layer_spec, name, train_mode=True, reuse=False, A=None, B=None, batch_size=8):
    layer_type = layer_spec['type']
    out_channels = layer_spec['out_channels']
    ksize = layer_spec['ksize']
    stride = layer_spec['stride']
    padding = layer_spec['padding']
    use_se = layer_spec['se-block']
    use_bias = layer_spec['use_bias']

    # activation function
    if (layer_spec['rectifier'] == 'lrelu') or (layer_spec['rectifier'] == 'LReLU'):
        activation = lrelu
    elif (layer_spec['rectifier'] == 'prelu') or (layer_spec['rectifier'] == 'PReLU'):
        activation = prelu
    elif (layer_spec['rectifier'] == 'relu') or (layer_spec['rectifier'] == 'ReLU'):
        activation = relu
    elif (layer_spec['rectifier'] == 'sigmoid') or (layer_spec['rectifier'] == 'Sigmoid'):
        activation = sigmoid
    elif (layer_spec['rectifier'] == 'tanh') or (layer_spec['rectifier'] == 'Sigmoid'):
        activation = tanh
    else:
        activation = pass_activation

    # normalization function
    if (layer_spec['norm'] == 'instance'):
        batch_norm = instance_normalization
        bn_name = 'instance_norm'
        bn_reuse = reuse
        #use_bias = False
    elif (layer_spec['norm'] == 'instance_sep') or (layer_spec['norm'] == 'instance_s'):
        batch_norm = instance_normalization
        bn_name = layer_spec['bn_name']+'instance_norm'
        bn_reuse = layer_spec['bn_reuse']
        use_bias = False
    elif (layer_spec['norm'] == 'batch'):
        batch_norm = batch_normalization
        bn_name = 'batch_norm'
        bn_reuse = reuse
        use_bias = False
    elif (layer_spec['norm'] == 'batch_sep') or (layer_spec['norm'] == 'batch_s'):
        batch_norm = batch_normalization
        bn_name = layer_spec['bn_name']+'batch_norm'
        bn_reuse = layer_spec['bn_reuse']
        use_bias = False
    else:
        bn_name = 'pass'
        bn_reuse = reuse
        batch_norm = pass_normalization

    with tf.variable_scope(name):
        # conv
        if (layer_type == 'conv') or (layer_type == 'Conv'):
            convolved = conv2d(x, out_channels, ksize=ksize, stride=stride, padding=padding, use_bias=use_bias,
                               reuse=reuse)
            normalized = batch_norm(convolved, train_mode=train_mode, name=bn_name, reuse=bn_reuse)
            if use_se:
                normalized = se_block(normalized, reduction_ratio=16, rect=layer_spec['rectifier'], reuse=reuse)
            rectified = activation(normalized, 0.2, reuse=reuse)
        # deconv
        elif (layer_type == 'deconv') or (layer_type == 'Deconv_o') or (layer_type == 'dconv_o'):
            convolved = deconv2d_old(x, out_channels, ksize=ksize, stride=stride, padding=padding,
                                 use_bias=use_bias, reuse=reuse)
            normalized = batch_norm(convolved, train_mode=train_mode, name=bn_name, reuse=bn_reuse)
            if use_se:
                normalized = se_block(normalized, reduction_ratio=16, rect=layer_spec['rectifier'], reuse=reuse)
            rectified = activation(normalized, 0.2, reuse=reuse)
        # residual_block
        elif (layer_type == 'res') or (layer_type == "Res") or (layer_type == "res_block"):
            rectified = residual_block(x, f_size=out_channels, ksize=ksize, stride=stride,
                                       rectifier=layer_spec['rectifier'], norm=layer_spec['norm'], padding=padding,
                                       train_mode=train_mode, reuse=reuse, use_se=use_se)
        # bottleneck_residual_block
        elif (layer_type == 'res_b') or (layer_type == "Res_b") or (layer_type == "res_bottleneck"):
            rectified = residual_bottleneck(x, f_size=out_channels, ksize=ksize, stride=stride,
                                            rectifier=layer_spec['rectifier'], bottleneck=4, norm=layer_spec['norm'],
                                            padding=padding, train_mode=train_mode, reuse=reuse, use_se=use_se)
        # Style block
        elif (layer_type == 'style') or (layer_type == "style_block") or (layer_type == "Style"):
            if A is not None:
                rectified = style_block(x, A, B, f_size=out_channels, ksize=ksize, use_bias=False, padding='SAME',
                                        reuse=reuse, batch_size=batch_size, rectifier=layer_spec['rectifier'], use_noise=layer_spec['use_noise'])
            else:
                print("Style_block error: No A")
                assert False
        # Style2 block
        elif (layer_type == 'style2') or (layer_type == "style2_block") or (layer_type == "Style2"):
            if A is not None:
                rectified = style2_block(x, A, B, f_size=out_channels, ksize=ksize, use_bias=False, padding='SAME',
                                         reuse=reuse, rectifier=layer_spec['rectifier'],
                                         use_noise=layer_spec['use_noise'], demodulate=True, fused_modconv=True)
            else:
                print("Style2_block error: No A")
                assert False
        # linear
        elif (layer_type == 'fc') or (layer_type == 'FC') or (layer_type == 'linear') or (layer_type == 'Linear'):
            shape = x.get_shape().as_list()
            if len(shape) != 2:
                reshaped = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
            else:
                reshaped = x
            if stride > 1:
                out_channel = out_channels * stride * stride
            else:
                out_channel = out_channels
            convolved = linear(reshaped, out_channel, use_bias=use_bias, reuse=reuse)
            normalized = batch_norm(convolved, train_mode=train_mode, name=bn_name, reuse=bn_reuse)
            rectified = activation(normalized, 0.2, reuse=reuse)
            if stride > 1:
                rectified = tf.reshape(rectified, [-1, stride, stride, out_channels])
        # max_pooling
        elif (layer_type == 'pool') or (layer_type == 'Pool'):
            if (out_channels == 'max') or (out_channels == 'Max'):
                rectified = tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=padding)
            elif (out_channels == 'average') or (out_channels == 'avg'):
                if ksize == -1:
                    shape = x.get_shape().as_list()
                    ksize = shape[1]
                    stride = shape[1]
                    rectified = tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding='VALID')
                else:
                    rectified = tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=padding)
        elif (layer_type == 'vectorize'):
            shape = x.get_shape().as_list()
            if len(shape) != 2:
                rectified = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
            else:
                rectified = x
        elif (layer_type == 'up'):
            rectified = upsample_2d(x, stride)
        elif (layer_type == 'pass') or (layer_type == 'Pass'):
            rectified = x
        else:
            assert False, 'Wrong layer_type in feature network'
        return rectified


def CNN_Encoder(inputs, config, train_mode=True, name="ENC_", reuse=False, share=None, share_name="ENC",
                share_reuse=False, A=None, B=None, batch_size=8, use_noise=False, output_all_layers=False, ref=None):
    layers = []
    layer_spec = {'use_bias': config['use_bias'], 'padding': config['padding'], 'se-block': config['se-block'],
                  'rectifier': config['rectifier'], 'norm': config['norm'], 'bn_name': name, 'bn_reuse': reuse,
                  'use_noise': use_noise}

    num_layer = len(config['layer_specs'])
    if share is None:
        share = 99999
    elif share < -100:
        share = 99999
    elif share < 0:
        tmp = name
        name = share_name
        share_name = tmp
        tmp = reuse
        reuse = share_reuse
        share_reuse = tmp
        share = -share

    # layers
    layers.append(inputs)
    i = 0
    for layer_type, out_channels, stride, ksize in config['layer_specs']:
        # weight sharing
        if i == share:
            name = share_name
            reuse = share_reuse
        # layer spec
        layer_spec['type'] = layer_type
        layer_spec['out_channels'] = out_channels
        layer_spec['stride'] = stride
        layer_spec['ksize'] = ksize
        layer_name = name + "/layer_%d" % len(layers)

        if i + 1 == num_layer:
            if config['output_activation'] is not None:
                if config['output_activation'] == 'no_norm':
                    layer_spec['norm'] = 'pass'
                elif config['output_activation'] == 'no_norm_tanh':
                    layer_spec['norm'] = 'pass'
                    layer_spec['rectifier'] = 'tanh'
                else:
                    layer_spec['rectifier'] = config['output_activation']

        # build a layer
        layers.append(build_layer(layers[-1], layer_spec, name=layer_name, train_mode=train_mode, reuse=reuse, A=A, B=B, batch_size=batch_size))

        i += 1

    if output_all_layers:
        return layers[-1], layers
    else:
        return layers[-1]


#U-Net
def Unet_decoder(inputs, config, train_mode=True, name="ENC_", reuse=False, share=None, share_name="unet_dec",
                 share_reuse=False, A=None, B=None, batch_size=8, use_noise=False, output_all_layers=False, ref=None):
    layers = ref
    layer_spec = {'use_bias': config['use_bias'], 'padding': config['padding'], 'se-block': config['se-block'],
                  'rectifier': config['rectifier'], 'norm': config['norm'], 'bn_name': name, 'bn_reuse': reuse}

    num_layer = len(config['layer_specs'])
    if share is None:
        share = 99999999
    elif share < 0:
        tmp = name
        name = share_name
        share_name = tmp
        tmp = reuse
        reuse = share_reuse
        share_reuse = tmp
        share = -share

    # Config for decoder
    skip_i = len(layers) - 1
    i = 0
    # Decoder
    for layer_type, out_channels, stride, ksize in config['layer_specs']:
        # weight sharing
        if i == share:
            name = share_name
            reuse = share_reuse
        # layer spec
        layer_spec['type'] = layer_type
        layer_spec['out_channels'] = out_channels
        layer_spec['stride'] = stride
        layer_spec['ksize'] = ksize
        layer_name = name + "/layer_%d" % len(layers)

        # skip connection
        if (skip_i < len(layers) - 1) and (skip_i > 0):
            x = tf.concat([layers[-1], layers[skip_i]], axis=3)
        else:
            x = layers[-1]

        if i + 1 == num_layer:
            if config['output_activation'] is not None:
                if config['output_activation'] == 'no_norm':
                    layer_spec['norm'] = 'pass'
                elif config['output_activation'] == 'no_norm_tanh':
                    layer_spec['norm'] = 'pass'
                    layer_spec['rectifier'] = 'tanh'
                else:
                    layer_spec['rectifier'] = config['output_activation']

        # build a layer
        layers.append(build_layer(x, layer_spec, name=layer_name, train_mode=train_mode, reuse=reuse))

        i += 1
        skip_i -= 1

    return layers[-1]


#U-Net
def Unet(inputs, config, train_mode=True, name="ENC_", reuse=False, share=None, share_name="Unet_", share_reuse=False, label=None):
    layers = []
    layer_spec = {'use_bias': config['use_bias'], 'padding': config['padding'], 'se-block': config['se-block'],
                  'rectifier': config['rectifier'], 'norm': config['norm'], 'bn_name': name, 'bn_reuse': reuse}

    num_layer = len(config['layer_specs'])
    if share is None:
        share = 99999999
    elif share < 0:
        tmp = name
        name = share_name
        share_name = tmp
        tmp = reuse
        reuse = share_reuse
        share_reuse = tmp
        share = -share

    # for skip connection
    tmp_channels = config['output_channels']
    decoder_specs = []
    # Encoder
    layers.append(inputs)
    i = 0
    for layer_type, out_channels, stride, ksize in config['layer_specs']:
        # weight sharing
        if i == share:
            name = share_name
            reuse = share_reuse
        # layer spec
        layer_spec['type'] = layer_type
        layer_spec['out_channels'] = out_channels
        layer_spec['stride'] = stride
        layer_spec['ksize'] = ksize
        layer_name = name+"/enc_layer_%d" % len(layers)

        # build a layer
        layers.append(build_layer(layers[-1], layer_spec, name=layer_name, train_mode=train_mode, reuse=reuse))

        i += 1

        # decoder spec
        if (layer_type == 'conv') or (layer_type == 'Conv'):
            layer_type = 'dconv_o'
        decoder_specs.append((layer_type, tmp_channels, stride, ksize))
        tmp_channels = out_channels

    if label is not None:
        with tf.variable_scope(name+"/label_enc_0"):
            label0 = linear(label, 1024, reuse=reuse)
            label0 = tf.reshape(label0, [-1,4,4,64])
            label0 = lrelu(label0, 0.2)
        # layer 1
        layer_spec['type'] = 'conv'
        layer_spec['out_channels'] = 128
        layer_spec['stride'] = 1
        layer_spec['ksize'] = 4
        layer_name = name + "/label_enc_1"
        label1 = build_layer(label0, layer_spec, name=layer_name, train_mode=train_mode, reuse=reuse)
        # layer 2
        layer_spec['out_channels'] = 256
        layer_name = name + "/label_enc_2"
        label2 = build_layer(label1, layer_spec, name=layer_name, train_mode=train_mode, reuse=reuse)
        label2 = tf.concat([layers[-1],label2],axis=3)
        # mid layer
        layer_spec['out_channels'] = 512
        layer_name = name + "/label_mid_layer_3"
        label3 = build_layer(label2, layer_spec, name=layer_name, train_mode=train_mode, reuse=reuse)
        layers.append(label3)

    # Config for decoder
    skip_i = num_layer
    decoder_specs.reverse()

    # Decoder
    for layer_type, out_channels, stride, ksize in decoder_specs:
        # weight sharing
        if i == share:
            name = share_name
            reuse = share_reuse
        # layer spec
        layer_spec['type'] = layer_type
        layer_spec['out_channels'] = out_channels
        layer_spec['stride'] = stride
        layer_spec['ksize'] = ksize
        layer_name = name + "/dec_layer_%d" % len(layers)

        # skip connection
        if skip_i < num_layer:
            x = tf.concat([layers[-1], layers[skip_i]], axis=3)
        else:
            x = layers[-1]

        if skip_i == 1:
            if config['output_activation'] is not None:
                layer_spec['rectifier'] = config['output_activation']

        # build a layer
        layers.append(build_layer(x, layer_spec, name=layer_name, train_mode=train_mode, reuse=reuse))

        i += 1
        skip_i -= 1

    return layers[-1]


# StyleGAN2
def Style2_skip_Generator(inputs, config, train_mode=True, name="SkipG_", reuse=False, share=None, share_name="SkipG_",
                         share_reuse=False, A=None, B=None, batch_size=8, use_noise=False, output_all_layers=False, ref=None):
    layers = []
    RGB_layers = []
    layer_spec = {'use_bias': config['use_bias'], 'padding': config['padding'], 'se-block': config['se-block'],
                  'rectifier': config['rectifier'], 'norm': config['norm'], 'bn_name': name, 'bn_reuse': reuse,
                  'use_noise': use_noise}

    num_layer = len(config['layer_specs'])
    if share is None:
        share = 99999
    elif share < -100:
        share = 99999
    elif share < 0:
        tmp = name
        name = share_name
        share_name = tmp
        tmp = reuse
        reuse = share_reuse
        share_reuse = tmp
        share = -share

    # layers
    layers.append(inputs)
    i = 0
    for layer_type, out_channels, stride, ksize in config['layer_specs']:
        # weight sharing
        if i == share:
            name = share_name
            reuse = share_reuse
        # layer spec
        layer_spec['type'] = layer_type
        layer_spec['out_channels'] = out_channels
        layer_spec['stride'] = stride
        layer_spec['ksize'] = ksize
        layer_name = name + "/layer_%d" % len(layers)

        # tRGB
        if (layer_type == 'up') or (layer_type == 'dconv') or (layer_type == 'dconv_o'):
            x = torgb(layers[-1], A, f_size=3, padding='SAME', reuse=reuse, fused_modconv=True, name=layer_name+'_trgb')
            if not RGB_layers:
                RGB_layers.append(x)
            else:
                y = upsample_2d(RGB_layers[-1], 2)
                RGB_layers.append(x + y)

        # build a layer
        layers.append(build_layer(layers[-1], layer_spec, name=layer_name, train_mode=train_mode, reuse=reuse, A=A, B=B, batch_size=batch_size))

        i += 1

    # output RGB image
    x = torgb(layers[-1], A, f_size=3, padding='SAME', reuse=reuse, fused_modconv=True, name=layer_name+'_trgb')
    y = upsample_2d(RGB_layers[-1], 2)
    layers.append(x + y)

    if output_all_layers:
        return layers[-1], layers
    else:
        return layers[-1]


# StyleGAN2
def Style2_residual_Discriminator(inputs, config, train_mode=True, name="ResD_", reuse=False, share=None,
                                  share_name="ResD_", share_reuse=False, A=None, B=None, batch_size=8, use_noise=False, output_all_layers=False, ref=None):
    layers = []
    layer_spec = {'use_bias': config['use_bias'], 'padding': config['padding'], 'se-block': config['se-block'],
                  'rectifier': config['rectifier'], 'norm': config['norm'], 'bn_name': name, 'bn_reuse': reuse,
                  'use_noise': use_noise}

    num_layer = len(config['layer_specs'])
    if share is None:
        share = 99999
    elif share < -100:
        share = 99999
    elif share < 0:
        tmp = name
        name = share_name
        share_name = tmp
        tmp = reuse
        reuse = share_reuse
        share_reuse = tmp
        share = -share

    # layers
    layers.append(inputs)
    # fRGB
    tmp = config['layer_specs'][0]
    if tmp[2] > 1:
        f_size = int(tmp[1] / 2)
    else:
        f_size = tmp[1]
    frgb = fromrgb(layers[-1], f_size=f_size, rectifier=layer_spec['rectifier'], reuse=reuse, name=name+'_I_frgb')
    layers.append(frgb)

    i = 0
    for layer_type, out_channels, stride, ksize in config['layer_specs']:
        # weight sharing
        if i == share:
            name = share_name
            reuse = share_reuse
        # layer spec
        layer_spec['type'] = layer_type
        layer_spec['out_channels'] = out_channels
        layer_spec['stride'] = stride
        layer_spec['ksize'] = ksize
        layer_name = name + "/layer_%d" % len(layers)

        if i + 1 == num_layer:
            if config['output_activation'] is not None:
                if config['output_activation'] == 'no_norm':
                    layer_spec['norm'] = 'pass'
                elif config['output_activation'] == 'no_norm_tanh':
                    layer_spec['norm'] = 'pass'
                    layer_spec['rectifier'] = 'tanh'
                else:
                    layer_spec['norm'] = 'pass'
                    layer_spec['rectifier'] = config['output_activation']

        # build a layer
        layers.append(build_layer(layers[-1], layer_spec, name=layer_name, train_mode=train_mode, reuse=reuse, A=A, B=B, batch_size=batch_size))
        # fRGB
        if stride > 1:
            frgb = fromrgb(frgb, f_size=out_channels, rectifier=layer_spec['rectifier'], reuse=reuse, name=layer_name+'_frgb')
            frgb = downsample_2d(frgb, 2)
            frgb = frgb + layers[-1]
            layers.append(frgb)

        i += 1

    if output_all_layers:
        return layers[-1], layers
    else:
        return layers[-1]


# arcface------------------------------------------------------------------
def AngularSoftmax(embds, labels, class_num, scale, angular_margin, cos_margin, reuse=False, scope='logits'):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        weights = tf.get_variable(name='classify_weight', shape=[embds.get_shape().as_list()[-1], class_num], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

    embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
    weights = tf.nn.l2_normalize(weights, axis=0)

    # cos(theta)
    cos_t = tf.matmul(embds, weights, name='cos_t')

    # ArcFace
    if (angular_margin == 0) or (angular_margin == None):
        cos_mt_arc = cos_t
    else:
        cos_m = math.cos(angular_margin)
        sin_m = math.sin(angular_margin)
        mm = sin_m * angular_margin  # issue 1
        threshold = math.cos(math.pi - angular_margin)

        # cos(theta+m)
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = scale * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta + m should in range [0, pi]
        #    0 <= theta + m <= pi
        #    -m <= theta <= pi -m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = scale*(cos_t - mm)
        cos_mt_arc = tf.where(cond, cos_mt, keep_val)

    # CosFace
    if (cos_margin == 0) or (cos_margin == None):
        cos_mt_temp = cos_mt_arc
    else:
        cos_mt_temp = cos_mt_arc - cos_margin

    # i!=j case
    mask = tf.one_hot(labels, depth=class_num, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')
    s_cos_t = tf.multiply(scale, cos_t, name='scalar_cos_t')
    # logits
    output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')

    # losses
    softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
    pred = tf.argmax(tf.nn.softmax(output), axis=-1, output_type=tf.int64)

    return output, softmax_loss, pred


#adacos------------------------------------------------------------------

def AdaCos(embds, labels, class_num, reuse=False, scope='logits', is_dynamic=True, weights_placeholder=False):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if weights_placeholder:
            weightsA = tf.placeholder(tf.float32, shape=[embds.get_shape().as_list()[-1], class_num], name='classify_weight')
        else:
            weightsA = tf.get_variable(name='classify_weight', shape=[embds.get_shape().as_list()[-1], class_num], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

        # s
        init_s = math.sqrt(2) * math.log(class_num - 1)
        adacos_s = tf.get_variable(name='adacos_s_value', dtype=tf.float32, initializer=tf.constant(init_s), trainable=False)

    embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
    print(embds.get_shape())
    weights = tf.nn.l2_normalize(weightsA, axis=0)
    # cos(theta)
    cos_t = tf.matmul(embds, weights, name='cos_t')
    #theta = tf.acos(tf.clip_by_value(cos_t, -1.0 + 1e-10, 1.0 - 1e-10))

    if is_dynamic == False:
        output = tf.multiply(init_s, cos_t, name='adacos_fixed_logits')
        # losses
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
        pred = tf.argmax(tf.nn.softmax(output), axis=-1, output_type=tf.int64)
        return output, softmax_loss, pred, tf.cos(math.pi/4), init_s, weightsA, class_num-1

    # B_avg: i!=j case
    mask = tf.one_hot(labels, depth=class_num, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')
    B_avg = tf.exp(adacos_s*cos_t)
    B_avg = tf.multiply(B_avg, inv_mask)
    #B_avg = tf.where(tf.less(mask, 1), tf.exp(adacos_s*cos_t), tf.zeros_like(cos_t))
    B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1), name='B_avg')
    #B_avg = tf.stop_gradient(B_avg)

    # cos median
    cos_class = tf.gather_nd(cos_t, tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1), name='cos_class')
    cos_med = tf.contrib.distributions.percentile(cos_class, 50.0)

    # scale
    with tf.control_dependencies([cos_med, B_avg]):
        temp_s = tf.log(B_avg) / tf.maximum(cos_med, tf.cos(math.pi/4.))
        adacos_s = tf.assign(adacos_s, temp_s, name='adacos_s')
        output = tf.multiply(adacos_s, cos_t, name='adacos_dynamic_logits')

    # losses
    softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
    #inv_softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=inv_mask, logits=output)
    #softmax_loss = softmax_loss - 0.0001*inv_softmax_loss
    pred = tf.argmax(tf.nn.softmax(output), axis=-1, output_type=tf.int64)

    return output, softmax_loss, pred, cos_med, adacos_s, weightsA, B_avg


def GAN_loss_bin(predict_fake, predict_real, mode='log', eps=1e-6):
    shape = predict_fake.get_shape().as_list()
    if len(shape) == 3:
        predict_fake = tf.reduce_mean(predict_fake, axis=[1,2])
        predict_real = tf.reduce_mean(predict_real, axis=[1,2])
    elif len(shape) > 3:
        predict_fake = tf.reduce_mean(predict_fake, axis=[1,2,3])
        predict_real = tf.reduce_mean(predict_real, axis=[1,2,3])
    print(predict_real.get_shape())

    if mode == 'ls':
        # least square adversarial loss
        d_loss = tf.add(tf.reduce_mean((predict_real-1.)**2), tf.reduce_mean(predict_fake**2), name='d_loss')
        g_loss = tf.reduce_mean((predict_fake-1)**2, name='g_loss')
        print("Least Square GAN")
    elif mode == 'hinge':
        # hinge loss
        print('hinge')
        d_loss = tf.add(-tf.reduce_mean(tf.minimum(0.,predict_real-1)), -tf.reduce_mean(tf.minimum(0.,-predict_fake-1)), name='d_loss')
        g_loss = -tf.reduce_mean(predict_fake, name='g_loss')
    elif (mode == 'wgan') or (mode == 'wgan-gp'):
        # wassertain loss
        print('wgan')
        d_loss = tf.add(-tf.reduce_mean(predict_real), tf.reduce_mean(predict_fake), name='d_loss')
        g_loss = -tf.reduce_mean(predict_fake, name='g_loss')
    else:
        #predict_real = tf.nn.sigmoid(predict_real)
        #predict_fake = tf.nn.sigmoid(predict_fake)
        # log adversarial loss
        d_loss = tf.add(-tf.reduce_mean(tf.log(predict_real + eps)), -tf.reduce_mean(tf.log(1 - predict_fake + eps)), name='d_loss')
        g_loss = -tf.reduce_mean(tf.log(predict_fake + eps), name='g_loss')

    Gen_loss_sum = tf.summary.scalar("Generator_adversarial_loss", g_loss)
    Dis_loss_sum = tf.summary.scalar("Discriminator_adversarial_loss", d_loss)

    return g_loss, d_loss


def GAN_loss_mul(predict_fake, predict_real, mode='log', eps=1e-6, label=None, f_label=None):
    shape = predict_fake.get_shape().as_list()
    if len(shape) > 2:
        predict_fake = tf.reduce_mean(predict_fake, axis=[1,2])
        predict_real = tf.reduce_mean(predict_real, axis=[1,2])
    print(predict_real.get_shape())

    assert label!=None, 'GAN_loss: label error'

    print("GAN loss with label")
    if f_label == None:
        f_label = label
    real_label = tf.concat([label,tf.zeros([tf.shape(label)[0],1])], axis=1)
    f_real_label = tf.concat([f_label,tf.zeros([tf.shape(f_label)[0],1])], axis=1)
    fake_label = tf.concat([tf.zeros_like(f_label),tf.ones([tf.shape(f_label)[0],1])], axis=1)
    print(real_label.get_shape())

    if mode == 'ls':
        # least square adversarial loss
        d_loss = tf.add(tf.reduce_mean((predict_real-real_label)**2), tf.reduce_mean((predict_fake-fake_label)**2), name='d_loss')
        g_loss = tf.reduce_mean((predict_fake-f_real_label)**2, name='g_loss')
        print("Least Square GAN")
    elif mode == 'hinge':
        # hinge loss
        # With loss for false label
        print('hinge')
        d_loss = tf.add(-tf.reduce_mean(real_label*tf.minimum(0.,predict_real-1) + (1-real_label)*tf.minimum(0.,-predict_real-1)), -tf.reduce_mean(fake_label*tf.minimum(0.,predict_fake-1)+(1-fake_label)*tf.minimum(0.,-predict_fake-1)), name='d_loss')
        g_loss = -tf.reduce_mean(f_real_label*predict_fake + (1-f_real_label)*(-predict_fake), name='g_loss')    # False label -> hinge? or all? / real label vs others || real label vs fake?
    elif (mode == 'wgan') or (mode == 'wgan-gp'):
        # wassertain loss
        print('wgan')
        d_loss = tf.add(-tf.reduce_mean(real_label*predict_real + (1-real_label)*(-predict_real)), -tf.reduce_mean(fake_label*predict_fake+(1-fake_label)*(-predict_fake)), name='d_loss')
        g_loss = -tf.reduce_mean(f_real_label*predict_fake + (1-f_real_label)*(-predict_fake), name='g_loss')
    else:
        # log adversarial loss
        # With loss for false label
        d_loss = tf.add(-tf.reduce_mean(real_label*tf.log(predict_real + eps) + (1-real_label)*tf.log(1-predict_real + eps)), -tf.reduce_mean(fake_label*tf.log(predict_fake + eps) + (1-fake_label)*tf.log(1-predict_fake + eps)), name='d_loss')
        g_loss = -tf.reduce_mean(f_real_label*tf.log(predict_fake + eps) + (1-f_real_label)*tf.log(1-predict_fake + eps), name='g_loss')

    Gen_loss_sum = tf.summary.scalar("Generator_adversarial_loss", g_loss)
    Dis_loss_sum = tf.summary.scalar("Discriminator_adversarial_loss", d_loss)

    return g_loss, d_loss

