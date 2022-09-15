import numpy as np
import time
import os
import tensorflow as tf
import model
import input_data
import gan
from utils import *

class matchnet(object):
    def __init__(self, config, from_zero=True):
        # input batch
        self.input_image_size = config.img_size
        self.padding_size = 272
        self.crop_size = 256    # None or same with image_size, if you don't want to crop
        self.random_crop = True
        self.photo_dim = 3
        self.sketch_dim = 3
        self.input_dir = config.tr_dir
        self.photo_txt = config.tr_dir+'/'+config.tr_list
        self.sketch_txt = config.ts_dir+'/'+config.ts_list
        self.num_style = 1    # excluding photo
        self.style_list = ['real_db']   #['real_db', 'syn_cufs', 'syn_cufsf', 'syn_iiitd']
        self.batch_size = config.batch_size
        self.flip = True
        self.num_identity = config.num_identity
        # network architecture
        self.encoder_model = config.enc_model    # Select model
        self.share = 0 # weight sharing is started at
        self.norm = config.enc_norm
        self.feature_scale = 10#64.
        self.method = 'adacos'
        self.angular_margin = 0.5
        self.cos_margin = 0. #0.35
        self.weight_decay = None#0.0005
        self.feature_dim = config.med_channels
        self.g_matching_lambda = config.g_matching_lambda
        self.se_block = False
        # trainer
        self.max_epoch = config.max_epoch
        self.learning_rate = config.enc_lr
        self.beta1 = 0.5
        self.beta2 = 0.999
        #self.train_strategy = 'siamese'#'fix_photo'
        # learning rate decay
        self.lr_decay = None#0.1
        self.lr_step = 2500
        self.step = tf.Variable(0, trainable=False, name='global_step')
        # records
        self.log_dir = config.log_dir
        self.summary_step = 50    # write tensorboard summary
        self.print_epoch = config.print_epoch    # print losses
        self.save_epoch = config.save_epoch    # save model
        # to continue training, you must use same network architecture
        self.continue_training = config.continue_tr
        self.load_dir = config.load_dir
        self.load_epoch = config.load_epoch
        self.from_zero = from_zero
        # Translation function translate images in ts_dir using model on load_dir, load_step(or epoch)
        # you must use same network architecture with training
        # test epochs
        self.ts_batch_size = config.ts_batch_size
        self.ts_min_epoch = config.ts_min_epoch
        self.ts_max_epoch = config.ts_max_epoch
        self.ts_unit_epoch = config.ts_unit_epoch
        self.test_gallery = config.use_gallery
        self.gallery_txt = config.ts_dir+'/'+config.gall_list
        self.load_baseline = False
        self.train_photo = config.train_photo
        if config.debug == 'True':
            self.debug = True
        else:
            self.debug = False

        # Select GPU
        self.gpu_num = config.gpu_num

    def config_network(self, model_key, norm='instance', output_channels=1, output_activation='lrelu'):
        # select network model
        # layer specs = (type, out_channels, stride, ksize)
        print(model_key)

        if (model_key == 'alexnet') or (model_key == 'alex'):
            build_func = model.CNN_Encoder
            config = {
                'layer_specs': [('conv', 96, 4, 11), ('pool', 'max', 2, 5), ('conv', 256, 1, 5), ('pool', 'max', 2, 3),
                                ('conv', 384, 1, 3), ('conv', 384, 1, 3), ('conv', 256, 1, 3), ('pool', 'max', 2, 3),
                                ('fc', 4096, 0, 0), ('fc', 4096, 0, 0), ('fc', output_channels, 0, 0)],
                'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                'se-block': self.se_block, 'output_channels': output_channels,
                'output_activation': output_activation}
        elif (model_key == 'patchgan') or (model_key == 'PatchGAN'):
            build_func = model.CNN_Encoder
            #  sigmoid at last layer
            config = {'layer_specs': [('conv', 64, 2, 4), ('conv', 128, 2, 4), ('conv', 256, 2, 4),
                                      ('conv', 512, 2, 4), ('conv', output_channels, 1, 4)],
                      'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'None'):
            build_func = model.CNN_Encoder
            config = {'layer_specs': [('pass', 64, 2, 4)],
                      'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'avg'):
            build_func = model.CNN_Encoder
            config = {'layer_specs': [('pool', 'avg', 1, -1), ('vectorize', 1, 1, 1)],
                      'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'vec'):
            build_func = model.CNN_Encoder
            config = {'layer_specs': [('vectorize', 1, 1, 1)],
                      'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'fc1'):
            build_func = model.CNN_Encoder
            config = {
                'layer_specs': [('fc', output_channels, 0, 0)],
                'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                'se-block': self.se_block, 'output_channels': output_channels,
                'output_activation': output_activation}
        elif (model_key == 'fc2'):
            build_func = model.CNN_Encoder
            config = {
                'layer_specs': [('fc', 1024, 0, 0), ('fc', output_channels, 0, 0)],
                'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                'se-block': self.se_block, 'output_channels': output_channels,
                'output_activation': output_activation}
        elif (model_key == 'fc3'):
            build_func = model.CNN_Encoder
            config = {
                'layer_specs': [('fc', 1024, 0, 0), ('fc', 2048, 0, 0), ('fc', output_channels, 0, 0)],
                'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                'se-block': self.se_block, 'output_channels': output_channels,
                'output_activation': output_activation}
        else:
            assert False, 'Config_network: Wrong model'

        return build_func, config

    def build_network(self, mode='train_with_gan'):
        # config matchnet
        Encoder, self.enc_config = self.config_network(self.encoder_model, self.norm, output_channels=self.feature_dim,
                                                       output_activation='lrelu')
        if (mode == 'train_with_gan') or (mode == 'train_gan') or (mode == 'train_with_fixed_gan') or (mode == 'train_with_fixed_gan_sk') or (mode == 'train_with_g1') or (mode == 'train_with_gan_v2'):
            print('Train_with_gan')
            # encoder
            self.photo_embds = Encoder(self.photo_inp, self.enc_config, self.train_mode, reuse=False,
                                            name='Enc_photo_', share=self.share, share_name="Enc_share_", share_reuse=False)
            self.sketch_embds = Encoder(self.sketch_inp, self.enc_config, self.train_mode, reuse=False,
                                             name='Enc_sketch_', share=self.share, share_name="Enc_share_", share_reuse=True)
            if (self.method == 'adacos') or (self.method == 'Adacos') or (self.method == 'AdaCos'):
                print('AdaCos')
                if (mode == 'train_with_fixed_gan') or (mode == 'train_with_g1'):
                    embds = self.photo_embds
                    identities = self.photo_identity
                    logits, softmax, preds, self.cos_med, self.dynamic_s, self.classify_weights, self.B_avg = model.AdaCos(embds, identities, self.num_identity, reuse=False, scope='Net_sketch_logits', is_dynamic=True)
                    self.photo_logits = logits
                    self.sketch_logits = logits
                    self.photo_softmax = softmax
                    self.sketch_softmax = softmax
                    self.photo_pred = preds
                    self.sketch_pred = preds
                    med_sum = tf.summary.scalar("cos_med", self.cos_med)
                    s_sum = tf.summary.scalar("dynamic_s", self.dynamic_s)
                    B_sum = tf.summary.scalar("B_avg", self.B_avg)
                else:
                    embds = tf.concat([self.photo_embds, self.sketch_embds], axis=0)
                    identities = tf.concat([self.photo_identity, self.sketch_identity], axis=0)
                    logits, softmax, preds, self.cos_med, self.dynamic_s, self.classify_weights, self.B_avg = model.AdaCos(embds, identities, self.num_identity, reuse=False, scope='Net_sketch_logits', is_dynamic=True)
                    self.photo_logits, self.sketch_logits = tf.split(logits, 2, axis=0)
                    self.photo_softmax, self.sketch_softmax = tf.split(softmax, 2, axis=0)
                    self.photo_pred, self.sketch_pred = tf.split(preds, 2, axis=0)
                    med_sum = tf.summary.scalar("cos_med", self.cos_med)
                    s_sum = tf.summary.scalar("dynamic_s", self.dynamic_s)
                    B_sum = tf.summary.scalar("B_avg", self.B_avg)
            else:
                print('build_network: Wrong method')
                assert False
            # loss functions
            if self.train_photo:
                self.loss = tf.reduce_mean(self.photo_softmax, axis=0)
            else:
                self.loss = tf.reduce_mean(softmax, axis=0)

        elif (mode == 'test_with_gan') or (mode == 'test'):
            # encoder
            self.photo_embds = Encoder(self.photo_inp, self.enc_config, self.train_mode, reuse=False,
                                            name='Enc_photo_', share=self.share, share_name="Enc_share_", share_reuse=False)
            self.sketch_embds = Encoder(self.sketch_inp, self.enc_config, self.train_mode, reuse=False,
                                             name='Enc_sketch_', share=self.share, share_name="Enc_share_", share_reuse=True)
            # normalize embds
            self.photo_embds = tf.nn.l2_normalize(self.photo_embds, axis=1, name='normed_embd_photo')
            self.sketch_embds = tf.nn.l2_normalize(self.sketch_embds, axis=1, name='normed_embd_sketch')
            # gallery
            if self.test_gallery:
                self.gallery_embds = Encoder(self.gallery_inp, self.enc_config, self.train_mode, reuse=True,
                                            name='Enc_photo_', share=self.share, share_name="Enc_share_", share_reuse=True)
                self.gallery_embds = tf.nn.l2_normalize(self.gallery_embds, axis=1, name='normed_embd_galls')

        else:
            print('build_network: Wrong mode')
            assert False

        return

    def build_trainer(self, mode='train_with_gan', gan_=None, gan_config=None):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if (mode == 'train_with_gan') or (mode == 'train_gan') or (mode == 'train_with_fixed_gan') or (mode == 'train_with_fixed_gan_sk') or (mode == 'train_with_g1') or (mode == 'train_with_gan_v2'):
            self.photo_batch_size = self.batch_size
            self.sketch_batch_size = self.batch_size
        else:
            print('build_Trainer: Wrong mode 1')
            assert False

        if (self.crop_size != None) and (self.crop_size != False):
            self.image_size = self.crop_size
        else:
            self.image_size = self.input_image_size

        # placeholder
        self.train_mode = tf.placeholder(tf.bool, name='train_mode')

        # Inputs
        if (mode == 'train_with_gan') or (mode == 'train_gan') or (mode == 'train_with_fixed_gan') or (mode == 'train_with_fixed_gan_sk') or (mode == 'train_with_g1') or (mode == 'train_with_gan_v2'):
            if gan_ == None:
                if gan_config is not None:
                    gan_ = gan.GAN(gan_config)
                    gan_.med_domain = True
                    gan_.build_trainer(mode='train_with_matching')
                else:
                    print('build_Trainer: Give GAN or gan_config')
                    assert False
                self.photo_inp = gan_.gen_med_p2s
                self.sketch_inp = gan_.gen_med_s2p
                self.photo_identity = gan_.photo_identity
                self.sketch_identity = gan_.sketch_identity
                self.photo_name = gan_.photo_name
                self.sketch_name = gan_.sketch_name
                self.tr_photo_num = gan_.tr_photo_num
                self.tr_sketch_num = gan_.tr_sketch_num
                self.ts_photo_num = gan_.ts_photo_num
                self.ts_sketch_num = gan_.ts_sketch_num
        else:
            # train batch
            self.tr_photo_inp, self.tr_sketch_inp, self.tr_photo_identity, self.tr_sketch_identity, self.tr_photo_name, self.tr_sketch_name, self.tr_photo_num, self.tr_sketch_num = \
                input_data.photo_sketch_batch_inputs(self.tr_inp_dir, self.tr_txt, self.tr_txt, self.num_identity, 1,
                                                     ['real_db'], self.batch_size, img_size=self.input_image_size,
                                                     name='tr_inp', photo_dim=self.img_channels,
                                                     sketch_dim=self.img_channels,
                                                     flip=self.flip, crop_size=self.crop_size,
                                                     padding_size=self.padding_size,
                                                     random_crop=self.random_crop, train_mode='train_gan',
                                                     concat_sketch_styles=True, log_dir=self.log_dir + '/tr_inputs.txt')
            # test batch
            self.ts_photo_inp, self.ts_sketch_inp, self.ts_photo_identity, self.ts_sketch_identity, self.ts_photo_name, self.ts_sketch_name, self.ts_photo_num, self.ts_sketch_num = \
                input_data.photo_sketch_batch_inputs(self.ts_inp_dir, self.ts_txt, self.ts_txt, self.num_identity, 1,
                                                     ['real_db'], self.ts_batch_size, img_size=self.input_image_size,
                                                     name='ts_inp', photo_dim=self.img_channels,
                                                     sketch_dim=self.img_channels,
                                                     flip=False, crop_size=self.crop_size,
                                                     padding_size=self.padding_size,
                                                     random_crop=False, train_mode='test_gan',
                                                     concat_sketch_styles=True,
                                                     log_dir=self.log_dir + '/ts_inputs.txt')
            '''
            # sketch inputs in case of DA # concat_sketch_styles = False 
            if self.DA:
                self.sketch_inps = self.sketch_inp
                self.sketch_identities = self.sketch_identity
                self.sketch_inp = self.sketch_inps[0]
                self.sketch_identity = self.sketch_identities[0]
                for i in range(1, self.num_style):
                    self.sketch_inp = tf.concat([self.sketch_inp, self.sketch_inps[i]], axis=0)
                    self.sketch_identity = tf.concat([self.sketch_identity, self.sketch_identities[i]], axis=0)
            '''
            # input condition train/test
            self.photo_inp = tf.cond(self.train_mode, lambda: self.tr_photo_inp, lambda: self.ts_photo_inp)
            self.sketch_inp = tf.cond(self.train_mode, lambda: self.tr_sketch_inp, lambda: self.ts_sketch_inp)
            self.photo_identity = tf.cond(self.train_mode, lambda: self.tr_photo_identity, lambda: self.ts_photo_identity)
            self.sketch_identity = tf.cond(self.train_mode, lambda: self.tr_sketch_identity, lambda: self.ts_sketch_identity)
            self.photo_name = tf.cond(self.train_mode, lambda: self.tr_photo_name, lambda: self.ts_photo_name)
            self.sketch_name = tf.cond(self.train_mode, lambda: self.tr_sketch_name, lambda: self.ts_sketch_name)

            print('Photo')
            print(self.photo_inp.get_shape())
            print(self.photo_num)
            print('Sketch')
            print(self.sketch_inp.get_shape())
            print(self.sketch_num)

        epoch = int(self.tr_photo_num / self.batch_size)

        # build network
        self.build_network(mode=mode)

        # print settings
        txtfile = open(self.log_dir + '/' + mode + 'train_enc_config.txt', 'w')
        print(mode, file=txtfile)
        print('# Network settings=======================', file=txtfile)
        print('model: %s' % self.encoder_model, file=txtfile)
        print(self.enc_config, file=txtfile)
        print('weight sharing start at %d' % self.share, file=txtfile)
        print('feature_scale: %f' % self.feature_scale, file=txtfile)
        print('angular_margin: %f' % self.angular_margin, file=txtfile)
        print('cos_margin: %f' % self.cos_margin, file=txtfile)
        print('feature_dimension: %d' % self.feature_dim, file=txtfile)
        print('share: %d' % self.share, file=txtfile)
        print('g_lambda: %f' % self.g_matching_lambda, file=txtfile)
        '''
        print('DA: %r' % self.DA, file=txtfile)
        if self.DA:
            print('domain_lambda_G: %f' % self.domain_lambda_g, file=txtfile)
            print('domain_lambda_D: %f' % self.domain_lambda_d, file=txtfile)
            print('DA_strategy: %s' % self.DA_strategy, file=txtfile)
            print('learning_rate_D: %f' % self.learning_rate_d, file=txtfile)
            print('DA_loss: %s' % self.da_loss, file=txtfile)
            print('DA_model: %s' % self.DA_model, file=txtfile)
            print(self.DA_config, file=txtfile)
        '''
        print('# Input settings=======================', file=txtfile)
        print('(photo) batch_size: %d' % self.batch_size, file=txtfile)
        print('input_image_size: %d' % self.input_image_size, file=txtfile)
        print('padding_size: %d' % self.padding_size, file=txtfile)
        print('crop_size: %d' % self.crop_size, file=txtfile)
        print('random_crop: %r' % self.random_crop, file=txtfile)
        print('photo_dim: %d' % self.photo_dim, file=txtfile)
        print('sketch_dim: %d' % self.sketch_dim, file=txtfile)
        print('flip: %r' % self.flip, file=txtfile)
        print('num_identity: %d' % self.num_identity, file=txtfile)
        print('style_list: %s' % self.style_list, file=txtfile)
        print('# Trainer settings=======================', file=txtfile)
        print('max_epoch: %d' % self.max_epoch, file=txtfile)
        print('learning_rate: %f' % self.learning_rate, file=txtfile)
        print('beta1: %f' % self.beta1, file=txtfile)
        print('beta2: %f' % self.beta2, file=txtfile)
        if self.weight_decay != None:
            print('weight_decay: %f' % self.weight_decay, file=txtfile)
        if self.lr_decay != None:
            print('lr_decay: %f' % self.lr_decay, file=txtfile)
            print('lr_step: %f' % self.lr_step, file=txtfile)
        print('train_photo_only: %r' % self.train_photo)
        '''
        if mode == 'train_sketch':
            print('# Sketch training settings=======================', file=txtfile)
            print('strategy: %s' % self.train_strategy, file=txtfile)
            print('photo_ckpt_dir: %s' % self.photo_ckpt_dir, file=txtfile)
            print('photo_load_epoch: %s' % self.photo_load_epoch, file=txtfile)
        '''
        print("1 epoch: %d iterations" % epoch)
        txtfile.close()

        # add matching loss to gan
        if mode == 'train_with_gan':
            gan_.g_loss = gan_.g_loss + (self.g_matching_lambda * self.loss)
        if (mode == 'train_with_gan') or (mode == 'train_gan')  or (mode == 'train_with_fixed_gan_sk'):# or (mode == 'train_with_fixed_gan') or (mode == 'train_with_g1'):
            gan_.build_optimizer()

        # variable list
        self.t_vars = tf.trainable_variables()
        self.enc_vars = [var for var in self.t_vars if 'Enc_' in var.name]
        if (mode == 'train_with_g1') or (mode == 'train_with_gan_v2'):
            gp_vars = [var for var in self.t_vars if 'Gen_p2s_A_' in var.name]
            gs_vars = [var for var in self.t_vars if 'Gen_s2p_A_' in var.name]
            self.enc_vars = self.enc_vars + gp_vars + gs_vars

        # weight decay
        if (self.weight_decay != None) and (self.weight_decay != False) and (self.weight_decay != 0):
            self.original_loss = self.loss
            pure_loss_sum = tf.summary.scalar("loss_before_wd", self.original_loss)
            self.reg = tf.reduce_sum(
                [tf.nn.l2_loss(var, name='l2_weights') for var in self.enc_vars if 'weights' in var.name], name='l2reg')
            reg_sum = tf.summary.scalar("l2reg", self.reg)
            self.loss = tf.add(self.original_loss, self.weight_decay * self.reg, name='loss')

        loss_sum = tf.summary.scalar("loss", self.loss)

        # learning rate
        if self.lr_decay == None:
            self.lr = self.learning_rate
        else:
            self.lr = tf.train.exponential_decay(self.learning_rate, self.step, self.lr_step * epoch, self.lr_decay,
                                                 staircase=True)

        # Optimization
        if len(self.enc_vars) > 0:
            optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2, name='Adam')
            grad = optimizer.compute_gradients(self.loss, var_list=self.enc_vars)
            self.optim = optimizer.apply_gradients(grad, global_step=self.step)


        if (mode == 'train_with_gan') or (mode == 'train_gan') or (mode == 'train_with_fixed_gan') or (mode == 'train_with_fixed_gan_sk') or (mode == 'train_with_g1') or (mode == 'train_with_gan_v2'):
            return gan_
        else:
            return

    def train_with_gan(self, gan_, mode='train_with_gan'):
        epoch = int(self.tr_photo_num / gan_.batch_size)
        ts_epoch = int(self.ts_photo_num / gan_.ts_batch_size)
        print("1 epoch: %d iterations" % epoch)

        if not os.path.exists(self.log_dir + '/net_ckpt'):
            os.mkdir(self.log_dir + '/net_ckpt')
        if not os.path.exists(self.log_dir + '/net_image'):
            os.mkdir(self.log_dir + '/net_image')
        if not os.path.exists(self.log_dir + '/net_summary'):
            os.mkdir(self.log_dir + '/net_summary')
        if not os.path.exists(self.log_dir + '/net_summary/' + str(self.max_epoch)):
            os.mkdir(self.log_dir + '/net_summary/' + str(self.max_epoch))
        txtfile_m = open(self.log_dir + '/matching_log.txt', 'w')
        txtfile = open(self.log_dir + '/gan_log.txt', 'w')
        txtfile_ts = open(self.log_dir + '/gan_log_ts.txt', 'w')

        print("epoch\ttr_loss\tcos_med\tscale\tB_avg", file=txtfile_m)
        print("epoch\td_loss\tg_loss\tadv_loss\ts_loss\tssim\tcol-loss", file=txtfile)
        print("epoch\td_loss\tg_loss\tadv_loss\ts_loss\tssim\tcol-loss", file=txtfile_ts)

        # initializer
        init_op = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        if self.load_baseline:
            self.b_vars = [var for var in self.t_vars if '_med_' not in var.name]
            self.saver = tf.train.Saver(self.b_vars, max_to_keep=None)
        else:
            self.saver = tf.train.Saver(self.t_vars, max_to_keep=None)
        # Summary
        self.merged_summary = tf.summary.merge_all()

        # feed_dict
        with_gan = False
        if (gan_.med_step is None) or (gan_.med_step <= 10):
            med_lambda = gan_.med_lambda
            gan_.med_step = -1
        else:
            med_lambda = 0
        feed_dict = {self.train_mode: True, gan_.train_mode: True, gan_.med_lambda_p: med_lambda}
        feed_dict_ts = {self.train_mode: False, gan_.train_mode: False, gan_.med_lambda_p: med_lambda}
        # fetch dict
        tr_dict = {}
        ts_dict = {'inp_photo': gan_.photo_inp, 'inp_sketch': gan_.sketch_inp, 'gen_sketch': gan_.gen_sketch,
                   'gen_photo': gan_.gen_photo, 'med_p2s': gan_.gen_med_p2s, 'med_s2p': gan_.gen_med_s2p,
                   'name': gan_.photo_name, 'sk_name': gan_.sketch_name, 'd_loss': gan_.d_loss, 'g_loss': gan_.g_loss,
                   'adv_loss': gan_.adv_loss, 'col_loss': gan_.s_loss_med,
                   's_loss': gan_.s_loss, 'average_ssim': gan_.average_ssim, 'photo_ssim_score': gan_.ssim_score_photo,
                   'sketch_ssim_score': gan_.ssim_score_sketch}
        if (mode == 'train_with_gan') or (mode == 'train_gan'):
            tr_dict.update({'d_loss': gan_.d_loss, 'g_loss': gan_.g_loss, 'adv_loss': gan_.adv_loss, 's_loss': gan_.s_loss,
                            'average_ssim': gan_.average_ssim, 'photo_ssim_score': gan_.ssim_score_photo,
                            'sketch_ssim_score': gan_.ssim_score_sketch, 'name': gan_.photo_name, 'sk_name': gan_.sketch_name,
                            'col_loss': gan_.s_loss_med})
            tr_dict.update({'d_optim': gan_.d_optim, 'g_optim': gan_.g_optim})
            with_gan = True
        if (mode == 'train_with_gan') or (mode == 'train_with_fixed_gan') or (mode == 'train_with_fixed_gan_sk') or (mode == 'train_with_g1') or (mode == 'train_with_gan_v2'):
            tr_dict.update({'loss': self.loss})
            if len(self.enc_vars) > 0:
                tr_dict.update({'optim': self.optim})
            # adacos
            if (self.method == 'adacos') or (self.method == 'Adacos') or (self.method == 'AdaCos'):
                tr_dict.update({'cos_med': self.cos_med, 'dynamic_s': self.dynamic_s, 'B_avg': self.B_avg})

        # ConfigProto
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # Training
        with tf.Session(config=config) as sess:
            print("Start session")
            summary_writer = tf.summary.FileWriter(self.log_dir + '/net_summary/' + str(self.max_epoch), sess.graph)
            if self.continue_training:
                sess.run(init_op)
                self.saver.restore(sess, self.load_dir + "/net-" + str(self.load_epoch))
                print("Model restored from epoch %d." % self.load_epoch)
                if self.from_zero:
                    epoch_i = 0
                else:
                    epoch_i = self.load_epoch
            else:
                epoch_i = 0
                sess.run(init_op)
                print("Initialization done")

            if self.load_baseline:
                self.saver = tf.train.Saver(self.t_vars, max_to_keep=None)

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Queue started")

            # Do training
            print("Start training")
            for i in range(epoch * epoch_i + 1, int(self.max_epoch * epoch) + 1):
                # epoch check
                if i % epoch == 1:
                    epoch_i = epoch_i + 1
                if i == gan_.med_step:
                    feed_dict.update({gan_.med_lambda_p: gan_.med_lambda})
                    feed_dict_ts.update({gan_.med_lambda_p: gan_.med_lambda})
                    print("Use med_similarity now")
                    print("Use med_similarity now", file=txtfile)
                if with_gan and gan_.path_reg:
                    if i % gan_.lazy_reg == 0:
                        if gan_.path_reg:
                            tr_dict.update({'g_optim': gan_.g_reg_optim})
                    elif i % gan_.lazy_reg == 1:
                        if gan_.path_reg:
                            tr_dict.update({'g_optim': gan_.g_optim})

                # Update
                tr_result = sess.run(tr_dict, feed_dict=feed_dict)
                #assert not np.isnan(tr_result['loss']), 'Model diverged with loss = NaN'
                #assert not np.isnan(tr_result['g_loss']), 'Model diverged with g_loss = NaN'
                #assert not np.isnan(tr_result['d_loss']), 'Model diverged with d_loss = NaN'

                # records
                if i % self.summary_step == 0:
                    summary_str = sess.run(self.merged_summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)

                # epoch check
                if i % epoch == 0:
                    # print
                    if epoch_i % self.print_epoch == 0:
                        cur_time = time.localtime(time.time())
                        if mode != 'train_gan':
                            print(
                                "%4d.%02d.%02d %02d:%02d:%02d||epoch %03d. tr_loss: %.5f cos_med: %.5f scale: %.5f B_avg: %.5f" % (
                                cur_time.tm_year, cur_time.tm_mon, cur_time.tm_mday, cur_time.tm_hour, cur_time.tm_min,
                                cur_time.tm_sec, epoch_i, tr_result['loss'], tr_result['cos_med'], tr_result['dynamic_s'],
                                tr_result['B_avg']))
                            print("%03d\t%.5f\t%.5f\t%.5f\t%.5f" % (
                            epoch_i, tr_result['loss'], tr_result['cos_med'], tr_result['dynamic_s'], tr_result['B_avg']),
                                  file=txtfile_m)
                        if (mode == 'train_with_gan') or (mode == 'train_gan'):
                            print(
                                "%d.%d.%d %d:%d:%d||epoch %d. d_loss: %.5f g_loss: %.5f adv_loss: %.5f s_loss: %.5f ssim: %.5f c_loss: %.5f"
                                % (
                                cur_time.tm_year, cur_time.tm_mon, cur_time.tm_mday, cur_time.tm_hour, cur_time.tm_min,
                                cur_time.tm_sec, epoch_i, tr_result['d_loss'], tr_result['g_loss'],
                                tr_result['adv_loss'],
                                tr_result['s_loss'], tr_result['average_ssim'], tr_result['col_loss']))
                            print("%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f"
                                  % (epoch_i, tr_result['d_loss'], tr_result['g_loss'], tr_result['adv_loss'],
                                     tr_result['s_loss'], tr_result['average_ssim'], tr_result['col_loss']),
                                  file=txtfile)
                        print("-------------")

                    # save model
                    if epoch_i % self.save_epoch == 0:
                        save_path = self.saver.save(sess, self.log_dir + '/net_ckpt/net', global_step=epoch_i)
                        print("Model saved in file: %s" % save_path)

                    # visualization
                    if epoch_i % gan_.display_epoch == 0:
                        vis_dir = self.log_dir + '/net_image/ep' + str(epoch_i) + '_iter' + str(i)
                        print("============================")
                        print(self.log_dir)
                        g_loss = 0.
                        d_loss = 0.
                        adv_loss = 0.
                        s_loss = 0.
                        c_loss = 0.
                        ssim = 0.
                        for j in range(ts_epoch):
                            ts_result = sess.run(ts_dict, feed_dict=feed_dict_ts)
                            # for k in range(len(ts_result['name'])):
                            #     assert ts_result['name'][k] == ts_result['sk_name'][k], 'Ts_inputs sequence error'
                            visualize_results(vis_dir, ts_result, gan_.display_list)
                            g_loss = g_loss + ts_result['g_loss']
                            d_loss = d_loss + ts_result['d_loss']
                            adv_loss = adv_loss + ts_result['adv_loss']
                            s_loss = s_loss + ts_result['s_loss']
                            c_loss = c_loss + ts_result['col_loss']
                            ssim = ssim + ts_result['average_ssim']
                        print("%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f"
                              % (epoch_i, d_loss / ts_epoch, g_loss / ts_epoch, adv_loss / ts_epoch, s_loss / ts_epoch,
                                 ssim / ts_epoch, c_loss / ts_epoch), file=txtfile_ts)
                        print("epoch %d test|| d_loss: %.5f g_loss: %.5f adv_loss: %.5f s_loss: %.5f ssim: %.5f c_loss: %.5f"
                              % (epoch_i, d_loss / ts_epoch, g_loss / ts_epoch, adv_loss / ts_epoch, s_loss / ts_epoch,
                                 ssim / ts_epoch, c_loss / ts_epoch))
                        # display
                        write_html(vis_dir, gan_.display_list)
                        print("============================")

            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)
            sess.close()
        txtfile.close()
        txtfile_ts.close()

        return

    def build_inference(self, mode='test_with_gan', gan_=None, gan_config=None):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if (self.crop_size != None) and (self.crop_size != False):
            self.image_size = self.crop_size
        else:
            self.image_size = self.input_image_size
        # placeholder
        self.train_mode = tf.placeholder(tf.bool, name='train_mode')

        # Inputs
        if mode == 'test_with_gan':
            if gan_ == None:
                if gan_config is not None:
                    gan_ = gan.GAN(gan_config)
                    gan_.med_domain = True
                    gan_.build_inference(mode='test_with_matching', test_gallery=self.test_gallery, gallery_txt=self.gallery_txt)
                else:
                    print('build_Trainer: Give GAN or gan_config')
                    assert False

                self.photo_inp = gan_.gen_med_p2s
                self.sketch_inp = gan_.gen_med_s2p
                self.photo_identity = gan_.photo_identity
                self.sketch_identity = gan_.sketch_identity
                self.photo_name = gan_.photo_name
                self.sketch_name = gan_.sketch_name
                self.photo_num = gan_.photo_num
                self.sketch_num = gan_.sketch_num
                ph_embd_num = self.photo_num
                if self.test_gallery:
                    self.gallery_inp = gan_.gallery_med
                    self.gallery_identity = gan_.gallery_identity
                    self.gallery_name = gan_.gallery_name
                    self.gallery_num = gan_.gallery_num
                    self.extended_gall_num = self.photo_num + self.gallery_num
                    ph_embd_num = self.extended_gall_num
                    print('Test with enlarged gallery')
                    print('Gallery is extended to %d' % self.extended_gall_num)
        else:
            # test batch
            self.photo_inp, self.sketch_inp, self.photo_identity, self.sketch_identity, self.photo_name, self.sketch_name, self.photo_num, self.sketch_num = \
                input_data.photo_sketch_batch_inputs(self.ts_inp_dir, self.ts_txt, self.ts_txt, self.num_identity, 1,
                                                     ['real_db'], self.ts_batch_size, img_size=self.input_image_size,
                                                     name='ts_inp', photo_dim=self.img_channels,
                                                     sketch_dim=self.img_channels,
                                                     flip=False, crop_size=self.crop_size,
                                                     padding_size=self.padding_size,
                                                     random_crop=False, train_mode='test_gan',
                                                     concat_sketch_styles=True,
                                                     log_dir=self.log_dir + '/ts_inputs.txt')

            print('Photo')
            print(self.photo_inp.get_shape())
            print(self.photo_num)
            print('Sketch')
            print(self.sketch_inp.get_shape())
            print(self.sketch_num)
            ph_embd_num = self.photo_num

        if self.test_gallery:
            self.log_folder = '/test_results_gall' + str(self.extended_gall_num)
        else:
            self.log_folder = '/test_results'
        if not os.path.exists(self.log_dir + self.log_folder):
            os.mkdir(self.log_dir + self.log_folder)

        # build network
        self.build_network(mode=mode)

        # embd placeholder
        self.photo_embd = tf.placeholder(tf.float32, shape=(ph_embd_num, self.feature_dim), name='photo_inp')
        self.sketch_embd = tf.placeholder(tf.float32, shape=(1, self.feature_dim), name='sketch_inp')

        # cos distance (large cos value means near distance)
        self.cos_d = tf.matmul(self.sketch_embd, tf.transpose(self.photo_embd), name='cos')
        self.cos_d = tf.reshape(self.cos_d, [-1])
        # self.distance = tf.acos(self.cos, name='distance')
        self.top_k_values, self.top_k_indices = tf.nn.top_k(self.cos_d, ph_embd_num)

        # variable list
        self.t_vars = tf.trainable_variables()

        if (mode == 'test_with_gan'):
            return gan_
        else:
            return

    def test(self, gan_, mode='test_with_gan'):
        sketch_epoch = int(self.sketch_num/self.ts_batch_size)
        assert self.sketch_num % self.ts_batch_size == 0

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(self.t_vars, max_to_keep=None)
        # txt file
        txtlog = open(self.log_dir+self.log_folder+'/result_summary_'+str(self.ts_max_epoch)+'.txt', 'w')
        print("epoch\trank1\trank5\trank10\trank50", file=txtlog)

        # fetch dict
        load_sketch_dict = {'photo_embds': self.photo_embds, 'photo_identity': self.photo_identity, 'photo_name': self.photo_name,
                            'sketch_embds': self.sketch_embds, 'sketch_identity': self.sketch_identity, 'sketch_name': self.sketch_name}
        if self.test_gallery:
            gallery_epoch = int(self.gallery_num/self.ts_batch_size)
            load_gallery_dict = {'gallery_embds': self.gallery_embds, 'gallery_identity': self.gallery_identity, 'gallery_name': self.gallery_name}
        load_feed_dict = {self.train_mode: False, gan_.train_mode: False}

        matching_dict = {'values': self.top_k_values, 'indices': self.top_k_indices}
        matching_feed_dict = {self.train_mode: False, gan_.train_mode: False}

        # ConfigProto
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # Testing
        with tf.Session(config=config) as sess:
            print("Start session")

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Queue started")
            if self.debug:
                self.ts_min_epoch = self.ts_max_epoch
                load_sketch_dict.update({'photo_med': self.photo_inp, 'sketch_med': self.sketch_inp})
                load_gallery_dict.update({'gallery_med': self.gallery_inp})
                ph_file = open(self.log_dir + '/photo_embds.txt', 'w')
                sk_file = open(self.log_dir + '/sketch_embds.txt', 'w')
                gall_file = open(self.log_dir + '/gallery_embds.txt', 'w')
                phm_file = open(self.log_dir + '/photo_meds.txt', 'w')
                skm_file = open(self.log_dir + '/sketch_meds.txt', 'w')
                gallm_file = open(self.log_dir + '/gallery_meds.txt', 'w')
            for ep in range(self.ts_min_epoch, self.ts_max_epoch+1, self.ts_unit_epoch):
                # load model
                print(self.load_dir)
                self.load_epoch = ep
                self.saver.restore(sess, self.load_dir+"/net-"+str(self.load_epoch))
                print("Model restored from epoch %d." % self.load_epoch)

                # load images
                print("load images")
                # load photos and sketch
                load_result = sess.run(load_sketch_dict, load_feed_dict)
                sketch_embds = load_result['sketch_embds']
                sketch_identity = load_result['sketch_identity']
                sketch_name = []
                photo_embds = load_result['photo_embds']
                photo_identity = load_result['photo_identity']
                photo_name = []
                if self.debug:
                    for k in range(self.ts_batch_size):
                        print(load_result['photo_identity'][k], file=ph_file)
                        print(load_result['photo_identity'][k], file=phm_file)
                        print(load_result['photo_embds'][k], file=ph_file)
                        print(load_result['photo_med'][k], file=phm_file)
                        print('#--------------------', file=ph_file)
                        print('#--------------------', file=phm_file)
                        print(load_result['sketch_identity'][k], file=sk_file)
                        print(load_result['sketch_identity'][k], file=skm_file)
                        print(load_result['sketch_embds'][k], file=sk_file)
                        print(load_result['sketch_med'][k], file=skm_file)
                        print('#--------------------', file=sk_file)
                        print('#--------------------', file=skm_file)
                for j in range(len(load_result['sketch_name'])):
                    sketch_name.append(load_result['sketch_name'][j].decode('utf-8'))
                    photo_name.append(load_result['photo_name'][j].decode('utf-8'))
                for i in range(1, sketch_epoch):
                    load_result = sess.run(load_sketch_dict, load_feed_dict)
                    sketch_embds = np.concatenate([sketch_embds, load_result['sketch_embds']], 0)
                    sketch_identity = np.concatenate([sketch_identity, load_result['sketch_identity']], 0)
                    photo_embds = np.concatenate([photo_embds, load_result['photo_embds']], 0)
                    photo_identity = np.concatenate([photo_identity, load_result['photo_identity']], 0)
                    for j in range(len(load_result['sketch_name'])):
                        sketch_name.append(load_result['sketch_name'][j].decode('utf-8'))
                        photo_name.append(load_result['photo_name'][j].decode('utf-8'))
                    if self.debug:
                        for k in range(self.ts_batch_size):
                            print(load_result['photo_identity'][k], file=ph_file)
                            print(load_result['photo_identity'][k], file=phm_file)
                            print(load_result['photo_embds'][k], file=ph_file)
                            print(load_result['photo_med'][k], file=phm_file)
                            print('#--------------------', file=ph_file)
                            print('#--------------------', file=phm_file)
                            print(load_result['sketch_identity'][k], file=sk_file)
                            print(load_result['sketch_identity'][k], file=skm_file)
                            print(load_result['sketch_embds'][k], file=sk_file)
                            print(load_result['sketch_med'][k], file=skm_file)
                            print('#--------------------', file=sk_file)
                            print('#--------------------', file=skm_file)
                photo_name = np.asarray(photo_name)
                sketch_name = np.asarray(sketch_name)
                assert self.photo_num == len(photo_identity)
                assert self.sketch_num == len(sketch_identity)

                # load gallery
                if self.test_gallery:
                    print("load gallery")
                    # load photos
                    load_result = sess.run(load_gallery_dict, load_feed_dict)
                    gallery_embds = load_result['gallery_embds']
                    gallery_identity = load_result['gallery_identity']
                    gallery_name = []
                    if self.debug:
                        for k in range(self.ts_batch_size):
                            print(load_result['gallery_identity'][k], file=gall_file)
                            print(load_result['gallery_identity'][k], file=gallm_file)
                            print(load_result['gallery_embds'][k], file=gall_file)
                            print(load_result['gallery_med'][k], file=gallm_file)
                            print('#--------------------', file=gall_file)
                            print('#--------------------', file=gallm_file)
                    for j in range(len(load_result['gallery_name'])):
                        gallery_name.append(load_result['gallery_name'][j].decode('utf-8'))
                    for i in range(1, gallery_epoch):
                        load_result = sess.run(load_gallery_dict, load_feed_dict)
                        gallery_embds = np.concatenate([gallery_embds, load_result['gallery_embds']], 0)
                        gallery_identity = np.concatenate([gallery_identity, load_result['gallery_identity']], 0)
                        for j in range(len(load_result['gallery_name'])):
                            gallery_name.append(load_result['gallery_name'][j].decode('utf-8'))
                    gallery_name = np.asarray(gallery_name)
                    if self.debug:
                        for k in range(self.ts_batch_size):
                            print(load_result['gallery_identity'][k], file=gall_file)
                            print(load_result['gallery_identity'][k], file=gallm_file)
                            print(load_result['gallery_embds'][k], file=gall_file)
                            print(load_result['gallery_med'][k], file=gallm_file)
                            print('#--------------------', file=gall_file)
                            print('#--------------------', file=gallm_file)
                    assert self.gallery_num == len(gallery_identity)
                    # Add gallery to photo
                    photo_embds = np.concatenate([photo_embds, gallery_embds], axis=0)
                    photo_identity = np.concatenate([photo_identity, gallery_identity], axis=0)
                    photo_name = np.concatenate([photo_name, gallery_name], axis=0)
                if self.debug:
                    ph_file.close()
                    sk_file.close()
                    gall_file.close()
                    phm_file.close()
                    skm_file.close()
                    gallm_file.close()
                    exit()

                # matching
                print("matching test")
                txtfile = open(self.log_dir+self.log_folder+'/result_ep'+str(self.load_epoch)+'.txt', 'w')
                print("name\tr_gt\td_gt\td_max\trank1\trank2\trank3\trank4\trank5", file=txtfile)
                matching_feed_dict.update({self.photo_embd: photo_embds})
                r1 = 0
                r5 = 0
                r10 = 0
                r50 = 0
                for i in range(self.sketch_num):
                    matching_feed_dict.update({self.sketch_embd: [sketch_embds[i]]})
                    matching_result = sess.run(matching_dict, matching_feed_dict)
                    top_k = photo_identity[matching_result['indices']]
                    gt = np.argwhere(top_k==sketch_identity[i])[0][0]
                    if gt < 1:
                        r1 += 1
                    if gt < 5:
                        r5 += 1
                    if gt < 10:
                        r10 += 1
                    if gt < 50:
                        r50 += 1
                    top5_name = photo_name[matching_result['indices'][:5]]
                    print("%s\t%d\t%f\t%f\t%s\t%s\t%s\t%s\t%s" % (sketch_name[i], gt, matching_result['values'][gt], matching_result['values'][0], top5_name[0], top5_name[1], top5_name[2], top5_name[3], top5_name[4]), file=txtfile)

                r1_accuracy = r1/self.sketch_num
                r5_accuracy = r5/self.sketch_num
                r10_accuracy = r10/self.sketch_num
                r50_accuracy = r50/self.sketch_num
                print('rank1_accuracy:\t%.3f\nrank5_accuracy:\t%.3f' % (r1_accuracy, r5_accuracy), file=txtfile)
                print('rank10_accuracy:\t%.3f\nrank50_accuracy:\t%.3f' % (r10_accuracy, r50_accuracy), file=txtfile)
                txtfile.close()
                print('rank1_accuracy:\t%.3f\nrank5_accuracy:\t%.3f' % (r1_accuracy, r5_accuracy))
                print('rank10_accuracy:\t%.3f\nrank50_accuracy:\t%.3f' % (r10_accuracy, r50_accuracy))
                print("%d\t%.3f\t%.3f\t%.3f\t%.3f" % (self.load_epoch, r1_accuracy, r5_accuracy, r10_accuracy, r50_accuracy), file=txtlog)

            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)
            sess.close()
        txtlog.close()

        return


    def test_gan(self, gan_):
        ts_epoch = int(gan_.photo_num / gan_.ts_batch_size)

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(self.t_vars, max_to_keep=None)

        # feed_dict
        feed_dict_ts = {gan_.train_mode: False, self.train_mode: False}
        ts_dict = {'inp_photo': gan_.photo_inp, 'inp_sketch': gan_.sketch_inp, 'gen_sketch': gan_.gen_sketch,
                   'gen_photo': gan_.gen_photo, 'med_p2s': gan_.gen_med_p2s, 'med_s2p': gan_.gen_med_s2p,
                   'name': gan_.photo_name, 'sk_name': gan_.sketch_name}

        # ConfigProto
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # Training
        with tf.Session(config=config) as sess:
            print("Start session")
            epoch_i = self.load_epoch
            self.saver.restore(sess, self.load_dir + "/net-" + str(self.load_epoch))
            print("Model restored from epoch %d." % self.load_epoch)

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Queue started")
            print("Run GAN")

            # visualization
            vis_dir = self.log_dir + '/gan_test_image_ep' + str(epoch_i)
            print("============================")
            print(self.log_dir)
            for j in range(ts_epoch):
                ts_result = sess.run(ts_dict, feed_dict=feed_dict_ts)
                # for k in range(len(ts_result['name'])):
                #     assert ts_result['name'][k] == ts_result['sk_name'][k], 'Ts_inputs sequence error'
                visualize_results(vis_dir, ts_result, gan_.display_list)
            # display
            write_html(vis_dir, gan_.display_list)
            print("============================")


            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)
            sess.close()

        return

    def synthesis(self, gan_):
        epoch = int(self.photo_num / gan_.ts_batch_size)
        print("1 epoch: %d iterations" % epoch)

        if not os.path.exists(self.log_dir + '/tr_syn'):
            os.mkdir(self.log_dir + '/tr_syn')

        # initializer
        init_op = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        if self.load_baseline:
            self.b_vars = [var for var in self.t_vars if '_med_' not in var.name]
            self.saver = tf.train.Saver(self.b_vars, max_to_keep=None)
        else:
            self.saver = tf.train.Saver(self.t_vars, max_to_keep=None)

        # feed_dict
        feed_dict = {self.train_mode: False, gan_.train_mode: False}
        # fetch dict
        syn_dict = {'inp_photo': gan_.photo_inp, 'inp_sketch': gan_.sketch_inp, 'gen_sketch': gan_.gen_sketch,
                   'gen_photo': gan_.gen_photo, 'med_p2s': gan_.gen_med_p2s, 'med_s2p': gan_.gen_med_s2p,
                   'name': gan_.photo_name, 'sk_name': gan_.sketch_name}

        # ConfigProto
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # Training
        with tf.Session(config=config) as sess:
            print("Start session")

            sess.run(init_op)
            self.saver.restore(sess, self.load_dir + "/net-" + str(self.load_epoch))
            print("Model restored from epoch %d." % self.load_epoch)
            epoch_i = 0

            if self.load_baseline:
                self.saver = tf.train.Saver(self.t_vars, max_to_keep=None)

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Queue started")

            # visualization
            vis_dir = self.log_dir + '/tr_syn'
            print(vis_dir)
            print("============================")
            #print("tr_data")
            for j in range(epoch+1):
                syn_result = sess.run(syn_dict, feed_dict=feed_dict)
                # for k in range(len(ts_result['name'])):
                #     assert ts_result['name'][k] == ts_result['sk_name'][k], 'Ts_inputs sequence error'
                visualize_results(vis_dir, syn_result, ['inp_photo', 'gen_sketch'], ['photo', 'real_db'])
                if j % 100 == 0:
                    print('%d/%d' % (j, epoch+1))
            # print("ts_data")
            # for j in range(ts_epoch):
            #     syn_result = sess.run(syn_dict, feed_dict=feed_dict_ts)
            #     # for k in range(len(ts_result['name'])):
            #     #     assert ts_result['name'][k] == ts_result['sk_name'][k], 'Ts_inputs sequence error'
            #     visualize_results(vis_dir, syn_result, ['inp_photo', 'gen_sketch'], ['photo', 'real_db'])
            print("Done")

            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)
            sess.close()
        return
