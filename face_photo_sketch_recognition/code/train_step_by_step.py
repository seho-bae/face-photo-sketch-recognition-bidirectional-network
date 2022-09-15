import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import argparse

import matching
import gan
from utils import *

parser = argparse.ArgumentParser()
# training
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='record')
parser.add_argument('--load_dir', type=str, default='record')
parser.add_argument('--continue_tr', type=bool, default=False)
parser.add_argument('--load_epoch', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=5000)
parser.add_argument('--similarity_loss', type=str, default='L1')
parser.add_argument('--similarity_lambda', type=float, default=10)
parser.add_argument('--similarity_loss_med', type=str, default='L1')
parser.add_argument('--med_lambda', type=float, default=1)
parser.add_argument('--med_step', type=int, default=0)
# inputs
parser.add_argument('--img_size', type=int, default=272)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--tr_dir', type=str, default="../data/DB272prip")
parser.add_argument('--ts_dir', type=str, default="../data/DB272prip")
parser.add_argument('--tr_list', type=str, default='tr_list.txt')
parser.add_argument('--ts_list', type=str, default='ts_list.txt')
parser.add_argument('--num_identity', type=int, default=48)
# network
parser.add_argument('--g_model', type=str, default='col_gen')
parser.add_argument('--d_model', type=str, default='PatchGan')
parser.add_argument('--d_method', type=str, default='col-cgan')
#201005
parser.add_argument('--use_enc_dec', type=str, default='True')
parser.add_argument('--g_enc_model', type=str, default='col_gen_enc') # col_gen_enc, col_gen_short
parser.add_argument('--g_dec_model', type=str, default='col_gen_dec') # col_gen_dec, col_gen_short
parser.add_argument('--med_channels', type=int, default=256)
#matching
parser.add_argument('--enc_model', type=str, default='alex')
parser.add_argument('--enc_norm', type=str, default='batch')
parser.add_argument('--g_matching_lambda', type=float, default=1)
#learning rate
parser.add_argument('--enc_lr', type=float, default=0.0002)
parser.add_argument('--g_lr', type=float, default=0.0002)
parser.add_argument('--d_lr', type=float, default=0.0002)
#record
parser.add_argument('--print_epoch', type=int, default=10)
parser.add_argument('--save_epoch', type=int, default=100)
#test
parser.add_argument('--ts_batch_size', type=int, default=4)
parser.add_argument('--ts_min_epoch', type=int, default=100)
parser.add_argument('--ts_max_epoch', type=int, default=5000)
parser.add_argument('--ts_unit_epoch', type=int, default=100)
parser.add_argument('--use_gallery', type=bool, default=False)
parser.add_argument('--gall_list', type=str, default='list_gallery_1500.txt')
#train mode
parser.add_argument('--train_mode', type=str, default='train_with_gan')
#d_meds
#parser.add_argument('--d_p2s', type=str, default='False')
#parser.add_argument('--d_s2p', type=str, default='False')
#parser.add_argument('--med_d_lambda', type=float, default=0)
parser.add_argument('--share_g1', type=str, default='False')
parser.add_argument('--cross_gen', type=str, default='False')
parser.add_argument('--s_lambda_cross', type=float, default=0)
#load
parser.add_argument('--load_baseline', type=str, default='False')
parser.add_argument('--noise_len', type=int, default=0)

parser.add_argument('--g_norm', type=str, default='instance')
parser.add_argument('--d_norm', type=str, default='instance')
parser.add_argument('--g_enc_norm', type=str, default='g')
parser.add_argument('--g_dec_norm', type=str, default='g')

parser.add_argument('--recon_lambda', type=float, default=0)
parser.add_argument('--gen_mode', type=str, default='all')
parser.add_argument('--debug', type=str, default='False')

parser.add_argument('--share_g2', type=str, default='False')

parser.add_argument('--path_reg', type=str, default='False')

config = parser.parse_args()
config.ts_dir = config.tr_dir
config.train_photo = False
if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

# mode
from_zero = True
synthesis_mode = False
if config.train_mode == 'train_gan':
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step0/net_ckpt'
        from_zero = False
    config.log_dir = config.log_dir + '/step0'
elif config.train_mode == 'train_gan_as_step1':
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step1/net_ckpt'
        from_zero = False
    config.log_dir = config.log_dir + '/step1'
elif config.train_mode == 'train_with_fixed_gan':
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step1/net_ckpt'
        from_zero = False
    else:
        config.continue_tr = True
        config.load_dir = config.log_dir + '/step0/net_ckpt'
    config.log_dir = config.log_dir + '/step1'
elif config.train_mode == 'train_with_g1':
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step1/net_ckpt'
        from_zero = False
    else:
        config.continue_tr = True
        config.load_dir = config.log_dir + '/step0/net_ckpt'
    config.log_dir = config.log_dir + '/step1'
elif config.train_mode == 'synthesis':
    config.train_mode = 'test_with_gan'
    synthesis_mode = True
    config.continue_tr = True
    config.load_dir = config.log_dir + '/step0/net_ckpt'
    config.log_dir = config.log_dir + '/step1'
    config.ts_list = config.tr_list
elif config.train_mode == 'train_with_g1_from_scratch':
    config.train_mode = 'train_with_g1'
    config.continue_tr = False
    config.log_dir = config.log_dir + '/step1'
elif config.train_mode == 'train_with_g1_as_step0':
    config.train_mode = 'train_with_g1'
    config.continue_tr = False
    config.log_dir = config.log_dir + '/step0'
elif config.train_mode == 'train_with_gan_pre1':
    config.train_mode = 'train_with_gan'
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step1/net_ckpt'
        from_zero = False
    else:
        config.continue_tr = True
        config.load_dir = config.log_dir + '/step0/net_ckpt'
    config.log_dir = config.log_dir + '/step1'
elif config.train_mode == 'train_with_gan_pre0':
    config.train_mode = 'train_with_gan'
    config.continue_tr = False
    config.log_dir = config.log_dir + '/step0'
elif config.train_mode == 'train_with_gan_as_step1':
    config.train_mode = 'train_with_gan'
    config.continue_tr = False
    config.log_dir = config.log_dir + '/step1'
elif config.train_mode == 'train_with_gan_photo':
    config.train_mode = 'train_with_gan'
    config.train_photo = True
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step1/net_ckpt'
        from_zero = False
    else:
        config.continue_tr = True
        config.load_dir = config.log_dir + '/step0/net_ckpt'
    config.log_dir = config.log_dir + '/step1'
elif config.train_mode == 'train_with_gan_step1':
    config.train_mode = 'train_with_gan'
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step1/net_ckpt'
        from_zero = False
    else:
        config.continue_tr = True
        config.load_dir = config.log_dir + '/step0/net_ckpt'
    config.log_dir = config.log_dir + '/step1'
elif config.train_mode == 'train_gan_step1':
    config.train_mode = 'train_gan'
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step1/net_ckpt'
        from_zero = False
    else:
        config.continue_tr = True
        config.load_dir = config.log_dir + '/step0/net_ckpt'
    config.log_dir = config.log_dir + '/step1'
elif (config.train_mode == 'train_with_gan') or (config.train_mode == 'train_with_gan_from_scratch'):
    if config.train_mode == 'train_with_gan_from_scratch':
        config.continue_tr = False
        config.train_mode = 'train_with_gan'
    else:
        config.continue_tr = True
    config.load_dir = config.log_dir + '/step1/net_ckpt'
    if config.tr_list == 'tr_list_F.txt':
        config.log_dir = config.log_dir + '/step2_F'
    elif config.tr_list == 'tr_list_I.txt':
        config.log_dir = config.log_dir + '/step2_I'
    elif config.tr_list == 'tr_list_F1.txt':
        config.log_dir = config.log_dir + '/step2_F1'
    elif config.tr_list == 'tr_list_I1.txt':
        config.log_dir = config.log_dir + '/step2_I1'
    elif config.tr_list == 'tr_list_F2.txt':
        config.log_dir = config.log_dir + '/step2_F2'
    elif config.tr_list == 'tr_list_I2.txt':
        config.log_dir = config.log_dir + '/step2_I2'
    elif config.tr_list == 'tr_list_F3.txt':
        config.log_dir = config.log_dir + '/step2_F3'
    elif config.tr_list == 'tr_list_I3.txt':
        config.log_dir = config.log_dir + '/step2_I3'
    elif config.tr_list == 'tr_list_F4.txt':
        config.log_dir = config.log_dir + '/step2_F4'
    elif config.tr_list == 'tr_list_I4.txt':
        config.log_dir = config.log_dir + '/step2_I4'
    elif config.tr_list == 'tr_list_F5.txt':
        config.log_dir = config.log_dir + '/step2_F5'
    elif config.tr_list == 'tr_list_I5.txt':
        config.log_dir = config.log_dir + '/step2_I5'
    else:
        config.log_dir = config.log_dir + '/step2'
else:
    print('Wrong train mode')
    print(config.train_mode)
    assert False

# Build
net = matching.matchnet(config, from_zero)
if config.load_baseline == 'True':
    net.load_baseline = True

if net.gpu_num is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(net.gpu_num)
print(device_lib.list_local_devices())

if synthesis_mode:
    gan_ = net.build_inference(mode='test_with_gan', gan_config=config)
else:
    gan_ = net.build_trainer(mode=config.train_mode, gan_config=config)

# Syn first
if synthesis_mode:
    net.synthesis(gan_)
    exit()

# Variables
if not os.path.exists(net.log_dir):
    os.mkdir(net.log_dir)
txtfile = open(net.log_dir+'/variables.txt', 'w')
print("Enc_vars")
for i in range(len(net.enc_vars)):
    print(net.enc_vars[i].name)
    print(net.enc_vars[i].name, file=txtfile)
print("Gen_vars")
for i in range(len(gan_.Gen_vars)):
    print(gan_.Gen_vars[i].name)
    print(gan_.Gen_vars[i].name, file=txtfile)
print("Dis_vars")
for i in range(len(gan_.Dis_vars)):
    print(gan_.Dis_vars[i].name)
    print(gan_.Dis_vars[i].name, file=txtfile)
print("build model finished")
print("Tr_num: "+str(net.tr_photo_num))
print("Ts_num: "+str(net.ts_photo_num))
txtfile.close()

# Train
net.train_with_gan(gan_, mode=config.train_mode)
