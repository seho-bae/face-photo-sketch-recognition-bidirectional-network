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
parser.add_argument('--med_lambda', type=float, default=0.1)
parser.add_argument('--med_step', type=int, default=0)
# inputs
parser.add_argument('--img_size', type=int, default=272)
parser.add_argument('--batch_size', type=int, default=1)
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
parser.add_argument('--ts_batch_size', type=int, default=1)
parser.add_argument('--ts_min_epoch', type=int, default=100)
parser.add_argument('--ts_max_epoch', type=int, default=5000)
parser.add_argument('--ts_unit_epoch', type=int, default=100)
parser.add_argument('--use_gallery', type=bool, default=False)
parser.add_argument('--gall_list', type=str, default='list_gallery_1500.txt')
#d_meds
#parser.add_argument('--d_p2s', type=str, default='False')
#parser.add_argument('--d_s2p', type=str, default='False')
#parser.add_argument('--med_d_lambda', type=float, default=0)
parser.add_argument('--share_g1', type=str, default='False')
parser.add_argument('--noise_len', type=int, default=0)
parser.add_argument('--cross_gen', type=str, default='False')
parser.add_argument('--s_lambda_cross', type=float, default=0)
#test_mode
parser.add_argument('--test_mode', type=str, default='matching')

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
config.load_dir = config.log_dir + '/net_ckpt'
config.ts_dir = config.tr_dir
config.train_photo = False
# Build
net = matching.matchnet(config)

if net.gpu_num is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(net.gpu_num)
print(device_lib.list_local_devices())

gan_ = net.build_inference(mode='test_with_gan', gan_config=config)

# Test
if (config.test_mode == 'gan') or (config.test_mode == 'syn') or (config.test_mode == 'gen') or (config.test_mode == 'synthesis'):
    net.test_gan(gan_)
else:
    net.test(gan_)
