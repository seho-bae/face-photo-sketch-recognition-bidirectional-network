#!/bin/bash
gpu=0
#================================================================================================

dir="StyleGAN1"
g_model="res4"	#we don't use
d_model="PatchGAN"
use_enc_dec="True"
g_enc_model='alex'
g_dec_model='style_gen'
enc_model="None" #classifier
med_channels=512
share_g1='True'

max0=3000
max1=50
max2=3500
batch_size=8
step2_size=32
lambda=10 #lambda s
med_lambda=1  #lambda w
g_lambda=1  #lambda GAN
enc_lr=0.0005 #learning rate for CelebA
simmilarity_loss='L1'
simmilarity_loss_med='L1'

enc_norm='instance'
g_norm='instance'	#we don't use
g_enc_norm='instance'
g_dec_norm='instance'
d_norm='instance'

d_method='cgan' #discriminator method

cross_gen=False #p2p and s2s
s_lambda_cross=0
gen_mode='all'

debug=False

path_reg=False

share_g2=False

#=============================================================================================
mkdir $dir
#=============================================================================================

python3 train_step_by_step.py --gpu_num=$gpu --log_dir=$dir --tr_dir="../data/CUFS" --ts_dir="../data/CUFS"  --tr_list='tr_list_cufs.txt' --ts_list='ts_list_cufs.txt' --max_epoch=$max0 --g_model=$g_model --d_model=$d_model --batch_size=$batch_size --ts_batch_size=$batch_size --num_identity=10177 --similarity_lambda=$lambda --med_lambda=$med_lambda --med_step=0 --enc_model=$enc_model --train_mode='train_gan' --save_epoch=500 --g_matching_lambda=$g_lambda --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

python3 train_step_by_step.py --gpu_num=$gpu --log_dir=$dir --tr_dir="../data/CelebA" --ts_dir="../data/CelebA" --tr_list='tr_list_celeba.txt' --ts_list='ts_list_celeba.txt' --max_epoch=$max1 --g_model=$g_model --d_model=$d_model --similarity_lambda=$lambda --med_lambda=0 --med_step=0 --enc_model=$enc_model --train_mode='train_with_g1' --load_epoch=$max0 --batch_size=$step2_size --ts_batch_size=$step2_size --num_identity=10177 --img_size=256 --print_epoch=1 --save_epoch=25 --g_matching_lambda=$g_lambda --enc_lr=$enc_lr --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

#--------------------------------------------------------------------------------------
db='F1'

python3 train_step_by_step.py --gpu_num=$gpu --log_dir=$dir --tr_list='list_'$db'.txt' --ts_list='ts_list_'$db'.txt' --max_epoch=$max2 --g_model=$g_model --d_model=$d_model --batch_size=$batch_size --ts_batch_size=$batch_size --similarity_lambda=$lambda --enc_model=$enc_model --train_mode='train_with_gan' --load_epoch=$max1 --num_identity=10177 --g_matching_lambda=$g_lambda --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --med_lambda=$med_lambda --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

python3 test_with_gan.py --gpu_num=$gpu --log_dir=$dir'/step2_'$db --tr_list='tr_list_'$db'.txt' --ts_list='ts_list_'$db'.txt' --max_epoch=$max2 --g_model=$g_model --d_model=$d_model --batch_size=1 --ts_batch_size=1 --similarity_lambda=$lambda --enc_model=$enc_model --num_identity=10177 --g_matching_lambda=$g_lambda --ts_max_epoch=$max2 --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --med_lambda=$med_lambda --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

python3 test_with_gan.py --gpu_num=$gpu --log_dir=$dir'/step2_'$db --tr_list='tr_list_'$db'.txt' --ts_list='ts_list_'$db'.txt' --max_epoch=$max2 --g_model=$g_model --d_model=$d_model --batch_size=1 --ts_batch_size=1 --similarity_lambda=$lambda --enc_model=$enc_model --num_identity=10177 --g_matching_lambda=$g_lambda --ts_max_epoch=$max2 --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --use_gallery=1 --gall_list='list_gallery_1500.txt' --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --med_lambda=$med_lambda --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

python3 test_with_gan.py --gpu_num=$gpu --log_dir=$dir'/step2_'$db --tr_list='tr_list_'$db'.txt' --ts_list='ts_list_'$db'.txt' --max_epoch=$max2 --g_model=$g_model --d_model=$d_model --batch_size=1 --ts_batch_size=1 --similarity_lambda=$lambda --enc_model=$enc_model --num_identity=10177 --g_matching_lambda=$g_lambda --ts_max_epoch=$max2 --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --use_gallery=1 --gall_list='list_gallery_10000.txt' --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --med_lambda=$med_lambda --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

rm -r $dir'/step2_'$db'/net_ckpt'
#--------------------------------------------------------------------------------------
db='I1'

python3 train_step_by_step.py --gpu_num=$gpu --log_dir=$dir --tr_list='tr_list_'$db'.txt' --ts_list='ts_list_'$db'.txt' --max_epoch=$max2 --g_model=$g_model --d_model=$d_model --batch_size=$batch_size --ts_batch_size=$batch_size --similarity_lambda=$lambda --enc_model=$enc_model --train_mode='train_with_gan' --load_epoch=$max1 --num_identity=10177 --g_matching_lambda=$g_lambda --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --med_lambda=$med_lambda --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

python3 test_with_gan.py --gpu_num=$gpu --log_dir=$dir'/step2_'$db --tr_list='tr_list_'$db'.txt' --ts_list='ts_list_'$db'.txt' --max_epoch=$max2 --g_model=$g_model --d_model=$d_model --batch_size=1 --ts_batch_size=1 --similarity_lambda=$lambda --enc_model=$enc_model --num_identity=10177 --g_matching_lambda=$g_lambda --ts_max_epoch=$max2 --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --med_lambda=$med_lambda --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

python3 test_with_gan.py --gpu_num=$gpu --log_dir=$dir'/step2_'$db --tr_list='tr_list_'$db'.txt' --ts_list='ts_list_'$db'.txt' --max_epoch=$max2 --g_model=$g_model --d_model=$d_model --batch_size=1 --ts_batch_size=1 --similarity_lambda=$lambda --enc_model=$enc_model --num_identity=10177 --g_matching_lambda=$g_lambda --ts_max_epoch=$max2 --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --use_gallery=1 --gall_list='list_gallery_1500.txt' --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --med_lambda=$med_lambda --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

python3 test_with_gan.py --gpu_num=$gpu --log_dir=$dir'/step2_'$db --tr_list='tr_list_'$db'.txt' --ts_list='ts_list_'$db'.txt' --max_epoch=$max2 --g_model=$g_model --d_model=$d_model --batch_size=1 --ts_batch_size=1 --similarity_lambda=$lambda --enc_model=$enc_model --num_identity=10177 --g_matching_lambda=$g_lambda --ts_max_epoch=$max2 --similarity_loss=$simmilarity_loss --enc_norm=$enc_norm --similarity_loss_med=$simmilarity_loss_med --use_gallery=1 --gall_list='list_gallery_10000.txt' --d_method=$d_method --share_g1=$share_g1 --use_enc_dec=$use_enc_dec --g_enc_model=$g_enc_model --g_dec_model=$g_dec_model --med_channels=$med_channels --cross_gen=$cross_gen --s_lambda_cross=$s_lambda_cross --gen_mode=$gen_mode --med_lambda=$med_lambda --g_norm=$g_norm --d_norm=$d_norm --g_enc_norm=$g_enc_norm --g_dec_norm=$g_dec_norm --debug=$debug --share_g2=$share_g2 --path_reg=$path_reg

rm -r $dir'/step2_'$db'/net_ckpt'
#--------------------------------------------------------------------------------------
