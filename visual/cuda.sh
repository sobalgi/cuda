#!/bin/sh
###CUDA
#nohup python SB_main_00.py --dataroot ./data --exp=usps_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00.py --dataroot ./data --exp=mnist_usps --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00.py --dataroot ./data --exp=svhn_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00.py --dataroot ./data --exp=mnist_svhn --batch_size=128 --epoch_size=large --network_type=stn --gpus 1,2,3 --use_tied_gen --workers 6  &  wait 2 &  #
#nohup python SB_main_00.py --dataroot ./data --exp=cifar_stl --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00.py --dataroot ./data --exp=stl_cifar --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00.py --dataroot ./data --exp=syndigits_svhn --batch_size=128 --epoch_size=large --network_type=stn --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00.py --dataroot ./data --exp=synsigns_gtsrb --batch_size=128 --epoch_size=small --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#
#nohup python SB_main_00.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#
###Source only using supervised loss, test Target
#nohup python SB_main_00_ss.py --dataroot ./data --exp=usps_mnist --batch_size=128 --epoch_size=source --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss.py --dataroot ./data --exp=mnist_usps --batch_size=128 --epoch_size=source --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss.py --dataroot ./data --exp=svhn_mnist --batch_size=128 --epoch_size=source --network_type=se --gpus 0,1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss.py --dataroot ./data --exp=mnist_svhn --batch_size=128 --epoch_size=source --network_type=stn --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss.py --dataroot ./data --exp=cifar_stl --batch_size=128 --epoch_size=source --network_type=se --gpus 0,1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss.py --dataroot ./data --exp=stl_cifar --batch_size=128 --epoch_size=source --network_type=se --gpus 0,1,2 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss.py --dataroot ./data --exp=syndigits_svhn --batch_size=128 --epoch_size=source --network_type=stn --gpus 0,1 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss.py --dataroot ./data --exp=synsigns_gtsrb --batch_size=128 --epoch_size=source --network_type=se --gpus 2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss.py --dataroot ./data --exp=amazon_dslr --batch_size=32 --epoch_size=source --network_type=resnet --gpus 1,2,3 --workers 6  &  wait 2 &  #
#
#nohup python SB_main_00_ss.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#
###Target only using supervised loss, test Target
#nohup python SB_main_00_ts.py --dataroot ./data --exp=usps_mnist --batch_size=128 --epoch_size=source --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ts.py --dataroot ./data --exp=mnist_usps --batch_size=128 --epoch_size=source --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ts.py --dataroot ./data --exp=svhn_mnist --batch_size=128 --epoch_size=source --network_type=se --gpus 0,1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ts.py --dataroot ./data --exp=mnist_svhn --batch_size=128 --epoch_size=source --network_type=stn --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ts.py --dataroot ./data --exp=cifar_stl --batch_size=128 --epoch_size=source --network_type=se --gpus 0,1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ts.py --dataroot ./data --exp=stl_cifar --batch_size=128 --epoch_size=source --network_type=se --gpus 0,1,2 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ts.py --dataroot ./data --exp=syndigits_svhn --batch_size=128 --epoch_size=source --network_type=stn --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ts.py --dataroot ./data --exp=synsigns_gtsrb --batch_size=128 --epoch_size=source --network_type=se --gpus 1,2,3,0 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ts.py --dataroot ./data --exp=amazon_dslr --batch_size=32 --epoch_size=source --network_type=resnet --gpus 1,2,3 --workers 6  &  wait 2 &  #
#
#nohup python SB_main_00_ts.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ts.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ts.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ts.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ts.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ts.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#
#Supervised source and unsupervised target
#nohup python SB_main_00_ss_tu.py --dataroot ./data --exp=usps_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 0,1 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu.py --dataroot ./data --exp=mnist_usps --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu.py --dataroot ./data --exp=svhn_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu.py --dataroot ./data --exp=mnist_svhn --batch_size=128 --epoch_size=large --network_type=stn --gpus 1,2,3 --use_tied_gen --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu.py --dataroot ./data --exp=cifar_stl --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu.py --dataroot ./data --exp=stl_cifar --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu.py --dataroot ./data --exp=syndigits_svhn --batch_size=128 --epoch_size=large --network_type=stn --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu.py --dataroot ./data --exp=synsigns_gtsrb --batch_size=128 --epoch_size=small --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#
#nohup python SB_main_00_ss_tu.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#
#Supervised source and unsupervised target and unsupervised source
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --exp=usps_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 0,1 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --exp=mnist_usps --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --exp=svhn_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --exp=mnist_svhn --batch_size=128 --epoch_size=large --network_type=stn --gpus 1,2,3 --use_tied_gen --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --exp=cifar_stl --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --exp=stl_cifar --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --exp=syndigits_svhn --batch_size=128 --epoch_size=large --network_type=stn --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --exp=synsigns_gtsrb --batch_size=128 --epoch_size=small --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#
###Supervised source and unsupervised target and unsupervised source and adversarial source
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --exp=usps_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --exp=mnist_usps --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --exp=svhn_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --exp=mnist_svhn --batch_size=128 --epoch_size=large --network_type=stn --gpus 1,2,3 --use_tied_gen --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --exp=cifar_stl --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --exp=stl_cifar --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --exp=syndigits_svhn --batch_size=128 --epoch_size=large --network_type=stn --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --exp=synsigns_gtsrb --batch_size=128 --epoch_size=small --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#
###Supervised source and unsupervised target and unsupervised source and adversarial target and adversarial source
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --exp=usps_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --exp=mnist_usps --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --exp=svhn_mnist --batch_size=128 --epoch_size=large --network_type=se --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --exp=mnist_svhn --batch_size=128 --epoch_size=large --network_type=stn --gpus 1,2,3 --use_tied_gen --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --exp=cifar_stl --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --exp=stl_cifar --batch_size=128 --epoch_size=large --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --exp=syndigits_svhn --batch_size=128 --epoch_size=large --network_type=stn --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --exp=synsigns_gtsrb --batch_size=128 --epoch_size=small --network_type=se --use_tied_gen --gpus 1,2,3 --workers 6  &  wait 2 &  #
#
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=amazon_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=dslr_webcam #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_amazon #  --network_type=cdan --lr_decay_type cdan_inv
#nohup python SB_main_00_ss_tu_su_ta_sa.py --dataroot ./data --batch_size=50 --buffer_size 0.5 --epoch_size=large --network_type=cdan --lr_decay_type none --gpus 0,1 --workers 24 --exp=webcam_dslr #  --network_type=cdan --lr_decay_type cdan_inv
#
#wait
