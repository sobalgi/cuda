#!/bin/sh
## DANN/MT-Tri Network
# Target Supervised
#nohup python SB_lang_00_ts.py --exp=books_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=books_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=books_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=dvd_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=dvd_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=dvd_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=electronics_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=electronics_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=electronics_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=kitchen_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=kitchen_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ts.py --exp=kitchen_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #

# Source Supervised
#nohup python SB_lang_00_ss.py --exp=books_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=books_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=books_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=dvd_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=dvd_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=dvd_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=electronics_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=electronics_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=electronics_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=kitchen_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=kitchen_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss.py --exp=kitchen_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #

# Source Supervised, Target Unsupervised
#nohup python SB_lang_00_ss_tu.py --exp=books_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=books_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=books_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=dvd_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=dvd_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=dvd_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=electronics_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=electronics_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=electronics_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=kitchen_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=kitchen_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu.py --exp=kitchen_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #

# Source Supervised, Target Unsupervised, Source Unsupervised
#nohup python SB_lang_00_ss_tu_su.py --exp=books_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=books_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=books_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=dvd_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=dvd_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=dvd_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=electronics_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=electronics_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=electronics_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=kitchen_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=kitchen_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &
#nohup python SB_lang_00_ss_tu_su.py --exp=kitchen_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #

# Source Supervised, Target Unsupervised, Source Unsupervised, Target Adversarial
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=books_dvd --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=books_electronics --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=books_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=dvd_books --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=dvd_electronics --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=dvd_kitchen --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=electronics_books --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=electronics_dvd --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=electronics_kitchen --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=kitchen_books --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=kitchen_dvd --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=kitchen_electronics --batch_size=128 --network_type=dade --gpus 3 --workers 2 &  #

# Source Supervised, Target Unsupervised, Source Unsupervised, Source Adversarial
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=books_dvd --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=books_electronics --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=books_kitchen --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=dvd_books --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=dvd_electronics --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=dvd_kitchen --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=electronics_books --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=electronics_dvd --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=electronics_kitchen --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=kitchen_books --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=kitchen_dvd --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=kitchen_electronics --batch_size=128 --network_type=dade --gpus 4 --workers 2 &  #

# Source Supervised, Target Unsupervised, Source Unsupervised, Target Adversarial, Source Adversarial
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=books_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=books_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=books_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=dvd_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=dvd_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=dvd_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=electronics_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=electronics_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=electronics_kitchen --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=kitchen_books --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=kitchen_dvd --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=kitchen_electronics --batch_size=128 --network_type=dade --gpus 1 --workers 2 &  #

## MAN Network
# Source Supervised
#nohup python SB_lang_00_ss.py --exp=books_dvd --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=books_electronics --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=books_kitchen --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=dvd_books --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=dvd_electronics --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=dvd_kitchen --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=electronics_books --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=electronics_dvd --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=electronics_kitchen --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=kitchen_books --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=kitchen_dvd --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss.py --exp=kitchen_electronics --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #

# Target Supervised
#nohup python SB_lang_00_ts.py --exp=books_dvd --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=books_electronics --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=books_kitchen --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=dvd_books --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=dvd_electronics --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=dvd_kitchen --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=electronics_books --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=electronics_dvd --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=electronics_kitchen --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=kitchen_books --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=kitchen_dvd --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ts.py --exp=kitchen_electronics --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #

# Source Supervised, Target Unsupervised
#nohup python SB_lang_00_ss_tu.py --exp=books_dvd --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=books_electronics --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=books_kitchen --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=dvd_books --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=dvd_electronics --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=dvd_kitchen --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=electronics_books --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=electronics_dvd --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=electronics_kitchen --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=kitchen_books --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=kitchen_dvd --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu.py --exp=kitchen_electronics --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #

# Source Supervised, Target Unsupervised, Source Unsupervised
#nohup python SB_lang_00_ss_tu_su.py --exp=books_dvd --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=books_electronics --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=books_kitchen --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=dvd_books --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=dvd_electronics --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=dvd_kitchen --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=electronics_books --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=electronics_dvd --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=electronics_kitchen --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=kitchen_books --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=kitchen_dvd --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su.py --exp=kitchen_electronics --batch_size=128 --network_type=man --gpus 6 --workers 2 &  #

# Source Supervised, Target Unsupervised, Source Unsupervised, Target Adversarial
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=books_dvd --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=books_electronics --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=books_kitchen --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=dvd_books --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=dvd_electronics --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=dvd_kitchen --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=electronics_books --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=electronics_dvd --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=electronics_kitchen --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=kitchen_books --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=kitchen_dvd --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta.py --exp=kitchen_electronics --batch_size=128 --network_type=man --gpus 7 --workers 2 &  #

# Source Supervised, Target Unsupervised, Source Unsupervised, Source Adversarial
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=books_dvd --batch_size=128 --network_type=man --gpus 0 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=books_electronics --batch_size=128 --network_type=man --gpus 0 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=books_kitchen --batch_size=128 --network_type=man --gpus 0 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=dvd_books --batch_size=128 --network_type=man --gpus 0 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=dvd_electronics --batch_size=128 --network_type=man --gpus 0 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=dvd_kitchen --batch_size=128 --network_type=man --gpus 0 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=electronics_books --batch_size=128 --network_type=man --gpus 2 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=electronics_dvd --batch_size=128 --network_type=man --gpus 2 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=electronics_kitchen --batch_size=128 --network_type=man --gpus 2 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=kitchen_books --batch_size=128 --network_type=man --gpus 2 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=kitchen_dvd --batch_size=128 --network_type=man --gpus 2 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_sa.py --exp=kitchen_electronics --batch_size=128 --network_type=man --gpus 2 --workers 2 &  #

# Source Supervised, Target Unsupervised, Source Unsupervised, Target Adversarial, Source Adversarial
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=books_dvd --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=books_electronics --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=books_kitchen --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=dvd_books --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=dvd_electronics --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=dvd_kitchen --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=electronics_books --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=electronics_dvd --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=electronics_kitchen --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=kitchen_books --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=kitchen_dvd --batch_size=128 --gpus 6 --workers 2 &  #
#nohup python SB_lang_00_ss_tu_su_ta_sa.py --exp=kitchen_electronics --batch_size=128 --gpus 6 --workers 2 &  #

wait
