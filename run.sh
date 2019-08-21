#my.sh
#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 main.py --feature_type flow --use_tf_log 1 --train 1 --train 2 --train 3 --train 4 --train 5 --train 6 --train 7 --train 8 --train 9 --test 10 --test 11 --test 12
