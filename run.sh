#my.sh
#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 main.py --feature_type flow --use_tf_log 1 --train 2 --train 3 --train 4 --train 6 --train 9 --test 10 --test 12
