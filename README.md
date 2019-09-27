# EndoVis2019
action, phase and instrument
Newest code in update_dataset branch.

Instruction to run the code:
CUDA_VISIBLE_DEVICES=0 python main.py --feature_type rgb --fusion_mode 0 --use_tf_log 1 --learning_rate 1e-3 --concurrence_weight 0.5 --weighted 1

parameters explanation:
feature_type: feature files used for training and testing
rgb, rgb_oversample_4, rgb_oversample_4_norm
(resnet remains to test)

fusion_mode:
0 - rgb/flow single feature file;
1 - rgb and flow early fusion;
2 - rgb and flow late fusion;
3 - resnet;
4 - resnet and rgb early fusion;
5 - resnet and rgb late fusion;
6 - resnet and flow early fusion;
7 - resnet and flow late fusion;

concurrence_weight: action and instrument inside class dependency parameter: model prediction = net output*(identity matrix + concurrence_matrix * concurrence_weight)

weighted: whether to use pos_weight in loss functions
