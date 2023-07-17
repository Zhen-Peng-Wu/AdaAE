import os

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
data_root = "./data/chem_data/"
learning_type = "transfer_finetune"
pretrain_weight = "\"./logger/chembl_filtered_transfer_pretrain/['edge_perturbation', 'attribute_masking', 'gcn', 'add', '64', 'softplus', 'graphsage', 'mean', '256', 'tanh', 'gin', 'max', '128', 'elu', 'global_mean']_epoch_5.pth\""
load_weight_flag = True
train_epoch = 100


# datasets = ['bbbp', 'tox21', 'toxcast', 'muv']
# device = "cuda:0"
datasets = ['sider', 'clintox',  'hiv', 'bace']
device = "cuda:1"

for seed in seeds:
    for dataset in datasets:
        command = "python auto_test.py --data_root " + data_root + \
                  " --learning_type " + learning_type + \
                  " --pretrain_weight " + pretrain_weight + \
                  " --load_weight_flag " + str(load_weight_flag) + \
                  " --train_epoch " + str(train_epoch) + \
                  " --seed " + str(seed) + \
                  " --dataset " + dataset + \
                  " --device " + device
        print(command)
        os.system(command)
