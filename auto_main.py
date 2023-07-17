import argparse
import random
import torch
import numpy as np
import os
from utils.datasets import get_dataset
from utils.chem_datasets import MoleculeDataset
from utils.dataset_splitter import semisupervised_DATA, unsupervised_DATA, transfer_DATA
from AdaAE_core.auto_model import AutoModel

def arg_parse():
    str2bool = lambda x: x.lower() == "true"

    parser = argparse.ArgumentParser(description='AdaAE_core Arguments.')
    parser.add_argument('--data_root', type=str, default="./data/")
    parser.add_argument('--dataset', dest='dataset', default='MUTAG', help='Dataset')
    parser.add_argument('--learning_type', type=str, default="unsupervised", help='semisupervised/unsupervised/transfer_pretrain/transfer_finetune')
    parser.add_argument('--n_fold', type=int, default=10, help='for semisupervised learing')
    parser.add_argument('--semi_split', type=float, default=0.1, help='percent of semi training data, for semisupervised learing')
    parser.add_argument('--pretrain_weight', type=str, default="./logger/chembl_filtered_transfer_pretrain_test/['identical', 'attribute_masking', 'gcn', 'add', '64', 'softplus', 'graphsage', 'mean', '256', 'tanh', 'gin', 'max', '64', 'elu', 'global_max']_epoch_10.pth", help='the weight file of pretrain architecture, and used for transfer_finetune')
    parser.add_argument('--load_weight_flag', type=str2bool, default=True, help='True/False')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--gnn_drop_out', type=float, default=0.0)
    parser.add_argument('--bias', type=str2bool, default=True)
    parser.add_argument('--temperature', type=float, default=0.1, help='the temperature parameter of gumbel softmax trick')
    parser.add_argument('--search_epoch', type=int, default=200, help='the number of search epoch for gumbel')
    parser.add_argument('--learning_rate_gumbel', type=float, default=0.1, help='the learning rate for gumbel')
    parser.add_argument('--l2_regularization_strength_gumbel', type=float, default=0.001, help='the regularization strength for gumbel')
    parser.add_argument('--train_epoch', type=int, default=10, help='the number of train epoch for model sampled by gumbel')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='the learning rate for contrastive learning')
    parser.add_argument('--l2_regularization_strength', type=float, default=0.0, help='the regularization strength for contrastive learning')
    parser.add_argument('--return_top_k', type=int, default=5, help='the number of top model for testing')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = arg_parse()
set_seed(args.seed)

if args.learning_type == "semisupervised":
    dataset = get_dataset(args.dataset, sparse=True, feat_str='deg+odeg100', root=args.data_root)
    graph = semisupervised_DATA(dataset, args)
elif args.learning_type == "unsupervised":
    dataset = get_dataset(args.dataset, sparse=True, feat_str='deg+odeg100', root=args.data_root)
    graph = unsupervised_DATA(dataset, args)
elif args.learning_type == "transfer_pretrain":
    dataset = MoleculeDataset(os.path.join(args.data_root, args.dataset), dataset=args.dataset)
    graph = transfer_DATA(dataset, args)
elif args.learning_type == "transfer_finetune":
    dataset = MoleculeDataset(os.path.join(args.data_root, args.dataset), dataset=args.dataset)
    graph = transfer_DATA(dataset, args)
else:
    raise Exception("Wrong learning type:", args.learning_type)


AutoModel(graph, args)
