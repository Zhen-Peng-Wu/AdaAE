import torch
from sklearn.model_selection import StratifiedKFold

class unsupervised_DATA(object):
    def __init__(self,
                 dataset,
                 args):

        self.dataset = dataset
        self.num_features = dataset.num_node_features
        self.num_labels = dataset.num_classes
        self.data_name = dataset.name + '_' + args.learning_type
        self.len_dataset = len(dataset)
        withEdge = False if self.dataset.data.edge_attr is None else True
        self.withEdge = withEdge

class semisupervised_DATA(object):
    def __init__(self,
                 dataset,
                 args):

        self.dataset = dataset
        self.num_features = dataset.num_node_features
        self.num_labels = dataset.num_classes
        self.data_name = dataset.name + '_' + args.learning_type
        self.len_dataset = len(dataset)
        withEdge = False if self.dataset.data.edge_attr is None else True
        self.withEdge = withEdge

        skf = StratifiedKFold(args.n_fold, shuffle=True, random_state=12345)

        test_indices, train_indices = [], []
        for _, idx in skf.split(torch.zeros(self.len_dataset), self.dataset.data.y):
            test_indices.append(torch.from_numpy(idx))

        semi_size = int(self.len_dataset * args.semi_split)

        for i in range(args.n_fold):
            train_mask = torch.ones(self.len_dataset, dtype=torch.uint8)
            train_mask[test_indices[i].long()] = 0
            train_indice = torch.nonzero(train_mask, as_tuple=False).view(-1)

            # semi split
            train_size = train_indice.shape[0]
            select_idx = torch.randperm(train_size)[:semi_size]
            semi_indice = train_indice[select_idx]

            train_indices.append(semi_indice)

        self.train_indices = train_indices
        self.test_indices = test_indices


class transfer_DATA(object):
    def __init__(self,
                 dataset,
                 args):

        self.dataset = dataset
        self.num_features = dataset.num_node_features
        self.num_labels = dataset.num_classes
        self.data_name = dataset.dataset + '_' + args.learning_type
        self.len_dataset = len(dataset)
        withEdge = False if self.dataset.data.edge_attr is None else True
        self.withEdge = withEdge

        # Bunch of classification tasks
        if args.dataset == "tox21":
            num_tasks = 12
        elif args.dataset == "hiv":
            num_tasks = 1
        elif args.dataset == "pcba":
            num_tasks = 128
        elif args.dataset == "muv":
            num_tasks = 17
        elif args.dataset == "bace":
            num_tasks = 1
        elif args.dataset == "bbbp":
            num_tasks = 1
        elif args.dataset == "toxcast":
            num_tasks = 617
        elif args.dataset == "sider":
            num_tasks = 27
        elif args.dataset == "clintox":
            num_tasks = 2
        else:
            num_tasks = 1

        self.num_tasks = num_tasks
