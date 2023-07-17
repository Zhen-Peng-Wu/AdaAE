import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from AdaAE_core.util import cl_loss_function
from AdaAE_core.model.gnn_model import GnnModel
from AdaAE_core.model.classifier_model.transfer_classifier import TransferClassifier
from AdaAE_core.model.logger import model_weight_save, gnn_architecture_split
from utils.chem_splitters import scaffold_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os


def transfer_scratch_train_weight_save(graph_data, model_component, args, test_epoch=10, save_epoch=5):
    # train the model from the scratch and save the model
    print(25 * "#", "saving, train from the scratch", 25 * "#")

    batch_size = args.batch_size

    train_losses = []

    train_dataset = graph_data.dataset

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    model = GnnModel(model_component, args, graph_data.num_features, graph_data.withEdge).to(args.device)

    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=args.learning_rate,
                                 weight_decay=args.l2_regularization_strength)


    for epoch in range(1, test_epoch + 1):
        # training on the training dataset

        train_loss_one_epoch = scratch_train_each_epoch(model,
                                                        optimizer,
                                                        train_loader,
                                                        args)
        train_losses.append(train_loss_one_epoch)

        print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss_one_epoch))
        if epoch % save_epoch == 0:
            suffix = str(gnn_architecture_split(model_component, args.gnn_layers)) + '_epoch_' + str(epoch) + '.pth'
            model_weight_save(model, optimizer, graph_data.data_name, suffix)

    print("save model:\t", str(gnn_architecture_split(model_component, args.gnn_layers)))

def scratch_train_each_epoch(model,
                             optimizer,
                             train_loader,
                             args):
    model.train()
    train_loss_one_epoch = 0
    total_graphs = 0

    for train_data in tqdm(train_loader, desc="Iteration"):

        train_data = train_data.to(args.device)

        x1, x2 = model(train_data)
        cl_loss = cl_loss_function(x1, x2)

        optimizer.zero_grad()
        cl_loss.backward()
        optimizer.step()

        total_graphs += train_data.num_graphs
        train_loss_one_epoch += cl_loss.item() * train_data.num_graphs

    train_loss_one_epoch /= total_graphs
    return train_loss_one_epoch


def transfer_scratch_train_finetune(graph_data, model_component, args, test_epoch=100):
    # train the model from the scratch and test based on the test dataset
    print(25 * "#", "testing, train from the scratch", 25 * "#")

    batch_size = args.batch_size

    dataset = graph_data.dataset

    smiles_list = pd.read_csv(os.path.join(args.data_root, args.dataset, 'processed/smiles.csv'), header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0,
                                                                frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = GnnModel(model_component, args, graph_data.num_features, graph_data.withEdge).to(args.device)


    if args.load_weight_flag == True:
        pretrain_weight = torch.load(args.pretrain_weight, map_location=args.device)
        model.load_state_dict(pretrain_weight['gnn_model'])


    classifier_model = TransferClassifier(int(model_component[-3].split('-')[-1]), graph_data.num_tasks).to(args.device)

    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': classifier_model.parameters()}],
                                 lr=args.learning_rate,
                                 weight_decay=args.l2_regularization_strength)

    val_performances = []
    test_performances = []

    best_test_performance = 0

    for epoch in range(1, test_epoch + 1):


        # training on the training dataset
        scratch_train_with_classifier_each_epoch(model,
                                                 classifier_model,
                                                 optimizer,
                                                 train_loader,
                                                 args)

        val_performance = eval_dataset(model, classifier_model, valid_loader, args)
        val_performances.append(val_performance)

        test_performance = eval_dataset(model, classifier_model, test_loader, args)
        test_performances.append(test_performance)
        print("Epoch: {}, Test Performance: {:.2f}".format(epoch, test_performance * 100))

        if test_performance > best_test_performance:
            best_test_performance = test_performance

    print("Best Test Performance: {:.2f}".format(best_test_performance*100))

    print("test model:\t", str(gnn_architecture_split(model_component, args.gnn_layers)))


def scratch_train_with_classifier_each_epoch(model,
                                             classifier_model,
                                             optimizer,
                                             train_loader,
                                             args):
    model.train()
    classifier_model.train()

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    for train_data in tqdm(train_loader, desc="Iteration"):

        train_data = train_data.to(args.device)

        node_embedding_matrix = model.forward_encoder(train_data)

        pred = classifier_model(node_embedding_matrix)
        y = train_data.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        loss = torch.sum(loss_mat) / torch.sum(is_valid)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_dataset(model,
                 classifier_model,
                 data_loader,
                 args):
    model.eval()
    classifier_model.eval()
    y_true = []
    y_scores = []

    for data in data_loader:
        data = data.to(args.device)
        with torch.no_grad():
            node_embedding_matrix = model.forward_encoder(data)
            pred = classifier_model(node_embedding_matrix)

        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list)

