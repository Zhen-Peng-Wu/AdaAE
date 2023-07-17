import torch
from torch_geometric.loader import DataLoader
from AdaAE_core.util import cl_loss_function
from AdaAE_core.model.gnn_model import GnnModel
from AdaAE_core.model.classifier_model.unsupervised_classifier import UnsupervisedClassifier
from AdaAE_core.model.logger import gnn_architecture_split
import numpy as np
from tqdm import tqdm


def unsupervised_scratch_train(graph_data, model_component, args, test_epoch=30, log_interval=5):
    # train the model from the scratch and test based on the test dataset
    print(25 * "#", "testing, train from the scratch", 25 * "#")

    batch_size = args.batch_size

    train_losses = []

    train_dataset = graph_data.dataset

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

    test_dataset = graph_data.dataset
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = GnnModel(model_component, args, graph_data.num_features, graph_data.withEdge).to(args.device)

    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=args.learning_rate,
                                 weight_decay=args.l2_regularization_strength)

    best_test_acc = 0
    best_test_std = 0

    for epoch in range(1, test_epoch + 1):

        # training on the training dataset

        train_loss_one_epoch = scratch_train_each_epoch(model,
                                                        optimizer,
                                                        train_loader,
                                                        args)
        train_losses.append(train_loss_one_epoch)

        print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss_one_epoch))

        if epoch % log_interval == 0:
            # testing on the test dataset
            test_acc, test_std = eval_test_dataset(model,
                                                   test_loader,
                                                   args)
            print('Epoch: {}, Test Acc: {:.2f}±{:.2f}'.format(epoch, test_acc * 100, test_std * 100))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_std = test_std

    performance = 'Best Test Acc:\t {:.2f}±{:.2f}'.format(best_test_acc*100, best_test_std*100)
    print(performance)

    print("test model:\t", str(gnn_architecture_split(model_component, args.gnn_layers)))


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

def eval_test_dataset(model,
                      test_loader,
                      args):
    model.eval()


    emb = []
    y = []
    for test_data in test_loader:
        with torch.no_grad():
            test_data = test_data.to(args.device)
            node_embedding_matrix = model.forward_encoder(test_data)

            e = node_embedding_matrix.cpu().numpy()
            e[np.isnan(e)] = 0
            emb.append(e)
            y.append(test_data.y.cpu().numpy())
    emb = np.concatenate(emb, 0)
    y = np.concatenate(y, 0)

    acc, std = UnsupervisedClassifier(emb, y)
    return acc, std
