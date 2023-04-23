import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from AdaAE_core.util import cl_loss_function
from AdaAE_core.model.gnn_model import GnnModel
from AdaAE_core.model.classifier_model.semisupervised_classifier import SemisupervisedClassifier
from AdaAE_core.model.logger import gnn_architecture_split
from tqdm import tqdm


def semisupervised_scratch_train(graph_data, model_component, args, test_epoch=100):
    # train the model from the scratch and test based on the test dataset
    print(25 * "#", "testing, train from the scratch", 25 * "#")

    batch_size = args.batch_size
    folds = len(graph_data.train_indices)

    train_losses = []
    test_performances = []

    for fold in range(folds):
        train_dataset = graph_data.dataset[graph_data.train_indices[fold]]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

        test_dataset = graph_data.dataset[graph_data.test_indices[fold]]
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = GnnModel(model_component, args, graph_data.num_features, graph_data.withEdge).to(args.device)

        classifier_model = SemisupervisedClassifier(int(model_component[-3].split('-')[-1]), graph_data.num_labels).to(args.device)

        optimizer = torch.optim.Adam([{'params': model.parameters()},
                                      {'params': classifier_model.parameters()}],
                                     lr=args.learning_rate,
                                     weight_decay=args.l2_regularization_strength)

        for epoch in range(1, test_epoch + 1):
            # training on the training dataset, the epoch same as the previous work
            train_loss_one_epoch = scratch_train_each_epoch(model,
                                                            classifier_model,
                                                            optimizer,
                                                            train_loader,
                                                            args)
            train_losses.append(train_loss_one_epoch)

            # testing on the test dataset
            performance_one_epoch = eval_test_dataset(model,
                                                      classifier_model,
                                                      test_loader,
                                                      args)
            test_performances.append(performance_one_epoch)
            print('fold', fold, ', epoch', epoch)


    test_performances = torch.tensor(test_performances).view(folds, -1).max(dim=1)[0]*100
    test_performance_mean = round(test_performances.mean().item(), 2)
    test_performance_std = round(test_performances.std().item(), 2)
    performance = "test acc:\t" + str(test_performance_mean) + "±" + str(test_performance_std)
    print(performance)

    print("test model:\t", str(gnn_architecture_split(model_component, args.gnn_layers)))


def scratch_train_each_epoch(model,
                             classifier_model,
                             optimizer,
                             train_loader,
                             args):
    model.train()
    classifier_model.train()
    train_loss_one_epoch = 0
    total_graphs = 0

    for train_data in tqdm(train_loader, desc="Iteration"):

        train_data = train_data.to(args.device)

        x1, x2 = model(train_data)
        cl_loss = cl_loss_function(x1, x2)

        train_predict_y1 = classifier_model(x1)
        train_predict_y2 = classifier_model(x2)

        cls_loss1 = F.nll_loss(train_predict_y1, train_data.y)
        cls_loss2 = F.nll_loss(train_predict_y2, train_data.y)
        cls_loss = (cls_loss1 + cls_loss2) / 2

        train_loss = cl_loss + cls_loss

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_graphs += train_data.num_graphs
        train_loss_one_epoch += train_loss.item() * train_data.num_graphs

    train_loss_one_epoch /= total_graphs
    return train_loss_one_epoch


def eval_test_dataset(model,
                      classifier_model,
                      test_loader,
                      args):
    model.eval()
    classifier_model.eval()
    performance_one_epoch = 0

    for test_data in test_loader:
        test_data = test_data.to(args.device)
        with torch.no_grad():
            node_embedding_matrix = model.forward_encoder(test_data)
            test_predict_y = classifier_model(node_embedding_matrix).max(1)[1]
        performance_one_epoch += test_predict_y.eq(test_data.y.view(-1)).sum().item()
    performance_one_epoch = performance_one_epoch / len(test_loader.dataset)
    return performance_one_epoch
