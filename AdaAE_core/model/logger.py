import os
import torch

logger_path = os.path.split(os.path.realpath(__file__))[0][:-(7+len('AdaAE_core'))] + "/logger"

## save the name of model
def gnn_architecture_save(best_alpha_model_list, data_name):
    if not os.path.exists(logger_path + '/' + str(data_name)):
        os.makedirs(logger_path + '/' + str(data_name))

    with open(logger_path + '/' + str(data_name) + "/" + str(data_name) + "_gnn_logger.txt", "w") as f:
        for best_alpha_model in best_alpha_model_list:
            f.write(str(best_alpha_model) + "\n")

    print("gnn architecture save")
    print("save path: ", logger_path + '/' + str(data_name) + "/" + str(data_name) + "_gnn_logger.txt")
    print(50 * "=")

## load the name of model
def gnn_architecture_load(data_name, gnn_layers):
    best_alpha_model_split_list = []

    with open(logger_path + '/' + str(data_name) + '/' + str(data_name) + "_gnn_logger.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            best_alpha_model_split = eval(line.strip())
            best_alpha_model_split_list.append(best_alpha_model_split)

    best_alpha_model_list = []
    for best_alpha_model_split in best_alpha_model_split_list:
        best_alpha_model = gnn_architecture_merge(best_alpha_model_split, gnn_layers)
        best_alpha_model_list.append(best_alpha_model)

    return best_alpha_model_list

def gnn_architecture_merge(best_alpha_model_split, gnn_layers):
    best_alpha_model = best_alpha_model_split[:2]

    for layer in range(gnn_layers):
        attention = best_alpha_model_split[2 + layer * 4 + 0]
        aggregation = best_alpha_model_split[2 + layer * 4 + 1]
        hidden_dimension = str(best_alpha_model_split[2 + layer * 4 + 2])
        convolution = '-'.join([attention, aggregation, hidden_dimension])
        best_alpha_model.append(convolution)
        activation = best_alpha_model_split[2 + layer * 4 + 3]
        best_alpha_model.append(activation)
    best_alpha_model.append(best_alpha_model_split[-1])
    return best_alpha_model

## split the name of model into single component
def gnn_architecture_split(best_alpha_model, gnn_layers):
    best_alpha_model_split = best_alpha_model[:2]

    for layer in range(gnn_layers):
        convolution = best_alpha_model[2 + layer * 2 + 0]
        attention_type, aggregator_type, hidden_dimension = convolution.split('-')
        best_alpha_model_split += [attention_type, aggregator_type, hidden_dimension]

        activation = best_alpha_model[2 + layer * 2 + 1]
        best_alpha_model_split.append(activation)
    best_alpha_model_split.append(best_alpha_model[-1])

    return best_alpha_model_split

## save the weight of model
def model_weight_save(gnn_model, optimizer, data_name, suffix='model_weight.pth'):

    if not os.path.exists(logger_path + '/' + str(data_name)):
        os.makedirs(logger_path + '/' + str(data_name))

    state = {"gnn_model": gnn_model.state_dict(),
             "optimizer": optimizer.state_dict()}
    torch.save(state, logger_path + '/' + str(data_name) + "/" + suffix)

    print("gnn model and optimizer parameter save")
    print("save path: ", logger_path + '/' + str(data_name) + "/" + suffix)




