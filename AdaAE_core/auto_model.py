import torch.nn.functional as F
from AdaAE_core.util import data_information
from AdaAE_core.search_space.search_space_config import SearchSpace
from AdaAE_core.model.ArchitectureOptimizer import ArchitectureGradientOptimizer
from AdaAE_core.model.semisupervised_test import semisupervised_scratch_train
from AdaAE_core.model.unsupervised_test import unsupervised_scratch_train
from AdaAE_core.model.transfer_test import transfer_scratch_train_weight_save
from AdaAE_core.model.transfer_test import transfer_scratch_train_finetune
from AdaAE_core.model.logger import gnn_architecture_save, gnn_architecture_load, gnn_architecture_merge
from AdaAE_core.util import cl_loss_function
from torch_geometric.loader import DataLoader

class AutoModel(object):
    """
    The top API to realize gnn architecture search and model testing automatically.

    Using search algorithm samples gnn architectures and evaluate
    corresponding performance,testing the top k model from the sampled
    gnn architectures based on performance.

    """

    def __init__(self, graph_data, args):

        self.graph_data = graph_data
        self.args = args
        self.search_space = SearchSpace(self.args.gnn_layers)

        # data_information_dict = data_information(self.data)
        # print("the dataset information:\t", data_information_dict)

        self.architecture_gradient_optimizer = ArchitectureGradientOptimizer(self.search_space, self.args)

        if args.learning_type != "transfer_finetune":
            self.search_model()

            self.derive_target_model()
        else:
            self.derive_target_model_finetune()



    def search_model(self):
        # get the architecture alpha parameter sample distribution for gumbel softmax sample
        architecture_alpha_list = self.architecture_gradient_optimizer.architecture_alpha_list

        ## start search
        for epoch in range(self.args.search_epoch):
            print(32 * "=")
            print("Search Epoch:", epoch+1)
            gumbel_softmax_sample_output_list = []

            ## gumbel sample
            for architecture_alpha in architecture_alpha_list:
                gumbel_softmax_sample_output_list.append(self.hard_gumbel_softmax_sample(F.softmax(architecture_alpha, dim=-1)))

            # decode the architecture from gumbel_softmax_sample
            sample_candidate_index_list, sample_architecture = self.gnn_architecture_decode(gumbel_softmax_sample_output_list)

            # build the sampled gnn model based on the sampled architecture
            self.architecture_gradient_optimizer.build_optimize_gnn_model(sample_architecture, self.graph_data)


            # architecture alpha parameter optimization
            train_loader = DataLoader(self.graph_data.dataset, self.args.batch_size, shuffle=False)
            step_num = len(train_loader)

            self.architecture_gradient_optimizer.optimizer.zero_grad()
            nan_indicator = False

            for step, train_data in enumerate(train_loader):
                train_data = train_data.to(self.args.device)
                x1, x2 = self.architecture_gradient_optimizer(train_data, gumbel_softmax_sample_output_list, sample_candidate_index_list)
                # cl_loss = cl_loss_function(x1, x2) / step_num
                cl_loss = cl_loss_function(x1, x2)

                if step < step_num-1:
                    cl_loss.backward(retain_graph=True)
                else:
                    cl_loss.backward()

                if str(cl_loss.item()) == 'nan':
                    nan_indicator = True

            if nan_indicator == False:
                self.architecture_gradient_optimizer.optimizer.step()

                best_model = self.architecture_gradient_optimizer.best_alpha_gnn_architecture()
                print("Best Model:", best_model)

        print(32 * "=")
        print("Search Ending")

        ## get the last searched models, at which point the gumbel optimizer converge
        best_alpha_model_list = self.architecture_gradient_optimizer.get_top_architecture(self.args.return_top_k)

        gnn_architecture_save(best_alpha_model_list, self.graph_data.data_name)


    ## gumbel sample with hard sample
    def hard_gumbel_softmax_sample(self, sample_probability):

        hard_gumbel_softmax_sample_output = F.gumbel_softmax(logits=sample_probability,
                                                             tau=self.args.temperature,
                                                             hard=True)
        return hard_gumbel_softmax_sample_output

    ## decode the sampled architecture matrix to architecture embedding and architecture name
    def gnn_architecture_decode(self, gumbel_softmax_sample_ret_list):
        candidate_list = []
        candidate_index_list = []

        for i, component_one_hot in enumerate(gumbel_softmax_sample_ret_list):
            component_one_hot = component_one_hot.cpu().detach().numpy().tolist()[0]

            candidate_index = component_one_hot.index(max(component_one_hot))
            candidate_index_list.append(candidate_index)

            component = self.search_space.stack_gnn_architecture[i]
            candidate_list.append(self.search_space.space_dict[component][candidate_index])

        return candidate_index_list, candidate_list


    ## test the model and run the model scratch
    def derive_target_model(self):

        best_alpha_model_list = gnn_architecture_load(self.graph_data.data_name, self.args.gnn_layers)

        print(35*"=" + " the testing start " + 35*"=")

        for best_alpha_model in best_alpha_model_list:
            if self.args.learning_type == "semisupervised":
                semisupervised_scratch_train(graph_data=self.graph_data,
                                             model_component=best_alpha_model,
                                             args=self.args)

            elif self.args.learning_type == "unsupervised":
                unsupervised_scratch_train(graph_data=self.graph_data,
                                           model_component=best_alpha_model,
                                           args=self.args)

            elif self.args.learning_type == "transfer_pretrain":
                transfer_scratch_train_weight_save(graph_data=self.graph_data,
                                                   model_component=best_alpha_model,
                                                   args=self.args)

            else:
                raise Exception("Wrong learning type:", self.args.learning_type)

        print(35 * "=" + " the testing ending " + 35 * "=")


    def derive_target_model_finetune(self):
        best_alpha_model_split = eval(self.args.pretrain_weight.split('/')[-1].split('_epoch_')[0])
        best_alpha_model = gnn_architecture_merge(best_alpha_model_split, self.args.gnn_layers)

        transfer_scratch_train_finetune(graph_data=self.graph_data,
                                        model_component=best_alpha_model,
                                        args=self.args)

