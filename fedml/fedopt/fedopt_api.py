#################### for others ###################################################
import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from .optrepo import OptRepo
from fedml.simulation.sp.fedopt.Dowg import DoWG, CDoWG
from fedml.simulation.sp.fedopt.FEDDFW import DFW
# import torch_optimizer as tor_optim
# from dog import DoG, LDoG, PolynomialDecayAverager

class FedOptAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        # self.model_global = model
        # self.model_global.train()

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = create_model_trainer(model, args)
        self._instanciate_opt()
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                self.model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample/home/chaoyanghe/zhtang_FedML/python/fedml/simulation/sp/fedopt/__pycache___testset

    def _instanciate_opt(self):
        if self.args.server_optimizer == 'FedAvg':
            return
        if self.args.server_optimizer == 'DFW':
            self.opt = DFW(self.model_trainer.model.parameters(), eta=.001, momentum=0.9, weight_decay=.00001, eps=1e-5)
        # elif self.args.server_optimizer == 'Yogi':            
        #     self.opt = tor_optim.Yogi(self.model_trainer.model.parameters(), lr=1.0)
        # elif self.args.server_optimizer == 'Dog' or self.args.server_optimizer == 'LDog':
        #     opt_class = LDoG if self.args.server_optimizer == 'LDog' else DoG
        #     self.opt = opt_class(self.model_trainer.model.parameters(), reps_rel = 1e-6, lr = 1.0)
        #     self.avgr = PolynomialDecayAverager(self.model_trainer.model, gamma=8)  
        elif self.args.server_optimizer == 'Dowg':
            self.opt = DoWG(self.model_trainer.model.parameters(), eps=1e-4)
        else:
            self.opt = OptRepo.name2cls(self.args.server_optimizer)(
            # self.model_global.parameters(), lr=self.args.server_lr
            self.model_trainer.model.parameters(),
            lr=self.args.server_lr,
            # momentum=0.9 # for fedavgm
            # eps = 1e-3 for adaptive optimizer
        )
    def train(self):
        for round_idx in range(self.args.comm_round):
            w_global = self.model_trainer.get_model_params()
            logging.info("################ Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]

                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # train on new dataset
                w = client.train(w_global)
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                # loss_locals.append(copy.deepcopy(loss))
                # logging.info('Client {:3d}, loss {:.3f}'.format(client_idx, loss))

                       # reset weight after standalone simulation
            self.model_trainer.set_model_params(w_global)
            # update global weights
            w_avg = self._aggregate(w_locals)
            
            self.opt.zero_grad()
            opt_state = self.opt.state_dict()
            self._set_model_global_grads(w_avg)
            self._instanciate_opt()
            self.opt.load_state_dict(opt_state)
 
                        
            if self.args.server_optim and self.args.server_optimizer == 'Dowg':
                self.opt.step()
                
            else:
                self.model_trainer.set_model_params(w_avg)
                
            w_global = self.model_trainer.get_model_params()

            #test results
            #at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)
            
            
    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _set_model_global_grads(self, new_state):
        new_model = copy.deepcopy(self.model_trainer.model)
        new_model.load_state_dict(new_state)
        with torch.no_grad():
            for parameter, new_parameter in zip(self.model_trainer.model.parameters(), new_model.parameters()):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.model_trainer.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.model_trainer.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        self.model_trainer.set_model_params(new_model_state_dict)

    
    def _test_global_model_on_global_data(self, w_global, round_idx, return_val=False):
        logging.info("################test_global_model_on_global_dataset################")
        self.model_trainer.set_model_params(w_global)
        metrics_test = self.model_trainer.test(self.test_global, self.device, self.args)
        if return_val:
            return metrics_test
        # metrics_train = self.model_trainer.test(self.train_global, self.device, self.args)
        test_acc = metrics_test["test_correct"] / metrics_test["test_total"]
        test_loss = metrics_test["test_loss"] / metrics_test["test_total"]
        # train_acc = metrics_train["test_correct"] / metrics_train["test_total"]
        # train_loss = metrics_train["test_loss"] / metrics_train["test_total"]
        stats = {"test_acc": test_acc, "test_loss":test_loss}
        logging.info(stats)
        if self.args.enable_wandb:
            wandb.log({"Global Test Acc": test_acc, "round":round_idx})
            wandb.log({"Global Test Loss": test_loss, "round":round_idx})
    
    def _local_test_on_all_clients(self, round_idx):
        logging.info("################local_test_on_all_clients : {}".format(round_idx))
        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]
        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # Train and test might have different number of clients
            if self.test_data_local_dict[client_idx] is None:
                continue
            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):
        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)

######################## for dfw ###########################################
# class FedOptAPI(object):
#     def __init__(self, args, device, dataset, model):
#         self.device = device
#         self.args = args
#         [
#             train_data_num,
#             test_data_num,
#             train_data_global,
#             test_data_global,
#             train_data_local_num_dict,
#             train_data_local_dict,
#             test_data_local_dict,
#             class_num,
#         ] = dataset
#         self.train_global = train_data_global
#         self.test_global = test_data_global
#         self.val_global = None
#         self.train_data_num_in_total = train_data_num
#         self.test_data_num_in_total = test_data_num

#         # self.model_global = model
#         # self.model_global.train()

#         self.client_list = []
#         self.train_data_local_num_dict = train_data_local_num_dict
#         self.train_data_local_dict = train_data_local_dict
#         self.test_data_local_dict = test_data_local_dict

#         self.model_trainer = create_model_trainer(model, args)
#         self._instanciate_opt()
#         self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)

#     def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
#         logging.info("############setup_clients (START)#############")
#         for client_idx in range(self.args.client_num_per_round):
#             c = Client(
#                 client_idx,
#                 train_data_local_dict[client_idx],
#                 test_data_local_dict[client_idx],
#                 train_data_local_num_dict[client_idx],
#                 self.args,
#                 self.device,
#                 self.model_trainer,
#             )
#             self.client_list.append(c)
#         logging.info("############setup_clients (END)#############")

#     def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
#         if client_num_in_total == client_num_per_round:
#             client_indexes = [client_index for client_index in range(client_num_in_total)]
#         else:
#             num_clients = min(client_num_per_round, client_num_in_total)
#             np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
#             client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
#         logging.info("client_indexes = %s" % str(client_indexes))
#         return client_indexes

#     def _generate_validation_set(self, num_samples=10000):
#         test_data_num = len(self.test_global.dataset)
#         sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
#         subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
#         sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
#         self.val_global = sample/home/chaoyanghe/zhtang_FedML/python/fedml/simulation/sp/fedopt/__pycache___testset

#     def _instanciate_opt(self):
#         if self.args.server_optimizer == 'FedAvg':
#             return
#         if self.args.server_optimizer == 'DFW':
#             self.opt = DFW(self.model_trainer.model.parameters(), eta=.001, momentum=0.9, weight_decay=.00001, eps=1e-5)
#         # elif self.args.server_optimizer == 'Yogi':            
#         #     self.opt = tor_optim.Yogi(self.model_trainer.model.parameters(), lr=1.0)
#         # elif self.args.server_optimizer == 'Dog' or self.args.server_optimizer == 'LDog':
#         #     opt_class = LDoG if self.args.server_optimizer == 'LDog' else DoG
#         #     self.opt = opt_class(self.model_trainer.model.parameters(), reps_rel = 1e-6, lr = 1.0)
#         #     self.avgr = PolynomialDecayAverager(self.model_trainer.model, gamma=8)  
#         elif self.args.server_optimizer == 'Dowg':
#             self.opt = DoWG(self.model_trainer.model.parameters())
#         else:
#             self.opt = OptRepo.name2cls(self.args.server_optimizer)(
#             # self.model_global.parameters(), lr=self.args.server_lr
#             self.model_trainer.model.parameters(),
#             lr=self.args.server_lr,
#             # momentum=0.9 # for fedavgm
#             # eps = 1e-3 for adaptive optimizer
#         )
#     def train(self):
#         for round_idx in range(self.args.comm_round):
#             w_global = self.model_trainer.get_model_params()
#             logging.info("################ Communication round : {}".format(round_idx))

#             w_locals = []

#             """
#             for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
#             Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
#             """
#             client_indexes = self._client_sampling(
#                 round_idx, self.args.client_num_in_total, self.args.client_num_per_round
#             )
#             logging.info("client_indexes = " + str(client_indexes))

#             for idx, client in enumerate(self.client_list):
#                 # update dataset
#                 client_idx = client_indexes[idx]

#                 client.update_local_dataset(
#                     client_idx,
#                     self.train_data_local_dict[client_idx],
#                     self.test_data_local_dict[client_idx],
#                     self.train_data_local_num_dict[client_idx],
#                 )

#                 # train on new dataset
#                 w = client.train(w_global)
#                 w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
#                 # loss_locals.append(copy.deepcopy(loss))
#                 # logging.info('Client {:3d}, loss {:.3f}'.format(client_idx, loss))

#                        # reset weight after standalone simulation
#             self.model_trainer.set_model_params(w_global)
#             # update global weights
#             w_avg = self._aggregate(w_locals)
 
              
#             if self.args.server_optim and self.args.server_optimizer == 'Adam':
#                 self.opt.zero_grad()
#                 opt_state = self.opt.state_dict()
#                 self._set_model_global_grads(w_avg)
#                 self._instanciate_opt()
#                 self.opt.load_state_dict(opt_state)
#                 self.opt.step()
                
                             
#             elif self.args.server_optim and self.args.server_optimizer == 'Dowg':
#                 self.opt.zero_grad()
#                 opt_state = self.opt.state_dict()
#                 self._set_model_global_grads(w_avg)
#                 self._instanciate_opt()
#                 self.opt.load_state_dict(opt_state)
#                 self.opt.step()
                
#             else:
#                 self.model_trainer.set_model_params(w_avg)
                
#             w_global = self.model_trainer.get_model_params()

#             #test results
#             #at last round
#             if round_idx == self.args.comm_round - 1:
#                 self._local_test_on_all_clients(round_idx)
#             # per {frequency_of_the_test} round
#             elif round_idx % self.args.frequency_of_the_test == 0:
#                 if self.args.dataset.startswith("stackoverflow"):
#                     self._local_test_on_validation_set(round_idx)
#                 else:
#                     self._local_test_on_all_clients(round_idx)
            
            
#     def _aggregate(self, w_locals):
#         training_num = 0
#         for idx in range(len(w_locals)):
#             (sample_num, averaged_params) = w_locals[idx]
#             training_num += sample_num

#         (sample_num, averaged_params) = w_locals[0]
#         for k in averaged_params.keys():
#             for i in range(0, len(w_locals)):
#                 local_sample_number, local_model_params = w_locals[i]
#                 w = local_sample_number / training_num
#                 if i == 0:
#                     averaged_params[k] = local_model_params[k] * w
#                 else:
#                     averaged_params[k] += local_model_params[k] * w
#         return averaged_params

#     def _set_model_global_grads(self, new_state):
#         new_model = copy.deepcopy(self.model_trainer.model)
#         new_model.load_state_dict(new_state)
#         with torch.no_grad():
#             for parameter, new_parameter in zip(self.model_trainer.model.parameters(), new_model.parameters()):
#                 parameter.grad = parameter.data - new_parameter.data
#                 # because we go to the opposite direction of the gradient
#         model_state_dict = self.model_trainer.model.state_dict()
#         new_model_state_dict = new_model.state_dict()
#         for k in dict(self.model_trainer.model.named_parameters()).keys():
#             new_model_state_dict[k] = model_state_dict[k]
#         self.model_trainer.set_model_params(new_model_state_dict)

    
#     def _test_global_model_on_global_data(self, w_global, round_idx, return_val=False):
#         logging.info("################test_global_model_on_global_dataset################")
#         self.model_trainer.set_model_params(w_global)
#         metrics_test = self.model_trainer.test(self.test_global, self.device, self.args)
#         if return_val:
#             return metrics_test
#         # metrics_train = self.model_trainer.test(self.train_global, self.device, self.args)
#         test_acc = metrics_test["test_correct"] / metrics_test["test_total"]
#         test_loss = metrics_test["test_loss"] / metrics_test["test_total"]
#         # train_acc = metrics_train["test_correct"] / metrics_train["test_total"]
#         # train_loss = metrics_train["test_loss"] / metrics_train["test_total"]
#         stats = {"test_acc": test_acc, "test_loss":test_loss}
#         logging.info(stats)
#         if self.args.enable_wandb:
#             wandb.log({"Global Test Acc": test_acc, "round":round_idx})
#             wandb.log({"Global Test Loss": test_loss, "round":round_idx})
    
#     def _local_test_on_all_clients(self, round_idx):
#         logging.info("################local_test_on_all_clients : {}".format(round_idx))
#         train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#         test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#         client = self.client_list[0]
#         for client_idx in range(self.args.client_num_in_total):
#             """
#             Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
#             the training client number is larger than the testing client number
#             """
#             client.update_local_dataset(
#                 0,
#                 self.train_data_local_dict[client_idx],
#                 self.test_data_local_dict[client_idx],
#                 self.train_data_local_num_dict[client_idx],
#             )
#             # train data
#             train_local_metrics = client.local_test(False)
#             train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
#             train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
#             train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

#             # Train and test might have different number of clients
#             if self.test_data_local_dict[client_idx] is None:
#                 continue
#             # test data
#             test_local_metrics = client.local_test(True)
#             test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
#             test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
#             test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

#             """
#             Note: CI environment is CPU-based computing. 
#             The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
#             """
#             if self.args.ci == 1:
#                 break

#         # test on training dataset
#         train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
#         train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

#         # test on test dataset
#         test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
#         test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

#         stats = {"training_acc": train_acc, "training_loss": train_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Train/Acc": train_acc, "round": round_idx})
#             wandb.log({"Train/Loss": train_loss, "round": round_idx})
#         logging.info(stats)

#         stats = {"test_acc": test_acc, "test_loss": test_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Test/Acc": test_acc, "round": round_idx})
#             wandb.log({"Test/Loss": test_loss, "round": round_idx})
#         logging.info(stats)

#     def _local_test_on_validation_set(self, round_idx):
#         logging.info("################local_test_on_validation_set : {}".format(round_idx))

#         if self.val_global is None:
#             self._generate_validation_set()

#         client = self.client_list[0]
#         client.update_local_dataset(0, None, self.val_global, None)
#         # test data
#         test_metrics = client.local_test(True)

#         if self.args.dataset == "stackoverflow_nwp":
#             test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
#             test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
#             stats = {"test_acc": test_acc, "test_loss": test_loss}
#             if self.args.enable_wandb:
#                 wandb.log({"Test/Acc": test_acc, "round": round_idx})
#                 wandb.log({"Test/Loss": test_loss, "round": round_idx})
#         elif self.args.dataset == "stackoverflow_lr":
#             test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
#             test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
#             test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
#             test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
#             stats = {
#                 "test_acc": test_acc,
#                 "test_pre": test_pre,
#                 "test_rec": test_rec,
#                 "test_loss": test_loss,
#             }
#             if self.args.enable_wandb:
#                 wandb.log({"Test/Acc": test_acc, "round": round_idx})
#                 wandb.log({"Test/Pre": test_pre, "round": round_idx})
#                 wandb.log({"Test/Rec": test_rec, "round": round_idx})
#                 wandb.log({"Test/Loss": test_loss, "round": round_idx})
#         else:
#             raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

#         logging.info(stats)


# ############################ feddowg #############################################
# import copy
# import logging
# import random

# import numpy as np
# import torch
# import wandb

# from fedml.ml.trainer.trainer_creator import create_model_trainer
# from .client import Client
# from .optrepo import OptRepo
# from fedml.simulation.sp.fedopt.Dowg import DoWG, CDoWG
# from fedml.simulation.sp.fedopt.FEDDFW import DFW
# # import torch_optimizer as tor_optim
# # from dog import DoG, LDoG, PolynomialDecayAverager

# class FedOptAPI(object):
#     def __init__(self, args, device, dataset, model):
#         self.device = device
#         self.args = args
#         [
#             train_data_num,
#             test_data_num,
#             train_data_global,
#             test_data_global,
#             train_data_local_num_dict,
#             train_data_local_dict,
#             test_data_local_dict,
#             class_num,
#         ] = dataset
#         self.train_global = train_data_global
#         self.test_global = test_data_global
#         self.val_global = None
#         self.train_data_num_in_total = train_data_num
#         self.test_data_num_in_total = test_data_num

#         self.client_list = []
#         self.train_data_local_num_dict = train_data_local_num_dict
#         self.train_data_local_dict = train_data_local_dict
#         self.test_data_local_dict = test_data_local_dict

#         self.model_trainer = create_model_trainer(model, args)
        
#         # Instantiate the optimizer only once
#         self.opt = self._instantiate_opt()
#         self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)

#     def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
#         logging.info("############setup_clients (START)#############")
#         for client_idx in range(self.args.client_num_per_round):
#             c = Client(
#                 client_idx,
#                 train_data_local_dict[client_idx],
#                 test_data_local_dict[client_idx],
#                 train_data_local_num_dict[client_idx],
#                 self.args,
#                 self.device,
#                 self.model_trainer,
#             )
#             self.client_list.append(c)
#         logging.info("############setup_clients (END)#############")

#     def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
#         if client_num_in_total == client_num_per_round:
#             client_indexes = [client_index for client_index in range(client_num_in_total)]
#         else:
#             num_clients = min(client_num_per_round, client_num_in_total)
#             np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
#             client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
#         logging.info("client_indexes = %s" % str(client_indexes))
#         return client_indexes

#     def _generate_validation_set(self, num_samples=10000):
#         test_data_num = len(self.test_global.dataset)
#         sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
#         subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
#         sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
#         self.val_global = sample_testset

#     def _instantiate_opt(self):
#         if self.args.server_optimizer == 'FedAvg':
#             return None  # No optimizer needed for FedAvg
#         elif self.args.server_optimizer == 'DFW':
#             return DFW(self.model_trainer.model.parameters(), eta=.001, momentum=0.9, weight_decay=.00001, eps=1e-5)
#         elif self.args.server_optimizer == 'Dowg':
#             return DoWG(self.model_trainer.model.parameters())
#         else:
#             return OptRepo.name2cls(self.args.server_optimizer)(
#                 self.model_trainer.model.parameters(),
#                 lr=self.args.server_lr,
#             )

#     def train(self):
#         for round_idx in range(self.args.comm_round):
#             w_global = self.model_trainer.get_model_params()
#             logging.info("################ Communication round : {}".format(round_idx))

#             w_locals = []

#             client_indexes = self._client_sampling(
#                 round_idx, self.args.client_num_in_total, self.args.client_num_per_round
#             )
#             logging.info("client_indexes = " + str(client_indexes))

#             for idx, client in enumerate(self.client_list):
#                 # update dataset
#                 client_idx = client_indexes[idx]

#                 client.update_local_dataset(
#                     client_idx,
#                     self.train_data_local_dict[client_idx],
#                     self.test_data_local_dict[client_idx],
#                     self.train_data_local_num_dict[client_idx],
#                 )

#                 # train on new dataset
#                 w = client.train(w_global)
#                 w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

#             # Aggregate the local models
#             w_avg = self._aggregate(w_locals)

#             # Apply the optimizer if using DoWG or other server-side optimizers
#             if self.args.server_optimizer == 'Dowg':
#                 self._apply_dowg_optimizer(w_avg)
#             else:
#                 self.model_trainer.set_model_params(w_avg)
#                 w_global = self.model_trainer.get_model_params()

#             # Test results at the end of the round
#             if round_idx == self.args.comm_round - 1 or round_idx % self.args.frequency_of_the_test == 0:
#                 self._local_test_on_all_clients(round_idx)

#     def _apply_dowg_optimizer(self, w_avg):
#         # Save current optimizer state
#         opt_state = self.opt.state_dict()
        
#         # Set global model gradients based on the aggregated model
#         self._set_model_global_grads(w_avg)
        
#         # Restore the optimizer state to maintain `rt2` and `vt` accumulation
#         self.opt.load_state_dict(opt_state)
        
#         # Perform the optimization step
#         self.opt.step()

#     def _aggregate(self, w_locals):
#         training_num = 0
#         for idx in range(len(w_locals)):
#             (sample_num, averaged_params) = w_locals[idx]
#             training_num += sample_num

#         (sample_num, averaged_params) = w_locals[0]
#         for k in averaged_params.keys():
#             for i in range(0, len(w_locals)):
#                 local_sample_number, local_model_params = w_locals[i]
#                 w = local_sample_number / training_num
#                 if i == 0:
#                     averaged_params[k] = local_model_params[k] * w
#                 else:
#                     averaged_params[k] += local_model_params[k] * w
#         return averaged_params

#     def _set_model_global_grads(self, new_state):
#         new_model = copy.deepcopy(self.model_trainer.model)
#         new_model.load_state_dict(new_state)
#         with torch.no_grad():
#             for parameter, new_parameter in zip(self.model_trainer.model.parameters(), new_model.parameters()):
#                 parameter.grad = parameter.data - new_parameter.data
#                 # because we go to the opposite direction of the gradient
#         model_state_dict = self.model_trainer.model.state_dict()
#         new_model_state_dict = new_model.state_dict()
#         for k in dict(self.model_trainer.model.named_parameters()).keys():
#             new_model_state_dict[k] = model_state_dict[k]
#         self.model_trainer.set_model_params(new_model_state_dict)

#     def _test_global_model_on_global_data(self, w_global, round_idx, return_val=False):
#         logging.info("################test_global_model_on_global_dataset################")
#         self.model_trainer.set_model_params(w_global)
#         metrics_test = self.model_trainer.test(self.test_global, self.device, self.args)
#         if return_val:
#             return metrics_test
#         test_acc = metrics_test["test_correct"] / metrics_test["test_total"]
#         test_loss = metrics_test["test_loss"] / metrics_test["test_total"]
#         stats = {"test_acc": test_acc, "test_loss": test_loss}
#         logging.info(stats)
#         if self.args.enable_wandb:
#             wandb.log({"Global Test Acc": test_acc, "round": round_idx})
#             wandb.log({"Global Test Loss": test_loss, "round": round_idx})
    
#     def _local_test_on_all_clients(self, round_idx):
#         logging.info("################local_test_on_all_clients : {}".format(round_idx))
#         train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#         test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#         client = self.client_list[0]
#         for client_idx in range(self.args.client_num_in_total):
#             client.update_local_dataset(
#                 0,
#                 self.train_data_local_dict[client_idx],
#                 self.test_data_local_dict[client_idx],
#                 self.train_data_local_num_dict[client_idx],
#             )
#             train_local_metrics = client.local_test(False)
#             train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
#             train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
#             train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

#             if self.test_data_local_dict[client_idx] is None:
#                 continue

#             test_local_metrics = client.local_test(True)
#             test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
#             test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
#             test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

#             if self.args.ci == 1:
#                 break

#         train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
#         train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

#         test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
#         test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

#         stats = {"training_acc": train_acc, "training_loss": train_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Train/Acc": train_acc, "round": round_idx})
#             wandb.log({"Train/Loss": train_loss, "round": round_idx})
#         logging.info(stats)

#         stats = {"test_acc": test_acc, "test_loss": test_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Test/Acc": test_acc, "round": round_idx})
#             wandb.log({"Test/Loss": test_loss, "round": round_idx})
#         logging.info(stats)

#     def _local_test_on_validation_set(self, round_idx):
#         logging.info("################local_test_on_validation_set : {}".format(round_idx))

#         if self.val_global is None:
#             self._generate_validation_set()

#         client = self.client_list[0]
#         client.update_local_dataset(0, None, self.val_global, None)
#         test_metrics = client.local_test(True)

#         if self.args.dataset == "stackoverflow_nwp":
#             test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
#             test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
#             stats = {"test_acc": test_acc, "test_loss": test_loss}
#             if self.args.enable_wandb:
#                 wandb.log({"Test/Acc": test_acc, "round": round_idx})
#                 wandb.log({"Test/Loss": test_loss, "round": round_idx})
#         elif self.args.dataset == "stackoverflow_lr":
#             test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
#             test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
#             test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
#             test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
#             stats = {
#                 "test_acc": test_acc,
#                 "test_pre": test_pre,
#                 "test_rec": test_rec,
#                 "test_loss": test_loss,
#             }
#             if self.args.enable_wandb:
#                 wandb.log({"Test/Acc": test_acc, "round": round_idx})
#                 wandb.log({"Test/Pre": test_pre, "round": round_idx})
#                 wandb.log({"Test/Rec": test_rec, "round": round_idx})
#                 wandb.log({"Test/Loss": test_loss, "round": round_idx})
#         else:
#             raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

#         logging.info(stats)


# from fedml.ml.trainer.trainer_creator import create_model_trainer
# from .client import Client
# from .optrepo import OptRepo
# from fedml.simulation.sp.fedopt.Dowg import DoWG, CDoWG
# from fedml.simulation.sp.fedopt.FEDDFW import DFW
# # import torch_optimizer as tor_optim
# # from dog import DoG, LDoG, PolynomialDecayAverager

# class FedOptAPI(object):
#     def __init__(self, args, device, dataset, model):
#         self.device = device
#         self.args = args
#         [
#             train_data_num,
#             test_data_num,
#             train_data_global,
#             test_data_global,
#             train_data_local_num_dict,
#             train_data_local_dict,
#             test_data_local_dict,
#             class_num,
#         ] = dataset
#         self.train_global = train_data_global
#         self.test_global = test_data_global
#         self.val_global = None
#         self.train_data_num_in_total = train_data_num
#         self.test_data_num_in_total = test_data_num

#         self.client_list = []
#         self.train_data_local_num_dict = train_data_local_num_dict
#         self.train_data_local_dict = train_data_local_dict
#         self.test_data_local_dict = test_data_local_dict

#         self.model_trainer = create_model_trainer(model, args)
#         self.eta_g = args.learning_rate
        
#         # Instantiate the optimizer only once
#         self.opt = self._instantiate_opt()
#         self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)

#     def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
#         logging.info("############setup_clients (START)#############")
#         for client_idx in range(self.args.client_num_per_round):
#             c = Client(
#                 client_idx,
#                 train_data_local_dict[client_idx],
#                 test_data_local_dict[client_idx],
#                 train_data_local_num_dict[client_idx],
#                 self.args,
#                 self.device,
#                 self.model_trainer,
#             )
#             self.client_list.append(c)
#         logging.info("############setup_clients (END)#############")

#     def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
#         if client_num_in_total == client_num_per_round:
#             client_indexes = [client_index for client_index in range(client_num_in_total)]
#         else:
#             num_clients = min(client_num_per_round, client_num_in_total)
#             np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
#             client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
#         logging.info("client_indexes = %s" % str(client_indexes))
#         return client_indexes

#     def _generate_validation_set(self, num_samples=10000):
#         test_data_num = len(self.test_global.dataset)
#         sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
#         subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
#         sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
#         self.val_global = sample_testset

#     def _instantiate_opt(self):
#         if self.args.server_optimizer == 'FedAvg':
#             return None  # No optimizer needed for FedAvg
#         elif self.args.server_optimizer == 'DFW':
#             return DFW(self.model_trainer.model.parameters(), eta=.001, momentum=0.9, weight_decay=.00001, eps=1e-5)
#         elif self.args.server_optimizer == 'Dowg':
#             return DoWG(self.model_trainer.model.parameters(), eps=1e-4)
#         else:
#             return OptRepo.name2cls(self.args.server_optimizer)(
#                 self.model_trainer.model.parameters(),
#                 lr=self.args.server_lr,
#             )

#     def train(self):
#         for round_idx in range(self.args.comm_round):
#             w_global = self.model_trainer.get_model_params()
#             logging.info("################ Communication round : {}".format(round_idx))

#             w_locals = []

#             client_indexes = self._client_sampling(
#                 round_idx, self.args.client_num_in_total, self.args.client_num_per_round
#             )
#             logging.info("client_indexes = " + str(client_indexes))

#             for idx, client in enumerate(self.client_list):
#                 # Update dataset
#                 client_idx = client_indexes[idx]

#                 client.update_local_dataset(
#                     client_idx,
#                     self.train_data_local_dict[client_idx],
#                     self.test_data_local_dict[client_idx],
#                     self.train_data_local_num_dict[client_idx],
#                 )

#                 # Train on new dataset
#                 w = client.train(w_global, self.eta_g)
#                 w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

#             # Reset weight after standalone simulation
#             self.model_trainer.set_model_params(w_global)

#             # Aggregate local models
#             w_avg = self._aggregate(w_locals)

#             # Server optimizer
#             self.opt.zero_grad()
#             opt_state = self.opt.state_dict()

#             # Set global gradients and calculate eta_g
#             if self.args.server_optimizer == 'Dowg':
#                 self._apply_dowg_optimizer(w_avg)
#             else:
#                 self._set_model_global_grads(w_avg)
#                 self._instanciate_opt()
#                 self.opt.load_state_dict(opt_state)
#                 self.opt.step()

#             # Test results
#             if round_idx == self.args.comm_round - 1:
#                 self._local_test_on_all_clients(round_idx)
#             elif round_idx % self.args.frequency_of_the_test == 0:
#                 if self.args.dataset.startswith("stackoverflow"):
#                     self._local_test_on_validation_set(round_idx)
#                 else:
#                     self._local_test_on_all_clients(round_idx)

#     def _apply_dowg_optimizer(self, w_avg):
#         # Save the current optimizer state
#         opt_state = self.opt.state_dict()

#         # Set global model gradients based on the aggregated model
#         self._set_model_global_grads(w_avg)

#         # Restore the optimizer state to maintain `rt2` and `vt` accumulation
#         self._instantiate_opt()  # Reinstantiate the optimizer
#         self.opt.load_state_dict(opt_state)

#         # Perform the optimization step
#         self.opt.step()

#         # Calculate eta_g
#         rt2 = self.opt.state['rt2']
#         vt = self.opt.state['vt']
#         eta_g = rt2 / torch.sqrt(vt + self.opt.defaults['eps'])  # Calculate eta_g

#         # Store eta_g to pass to clients
#         self.eta_g = eta_g.item()

               
                
#     def _aggregate(self, w_locals):
#         training_num = 0
#         for idx in range(len(w_locals)):
#             (sample_num, averaged_params) = w_locals[idx]
#             training_num += sample_num

#         (sample_num, averaged_params) = w_locals[0]
#         for k in averaged_params.keys():
#             for i in range(0, len(w_locals)):
#                 local_sample_number, local_model_params = w_locals[i]
#                 w = local_sample_number / training_num
#                 if i == 0:
#                     averaged_params[k] = local_model_params[k] * w
#                 else:
#                     averaged_params[k] += local_model_params[k] * w
#         return averaged_params

#     def _set_model_global_grads(self, new_state):
#         new_model = copy.deepcopy(self.model_trainer.model)
#         new_model.load_state_dict(new_state)
#         with torch.no_grad():
#             for parameter, new_parameter in zip(self.model_trainer.model.parameters(), new_model.parameters()):
#                 parameter.grad = parameter.data - new_parameter.data
#                 # because we go to the opposite direction of the gradient
#         model_state_dict = self.model_trainer.model.state_dict()
#         new_model_state_dict = new_model.state_dict()
#         for k in dict(self.model_trainer.model.named_parameters()).keys():
#             new_model_state_dict[k] = model_state_dict[k]
#         self.model_trainer.set_model_params(new_model_state_dict)

#     def _test_global_model_on_global_data(self, w_global, round_idx, return_val=False):
#         logging.info("################test_global_model_on_global_dataset################")
#         self.model_trainer.set_model_params(w_global)
#         metrics_test = self.model_trainer.test(self.test_global, self.device, self.args)
#         if return_val:
#             return metrics_test
#         test_acc = metrics_test["test_correct"] / metrics_test["test_total"]
#         test_loss = metrics_test["test_loss"] / metrics_test["test_total"]
#         stats = {"test_acc": test_acc, "test_loss": test_loss}
#         logging.info(stats)
#         if self.args.enable_wandb:
#             wandb.log({"Global Test Acc": test_acc, "round": round_idx})
#             wandb.log({"Global Test Loss": test_loss, "round": round_idx})
    
#     def _local_test_on_all_clients(self, round_idx):
#         logging.info("################local_test_on_all_clients : {}".format(round_idx))
#         train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#         test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#         client = self.client_list[0]
#         for client_idx in range(self.args.client_num_in_total):
#             client.update_local_dataset(
#                 0,
#                 self.train_data_local_dict[client_idx],
#                 self.test_data_local_dict[client_idx],
#                 self.train_data_local_num_dict[client_idx],
#             )
#             train_local_metrics = client.local_test(False)
#             train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
#             train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
#             train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

#             if self.test_data_local_dict[client_idx] is None:
#                 continue

#             test_local_metrics = client.local_test(True)
#             test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
#             test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
#             test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

#             if self.args.ci == 1:
#                 break

#         train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
#         train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

#         test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
#         test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

#         stats = {"training_acc": train_acc, "training_loss": train_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Train/Acc": train_acc, "round": round_idx})
#             wandb.log({"Train/Loss": train_loss, "round": round_idx})
#         logging.info(stats)

#         stats = {"test_acc": test_acc, "test_loss": test_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Test/Acc": test_acc, "round": round_idx})
#             wandb.log({"Test/Loss": test_loss, "round": round_idx})
#         logging.info(stats)

#     def _local_test_on_validation_set(self, round_idx):
#         logging.info("################local_test_on_validation_set : {}".format(round_idx))

#         if self.val_global is None:
#             self._generate_validation_set()

#         client = self.client_list[0]
#         client.update_local_dataset(0, None, self.val_global, None)
#         test_metrics = client.local_test(True)

#         if self.args.dataset == "stackoverflow_nwp":
#             test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
#             test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
#             stats = {"test_acc": test_acc, "test_loss": test_loss}
#             if self.args.enable_wandb:
#                 wandb.log({"Test/Acc": test_acc, "round": round_idx})
#                 wandb.log({"Test/Loss": test_loss, "round": round_idx})
#         elif self.args.dataset == "stackoverflow_lr":
#             test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
#             test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
#             test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
#             test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
#             stats = {
#                 "test_acc": test_acc,
#                 "test_pre": test_pre,
#                 "test_rec": test_rec,
#                 "test_loss": test_loss,
#             }
#             if self.args.enable_wandb:
#                 wandb.log({"Test/Acc": test_acc, "round": round_idx})
#                 wandb.log({"Test/Pre": test_pre, "round": round_idx})
#                 wandb.log({"Test/Rec": test_rec, "round": round_idx})
#                 wandb.log({"Test/Loss": test_loss, "round": round_idx})
#         else:
#             raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

#         logging.info(stats)
