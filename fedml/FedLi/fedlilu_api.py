#################### for others ###################################################
import copy
import logging
import random

import numpy as np
import torch
import wandb
from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from .optrepo import OptRepo
from fedml.simulation.sp.fedopt.Dowg import DoWG, CDoWG
from fedml.simulation.sp.fedopt.FEDDFW import DFW

class FedLiLUAPI(object):
    def __init__(self, args, device, dataset, model):
        from fedml.ml.trainer.trainer_creator import create_model_trainer
        
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

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
        if args.server_optim:
                self._instanciate_opt()
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )

    def _setup_clients(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                copy.deepcopy(model_trainer),
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _set_model_global_grads(self, new_state):
        new_model = copy.deepcopy(self.model_trainer.model)
        new_model.load_state_dict(new_state)
        self.model_trainer.model.to(f'cuda:{self.args.gpu_id}')
        new_model.to(f'cuda:{self.args.gpu_id}')
        with torch.no_grad():
            for parameter, new_parameter in zip(self.model_trainer.model.parameters(), new_model.parameters()):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.model_trainer.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.model_trainer.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        self.model_trainer.set_model_params(new_model_state_dict)
    
    
    def _instanciate_opt(self):
        if self.args.server_optimizer == 'DFW':
            self.opt = DFW(self.model_trainer.model.parameters(), lr=0.01, momentum=0.9, weight_decay=.00001, eps=1e-5)
        elif self.args.server_optimizer == 'sgd':
            self.opt = torch.optim.SGD(self.model_trainer.model.parameters(), lr=0.01, momentum=0.9)
        else:
            self.opt = OptRepo.name2cls(self.args.server_optimizer)(
                self.model_trainer.model.parameters(),
                lr=self.args.server_lr,
                eps=1e-3
            )

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)

        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []

            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                w, loss = client.train(copy.deepcopy(w_global))
                mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w), copy.deepcopy(loss)))

            mlops.event("agg", event_started=True, event_value=str(round_idx))
            w_global, avg_loss = self._aggregate(w_locals)

            if self.args.server_optim and self.args.server_optimizer == 'DFW':
                self.opt.zero_grad()
                opt_state = self.opt.state_dict()
                self._set_model_global_grads(w_global)
                self._instanciate_opt()
                self.opt.load_state_dict(opt_state)
                
                # Ensure avg_loss is a floating point number before creating the tensor
                avg_loss_tensor = torch.tensor(float(avg_loss), requires_grad=True)
                avg_loss_tensor.backward()
                self.opt.step(avg_loss_tensor)
            else:
                self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))
            w_global = self.model_trainer.get_model_params()
            self._test_global_model_on_global_data(w_global, round_idx, False)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()




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
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params, local_loss) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params, local_loss) = w_locals[0]
        averaged_loss = 0  # Initialize averaged_loss
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params, local_loss = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                    if local_loss is not None:
                        averaged_loss = local_loss * w
                    else:
                        averaged_loss = 0  # or some other default value
                else:
                    averaged_params[k] += local_model_params[k] * w
                    if local_loss is not None:
                        averaged_loss += local_loss * w
                    # If local_loss is None, you might skip adding it to averaged_loss

        return averaged_params, averaged_loss

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            temp_w = []
            for (_, local_w) in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

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
            if self.test_data_local_dict[client_idx] is None:
                continue
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

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

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

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        mlops.log({"Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

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

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

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

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)