# import copy
# import logging
# import random

# import numpy as np
# import torch
# import wandb

# from fedml import mlops
# from fedml.ml.trainer.trainer_creator import create_model_trainer
# from .client import Client

# import copy
# import logging
# import random

# import numpy as np
# import torch
# import wandb

# from fedml import mlops
# from fedml.ml.trainer.trainer_creator import create_model_trainer
# from .client import Client


# class FedEXPAPI(object):
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

#         logging.info("model = {}".format(model))
#         self.model_trainer = create_model_trainer(model, args)
#         self.model = model
#         logging.info("self.model_trainer = {}".format(self.model_trainer))

#         self._setup_clients(
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
#         )

#         # Initialize epsilon for FedExP
#         self.epsilon = args.epsilon

#     def _setup_clients(
#         self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
#     ):
#         logging.info("############setup_clients (START)#############")
#         for client_idx in range(self.args.client_num_per_round):
#             c = Client(
#                 client_idx,
#                 train_data_local_dict[client_idx],
#                 test_data_local_dict[client_idx],
#                 train_data_local_num_dict[client_idx],
#                 self.args,
#                 self.device,
#                 model_trainer,
#             )
#             self.client_list.append(c)
#         logging.info("############setup_clients (END)#############")
        
#     def train(self):
#         logging.info("self.model_trainer = {}".format(self.model_trainer))
#         w_global = self.model_trainer.get_model_params()
#         mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
#         mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
#         mlops.log_round_info(self.args.comm_round, -1)
#         for round_idx in range(self.args.comm_round):

#             logging.info("################Communication round : {}".format(round_idx))

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
#                 mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
#                 w = client.train(copy.deepcopy(w_global))
#                 mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
#                 w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

#             # Update global weights using FedExP
#             mlops.event("agg", event_started=True, event_value=str(round_idx))
#             w_global = self._aggregate_fedexp(w_locals)
#             self.model_trainer.set_model_params(w_global)
#             mlops.event("agg", event_started=False, event_value=str(round_idx))

#             # test results
#             if round_idx == self.args.comm_round - 1:
#                 self._local_test_on_all_clients(round_idx)
#             elif round_idx % self.args.frequency_of_the_test == 0:
#                 if self.args.dataset.startswith("stackoverflow"):
#                     self._local_test_on_validation_set(round_idx)
#                 else:
#                     self._local_test_on_all_clients(round_idx)

#             mlops.log_round_info(self.args.comm_round, round_idx)

#         mlops.log_training_finished_status()
#         mlops.log_aggregation_finished_status()

#     def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
#         if client_num_in_total == client_num_per_round:
#             client_indexes = [client_index for client_index in range(client_num_in_total)]
#         else:
#             num_clients = min(client_num_per_round, client_num_in_total)
#             np.random.seed(round_idx)
#             client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
#         logging.info("client_indexes = %s" % str(client_indexes))
#         return client_indexes

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
    
#     def _aggregate_fedexp(self, w_locals):
#         # Calculate the average of local model updates
#         w_avg = self._aggregate(w_locals)

#         # Calculate delta (difference) between local and global models
#         deltas = []
#         for (_, local_w) in w_locals:
#             delta = {}
#             for k in local_w.keys():
#                 delta[k] = local_w[k] - w_avg[k]  # Calculate difference for each parameter
#             deltas.append(delta)

#         # Calculate average delta
#         avg_delta = {k: torch.mean(torch.stack([delta[k] for delta in deltas]), dim=0) for k in w_avg.keys()}

#         # Calculate eta_g (step size)
#         delta_norms_sum = np.sum([torch.norm(delta[k]) ** 2 for delta in deltas for k in delta.keys()])
#         avg_delta_norm = np.sum([torch.norm(avg_delta[k]) ** 2 for k in avg_delta.keys()])

#         eta_g = max(1, delta_norms_sum / (2 * len(deltas) * (avg_delta_norm + self.epsilon)))

#         # Update global model using FedExP rule
#         for k in w_avg.keys():
#             w_avg[k] = w_avg[k] - eta_g * avg_delta[k]

#         return w_avg


#     def _update_global_model(self, w_global, avg_update, eta_g):
#         new_weights = {k: v - self.args.learning_rate * eta_g * avg_update[k] for k, v in w_global.items()}
#         return new_weights
    
    
#     def _local_test_on_all_clients(self, round_idx):
#             logging.info("################local_test_on_all_clients : {}".format(round_idx))

#             train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#             test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#             client = self.client_list[0]

#             for client_idx in range(self.args.client_num_in_total):
#                 if self.test_data_local_dict[client_idx] is None:
#                     continue
#                 client.update_local_dataset(
#                     0,
#                     self.train_data_local_dict[client_idx],
#                     self.test_data_local_dict[client_idx],
#                     self.train_data_local_num_dict[client_idx],
#                 )
#                 train_local_metrics = client.local_test(False)
#                 train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
#                 train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
#                 train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

#                 test_local_metrics = client.local_test(True)
#                 test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
#                 test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
#                 test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

#             train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
#             train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

#             test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
#             test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

#             stats = {"training_acc": train_acc, "training_loss": train_loss}
#             if self.args.enable_wandb:
#                 wandb.log({"Train/Acc": train_acc, "round": round_idx})
#                 wandb.log({"Train/Loss": train_loss, "round": round_idx})

#             mlops.log({"Train/Acc": train_acc, "round": round_idx})
#             mlops.log({"Train/Loss": train_loss, "round": round_idx})
#             logging.info(stats)

#             stats = {"test_acc": test_acc, "test_loss": test_loss}
#             if self.args.enable_wandb:
#                 wandb.log({"Test/Acc": test_acc, "round": round_idx})
#                 wandb.log({"Test/Loss": test_loss, "round": round_idx})

#             mlops.log({"Test/Acc": test_acc, "round": round_idx})
#             mlops.log({"Test/Loss": test_loss, "round": round_idx})
#             logging.info(stats)

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

#             mlops.log({"Test/Acc": test_acc, "round": round_idx})
#             mlops.log({"Test/Loss": test_loss, "round": round_idx})

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

#             mlops.log({"Test/Acc": test_acc, "round": round_idx})
#             mlops.log({"Test/Pre": test_pre, "round": round_idx})
#             mlops.log({"Test/Rec": test_rec, "round": round_idx})
#             mlops.log({"Test/Loss": test_loss, "round": round_idx})
#         else:
#             raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

#         logging.info(stats)

# import copy
# import logging
# import random

# import numpy as np
# import torch
# import wandb

# from fedml import mlops
# from fedml.ml.trainer.trainer_creator import create_model_trainer
# from .client import Client


# class FedEXPAPI(object):
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

#         logging.info("model = {}".format(model))
#         self.model_trainer = create_model_trainer(model, args)
#         self.model = model
#         logging.info("self.model_trainer = {}".format(self.model_trainer))

#         self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer)

#         # Initialize epsilon for FedExP
#         self.epsilon = args.epsilon

#     def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
#         logging.info("############setup_clients (START)#############")
#         for client_idx in range(self.args.client_num_per_round):
#             c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx], train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
#             self.client_list.append(c)
#         logging.info("############setup_clients (END)#############")

#     def train(self):
#         logging.info("self.model_trainer = {}".format(self.model_trainer))
#         w_global = self.model_trainer.get_model_params()
#         mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
#         mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
#         mlops.log_round_info(self.args.comm_round, -1)
#         for round_idx in range(self.args.comm_round):

#             logging.info("################Communication round : {}".format(round_idx))

#             w_locals = []

#             client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total, self.args.client_num_per_round)
#             logging.info("client_indexes = " + str(client_indexes))

#             for idx, client in enumerate(self.client_list):
#                 # update dataset
#                 client_idx = client_indexes[idx]
#                 client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx], self.test_data_local_dict[client_idx], self.train_data_local_num_dict[client_idx])

#                 # train on new dataset
#                 mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
#                 w = client.train(copy.deepcopy(w_global))
#                 mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
#                 w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

#             # Update global weights using FedExP
#             mlops.event("agg", event_started=True, event_value=str(round_idx))
#             w_global = self._aggregate_fedexp(w_locals)
#             self.model_trainer.set_model_params(w_global)
#             mlops.event("agg", event_started=False, event_value=str(round_idx))

#             # test results
#             if round_idx == self.args.comm_round - 1:
#                 self._local_test_on_all_clients(round_idx)
#             elif round_idx % self.args.frequency_of_the_test == 0:
#                 if self.args.dataset.startswith("stackoverflow"):
#                     self._local_test_on_validation_set(round_idx)
#                 else:
#                     self._local_test_on_all_clients(round_idx)

#             mlops.log_round_info(self.args.comm_round, round_idx)

#         mlops.log_training_finished_status()
#         mlops.log_aggregation_finished_status()

#     def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
#         if client_num_in_total == client_num_per_round:
#             client_indexes = [client_index for client_index in range(client_num_in_total)]
#         else:
#             num_clients = min(client_num_per_round, client_num_in_total)
#             np.random.seed(round_idx)
#             client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
#         logging.info("client_indexes = %s" % str(client_indexes))
#         return client_indexes

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

#     def _aggregate_fedexp(self, w_locals):
#         # Calculate the average of local model updates
#         w_avg = self._aggregate(w_locals)

#         # Calculate delta (difference) between local and global models
#         deltas = []
#         for (_, local_w) in w_locals:
#             delta = {}
#             for k in local_w.keys():
#                 delta[k] = local_w[k] - w_avg[k]  # Calculate difference for each parameter
#             deltas.append(delta)

#         # Calculate average delta
#         avg_delta = {k: torch.mean(torch.stack([delta[k] for delta in deltas]), dim=0) for k in w_avg.keys()}

#         # Calculate eta_g (step size)
#         delta_norms_sum = np.sum([torch.norm(delta[k]).item() ** 2 for delta in deltas for k in delta.keys()])
#         avg_delta_norm = np.sum([torch.norm(avg_delta[k]).item() ** 2 for k in avg_delta.keys()])

#         eta_g = max(1, delta_norms_sum / (2 * len(deltas) * (avg_delta_norm + self.epsilon)))

#         # Update global model using FedExP rule
#         for k in w_avg.keys():
#             w_avg[k] = w_avg[k] - eta_g * avg_delta[k]

#         return w_avg


#     def _local_test_on_all_clients(self, round_idx):
#         logging.info("################local_test_on_all_clients : {}".format(round_idx))

#         train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
#         test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#         client = self.client_list[0]

#         for client_idx in range(self.args.client_num_in_total):
#             if self.test_data_local_dict[client_idx] is None:
#                 continue
#             client.update_local_dataset(0, self.train_data_local_dict[client_idx], self.test_data_local_dict[client_idx], self.train_data_local_num_dict[client_idx])
#             train_local_metrics = client.local_test(False)
#             train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
#             train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
#             train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

#             test_local_metrics = client.local_test(True)
#             test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
#             test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
#             test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

#         train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
#         train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

#         test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
#         test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

#         stats = {"training_acc": train_acc, "training_loss": train_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Train/Acc": train_acc, "round": round_idx})
#             wandb.log({"Train/Loss": train_loss, "round": round_idx})

#         mlops.log({"Train/Acc": train_acc, "round": round_idx})
#         mlops.log({"Train/Loss": train_loss, "round": round_idx})
#         logging.info(stats)

#         stats = {"test_acc": test_acc, "test_loss": test_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Test/Acc": test_acc, "round": round_idx})
#             wandb.log({"Test/Loss": test_loss, "round": round_idx})

#         mlops.log({"Test/Acc": test_acc, "round": round_idx})
#         mlops.log({"Test/Loss": test_loss, "round": round_idx})
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

#             mlops.log({"Test/Acc": test_acc, "round": round_idx})
#             mlops.log({"Test/Loss": test_loss, "round": round_idx})

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

#             mlops.log({"Test/Acc": test_acc, "round": round_idx})
#             mlops.log({"Test/Pre": test_pre, "round": round_idx})
#             mlops.log({"Test/Rec": test_rec, "round": round_idx})
#             mlops.log({"Test/Loss": test_loss, "round": round_idx})
#         else:
#             raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

#         logging.info(stats)

# import copy
# import logging
# import random

# import numpy as np
# import torch
# import wandb

# from fedml import mlops
# from fedml.ml.trainer.trainer_creator import create_model_trainer
# from .client import Client


# class FedEXPAPI(object):
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

#         logging.info("model = {}".format(model))
#         self.model_trainer = create_model_trainer(model, args)
#         self.model = model
#         logging.info("self.model_trainer = {}".format(self.model_trainer))

#         self._setup_clients(
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
#         )

#     def _setup_clients(
#         self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
#     ):
#         logging.info("############setup_clients (START)#############")
#         for client_idx in range(self.args.client_num_per_round):
#             c = Client(
#                 client_idx,
#                 train_data_local_dict[client_idx],
#                 test_data_local_dict[client_idx],
#                 train_data_local_num_dict[client_idx],
#                 self.args,
#                 self.device,
#                 model_trainer,
#             )
#             self.client_list.append(c)
#         logging.info("############setup_clients (END)#############")

#     def train(self):
#         logging.info("self.model_trainer = {}".format(self.model_trainer))
#         w_global = self.model_trainer.get_model_params()
#         mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
#         mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
#         mlops.log_round_info(self.args.comm_round, -1)

#         for round_idx in range(self.args.comm_round):
#             logging.info("################Communication round : {}".format(round_idx))

#             w_locals = []

#             client_indexes = self._client_sampling(
#                 round_idx, self.args.client_num_in_total, self.args.client_num_per_round
#             )
#             logging.info("client_indexes = " + str(client_indexes))

#             for idx, client in enumerate(self.client_list):
#                 client_idx = client_indexes[idx]
#                 client.update_local_dataset(
#                     client_idx,
#                     self.train_data_local_dict[client_idx],
#                     self.test_data_local_dict[client_idx],
#                     self.train_data_local_num_dict[client_idx],
#                 )

#                 mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
#                 w = client.train(copy.deepcopy(w_global))
#                 mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
#                 w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

#             mlops.event("agg", event_started=True, event_value=str(round_idx))
#             avg_update, eta_g = self._aggregate(w_locals)
#             w_global = self._update_global_model(w_global, avg_update, eta_g)
#             self.model_trainer.set_model_params(w_global)
#             mlops.event("agg", event_started=False, event_value=str(round_idx))

#             if round_idx == self.args.comm_round - 1:
#                 self._local_test_on_all_clients(round_idx)
#             elif round_idx % self.args.frequency_of_the_test == 0:
#                 if self.args.dataset.startswith("stackoverflow"):
#                     self._local_test_on_validation_set(round_idx)
#                 else:
#                     self._local_test_on_all_clients(round_idx)

#             mlops.log_round_info(self.args.comm_round, round_idx)

#         mlops.log_training_finished_status()
#         mlops.log_aggregation_finished_status()

#     def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
#         if client_num_in_total == client_num_per_round:
#             client_indexes = [client_index for client_index in range(client_num_in_total)]
#         else:
#             num_clients = min(client_num_per_round, client_num_in_total)
#             np.random.seed(round_idx)
#             client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
#         logging.info("client_indexes = %s" % str(client_indexes))
#         return client_indexes

#     def _aggregate(self, w_locals):
#         updates = [local_weights for _, local_weights in w_locals]
        
#         # Initialize the average update with zeros
#         avg_update = copy.deepcopy(updates[0])
#         for k in avg_update.keys():
#             avg_update[k] = torch.zeros_like(avg_update[k], dtype=torch.float32)
        
#         # Sum all updates
#         for update in updates:
#             for k in avg_update.keys():
#                 avg_update[k] += update[k].float()
        
#         # Divide by the number of updates to get the average
#         for k in avg_update.keys():
#             avg_update[k] /= len(updates)
        
#         # Compute eta_g
#         norm_avg_update = sum([torch.norm(v.float())**2 for v in avg_update.values()])
#         sum_norm_updates = sum([torch.norm(v.float())**2 for update in updates for v in update.values()])
#         eta_g = max(1, sum_norm_updates / (2 * len(updates) * (norm_avg_update + self.args.epsilon)))
        
#         return avg_update, eta_g

#     def _update_global_model(self, w_global, avg_update, eta_g):
#         new_weights = {k: v - self.args.lr_global * eta_g * avg_update[k] for k, v in w_global.items()}
#         return new_weights

#     def _local_test_on_all_clients(self, round_idx):
#         logging.info("################local_test_on_all_clients : {}".format(round_idx))

#         train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
#         test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

#         client = self.client_list[0]

#         for client_idx in range(self.args.client_num_in_total):
#             if self.test_data_local_dict[client_idx] is None:
#                 continue
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

#             test_local_metrics = client.local_test(True)
#             test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
#             test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
#             test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

#         train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
#         train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

#         test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
#         test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

#         stats = {"training_acc": train_acc, "training_loss": train_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Train/Acc": train_acc, "round": round_idx})
#             wandb.log({"Train/Loss": train_loss, "round": round_idx})

#         mlops.log({"Train/Acc": train_acc, "round": round_idx})
#         mlops.log({"Train/Loss": train_loss, "round": round_idx})
#         logging.info(stats)

#         stats = {"test_acc": test_acc, "test_loss": test_loss}
#         if self.args.enable_wandb:
#             wandb.log({"Test/Acc": test_acc, "round": round_idx})
#             wandb.log({"Test/Loss": test_loss, "round": round_idx})

#         mlops.log({"Test/Acc": test_acc, "round": round_idx})
#         mlops.log({"Test/Loss": test_loss, "round": round_idx})
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

#             mlops.log({"Test/Acc": test_acc, "round": round_idx})
#             mlops.log({"Test/Loss": test_loss, "round": round_idx})

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

#             mlops.log({"Test/Acc": test_acc, "round": round_idx})
#             mlops.log({"Test/Pre": test_pre, "round": round_idx})
#             mlops.log({"Test/Rec": test_rec, "round": round_idx})
#             mlops.log({"Test/Loss": test_loss, "round": round_idx})
#         else:
#             raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

#         logging.info(stats)

import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client


class FedEXPAPI(object):
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

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
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
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            delta_locals = []

            # Client sampling
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
                mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                w_local = client.train(copy.deepcopy(w_global))   #### local training update rule
                mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))

                delta_local = self._compute_model_difference(w_global, w_local)
                delta_locals.append((client.get_sample_number(), delta_local))

            # Compute global model difference
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            delta_global = self._aggregate_deltas(delta_locals)
            eta_g = self._compute_dynamic_learning_rate(delta_locals, delta_global)
            w_global = self._update_global_model(w_global, delta_global, eta_g)

            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            # Test results
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _compute_model_difference(self, w_global, w_local):
        delta = copy.deepcopy(w_global)
        for k in delta.keys():
            delta[k] = w_global[k] - w_local[k]
        return delta

    # def _aggregate_deltas(self, delta_locals):
    #     training_num = 0
    #     num_clients = len(delta_locals)
    #     for idx in range(len(delta_locals)):
    #         (sample_num, delta) = delta_locals[idx]
    #         training_num += sample_num

    #     (sample_num, delta) = delta_locals[0]
    #     aggregated_delta = copy.deepcopy(delta)
    #     for k in aggregated_delta.keys():
    #         aggregated_delta[k] = torch.zeros_like(aggregated_delta[k], dtype=torch.float32)

    #     for idx in range(len(delta_locals)):
    #         (sample_num, delta) = delta_locals[idx]
    #         for k in aggregated_delta.keys():
    #             # Ensure the types are float32 for both aggregated_delta and delta
    #             aggregated_delta[k] = aggregated_delta[k].to(torch.float32)
    #             delta[k] = delta[k].to(torch.float32)

    #             # Perform the weighted aggregation
    #             aggregated_delta[k] += (sample_num / training_num) * delta[k]
    #     return aggregated_delta
    
    def _aggregate_deltas(self, delta_locals):
        num_clients = len(delta_locals)  # This is M, the number of clients

        # Use the first delta to initialize the aggregated_delta with zeros of the same shape
        (sample_num, delta) = delta_locals[0]
        aggregated_delta = copy.deepcopy(delta)
        for k in aggregated_delta.keys():
            aggregated_delta[k] = torch.zeros_like(aggregated_delta[k], dtype=torch.float32)

        # Sum up the deltas
        for _, delta in delta_locals:
            for k in aggregated_delta.keys():
                aggregated_delta[k] += delta[k]

        # Average the summed deltas
        for k in aggregated_delta.keys():
            aggregated_delta[k] /= num_clients  # This averages the delta updates

        return aggregated_delta




    def _compute_dynamic_learning_rate(self, delta_locals, delta_global):
        norm_square = lambda x: sum(torch.norm(v.to(torch.float32)) ** 2 for v in x.values())
        delta_global_norm = norm_square(delta_global)
        delta_locals_norm_sum = sum(norm_square(delta) for _, delta in delta_locals)
        eta_g = max(1, delta_locals_norm_sum / (2 * len(delta_locals) * (delta_global_norm + self.args.epsilon)))
        return eta_g


    def _update_global_model(self, w_global, delta_global, eta_g):
        updated_global = copy.deepcopy(w_global)
        for k in updated_global.keys():
            updated_global[k] = w_global[k] - eta_g * delta_global[k]
        return updated_global

    def _local_test_on_all_clients(self, round_idx):
        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

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

    def _local_test_on_validation_set(self, round_idx):
        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
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
