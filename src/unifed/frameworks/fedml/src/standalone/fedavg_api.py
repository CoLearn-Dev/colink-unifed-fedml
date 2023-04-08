import copy
import logging
import random
from time import sleep, time

import flbenchmark.logging
import torch
from fedml.simulation.sp import fedavg
from sklearn.metrics import roc_auc_score

from .client import Client

AUC = [
    'breast_horizontal',
    'default_credit_horizontal',
    'give_credit_horizontal',
    'breast_vertical',
    'default_credit_vertical',
    'give_credit_vertical',
]


def getbyte(w):
    ret = 0
    for key, value in w.items():
        ret += value.numel() * value.element_size()
    return ret


class FedAvgAPI(fedavg.FedAvgAPI):
    def __init__(self, dataset, device, config, model_trainer, is_regression):
        self.device = device
        self.config = config
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

        self.is_regression = is_regression

        self.model_trainer = model_trainer

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            self.model_trainer,
        )

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.config['training']['client_per_round']):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.config['training'], self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        communication_time, communication_bytes = 0, 0
        report_my = 0

        w_global = self.model_trainer.get_model_params()
        with flbenchmark.logging.BasicLogger(id=0, agent_type='aggregator') as logger:
            with logger.training():
                for idx, client in enumerate(self.client_list):
                    # client.logger.start()
                    client.logger.training_start()

                for round_idx in range(self.config['training']['epochs']):
                    with logger.training_round() as tr:
                        tr.report_metric(
                            'client_num', self.config["training"]["client_per_round"])
                        logging.info(
                            "################Communication round : {}".format(round_idx))

                        for idx, client in enumerate(self.client_list):
                            # client.logger.start()
                            client.logger.training_round_start()

                        w_locals = []

                        """
                        for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                        Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                        """
                        client_indexes = self._client_sampling(round_idx, self.config['training']['tot_client_num'],
                                                               self.config['training']['client_per_round'])
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
                            w = client.train(copy.deepcopy(w_global))
                            communication_bytes += getbyte(w)
                            # self.logger.info("local weights = " + str(w))
                            w_locals.append(
                                (client.get_sample_number(), copy.deepcopy(w)))

                            # client back to server
                            communication_time += time() - client.time

                        # update global weights
                        with logger.computation() as c:
                            w_global = self._aggregate(w_locals)

                        for idx, client in enumerate(self.client_list):
                            with logger.communication(target_id=idx + 1) as c:
                                c.report_metric('byte', getbyte(w_global))

                        timea = time()
                        self.model_trainer.set_model_params(w_global)
                        # server to client time
                        communication_time += time() - timea

                        # test results
                        # at last round
                        if round_idx == self.config['training']['epochs'] - 1:
                            btime = time()
                            report_my = self._local_test_on_all_clients(
                                round_idx)
                            finaltime = time() - btime
                        # per {frequency_of_the_test} round
                        elif round_idx % self.config['training'].get('frequency_of_the_test', 1) == 0:
                            # if self.args.dataset.startswith("stackoverflow"):
                            pass
                            # self._local_test_on_validation_set(round_idx)
                            # else:
                            # self._local_test_on_all_clients(round_idx)

                        for idx, client in enumerate(self.client_list):
                            # client.logger.start()
                            client.logger.training_round_end()

                for idx, client in enumerate(self.client_list):
                    client.logger.end()

                # print("Total communication time is {}".format(communication_time))
                # print("Total communication cost is {}".format(communication_bytes * 2))
                # print("Total communication round is {}".format(self.args.comm_round))
                for idx, client in enumerate(self.client_list):
                    # client.logger.start()
                    client.logger.training_end()

            with logger.model_evaluation() as e:
                sleep(finaltime)
                if self.is_regression:
                    e.report_metric('mse', report_my)
                elif self.config['dataset'] in AUC:
                    # e.report_metric('accuracy', report_my[1] * 100)
                    e.report_metric('auc', report_my)
                else:
                    e.report_metric('accuracy', report_my)

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(
            range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(
            self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(
            subset, batch_size=self.config['training']['batch_size'])
        self.val_global = sample_testset

    def _local_test_on_all_clients(self, round_idx):

        logging.info(
            "################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        predict_my, target_my = None, None

        client = self.client_list[0]

        for client_idx in range(self.config['training']['tot_client_num']):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(
                copy.deepcopy(train_local_metrics['test_total']))
            if not self.is_regression:
                train_metrics['num_correct'].append(
                    copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(
                copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(
                copy.deepcopy(test_local_metrics['test_total']))
            if not self.is_regression:
                test_metrics['num_correct'].append(
                    copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(
                copy.deepcopy(test_local_metrics['test_loss']))

            if self.config['dataset'] in AUC:
                if predict_my == None:
                    predict_my = test_local_metrics['predict']
                else:
                    predict_my = torch.cat(
                        (predict_my, test_local_metrics['predict']))
                if target_my == None:
                    target_my = test_local_metrics['targety']
                else:
                    target_my = torch.cat(
                        (target_my, test_local_metrics['targety']))
                # predict_my += test_local_metrics['predict']
                # target_my += test_local_metrics['targety']

        # test on training dataset
        train_acc = 0
        if not self.is_regression:
            train_acc = sum(train_metrics['num_correct']) / \
                sum(train_metrics['num_samples'])

        train_loss = sum(train_metrics['losses']) / \
            sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = 0
        if not self.is_regression:
            test_acc = sum(test_metrics['num_correct']) / \
                sum(test_metrics['num_samples'])

        test_loss = sum(test_metrics['losses']) / \
            sum(test_metrics['num_samples'])

        if self.is_regression:
            return test_loss
        elif self.config['dataset'] not in AUC:
            return test_acc
        else:
            return roc_auc_score(target_my.cpu(), predict_my.cpu())

    def _local_test_on_validation_set(self, round_idx):

        logging.info(
            "################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.config['dataset'] == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / \
                test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.config['dataset'] == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / \
                test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / \
                test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre,
                     'test_rec': test_rec, 'test_loss': test_loss}
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Pre": test_pre, "round": round_idx})
            # wandb.log({"Test/Rec": test_rec, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception(
                "Unknown format to log metrics for dataset {}!" % self.config['dataset'])

        logging.info(stats)
