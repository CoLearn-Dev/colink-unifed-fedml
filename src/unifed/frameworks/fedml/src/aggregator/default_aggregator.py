import copy
import logging
import traceback

import numpy as np
import torch
import wandb
from fedml import mlops
from fedml.ml.aggregator.default_aggregator import DefaultServerAggregator
from sklearn.metrics import roc_auc_score
from torch import nn

from ..logger import LoggerManager

AUC = [
    'breast_horizontal',
    'default_credit_horizontal',
    'give_credit_horizontal',
    'breast_vertical',
    'default_credit_vertical',
    'give_credit_vertical',
]


class UniFedServerAggregator(DefaultServerAggregator):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.cpu_transfer = False \
            if not hasattr(self.args, "cpu_transfer") \
            else self.args.cpu_transfer
        self.logger = LoggerManager().get_logger(0, 'aggregator')

    def _test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
            "predicted": [],
            "truth": [],
        }

        """
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        """
        if args.dataset == "stackoverflow_lr":
            criterion = nn.BCELoss(reduction="sum").to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                if args.dataset == "stackoverflow_lr":
                    predicted = (pred > 0.5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > 0.1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics["test_precision"] += precision.sum().item()
                    metrics["test_recall"] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                metrics['predicted'].append(pred[:, -1].reshape(-1).detach().cpu().numpy())
                metrics['truth'].append(target.reshape(-1).detach().cpu().numpy())

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)

            metrics['predicted'] = np.concatenate(metrics['predicted'])
            metrics['truth'] = np.concatenate(metrics['truth'])
        return metrics

    def test(self, test_data, device, args):
        with self.logger.model_evaluation() as e:
            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            metrics = self._test(test_data, device, args)

            test_tot_correct, test_num_sample, test_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": args.round_idx})
                wandb.log({"Test/Loss": test_loss, "round": args.round_idx})

            mlops.log({"Test/Acc": test_acc, "round": args.round_idx})
            mlops.log({"Test/Loss": test_loss, "round": args.round_idx})

            stats = {"test_acc": test_acc, "test_loss": test_loss}
            logging.info(stats)

            e.report_metric('loss', test_loss)
            e.report_metric('accuracy', test_acc)
            if args.dataset in AUC:
                try:
                    auc = roc_auc_score(metrics['truth'], metrics['predicted'])
                    e.report_metric('auc', auc)
                except Exception as e:
                    traceback.print_exc()

        return (test_acc, test_loss, None, None)
