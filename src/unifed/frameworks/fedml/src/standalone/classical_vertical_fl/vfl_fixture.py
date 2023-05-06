import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

import flbenchmark.logging

from .vfl import VerticalMultiplePartyLogisticRegressionFederatedLearning

from ...logger import LoggerManager


def compute_correct_prediction(*, y_targets, y_prob_preds, threshold=0.5):
    y_hat_lbls = []
    pred_pos_count = 0
    pred_neg_count = 0
    correct_count = 0
    for y_prob, y_t in zip(y_prob_preds, y_targets):
        if y_prob <= threshold:
            pred_neg_count += 1
            y_hat_lbl = 0
        else:
            pred_pos_count += 1
            y_hat_lbl = 1
        y_hat_lbls.append(y_hat_lbl)
        if y_hat_lbl == y_t:
            correct_count += 1

    return np.array(y_hat_lbls), [pred_pos_count, pred_neg_count, correct_count]


class FederatedLearningFixture(object):
    def __init__(
        self,
        federated_learning: VerticalMultiplePartyLogisticRegressionFederatedLearning,
    ) -> None:
        self.federated_learning = federated_learning

    def fit(self, train_data, test_data, epochs=50, batch_size=-1):
        main_party_id = self.federated_learning.get_main_party_id()
        Xa_train = train_data[main_party_id]["X"]
        y_train = train_data[main_party_id]["Y"]
        Xa_test = test_data[main_party_id]["X"]
        y_test = test_data[main_party_id]["Y"]

        N = Xa_train.shape[0]
        residual = N % batch_size
        if residual == 0:
            n_batches = N // batch_size
        else:
            n_batches = N // batch_size + 1

        print("number of samples:", N)
        print("batch size:", batch_size)
        print("number of batches:", n_batches)

        global_step = -1
        recording_period = n_batches
        recording_step = -1
        threshold = 0.5

        loss_list = []
        # running_time_list = []

        logger0 = LoggerManager.get_logger(0)
        logger1 = LoggerManager.get_logger(1)
        self.federated_learning.set_logger(logger0, logger1)

        logger0.training_start()
        logger1.training_start()
        ROUND = 0
        for ep in range(epochs):
            logger0.training_round_start()
            logger1.training_round_start()

            for batch_idx in range(n_batches):
                global_step += 1

                # prepare batch data for party A, which has both X and y.
                Xa_batch = Xa_train[
                    batch_idx * batch_size:(batch_idx + 1) * batch_size
                ]
                Y_batch = y_train[
                    batch_idx * batch_size:(batch_idx + 1) * batch_size
                ]

                # prepare batch data for all other parties, which only has both X.
                party_X_train_batch_dict = dict()
                for party_id, party_X in train_data["party_list"].items():
                    party_X_train_batch_dict[party_id] = party_X[
                        batch_idx * batch_size: (batch_idx + 1) * batch_size
                    ]

                loss = self.federated_learning.fit(Xa_batch, Y_batch,
                                                   party_X_train_batch_dict,
                                                   global_step)
                loss_list.append(loss)
                if (global_step + 1) % recording_period == 0:
                    if ep == epochs - 1:
                        break
                    else:
                        recording_step += 1
                        ave_loss = np.mean(loss_list)
                        loss_list = list()
                        party_X_test_dict = dict()
                        for party_id, party_X in test_data["party_list"].items():
                            party_X_test_dict[party_id] = party_X
                        y_prob_preds = self.federated_learning.predict(
                            Xa_test,
                            party_X_test_dict
                        )
                        y_hat_lbls, statistics = compute_correct_prediction(
                            y_targets=y_test,
                            y_prob_preds=y_prob_preds,
                            threshold=threshold
                        )
                        acc = accuracy_score(y_test, y_hat_lbls)
                        # auc = roc_auc_score(y_test, y_prob_preds)
                        # print("--- epoch: {0}, batch: {1}, loss: {2}, acc: {3}, auc: {4}"
                        #       .format(ep, batch_idx, ave_loss, acc, auc))
                        ROUND += 1
                        # wandb.log({"Test/Acc": acc, "round": ROUND})
                        # wandb.log({"Test/AVE_Loss": ave_loss, "round": ROUND})
                        print({"Test/Acc": acc, "round": ROUND})
                        print({"Test/AVE_Loss": ave_loss, "round": ROUND})

            logger0.training_round_end()
            logger1.training_round_end()
        logger0.training_end()
        logger1.training_end()

        with logger0.model_evaluation() as e:
            loss_list = list()
            party_X_test_dict = dict()
            for party_id, party_X in test_data["party_list"].items():
                party_X_test_dict[party_id] = party_X
            y_prob_preds = self.federated_learning.predict(
                Xa_test,
                party_X_test_dict
            )
            y_hat_lbls, statistics = compute_correct_prediction(
                y_targets=y_test,
                y_prob_preds=y_prob_preds,
                threshold=threshold
            )
            # acc = accuracy_score(y_test, y_hat_lbls)
            auc = roc_auc_score(y_test, y_hat_lbls)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print("Final AUC = {}".format(auc))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            e.report_metric('auc', auc)
        logger0.end()
        logger1.end()
