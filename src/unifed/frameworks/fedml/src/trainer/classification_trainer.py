import torch
from torch import nn

from fedml.core.alg_frame.client_trainer import ClientTrainer
from ..logger import LoggerManager


class ClassificationTrainer(ClientTrainer):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.logger = LoggerManager.get_logger(args.rank, 'client')

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                dampening=args.dampening,
                nesterov=args.nesterov,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        self.logger.training_start()
        epoch_loss = []
        for epoch in range(args.epochs):
            self.logger.training_round_start()
            with self.logger.computation() as c:
                batch_loss = []
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    model.zero_grad()
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()

                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                c.report_metric('loss', sum(epoch_loss) / len(epoch_loss))
            self.logger.training_round_end()
        self.logger.training_end()

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            'predict': None,
            'targety': None
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                if metrics['predict'] == None:
                    metrics['predict'] = pred[:, -1].reshape(-1)
                else:
                    metrics['predict'] = torch.cat(
                        (metrics['predict'], pred[:, -1].reshape(-1)))

                if metrics['targety'] == None:
                    metrics['targety'] = target.reshape(-1)
                else:
                    metrics['targety'] = torch.cat(
                        (metrics['targety'], target.reshape(-1)))
        return metrics
