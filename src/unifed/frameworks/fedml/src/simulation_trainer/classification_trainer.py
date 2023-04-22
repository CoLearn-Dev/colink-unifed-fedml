import torch
from torch import nn


class ClassificationTrainer():
    def __init__(self, model, config=None):
        self.model = model
        self.id = 0
        self.config = config

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, config, logger):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        optim_param = config['optimizer_param']
        if config['optimizer'] == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters(
            )), lr=config['learning_rate'], momentum=optim_param['momentum'], weight_decay=optim_param['weight_decay'], dampening=optim_param['dampening'], nesterov=optim_param['nesterov'])
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config['learning_rate'],
                                         weight_decay=optim_param['weight_decay'], amsgrad=True)

        epoch_loss = []
        for epoch in range(config['epochs']):
            with logger.computation() as c:
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

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
