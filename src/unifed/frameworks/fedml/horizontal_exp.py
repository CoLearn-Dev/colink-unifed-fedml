import os

from fedml.model.linear.lr import LogisticRegression

from unifed.frameworks.fedml.src.logger import LoggerManager

from .src.data.breast_horizontal.data_loader import \
    load_partition_data_breast_horizontal
from .src.data.default_credit_horizontal.data_loader import \
    load_partition_data_default_credit_horizontal
from .src.data.give_credit_horizontal.data_loader import \
    load_partition_data_give_credit_horizontal
from .src.data.our_femnist.data_loader import load_partition_data_femnist
from .src.data.student_horizontal.data_loader import \
    load_partition_data_student_horizontal
from .src.data.vehicle_scale_horizontal.data_loader import \
    load_partition_data_vehicle_scale_horizontal
from .src.model.cv.lenet import lenet
from .src.model.linear.lr import LinearRegression
from .src.model.non_linear.mlp import MLP
from .src.simulation_trainer.classification_trainer import \
    ClassificationTrainer
from .src.simulation_trainer.nwp_trainer import NWPTrainer
from .src.simulation_trainer.regression_trainer import RegressionTrainer
from .src.standalone.fedavg_api import FedAvgAPI

FATE_DATASETS = (
    'student_horizontal',
    'breast_horizontal',
    'default_credit_horizontal',
    'give_credit_horizontal',
    'vehicle_scale_horizontal'
)

LEAF_DATASETS = (
    'femnist',
    'reddit',
    'celeba',
    'shakespeare',
)

DATA_DIR = os.path.expanduser('~/flbenchmark.working/data/')


def load_data(dataset_name, batch_size, data_dir):
    if dataset_name in FATE_DATASETS:
        train_dir = f'{data_dir}/{dataset_name}_train/'
        test_dir = f'{data_dir}/{dataset_name}_test/'

        if dataset_name == 'student_horizontal':
            load_partition_data = load_partition_data_student_horizontal
            input_dim = 13
        elif dataset_name == 'breast_horizontal':
            load_partition_data = load_partition_data_breast_horizontal
            input_dim = 30
        elif dataset_name == 'default_credit_horizontal':
            load_partition_data = load_partition_data_default_credit_horizontal
            input_dim = 23
        elif dataset_name == 'give_credit_horizontal':
            load_partition_data = load_partition_data_give_credit_horizontal
            input_dim = 10
        elif dataset_name == 'vehicle_scale_horizontal':
            load_partition_data = load_partition_data_vehicle_scale_horizontal
            input_dim = 18
        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
    elif dataset_name in LEAF_DATASETS:
        train_dir = f'{data_dir}/{dataset_name}/train/'
        test_dir = f'{data_dir}/{dataset_name}/test/'

        if dataset_name == 'femnist':
            load_partition_data = load_partition_data_femnist
            input_dim = 28 * 28
        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    [
        client_num,
        train_data_num, test_data_num,
        train_data_global, test_data_global,
        train_data_local_num_dict,
        train_data_local_dict, test_data_local_dict,
        class_num
    ] = load_partition_data(
        batch_size,
        train_dir,
        test_dir,
    )

    return [
        train_data_num, test_data_num,
        train_data_global, test_data_global,
        train_data_local_num_dict,
        train_data_local_dict, test_data_local_dict,
        class_num,
    ], input_dim, class_num


def create_model(model_name, input_dim, output_dim):
    if model_name.startswith('mlp'):
        hidden = [int(x) for x in model_name.split('_')[1:]]
        model = MLP(input_dim, output_dim, hidden)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(input_dim, output_dim)
    elif model_name == 'lenet':
        model = lenet(output_dim)
    elif model_name == 'linear_regression':
        model = LinearRegression(input_dim, 1)
    else:
        raise ValueError(f'Unknown model {model_name}')
    return model


def custom_model_trainer(config, model):
    if config['dataset'] == "stackoverflow_logistic_regression":
        return MyModelTrainerTAG(model)
    elif config['dataset'] in ["fed_shakespeare", "stackoverflow_nwp", "reddit"]:
        return NWPTrainer(model)
    elif config['dataset'] in ['student_horizontal']:
        return RegressionTrainer(model)
    else:  # default model trainer is for classification problem
        return ClassificationTrainer(model)


def run_simulation_horizontal(config, output_dir):
    # device = torch.device(
    #     "cuda:" + str(config.get('gpu', 0))
    #     if torch.cuda.is_available()
    #     else "cpu"
    # )

    # dataset = load_data(config['training'], config['dataset'])
    # model = create_model(
    #     config, model_name=config['model'], output_dim=dataset[7])

    # init device
    device = 'cpu'

    # load dataset
    dataset, input_dim, output_dim = load_data(
        config['dataset'],
        config['training']['batch_size'],
        DATA_DIR
    )

    # load model
    model = create_model(config['model'], input_dim, output_dim)

    model_trainer = custom_model_trainer(config, model)

    fedavgAPI = FedAvgAPI(
        dataset,
        device,
        config,
        model_trainer,
        output_dir,
        is_regression=(config['dataset'] == 'student_horizontal'),
    )
    fedavgAPI.train()

    LoggerManager.reset()
