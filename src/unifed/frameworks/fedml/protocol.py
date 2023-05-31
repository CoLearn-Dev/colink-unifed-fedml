import argparse
import json
import os
from typing import List
import time

import colink as CL
import fedml
import flbenchmark.datasets
import yaml
from fedml import FedMLRunner
from fedml.arguments import Arguments

from unifed.frameworks.fedml.util import (GetTempFileName, get_local_ip,
                                          store_error, store_return)

from .horizontal_exp import create_model
from .horizontal_exp import load_data as load_data_horizontal
from .horizontal_exp import run_simulation_horizontal
from .src.aggregator.default_aggregator import UniFedServerAggregator
from .src.logger import LoggerManager
from .src.trainer.classification_trainer import ClassificationTrainer
from .vertical_exp import run_simulation_vertical

pop = CL.ProtocolOperator(__name__)
UNIFED_TASK_DIR = "unifed:task"

LOGGING_DIR = 'output'
FRAMEWORK = 'fedml'
DATA_DIR = os.path.expanduser('~/flbenchmark.working/data')

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

VERTICAL_DATASETS = (
    'breast_vertical',
    'give_credit_vertical',
    'default_credit_vertical'
)


def load_config_from_param_and_check(param: bytes):
    unifed_config = json.loads(param.decode())
    framework = unifed_config["framework"]
    assert framework == "fedml"
    deployment = unifed_config["deployment"]
    if deployment["mode"] != "colink":
        raise ValueError("Deployment mode must be colink")
    return unifed_config


def download_data(dataset_name):
    flbd = flbenchmark.datasets.FLBDatasets(DATA_DIR)

    if dataset_name in FATE_DATASETS:
        train_dataset, test_dataset = flbd.fateDatasets(dataset_name)

        flbenchmark.datasets.convert_to_csv(
            train_dataset,
            out_dir=f'{DATA_DIR}/{dataset_name}_train',
        )

        if dataset_name == 'vehicle_scale_horizontal':
            return

        flbenchmark.datasets.convert_to_csv(
            test_dataset,
            out_dir=f'{DATA_DIR}/{dataset_name}_test',
        )
    elif dataset_name in LEAF_DATASETS:
        flbd.leafDatasets(dataset_name)
    elif dataset_name in VERTICAL_DATASETS:
        train_dataset, test_dataset = flbd.fateDatasets(dataset_name)

        flbenchmark.datasets.convert_to_csv(
            train_dataset,
            out_dir=f'{DATA_DIR}/{dataset_name}',
        )

        if test_dataset is None:
            return

        flbenchmark.datasets.convert_to_csv(
            test_dataset,
            out_dir=f'{DATA_DIR}/{dataset_name}',
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def write_file(participant_id, output_dir):
    with GetTempFileName() as temp_log_filename, \
            GetTempFileName() as temp_output_filename:

        # print(f"Writing to {temp_output_filename} and {temp_log_filename}...")
        # with open(temp_output_filename, 'w') as f:
        #     f.write(f"Some output for {participant_id} here.")

        with open(temp_log_filename, 'w') as f:
            with open(f"{output_dir}/log/{participant_id}.log", 'r') as f2:
                f.write(f2.read())

        with open(temp_output_filename, "rb") as f:
            output = f.read()

        with open(temp_log_filename, "rb") as f:
            log = f.read()
    return output, log


def run_simulation(config, output_dir):
    if config['dataset'].split('_')[-1] == 'horizontal' or config['dataset'] == 'femnist':
        run_simulation_horizontal(config, output_dir)
    elif config['dataset'].split('_')[-1] == 'vertical':
        run_simulation_vertical(config, output_dir)
    else:
        raise Exception("Not handling this mode yet!")


def add_args(cf, rank, role):
    cmd_args = argparse.Namespace()
    cmd_args.yaml_config_file = cf
    cmd_args.rank = rank
    cmd_args.role = role
    cmd_args.local_rank = 0
    cmd_args.node_rank = 0
    cmd_args.run_id = "0"
    return cmd_args


def load_arguments(cf, rank, role, training_type=None, comm_backend=None):
    cmd_args = add_args(cf, rank, role)

    # Load all arguments from YAML config file
    args = Arguments(cmd_args, training_type, comm_backend)

    if not hasattr(args, "worker_num"):
        args.worker_num = args.client_num_per_round

    # os.path.expanduser() method in Python is used
    # to expand an initial path component ~( tilde symbol)
    # or ~user in the given path to user’s home directory.
    if hasattr(args, "data_cache_dir"):
        args.data_cache_dir = os.path.expanduser(args.data_cache_dir)
    if hasattr(args, "data_file_path"):
        args.data_file_path = os.path.expanduser(args.data_file_path)
    if hasattr(args, "partition_file_path"):
        args.partition_file_path = os.path.expanduser(args.partition_file_path)
    if hasattr(args, "part_file"):
        args.part_file = os.path.expanduser(args.part_file)

    args.rank = int(args.rank)
    return args


def init_fedml_client(fedml_config, rank, output_dir):
    os.makedirs(output_dir + '/config/', exist_ok=True)
    config_fn = output_dir + '/config/fedm_config.yaml'
    with open(config_fn, 'w') as f:
        f.write(yaml.dump(fedml_config))
    args = fedml.init(load_arguments(config_fn, rank, 'client'))
    return args


def init_fedml_server(fedml_config, output_dir):
    os.makedirs(output_dir + '/config/', exist_ok=True)
    config_fn = output_dir + '/config/fedm_config.yaml'
    with open(config_fn, 'w') as f:
        f.write(yaml.dump(fedml_config))
    args = fedml.init(load_arguments(config_fn, 0, 'server'))
    return args


def config_fedml(config, ipconfig_fn):
    return {
        'common_args': {
            'training_type': 'cross_silo',
            'scenario': 'horizontal',
            'using_mlops': False,
            'random_seed': int(time.time()),
        },
        'data_args': {
            'dataset': config['dataset'],
        },
        'model_args': {
            'model': config['model'],
        },
        'train_args': {
            'federated_optimizer': 'FedAvg',
            'client_id_list': None,
            'client_num_in_total': config['training']['tot_client_num'],
            'client_num_per_round': config['training']['client_per_round'],
            'comm_round': config['training']['epochs'],
            'epochs': config['training']['inner_step'],
            'batch_size': config['training']['batch_size'],
            'client_optimizer': config['training']['optimizer'],
            'learning_rate': config['training']['learning_rate'],
            'momentum': config['training']['optimizer_param']['momentum'],
            'dampening': config['training']['optimizer_param']['dampening'],
            'weight_decay': config['training']['optimizer_param']['weight_decay'],
            'nesterov': config['training']['optimizer_param']['nesterov'],
        },
        'validate_args': {
            'frequency_of_the_test': 1,
        },
        'device_args': {
            'using_gpu': False,
        },
        'comm_args': {
            'backend': 'GRPC',
            'grpc_ipconfig_path': ipconfig_fn,
        },
        'tracking_args': {
            'enable_wandb': False,
        },
    }


def config_fedml_client(config, rank, server_ip, output_dir):
    os.makedirs(output_dir + '/config/', exist_ok=True)
    ipconfig_fn = output_dir + '/config/grpc_ipconfig.csv'
    with open(ipconfig_fn, 'w') as f:
        f.write('receiver_id,ip\n')
        f.write(f'0,{server_ip}\n')

    return config_fedml(config, ipconfig_fn)


def config_fedml_server(config, clients_ip, output_dir):
    os.makedirs(output_dir + '/config/', exist_ok=True)
    ipconfig_fn = output_dir + '/config/grpc_ipconfig.csv'
    with open(ipconfig_fn, 'w') as f:
        f.write('receiver_id,ip\n')
        f.write('0,127.0.0.1\n')
        for i, client_ip in enumerate(clients_ip):
            f.write(f'{i+1},{client_ip}\n')
    print('IP config file saved to', ipconfig_fn)

    return config_fedml(config, ipconfig_fn)


def run_fedml_client(args, model, device, dataset, output_dir):
    def custom_client_trainer(model, args):
        if args.dataset == "stackoverflow_logistic_regression":
            trainer = MyModelTrainerTAG(model, args)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp", "reddit"]:
            trainer = NWPTrainer(model, args)
        elif args.dataset in ['student_horizontal']:
            trainer = RegressionTrainer(model, args)
        else:
            trainer = ClassificationTrainer(model, args)
        return trainer

    logger = LoggerManager.get_logger(args.rank, 'client', output_dir)
    trainer = custom_client_trainer(model, args)
    fedml_runner = FedMLRunner(
        args, device, dataset, model,
        client_trainer=trainer)
    fedml_runner.run()
    logger.end()
    LoggerManager.reset()


def run_fedml_server(args, model, device, dataset, output_dir):
    def custom_server_aggregator(model, args):
        if args.dataset == "stackoverflow_lr":
            aggregator = MyServerAggregatorTAGPred(model, args)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            aggregator = MyServerAggregatorNWP(model, args)
        else:
            aggregator = UniFedServerAggregator(model, args)
        return aggregator

    logger = LoggerManager.get_logger(0, 'aggregator', output_dir)
    aggregator = custom_server_aggregator(model, args)
    fedml_runner = FedMLRunner(
        args, device, dataset, model,
        server_aggregator=aggregator)
    fedml_runner.run()
    logger.end()
    LoggerManager.reset()


def run_fedml(args, output_dir):
    # init device
    device = fedml.device.get_device(args)

    # load dataset
    dataset, input_dim, output_dim = \
        load_data_horizontal(args.dataset, args.batch_size, DATA_DIR)

    # load model
    model = create_model(args.model, input_dim, output_dim)

    if args.role == "client":
        run_fedml_client(args, model, device, dataset, output_dir)
    elif args.role == "server":
        run_fedml_server(args, model, device, dataset, output_dir)


@pop.handle("unifed.fedml:server")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_server(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    # Get task ID
    task_id = cl.get_task_id()
    user_id = cl.get_user_id()
    print(f'Running {user_id} as server for task {task_id}...')

    # Get participant ID
    participant_id = [
        i for i, p in enumerate(participants)
        if p.user_id == cl.get_user_id()
    ][0]

    # Create task directory
    task_root_dir = f'{LOGGING_DIR}/{FRAMEWORK}/{task_id}'
    participant_root_dir = f'{task_root_dir}/{participant_id}'
    os.makedirs(participant_root_dir, exist_ok=True)

    # Get client IPs from the clients
    clients_ip = [
        cl.recv_variable(f'client_ip', p).decode()
        for p in participants if p.role == 'client'
    ]

    # Get server IP (current machine's IP) and send to the clients
    server_ip = get_local_ip()
    cl.send_variable(
        "server_ip",
        server_ip,
        [p for p in participants if p.role == "client"]
    )

    # Load configuration and setup server
    config = load_config_from_param_and_check(param)

    # Download data
    download_data(config['dataset'])

    if config.get('mode', '') == 'simulation':
        run_simulation(config, participant_root_dir)
    else:
        try:
            # Configure FedML
            print('Setup server...')
            fedml_config = config_fedml_server(
                config, clients_ip, participant_root_dir)
            args = init_fedml_server(fedml_config, participant_root_dir)

            # Send notification to the clients that the server is ready
            cl.send_variable(
                "server_setup_done",
                True,
                [p for p in participants if p.role == "client"]
            )
        except Exception as e:
            # Send notification to the clients that the server raised an error
            cl.send_variable(
                "server_setup_done",
                False,
                [p for p in participants if p.role == "client"]
            )
            raise e

        # Run FedML
        run_fedml(args, participant_root_dir)

    # Write output and log
    output, log = write_file(participant_id, participant_root_dir)
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)

    return json.dumps({
        "server_ip": server_ip,
        "ip": get_local_ip(),
        "returncode": 0,
    })


@pop.handle("unifed.fedml:client")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_client(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    # Get task ID
    task_id = cl.get_task_id()
    user_id = cl.get_user_id()
    print(f'Running {user_id} as client for task {task_id}...')

    # Get participant ID
    participant_id = [
        i for i, p in enumerate(participants)
        if p.user_id == cl.get_user_id()
    ][0]

    # Create task directory
    task_root_dir = f'{LOGGING_DIR}/{FRAMEWORK}/{task_id}'
    participant_root_dir = f'{task_root_dir}/{participant_id}'
    os.makedirs(participant_root_dir, exist_ok=True)

    # Get client IP (current machine's IP) and send to the server
    client_ip = get_local_ip()
    cl.send_variable(
        f'client_ip',
        client_ip,
        [p for p in participants if p.role == 'server']
    )

    # Get server IP from the server
    server_in_list = [p for p in participants if p.role == "server"]
    assert len(server_in_list) == 1
    p_server = server_in_list[0]
    server_ip = cl.recv_variable("server_ip", p_server).decode()

    # Load configuration and setup server
    config = load_config_from_param_and_check(param)

    if config.get('mode', '') == 'simulation':
        pass
    else:
        # Receive notification from the server that the server is ready
        print(f'[{participant_id}] Waiting for server setup...')
        is_server_done = cl.recv_variable(
            "server_setup_done", p_server).decode()
        if is_server_done == 'False':
            raise Exception('Server raised an error')

        # Download data
        download_data(config['dataset'])

        # Configure FedML
        fedml_config = config_fedml_client(
            config, participant_id, server_ip, participant_root_dir)
        args = init_fedml_client(
            fedml_config, participant_id, participant_root_dir)

        # Run FedML
        run_fedml(args, participant_root_dir)

        # Write output and log
        output, log = write_file(participant_id, participant_root_dir)
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)

    return json.dumps({
        "server_ip": server_ip,
        "ip": get_local_ip(),
        "returncode": 0,
    })
