import argparse
import json
import os
from typing import List

import colink as CL
import fedml
import flbenchmark.datasets
import torch
import yaml
from fedml import FedMLRunner
from fedml.arguments import Arguments
from sklearn.utils import shuffle

from unifed.frameworks.fedml.util import (GetTempFileName, get_local_ip,
                                          store_error, store_return)

from .horizontal_exp import create_model, custom_model_trainer
from .horizontal_exp import load_data as load_data_horizontal
from .src.aggregator.default_aggregator import UniFedServerAggregator
from .src.standalone.fedavg_api import FedAvgAPI
from .src.trainer.classification_trainer import ClassificationTrainer
from .vertical_exp import load_data as load_data_vertical
from .vertical_exp import run_experiment

pop = CL.ProtocolOperator(__name__)
UNIFED_TASK_DIR = "unifed:task"


def load_config_from_param_and_check(param: bytes):
    unifed_config = json.loads(param.decode())
    framework = unifed_config["framework"]
    assert framework == "fedml"
    deployment = unifed_config["deployment"]
    if deployment["mode"] != "colink":
        raise ValueError("Deployment mode must be colink")
    return unifed_config


def download_data(config):
    flbd = flbenchmark.datasets.FLBDatasets('data')

    print("Downloading Data...")

    dataset_name = (
        'student_horizontal',
        'breast_horizontal',
        'default_credit_horizontal',
        'give_credit_horizontal',
        'vehicle_scale_horizontal'
    )

    for x in dataset_name:
        if config["dataset"] == x:
            train_dataset, test_dataset = flbd.fateDatasets(x)
            flbenchmark.datasets.convert_to_csv(
                train_dataset, out_dir='data/{}_train'.format(x))
            if x != 'vehicle_scale_horizontal':
                flbenchmark.datasets.convert_to_csv(
                    test_dataset, out_dir='data/{}_test'.format(x))

    vertical = (
        'breast_vertical',
        'give_credit_vertical',
        'default_credit_vertical'
    )

    for x in vertical:
        if config["dataset"] == x:
            my_dataset = flbd.fateDatasets(x)
            flbenchmark.datasets.convert_to_csv(
                my_dataset[0], out_dir='data/{}'.format(x))
            if my_dataset[1] != None:
                flbenchmark.datasets.convert_to_csv(
                    my_dataset[1], out_dir='data/{}'.format(x))

    leaf = (
        'femnist',
        'reddit',
        'celeba',
        'shakespeare',
    )

    for x in leaf:
        if config["dataset"] == x:
            my_dataset = flbd.leafDatasets(x)


def write_file(participant_id):
    with GetTempFileName() as temp_log_filename, \
            GetTempFileName() as temp_output_filename:

        # print(f"Writing to {temp_output_filename} and {temp_log_filename}...")
        # with open(temp_output_filename, 'w') as f:
        #     f.write(f"Some output for {participant_id} here.")

        with open(temp_log_filename, 'w') as f:
            with open(f"./log/{participant_id}.log", 'r') as f2:
                f.write(f2.read())

        with open(temp_output_filename, "rb") as f:
            output = f.read()

        with open(temp_log_filename, "rb") as f:
            log = f.read()
    return output, log


def run_simulation(config):
    device = torch.device(
        "cuda:" + str(config.get('gpu', 0))
        if torch.cuda.is_available()
        else "cpu"
    )

    if config['dataset'].split('_')[-1] == 'horizontal':
        dataset = load_data_horizontal(config['training'], config['dataset'])
        model = create_model(
            config, model_name=config['model'], output_dim=dataset[7])
        model_trainer = custom_model_trainer(config, model)

        fedavgAPI = FedAvgAPI(
            dataset,
            device,
            config,
            model_trainer,
            is_regression=(config['dataset'] == 'student_horizontal')
        )
        fedavgAPI.train()
    elif config['dataset'].split('_')[-1] == 'vertical':
        train, test = load_data_vertical(config['dataset'])
        Xa_train, Xb_train, y_train = train
        Xa_test, Xb_test, y_test = test

        Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
        Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
        train = [Xa_train, Xb_train, y_train]
        test = [Xa_test, Xb_test, y_test]
        run_experiment(
            train_data=train,
            test_data=test,
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            epoch=config['training']['epochs'],
            config=config,
        )
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


def config_fedml_client(config, rank, server_ip):
    with open('src/unifed/frameworks/fedml/config/fedml_config.yaml', 'r') as f:
        try:
            fedml_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise Exception(exc.message)

    ipconfig_fn = fedml_config['comm_args']['grpc_ipconfig_path'].replace(
        '.csv', f'_{rank}.csv')
    with open(ipconfig_fn, 'w') as f:
        f.write('receiver_id,ip\n')
        f.write(f'0,{server_ip}\n')

    fedml_config['data_args']['dataset'] = config['dataset']
    fedml_config['model_args']['model'] = config['model']
    fedml_config['train_args']['client_num_in_total'] = config['training']['tot_client_num']
    fedml_config['train_args']['client_num_per_round'] = config['training']['client_per_round']
    fedml_config['train_args']['comm_round'] = config['training']['epochs']
    fedml_config['train_args']['epochs'] = config['training']['inner_step']
    fedml_config['train_args']['batch_size'] = config['training']['batch_size']
    fedml_config['train_args']['client_optimizer'] = config['training']['optimizer']
    fedml_config['train_args']['learning_rate'] = config['training']['learning_rate']
    fedml_config['train_args']['weight_decay'] = config['training']['optimizer_param']['weight_decay']
    fedml_config['comm_args']['grpc_ipconfig_path'] = ipconfig_fn

    with GetTempFileName() as temp_filename:
        with open(temp_filename, 'w') as f:
            f.write(yaml.dump(fedml_config))
        args = fedml.init(load_arguments(temp_filename, rank, 'client'))

    return args


def config_fedml_server(config, rank, clients_ip):
    with open('src/unifed/frameworks/fedml/config/fedml_config.yaml', 'r') as f:
        try:
            fedml_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise Exception(exc.message)
    print('Loaded fedml config')

    ipconfig_fn = fedml_config['comm_args']['grpc_ipconfig_path'].replace(
        '.csv', f'_{rank}.csv')
    with open(ipconfig_fn, 'w') as f:
        f.write('receiver_id,ip\n')
        f.write('0,127.0.0.1\n')
        for i, client_ip in enumerate(clients_ip):
            f.write(f'{i+1},{client_ip}\n')
    print('IP config file saved to', ipconfig_fn)

    fedml_config['data_args']['dataset'] = config['dataset']
    fedml_config['model_args']['model'] = config['model']
    fedml_config['train_args']['client_num_in_total'] = config['training']['tot_client_num']
    fedml_config['train_args']['client_num_per_round'] = config['training']['client_per_round']
    fedml_config['train_args']['comm_round'] = config['training']['epochs']
    fedml_config['train_args']['epochs'] = config['training']['inner_step']
    fedml_config['train_args']['batch_size'] = config['training']['batch_size']
    fedml_config['train_args']['client_optimizer'] = config['training']['optimizer']
    fedml_config['train_args']['learning_rate'] = config['training']['learning_rate']
    fedml_config['train_args']['weight_decay'] = config['training']['optimizer_param']['weight_decay']
    fedml_config['comm_args']['grpc_ipconfig_path'] = ipconfig_fn

    with GetTempFileName() as temp_filename:
        with open(temp_filename, 'w') as f:
            f.write(yaml.dump(fedml_config))
        args = fedml.init(load_arguments(temp_filename, rank, 'server'))
    return args


def run_fedml(config, args):
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

    def custom_server_aggregator(model, args):
        if args.dataset == "stackoverflow_lr":
            aggregator = MyServerAggregatorTAGPred(model, args)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            aggregator = MyServerAggregatorNWP(model, args)
        else:
            aggregator = UniFedServerAggregator(model, args)
        return aggregator

    # init device
    device = fedml.device.get_device(args)

    # load data
    download_data(config)
    dataset = load_data_horizontal(config['training'], config['dataset'])

    # load model
    model = create_model(
        config,
        model_name=config['model'],
        output_dim=dataset[7],
    )

    if args.role == "client":
        trainer = custom_client_trainer(model, args)
        fedml_runner = FedMLRunner(
            args, device, dataset, model,
            client_trainer=trainer)
        fedml_runner.run()
        trainer.end()
    elif args.role == "server":
        aggregator = custom_server_aggregator(model, args)
        fedml_runner = FedMLRunner(
            args, device, dataset, model,
            server_aggregator=aggregator)
        fedml_runner.run()
        aggregator.end()


@pop.handle("unifed.fedml:server")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_server(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
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

    # Get participant ID
    participant_id = [
        i for i, p in enumerate(participants)
        if p.user_id == cl.get_user_id()
    ][0]

    # Load configuration and setup server
    config = load_config_from_param_and_check(param)

    # # Run simulation
    # run_simulation(config)

    # Configure FedML
    print('Setup server...')
    args = config_fedml_server(config, 0, clients_ip)

    # Send notification to the clients that the server is ready
    cl.send_variable(
        "server_setup_done",
        True,
        [p for p in participants if p.role == "client"]
    )

    # Run FedML
    run_fedml(config, args)

    # Write output and log
    output, log = write_file(participant_id)
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

    # Get participant ID
    participant_id = [
        i for i, p in enumerate(participants)
        if p.user_id == cl.get_user_id()
    ][0]

    # Load configuration and setup server
    config = load_config_from_param_and_check(param)

    # Receive notification from the server that the server is ready
    print(f'[{participant_id}] Waiting for server setup...')
    cl.recv_variable("server_setup_done", p_server).decode()

    # Configure FedML
    args = config_fedml_client(config, participant_id, server_ip)

    # Run FedML
    run_fedml(config, args)

    # Write output and log
    output, log = write_file(participant_id)
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)

    return json.dumps({
        "server_ip": server_ip,
        "ip": get_local_ip(),
        "returncode": 0,
    })
