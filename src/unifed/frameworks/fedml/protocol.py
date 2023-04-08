import json
from typing import List

import colink as CL
import flbenchmark.datasets
import torch
from sklearn.utils import shuffle

from unifed.frameworks.fedml.util import (GetTempFileName, get_local_ip,
                                          store_error, store_return)

from .horizontal_exp import create_model, custom_model_trainer
from .horizontal_exp import load_data as load_data_horizontal
from .src.standalone.fedavg_api import FedAvgAPI
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

        print(f"Writing to {temp_output_filename} and {temp_log_filename}...")
        with open(temp_output_filename, 'w') as f:
            f.write(f"Some output for {participant_id} here.")

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


@pop.handle("unifed.fedml:server")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_server(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    # Get server IP (current machine's IP) and send to the clients
    server_ip = get_local_ip()
    cl.send_variable(
        "server_ip",
        server_ip,
        [p for p in participants if p.role == "client"]
    )
    participant_id = [
        i for i, p in enumerate(participants)
        if p.user_id == cl.get_user_id()
    ][0]

    # Load configuration and setup server
    config = load_config_from_param_and_check(param)

    # Prepare data
    download_data(config)

    # Run simulation
    run_simulation(config)

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
    # get the ip of the server
    server_in_list = [p for p in participants if p.role == "server"]
    assert len(server_in_list) == 1
    p_server = server_in_list[0]
    server_ip = cl.recv_variable("server_ip", p_server).decode()
    participant_id = [
        i for i, p in enumerate(participants)
        if p.user_id == cl.get_user_id()
    ][0]

    # Load configuration and setup server
    config = load_config_from_param_and_check(param)

    # Prepare data
    download_data(config)

    # Write output and log
    output, log = write_file(participant_id)
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)

    return json.dumps({
        "server_ip": server_ip,
        "ip": get_local_ip(),
        "returncode": 0,
    })
