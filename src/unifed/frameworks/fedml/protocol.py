import os
import json
import sys
import subprocess
import tempfile
from typing import List

import colink as CL

from unifed.frameworks.fedml.util import store_error, store_return, GetTempFileName, get_local_ip

from .main_fedavg import load_data, create_model, custom_model_trainer
import flbenchmark.datasets
import torch
from .src.standalone.fedavg_api import FedAvgAPI

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


def download(config):
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
                train_dataset, out_dir='../csv_data/{}_train'.format(x))
            if x != 'vehicle_scale_horizontal':
                flbenchmark.datasets.convert_to_csv(
                    test_dataset, out_dir='../csv_data/{}_test'.format(x))

    vertical = (
        'breast_vertical',
        'give_credit_vertical',
        'default_credit_vertical'
    )

    for x in vertical:
        if config["dataset"] == x:
            my_dataset = flbd.fateDatasets(x)
            flbenchmark.datasets.convert_to_csv(
                my_dataset[0], out_dir='../csv_data/{}'.format(x))
            if my_dataset[1] != None:
                flbenchmark.datasets.convert_to_csv(
                    my_dataset[1], out_dir='../csv_data/{}'.format(x))

    leaf = (
        'femnist',
        'reddit',
        'celeba',
        'shakespeare',
    )

    for x in leaf:
        if config["dataset"] == x:
            my_dataset = flbd.leafDatasets(x)


@pop.handle("unifed.fedml:server")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_server(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    # for certain frameworks, clients need to learn the ip of the server
    # in that case, we get the ip of the current machine and send it to the clients
    server_ip = get_local_ip()
    cl.send_variable(
        "server_ip",
        server_ip,
        [p for p in participants if p.role == "client"]
    )

    # ============================================
    config = load_config_from_param_and_check(param)

    download(config)

    device = torch.device("cuda:" + str(config.get('gpu', 0))
                          if torch.cuda.is_available() else "cpu")

    dataset = load_data(config['training'], config['dataset'])
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
    # fedavgAPI.train(config)


@pop.handle("unifed.fedml:client")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_client(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    # get the ip of the server
    server_in_list = [p for p in participants if p.role == "server"]
    assert len(server_in_list) == 1
    p_server = server_in_list[0]
    server_ip = cl.recv_variable("server_ip", p_server).decode()
    # run external program
    participant_id = [
        i for i, p in enumerate(participants)
        if p.user_id == cl.get_user_id()
    ][0]
    return run_external_process_and_collect_result(cl, participant_id, "client", server_ip)
