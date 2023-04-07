import json
import subprocess
from typing import List

import colink as CL
import flbenchmark.datasets

from unifed.frameworks.fedml.util import (GetTempFileName, get_local_ip,
                                          store_error, store_return)

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


def run_external_process_and_collect_result(cl: CL.CoLink, participant_id,  role: str, server_ip: str, config: dict):
    with GetTempFileName() as temp_log_filename, \
            GetTempFileName() as temp_output_filename, \
            GetTempFileName() as temp_config_filename:
        # Dump config
        with open(temp_config_filename, 'w') as f:
            json.dump(config, f)

        # start training procedure
        process = subprocess.Popen(
            [
                "unifed-fedml-workload",
                role,
                str(participant_id),
                temp_config_filename,
                temp_output_filename,
                temp_log_filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Gather result
        stdout, stderr = process.communicate()
        returncode = process.returncode

        with open(temp_output_filename, "rb") as f:
            output = f.read()
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)

        with open(temp_log_filename, "rb") as f:
            log = f.read()
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)

        return json.dumps({
            "server_ip": server_ip,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": returncode,
        })


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
    # Get server IP (current machine's IP) and send to the clients
    server_ip = get_local_ip()
    cl.send_variable(
        "server_ip",
        server_ip,
        [p for p in participants if p.role == "client"]
    )

    # Load configuration and setup server
    config = load_config_from_param_and_check(param)
    download(config)

    # run external program
    participant_id = [
        i for i, p in enumerate(participants)
        if p.user_id == cl.get_user_id()
    ][0]
    return run_external_process_and_collect_result(cl, participant_id, "server", server_ip, config)


@pop.handle("unifed.fedml:client")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_client(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    # get the ip of the server
    server_in_list = [p for p in participants if p.role == "server"]
    assert len(server_in_list) == 1
    p_server = server_in_list[0]
    server_ip = cl.recv_variable("server_ip", p_server).decode()

    # Prepare data
    config = load_config_from_param_and_check(param)
    download(config)

    # run external program
    participant_id = [
        i for i, p in enumerate(participants)
        if p.user_id == cl.get_user_id()
    ][0]
    return run_external_process_and_collect_result(cl, participant_id, "client", server_ip, config)
