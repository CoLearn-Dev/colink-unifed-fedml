import json
import sys

import torch

from .horizontal_exp import create_model, custom_model_trainer, load_data
from .src.standalone.fedavg_api import FedAvgAPI


def load_config_from_param_and_check(param: bytes):
    unifed_config = json.loads(param.decode())
    framework = unifed_config["framework"]
    assert framework == "fedml"
    deployment = unifed_config["deployment"]
    if deployment["mode"] != "colink":
        raise ValueError("Deployment mode must be colink")
    return unifed_config


def simulate_workload():
    if len(sys.argv) != 6:
        raise ValueError(f'Invalid arguments. Got {sys.argv}')
    role, participant_id, config_path, output_path, log_path = sys.argv[1:6]
    print('Simulated workload here begin.')

    with open(config_path, 'r') as f:
        config = json.load(f)
    simulate_logging(participant_id, role, config)

    print(f"Writing to {output_path} and {log_path}...")
    with open(output_path, 'w') as f:
        f.write(f"Some output for {role} here.")

    with open(log_path, 'w') as f:
        with open(f"./log/{participant_id}.log", 'r') as f2:
            f.write(f2.read())

    print('Simulated workload here end.')


def simulate_logging(participant_id, role, config):
    if role == 'server':
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
        fedavgAPI.train(config)
