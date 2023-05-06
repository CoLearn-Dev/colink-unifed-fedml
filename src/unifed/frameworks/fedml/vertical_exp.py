import os

from fedml.model.finance.vfl_models_standalone import DenseModel, LocalModel
from fedml.simulation.sp.classical_vertical_fl.party_models import (
    VFLGuestModel, VFLHostModel)
from fedml.simulation.sp.classical_vertical_fl.vfl import \
    VerticalMultiplePartyLogisticRegressionFederatedLearning
from fedml.simulation.sp.classical_vertical_fl.vfl_fixture import \
    FederatedLearningFixture
from sklearn.utils import shuffle

from .src.data.preprocessing_data import (breast_load_two_party_data,
                                          default_credit_load_two_party_data,
                                          give_credit_load_two_party_data)
from .src.logger import LoggerManager

# from fedml_api.data_preprocessing.preprocessing_data import dvisits_load_two_party_data, motor_load_two_party_data, vehicle_scale_load_two_party_data

DATA_DIR = os.path.expanduser('~/flbenchmark.working/data/')


def load_data(dataset_name):
    ori_dataset_name = dataset_name.replace('_vertical', '')
    data_dir = DATA_DIR + f'{dataset_name}/{ori_dataset_name}_hetero_'
    if dataset_name == 'breast_vertical':
        train, test = breast_load_two_party_data(data_dir)
    elif dataset_name == 'default_credit_vertical':
        train, test = default_credit_load_two_party_data(data_dir)
    elif dataset_name == 'give_credit_vertical':
        train, test = give_credit_load_two_party_data(data_dir)
    return train, test


def run_experiment(train_data, test_data, batch_size, learning_rate, epoch, config, output_dir):
    dimension_ab = {
        'breast_vertical': (10, 20),
        'default_credit_vertical': (13, 10),
        'give_credit_vertical': (5, 5)
    }

    print("hyper-parameters:")
    print("batch size: {0}".format(batch_size))
    print("learning rate: {0}".format(learning_rate))

    Xa_train, Xb_train, y_train = train_data
    Xa_test, Xb_test, y_test = test_data

    print("################################ Wire Federated Models ############################")

    party_a_local_model = LocalModel(
        input_dim=Xa_train.shape[1],
        output_dim=dimension_ab[config["dataset"]][0],
        learning_rate=learning_rate,
        optim_param=config["training"]["optimizer_param"],
    )
    party_b_local_model = LocalModel(
        input_dim=Xb_train.shape[1],
        output_dim=dimension_ab[config["dataset"]][1],
        learning_rate=learning_rate,
        optim_param=config["training"]["optimizer_param"],
    )

    party_a_dense_model = DenseModel(
        party_a_local_model.get_output_dim(),
        1,
        learning_rate=learning_rate,
        optim_param=config["training"]["optimizer_param"],
        bias=True,
    )
    party_b_dense_model = DenseModel(
        party_b_local_model.get_output_dim(),
        1,
        learning_rate=learning_rate,
        optim_param=config["training"]["optimizer_param"],
        bias=False,
    )
    partyA = VFLGuestModel(local_model=party_a_local_model)
    partyA.set_dense_model(party_a_dense_model)
    partyB = VFLHostModel(local_model=party_b_local_model)
    partyB.set_dense_model(party_b_dense_model)

    party_B_id = "B"
    federatedLearning = \
        VerticalMultiplePartyLogisticRegressionFederatedLearning(
            partyA)
    federatedLearning.add_party(id=party_B_id, party_model=partyB)
    federatedLearning.set_debug(is_debug=False)

    print("################################ Train Federated Models ############################")

    LoggerManager.get_logger(0, "client", output_dir)
    LoggerManager.get_logger(1, "client", output_dir)
    fl_fixture = FederatedLearningFixture(federatedLearning)

    train_data = {federatedLearning.get_main_party_id(): {"X": Xa_train, "Y": y_train},
                  "party_list": {party_B_id: Xb_train}}
    test_data = {federatedLearning.get_main_party_id(): {"X": Xa_test, "Y": y_test},
                 "party_list": {party_B_id: Xb_test}}

    print(epoch, batch_size)
    fl_fixture.fit(train_data=train_data, test_data=test_data,
                   epochs=epoch, batch_size=batch_size)

    LoggerManager.reset()


def run_simulation_vertical(config, output_dir):
    train, test = load_data(config['dataset'])
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
        output_dir=output_dir,
    )
