import json
import logging
import os

import numpy as np
import torch
import pandas as pd


# def read_data(train_data_dir, test_data_dir):
#     '''parses data in given train and test data directories

#     assumes:
#     - the data in the input directories are .json files with
#         keys 'users' and 'user_data'
#     - the set of train set users is the same as the set of test set users

#     Return:
#         clients: list of non-unique client ids
#         groups: list of group ids; empty list if none found
#         train_data: dictionary of train data
#         test_data: dictionary of test data
#     '''
#     clients = []
#     groups = []
#     train_data = {}
#     test_data = {}

#     train_files = os.listdir(train_data_dir)
#     train_files = [f for f in train_files if f.endswith('.json')]
#     for f in train_files:
#         file_path = os.path.join(train_data_dir, f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         clients.extend(cdata['users'])
#         if 'hierarchies' in cdata:
#             groups.extend(cdata['hierarchies'])
#         train_data.update(cdata['user_data'])

#     test_files = os.listdir(test_data_dir)
#     test_files = [f for f in test_files if f.endswith('.json')]
#     for f in test_files:
#         file_path = os.path.join(test_data_dir, f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         test_data.update(cdata['user_data'])

#     clients = sorted(cdata['users'])

#     return clients, groups, train_data, test_data

def read_data(train_data_dir, test_data_dir):
    clients = ['guest', 'host']
    groups = []
    train_data, test_data = {'host': {}, 'guest': {}}, {
        'host': {}, 'guest': {}}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.csv')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        my_numpy = np.array(pd.read_csv(file_path))
        if file_path[-8] == 'h':
            train_data['host']['x'], train_data['host']['y'] = my_numpy[:,
                                                                        2:], my_numpy[:, 1]
        else:
            train_data['guest']['x'], train_data['guest']['y'] = my_numpy[:,
                                                                          2:], my_numpy[:, 1]

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.csv')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        my_numpy = np.array(pd.read_csv(file_path))
        test_data['host']['x'], test_data['host']['y'] = test_data['guest']['x'], test_data['guest']['y'] = my_numpy[:, 2:], my_numpy[:, 1]

    return clients, groups, train_data, test_data


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


# def load_partition_data_mnist_by_device_id(batch_size,
#                                            device_id,
#                                            train_path="MNIST_mobile",
#                                            test_path="MNIST_mobile"):
#     train_path += '/' + device_id + '/' + 'train'
#     test_path += '/' + device_id + '/' + 'test'
#     return load_partition_data_mnist(batch_size, train_path, test_path)


def load_partition_data_breast_horizontal(batch_size,
                                          train_path="data/breast_horizontal_train/",
                                          test_path="data/breast_horizontal_test/"):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])
        user_test_data_num = len(test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        test_batch = batch_data(test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 2  # to update the number of classes

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

# print(load_partition_data_breast_horizontal(16))
