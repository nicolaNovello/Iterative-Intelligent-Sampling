import json
import cv2
import torchvision.transforms as transforms
from classes import *
from main_functions import *


# Parameters of the position-predicting neural network (PPNN)
ppnn_hyperparams = {
    "Ni": 4,
    "No": 2,
    "num_hid_nodes": [128, 256],
    "activation_name": "ReLU",
    "dropout_vec": [1e-2, 1e-2],
    "batch_size": 50,
    "learning_rate": 1e-3,
    "samples_for_sampling": 15,  # Distance between samples collected in the environment (in pixels)
    "composed_transform": transforms.Compose([ToTensor()]),
    "max_num_training_points_cluster": 10  # Used for pre-training
}

utils_dict = {
    "dataset_path": 'NetsImages/Datasets/RSSIDataframe.csv',
    "scaler_path": 'NetsImages/scalers/scaler.pkl',
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "cluster_letters": ["A", "B", "C", "D", "E", "F"], 
    "n_retrainings": 3
}

dqn_hyperparams = {
    "cluster_letter_training_dqn": "C",
    "initial_value": 40,
    "num_iterations": 1000,
    "gamma": 0.92,
    "replay_memory_capacity": 1000,
    "lr": 1e-3,
    "target_net_update_steps": 10,
    "batch_size": 256,
    "min_samples_for_training": 512,
    "policy_type": 2,
    "sample_length": 4,
    "embedding_size": 8,
    "batch_size_lstm": 128,
    "path_length": 40
}

train_cluster_dict = {
    "A": True,
    "B": True,
    "C": True,
    "D": True,
    "E": True,
    "F": True
}

if __name__ == '__main__':

    test_dataset_dict = create_train_test_dataset_dict(ppnn_hyperparams, utils_dict)
    pretraining(ppnn_hyperparams, utils_dict)
    for k in range(utils_dict["n_retrainings"]):
        test_lnn(test_dataset_dict, train_cluster_dict, ppnn_hyperparams, utils_dict, k)
        if k == 0:
            train_dqn_lstm(ppnn_hyperparams, utils_dict, dqn_hyperparams, k)
        test_dqn_lstm(train_cluster_dict, ppnn_hyperparams, utils_dict, dqn_hyperparams, k)
        train_lnn_after_dqn(train_cluster_dict, ppnn_hyperparams, utils_dict, k)
        test_lnn_after_dqn(test_dataset_dict, train_cluster_dict, ppnn_hyperparams, utils_dict, k+1)

