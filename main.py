import torchvision.transforms as transforms
from main_functions import *


ppnn_hyperparams = {
    "Ni": 4,
    "No": 2,
    "num_hid_nodes": [128, 256],
    "activation_name": "ReLU",
    "dropout_vec": [1e-2, 1e-2],
    "batch_size": 50,
    "learning_rate": 1e-3,
    "samples_for_sampling": 15,
    "composed_transform": transforms.Compose([ToTensor()]),
    "max_num_training_points_cluster": 10
}

utils_dict = {
    "dataset_path": 'NetsImages/Datasets/RSSIDataframe.csv',
    "scaler_path": 'NetsImages/scalers/scaler.pkl',
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "cluster_letters": ["A", "B", "C", "D", "E", "F"],
    "n_retrainings": 3
}

drqn_hyperparams = {
    "lstm": True,
    "cluster_letter_training_drqn": "C",
    "initial_value": 40,
    "num_iterations": 1000,
    "gamma": 0.92,
    "replay_memory_capacity": 1000,
    "lr": 1e-3,
    "target_net_update_steps": 10,
    "min_samples_for_training": 512,
    "policy_type": 2,
    "sample_length": 4,
    "embedding_size": 8,
    "batch_size": 128,
    "path_length": 45  # 30 -> 24m; 40 -> 32m; 45 -> 36m
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
        test_ppnn(test_dataset_dict, train_cluster_dict, ppnn_hyperparams, utils_dict, k)
        if k == 0:
            train_drqn_lstm(ppnn_hyperparams, drqn_hyperparams, k)
        test_drqn_lstm(train_cluster_dict, ppnn_hyperparams, utils_dict, drqn_hyperparams, k)
        train_ppnn_after_drqn(train_cluster_dict, ppnn_hyperparams, utils_dict, k)
        test_ppnn_after_drqn(test_dataset_dict, train_cluster_dict, ppnn_hyperparams, utils_dict, drqn_hyperparams, k+1)


