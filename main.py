import json
import cv2
import torchvision.transforms as transforms
from classes import *
from main_functions import *


# Constants which identify different parts of the code.
RL = True  # True if using the dataset obtained from the RL path
GT = False  # True if doing the ground truth test

# Parameters of the neural networks used for localization (LNN)
lnn_hyperparams = {
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
    "cluster_letters": ["A", "B", "C", "D", "E", "F"],  # The cluster letters A and C are inverted between paper and code  (-> Train on C)
    "n_retrainings": 3,
    "random_action_test": False,  # Set this True if you want to collect the random path
    "RL": True,  # True if using the dataset obtained from the RL path
    "GT": False,  # True if doing the ground truth test
    "generalization_vec": [False]
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
    "starting_point_dqn_test": {
        "A": None,
        "B": None,
        "C": None,
        "D": None,
        "E": None,
        "F": None
    },
    "sample_length": [4],  # [2,3,4,5,6,7,8,9,10],
    "embedding_size": [8],  # [8,16,32],
    "batch_size_lstm": [128],  # [16,32,64,128],
    "path_length": [40] #[25, 30, 35, 40, 45, 50]
}

env_params = {
    "collision_reward": [-1000],
    "already_passed_reward": [-120],
    "prediction_error_reward": [1000],
}

train_cluster_dict = {
    "A": True,
    "B": True,
    "C": True,
    "D": True,
    "E": True,
    "F": True
}

## sam_length 2, emb size 8, bs 64 sembra dare buoni path del rl
## anche 4, 8, 128

if __name__ == '__main__':

    lstm = True
    grid_search = True
    test_dataset_dict = create_train_test_dataset_dict(lnn_hyperparams, utils_dict)
    #pretraining(lnn_hyperparams, utils_dict)
    if lstm:
        if grid_search:
            #accuracy_dict = {}
            for sam_length in dqn_hyperparams["sample_length"]:
                for emb_size in dqn_hyperparams["embedding_size"]:
                    for bs in dqn_hyperparams["batch_size_lstm"]:
                        for pl in dqn_hyperparams["path_length"]:
                            print("sam_length: ", sam_length)
                            print("emb_size: ", emb_size)
                            print("bs: ", bs)
                            print("path_length: ", pl)
                            train_cluster_dict = {
                                "A": True,
                                "B": True,
                                "C": True,
                                "D": True,
                                "E": True,
                                "F": True
                            }
                            pretraining(lnn_hyperparams, utils_dict, sam_length, emb_size, bs, pl,)
                            for k in range(utils_dict["n_retrainings"]):
                                test_lnn(test_dataset_dict, train_cluster_dict, lnn_hyperparams, utils_dict, k, sam_length, emb_size, bs, pl,)
                                if k == 0:
                                    train_dqn_lstm(lnn_hyperparams, utils_dict, dqn_hyperparams, k, sam_length, emb_size, bs, pl)
                                test_dqn_lstm(train_cluster_dict, lnn_hyperparams, utils_dict, dqn_hyperparams, k, sam_length, emb_size, bs, pl)
                                train_lnn_after_dqn(train_cluster_dict, lnn_hyperparams, utils_dict, k, sam_length, emb_size, bs, pl)
                                test_lnn_after_dqn(test_dataset_dict, train_cluster_dict, lnn_hyperparams, utils_dict, k+1, sam_length, emb_size, bs, pl)
                                #accuracy_dict["acc_samLength{}_embSize{}_bs{}_k{}".format(sam_length, emb_size, bs, k)] =
        else:
            pretraining(lnn_hyperparams, utils_dict)
            for k in range(utils_dict["n_retrainings"]):
                test_lnn(test_dataset_dict, train_cluster_dict, lnn_hyperparams, utils_dict, k)
                if k == 0:
                    train_dqn_lstm(lnn_hyperparams, utils_dict, dqn_hyperparams, k, dqn_hyperparams["sample_length"],
                                   dqn_hyperparams["embedding_size"], dqn_hyperparams["batch_size_lstm"])
                test_dqn_lstm(train_cluster_dict, lnn_hyperparams, utils_dict, dqn_hyperparams, k, dqn_hyperparams["embedding_size"])
                train_lnn_after_dqn(train_cluster_dict, lnn_hyperparams, utils_dict, k)
                test_lnn_after_dqn(test_dataset_dict, train_cluster_dict, lnn_hyperparams, utils_dict, k+1,
                                   dqn_hyperparams["sample_length"], dqn_hyperparams["embedding_size"], dqn_hyperparams["batch_size_lstm"])
    else:
        for k in range(utils_dict["n_retrainings"]):
            test_lnn(test_dataset_dict, train_cluster_dict, lnn_hyperparams, utils_dict, k)
            if k == 0:
                train_dqn(lnn_hyperparams, utils_dict, dqn_hyperparams, k)
            test_dqn(train_cluster_dict, lnn_hyperparams, utils_dict, dqn_hyperparams, k)
            train_lnn_after_dqn(train_cluster_dict, lnn_hyperparams, utils_dict, k)
            test_lnn_after_dqn(test_dataset_dict, train_cluster_dict, lnn_hyperparams, utils_dict, k+1)

