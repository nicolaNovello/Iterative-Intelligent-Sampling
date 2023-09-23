import json
import cv2
import torchvision.transforms as transforms
from classes import *


def run_test_lstm(env, RANDOM_ACTION_TEST, cluster_letter, policy_net, dqn_hyperparams, k, sam_length, emb_size, bs, pl):
    letter_num_dict = {"A": 1, "B": 10, "C": 100, "D": 1000, "E": 10000, "F": 100000}
    for num_episode in range(3):
        episode_reward = 0
        hidden = None
        #startingPoint = None
        #state = env.reset(test=True, pt=None)  # pt=dqn_hyperparams["starting_point_dqn_test"][cluster_letter]
        last_observation = env.reset(test=True, pt=None)
        action_space_dim = env.action_space_dim
        last_action = 0
        done = False
        if RANDOM_ACTION_TEST:
            random.seed(letter_num_dict[cluster_letter])
        else:
            random.seed(0)
        while not done:
            if RANDOM_ACTION_TEST:
                action = torch.tensor([random.randint(0, 2)])
            else:
                action, hidden = select_action_lstm(policy_net, torch.tensor(last_observation).float().view(1, 1, -1),
                                            F.one_hot(torch.tensor(last_action), action_space_dim).view(1, 1,-1).float(),
                                                    hidden, num_episode, ACTION_SELECTION=0, temperature=0, EPS_END=0)
                #action, hidden = policy_net.act(torch.tensor(last_observation).float().view(1, 1, -1),
                #                                     F.one_hot(torch.tensor(last_action), action_space_dim).view(1, 1,
                #                                                                                                 -1).float(),
                #                                     hidden=hidden, epsilon=0)
            observation, reward, done = env.play_step(action)
            episode_reward += reward
            last_observation = observation
            last_action = action
    # Save last position of the agent as starting point of the next trajectory (for each cluster)
    save_display_path = "NetsImages/OptimalPaths/cluster{}_iter{}_sampleLength{}_embSize{}_bs{}_pl{}_random{}.svg".format(cluster_letter, k,
                                                                        sam_length, emb_size, bs, pl, RANDOM_ACTION_TEST)
    env.save_display(save_display_path)
    ##### QUI
    dqn_hyperparams["starting_point_dqn_test"][cluster_letter] = env.return_snake()[-1]  # env.return_snake()[0] SAREBBE L ULTIMO PUNTO IN CUI PASSA, MA DEVO PRIMA FAR SI CHE NON SIA MAI FUORI DAI BORDI!!!


def run_test(env, RANDOM_ACTION_TEST, cluster_letter, policy_net, dqn_hyperparams):
    letter_num_dict = {"A": 1, "B": 10, "C": 100, "D": 1000, "E": 10000, "F": 100000}
    for num_episode in range(3):
        episode_reward = 0
        #startingPoint = None
        state = env.reset(test=True, pt=None)  # pt=dqn_hyperparams["starting_point_dqn_test"][cluster_letter
        done = False
        if RANDOM_ACTION_TEST:
            random.seed(letter_num_dict[cluster_letter])
        else:
            random.seed(0)
        while not done:
            if RANDOM_ACTION_TEST:
                action = torch.tensor([random.randint(0, 2)])
            else:
                action = select_action(policy_net, state, num_episode, ACTION_SELECTION=0, temperature=0, EPS_END=0)
            next_state, reward, done = env.play_step(action)
            episode_reward += reward
            state = next_state
    # Save last position of the agent as starting point of the next trajectory (for each cluster)
    dqn_hyperparams["starting_point_dqn_test"][cluster_letter] = env.return_snake()[-1]


def save_optimal_path(env, env_w, env_h, cluster_letter, coord, lnn_hyperparams, k):
    snake = env.return_snake()
    # Load the standard scaler to normalize data after having collected them
    sc_x = pickle.load(open('NetsImages/scalers/scaler.pkl', 'rb'))
    print(" load training dataset NetsImages/Datasets/TrainLNN/train_dataset_cluster{}_iter_{}.csv".format(cluster_letter, k))
    training_df = pd.read_csv('NetsImages/Datasets/TrainLNN/train_dataset_cluster{}_iter_{}.csv'.format(cluster_letter, k))
    print("len(training_df) before save optimal path: ", len(training_df))
    for idx, (x, y) in enumerate(snake):
        ind = x + y * env_w
        if ind >= env_h * env_w:  # Enters here when an action leads the agent outside the cluster
            RSSIs = [0, 0, 0, 0, 0, 0]
        else:
            RSSIs = coord[ind]
        RSSIs_df = pd.DataFrame(columns=["RSSI1", "RSSI2", "RSSI3", "RSSI4", "RSSI5", "RSSI6"])
        RSSIs_df.loc[0] = RSSIs
        RSSIs_df = RSSIs_df.drop(["RSSI1", "RSSI5"], axis=1)
        data_new = sc_x.transform(RSSIs_df)[0].tolist()
        sample = [x, y]
        sample.extend(data_new)
        #training_df.loc[lnn_hyperparams["max_num_training_points_cluster"] + idx] = sample
        training_df.loc[len(training_df) + idx] = sample
    #print("training_df: ", training_df)
    print("len(training_df) after save optimal path: ", len(training_df))
    #training_df.to_csv("NetsImages/Datasets/TrainLNNAfterRL/dataset_after_RL_sampling_cluster{}.csv".format(cluster_letter),
    #    index=False, header=False)
    print("save new dataset NetsImages/Datasets/TrainLNN/train_dataset_cluster{}_iter_{}.csv".format(cluster_letter, k+1))
    training_df.to_csv(
        "NetsImages/Datasets/TrainLNN/train_dataset_cluster{}_iter_{}.csv".format(cluster_letter, k+1), index=False, header=False)


def from_dataframe_to_dataset(dataframe):
    dataset_list = []
    for i in range(len(dataframe)):
        dataset_list.append(dataframe[i])
    return dataset_list


def create_cluster_heatmap(empty_img, empty_img_tmp, mse_array_cluster):
    # Max mse on the cluster identified by cluster_letter
    max_mse = np.max(mse_array_cluster)
    for i in range(empty_img.shape[0]):
        for j in range(empty_img.shape[1]):
            # Because remember that uint8(256) = 1, thus I have to saturate the values manually
            if 255 * (empty_img_tmp[i, j] / max_mse) >= 255:
                empty_img[i, j] = 255
            else:
                empty_img[i, j] = int(255 * (empty_img_tmp[i, j] / max_mse))
    return empty_img

