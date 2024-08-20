from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from sklearn.preprocessing import StandardScaler
import os
import math
import pandas as pd
import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F


pygame.init()
font = pygame.font.SysFont('arial', 25)
Point = namedtuple('Point', 'x, y')
device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Direction(Enum):
    RIGHT_S = 1
    LEFT_S = 2
    UP_S = 3
    DOWN_S = 4


class SamplingOperator:

    def __init__(self, cluster_letter, err_cluster_image, coord, localization_model,
                 scaler_path, max_steps=40, w=926, h=618):
        self.measurements = []
        self.scaler_path = scaler_path
        self.cluster_letter = cluster_letter
        self.clusterBorders = return_cluster_border(cluster_letter)
        self.err_cluster_image = err_cluster_image
        self.coord = coord
        self.localization_model = localization_model
        self.max_steps = max_steps
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Localization')
        self.clock = pygame.time.Clock()
        self.observation_space_dim = 13  # The size of the state
        self.action_space_dim = 3  # The number of actions that the agent can do
        self.reset()

    def reset(self):
        self.prediction_errors = [0]
        self.last_actions = deque(maxlen=10)
        self.last_errors = deque(maxlen=3)
        self.direction = Direction.RIGHT_S

        self.head = Point(int((self.clusterBorders[1] + self.clusterBorders[3]) / 2),
                          int((self.clusterBorders[0] + self.clusterBorders[2]) / 2))

        self.measurements = [self.head]
        self.iter = 0
        # List containing all the visited places of the agent
        self.places = []
        state_after_reset = self.get_state()
        return state_after_reset

    def return_measurements(self):
        return self.measurements

    def return_places(self):
        return self.places

    def save_display(self, path):
        pygame.image.save(self.display, path)

    def play_step(self, action, speed=30):
        self.last_actions.append(action)
        self.iter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move(action)
        self.measurements.insert(0, self.head)
        self.places.append(self.head)
        reward = 0
        game_over = False
        # Compute the error between predicted position and true one
        instant_error = self.compute_error(self.head)
        self.last_errors.append(instant_error)
        self.prediction_errors.append(instant_error)
        next_state = self.get_state()

        if self.is_collision(self.head):
            game_over = True
            reward = -1000
            return next_state, torch.tensor([reward], dtype=torch.float32), game_over
        if self.iter > self.max_steps:
            game_over = True
            return next_state, torch.tensor([reward], dtype=torch.float32), game_over
        weight = 1
        for i in range(len(self.places) - 1):
            if self.head == self.places[i]:
                reward -= 120 * weight
                weight += 1
        if weight > 2:
            return next_state, torch.tensor([reward], dtype=torch.float32), game_over

        reward += 1 * instant_error + np.sum(self.prediction_errors)/1000

        self._update_ui(self.err_cluster_image)
        self.clock.tick(speed)
        return next_state, torch.tensor([reward], dtype=torch.float32), game_over

    def get_RSSI(self, point):
        ind = point.x + point.y * 926
        RSSIs = self.coord[ind]
        RSSIs_df = pd.DataFrame(columns=["RSSI1", "RSSI2", "RSSI3", "RSSI4", "RSSI5", "RSSI6"])
        RSSIs_df.loc[0] = RSSIs
        RSSIs_df = RSSIs_df.drop(["RSSI1", "RSSI5"], axis=1).copy()  # Remove 2 extra columns present in original dataset
        return RSSIs_df

    def compute_error(self, pt):
        if pt.x + pt.y * 926 < self.w * self.h:
            RSSIs_df = self.get_RSSI(pt)
        else:
            return 0
        with torch.no_grad():
            sc_x = pickle.load(open(self.scaler_path, 'rb'))
            data_new = torch.tensor(sc_x.transform(RSSIs_df), dtype=torch.float32)
            data_new = data_new.to(device)
            output = self.localization_model(data_new)
            error = torch.sqrt((output[0][0] - pt.x) ** 2 + (output[0][1] - pt.y) ** 2)
            error = error.cpu().detach()
        return error

    def is_collision(self, pt):
        if ((pt.x > self.clusterBorders[1]) or (pt.x < self.clusterBorders[3]) or
                (pt.y > self.clusterBorders[2]) or (pt.y < self.clusterBorders[0])):
            return True
        else:
            return False

    def _update_ui(self, err_cluster_image):
        CUSTOM = (200, 100, 200)
        BLACK = (0, 0, 0)
        self.display.blit(err_cluster_image, (0, 0))  # Draw one image onto one other
        draw_cluster_borders(self.display)
        for pt in self.measurements:
            pygame.draw.circle(self.display, CUSTOM, (pt.x, pt.y), 8)
            pygame.draw.circle(self.display, BLACK, (pt.x, pt.y), 8, 2)
        pygame.display.flip()  # Update

    def _move(self, action, action_step_length=25):
        clock_wise1 = [Direction.RIGHT_S, Direction.DOWN_S, Direction.LEFT_S, Direction.UP_S]
        if self.direction in clock_wise1:
            idx = clock_wise1.index(self.direction)  # Save the index of the direction
        if action == 0:
            # If the action is straight, then there is no change in the direction
            new_dir = clock_wise1[idx]
        elif action == 1:
            # Action -> Turn right
            next_idx = (idx + 1) % 4  # Shift the direction clockwise in the array of the possible directions
            new_dir = clock_wise1[next_idx]
        elif action == 2:
            # Action -> Turn Left
            next_idx = (idx - 1) % 4
            new_dir = clock_wise1[next_idx]

        self.direction = new_dir  # Update the direction after the action of the agent
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT_S:
            x += action_step_length
        elif self.direction == Direction.LEFT_S:
            x -= action_step_length
        elif self.direction == Direction.DOWN_S:
            y += action_step_length
        elif self.direction == Direction.UP_S:
            y -= action_step_length
        self.head = Point(x, y)

    def get_state(self, action_step_length=25):
        head = self.measurements[0]
        point_l1 = Point(head.x - action_step_length, head.y)
        point_r1 = Point(head.x + action_step_length, head.y)
        point_u1 = Point(head.x, head.y - action_step_length)
        point_d1 = Point(head.x, head.y + action_step_length)
        # Compute the prediction error in nearby points
        error_r = self.compute_error(point_r1)
        error_l = self.compute_error(point_l1)
        error_up = self.compute_error(point_u1)
        error_down = self.compute_error(point_d1)
        # Save the direction of the last action
        dir_l = self.direction == Direction.LEFT_S
        dir_r = self.direction == Direction.RIGHT_S
        dir_u = self.direction == Direction.UP_S
        dir_d = self.direction == Direction.DOWN_S

        # Find the maximum error, to find the direction of maximum error and to normalize the errors in the state vector
        max_error = max(error_up, error_down, error_r, error_l)
        if dir_u:
            err_straight = error_up
            err_right = error_r
            err_left = error_l
        if dir_r:
            err_straight = error_r
            err_right = error_down
            err_left = error_up
        if dir_d:
            err_straight = error_down
            err_right = error_l
            err_left = error_r
        if dir_l:
            err_straight = error_l
            err_right = error_up
            err_left = error_down

        err_straight = err_straight / max_error
        err_right = err_right / max_error
        err_left = err_left / max_error

        state = [
            # Collision
            (dir_r and self.is_collision(Point(head.x + action_step_length, head.y))) or
            (dir_l and self.is_collision(Point(head.x - action_step_length, head.y))) or
            (dir_u and self.is_collision(Point(head.x, head.y - action_step_length))) or
            (dir_d and self.is_collision(Point(head.x, head.y + action_step_length))),

            (dir_u and self.is_collision(Point(head.x + action_step_length, head.y))) or
            (dir_d and self.is_collision(Point(head.x - action_step_length, head.y))) or
            (dir_l and self.is_collision(Point(head.x, head.y - action_step_length))) or
            (dir_r and self.is_collision(Point(head.x, head.y + action_step_length))),

            (dir_d and self.is_collision(Point(head.x + action_step_length, head.y))) or
            (dir_u and self.is_collision(Point(head.x - action_step_length, head.y))) or
            (dir_r and self.is_collision(Point(head.x, head.y - action_step_length))) or
            (dir_l and self.is_collision(Point(head.x, head.y + action_step_length))),

            # Already measured straight
            (dir_r and point_r1 in self.measurements) or
            (dir_l and point_l1 in self.measurements) or
            (dir_u and point_u1 in self.measurements) or
            (dir_d and point_d1 in self.measurements),

            # Already measured right (recall the direction is in absolute terms!)
            (dir_r and point_d1 in self.measurements) or
            (dir_l and point_u1 in self.measurements) or
            (dir_u and point_r1 in self.measurements) or
            (dir_d and point_l1 in self.measurements),

            # Already measured left
            (dir_r and point_u1 in self.measurements) or
            (dir_l and point_d1 in self.measurements) or
            (dir_u and point_l1 in self.measurements) or
            (dir_d and point_r1 in self.measurements),

            # Err value
            err_straight,
            err_right,
            err_left,
        ]
        errors_state_list = [0] * 3
        for i in range(len(list(self.last_errors))):
            errors_state_list[i] = self.last_errors[i]
        max_tmp = max(errors_state_list)
        if max_tmp > 0:
            errors_state_list_normalized = [el / max_tmp for el in errors_state_list]
        else:
            errors_state_list_normalized = errors_state_list
        state.extend(errors_state_list_normalized)
        state.append(np.sum(self.prediction_errors)/1000)

        return torch.tensor([state], dtype=torch.float32)


class CsvDataset(Dataset):

    def __init__(self, csv_file, scaler_path, transform=None, sample=0, normalize=True, reduced=False,
                 create_scaler=True):
        self.transform = transform
        if reduced:
            columns_names = ["x", "y", "RSSI2", "RSSI3", "RSSI4", "RSSI6"]
            df = pd.read_csv(csv_file, names=columns_names)
        else:
            columns_names = ["x", "y", "RSSI1", "RSSI2", "RSSI3", "RSSI4", "RSSI5", "RSSI6"]
            df = pd.read_csv(csv_file, names=columns_names)
            columns2remove = ["RSSI1", "RSSI5"]
            df = df.drop(columns2remove, axis=1)
        df = df.drop_duplicates()
        self.train_index = 0
        if sample:
            df1 = df[(df['x'] % sample == 0) & (df['y'] % sample == 0)].copy()
            df2 = df[(df['x'] % sample != 0) | (df['y'] % sample != 0)].copy()
            df = pd.concat([df1, df2])
            df = df.reset_index(drop=True)
            self.train_index = df1.shape[0]
        else:
            self.train_index = df.shape[0]
        label_cols = ["x", "y"]
        features_cols = ["RSSI2", "RSSI3", "RSSI4", "RSSI6"]
        X = df[features_cols].copy()
        self.y = df[label_cols].copy()
        if normalize:
            if create_scaler:
                sc_X = StandardScaler()
                sc_X.fit(X)
                pickle.dump(sc_X, open('NetsImages/scalers/scaler.pkl', 'wb'))
            else:
                sc_X = pickle.load(open(scaler_path, 'rb'))
            X = sc_X.transform(X)
        self.X = pd.DataFrame(X, columns=features_cols)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = (self.X.iloc[idx], self.y.iloc[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def return_train_index(self):
        return self.train_index


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample
        return (torch.tensor([x]).float(),
                torch.tensor([y]).float())


class Net(nn.Module):
    def __init__(self, Ni, num_hid_nodes, No, activation_name, dropout_vec):
        super(Net, self).__init__()
        self.activation_name = activation_name
        activation = getattr(nn, self.activation_name)()
        dimension = []
        dimension.append(Ni)
        dimension.extend(num_hid_nodes)
        dimension.append(No)
        linear_layers = [nn.Linear(in_features=dimension[counter],
                                   out_features=dimension[counter + 1])
                         for counter in range(len(dimension) - 1)]
        if (self.activation_name != "Sigmoid" and self.activation_name != "Tanh"):
            [torch.nn.init.kaiming_normal_(layer.weight) for layer in linear_layers]
        layers = []
        layers.append(linear_layers[0])
        for i in range(1, len(linear_layers)):
            layers.append(activation)
            layers.append(nn.Dropout(dropout_vec[i - 1]))
            layers.append(linear_layers[i])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

    def save(self, file_name):
        model_folder_path = 'NetsImages/RegModels'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


def split_dataset_clusters(dataset, cluster_borders):
    dataset_cluster = []
    for sample in dataset:
        x_coordinate = sample[1][0][0].item()
        y_coordinate = sample[1][0][1].item()
        if ((x_coordinate < cluster_borders[1]) and (x_coordinate > cluster_borders[3]) and
                (y_coordinate < cluster_borders[2]) and (y_coordinate > cluster_borders[0])):
            dataset_cluster.append(sample)

    return dataset_cluster


def draw_cluster_borders(image):
    Point = namedtuple('Point', 'x, y')
    antenna_1 = Point(324, 213)
    antenna_2 = Point(714, 213)
    antenna_3 = Point(335, 443)
    antenna_4 = Point(747, 455)

    color = (100, 100, 100)
    pygame.draw.line(image, color, (0, int(618/2)), (925, int(618/2)), width=3)
    pygame.draw.line(image, color, (int(926 / 3), 0), (int(926/3), 617), width=3)
    pygame.draw.line(image, color, (int(2*926 / 3), 0), (int(2*926 / 3), 617), width=3)

    ## And draw the circles in correspondence to the beacons
    pygame.draw.circle(image, color, antenna_1, 40, width=1)
    pygame.draw.circle(image, color, antenna_2, 40, width=1)
    pygame.draw.circle(image, color, antenna_3, 40, width=1)
    pygame.draw.circle(image, color, antenna_4, 40, width=1)
    pygame.draw.circle(image, color, antenna_1, 30, width=2)
    pygame.draw.circle(image, color, antenna_2, 30, width=2)
    pygame.draw.circle(image, color, antenna_3, 30, width=2)
    pygame.draw.circle(image, color, antenna_4, 30, width=2)
    pygame.draw.circle(image, color, antenna_1, 20, width=3)
    pygame.draw.circle(image, color, antenna_2, 20, width=3)
    pygame.draw.circle(image, color, antenna_3, 20, width=3)
    pygame.draw.circle(image, color, antenna_4, 20, width=3)
    pygame.draw.circle(image, color, antenna_1, 8, width=4)
    pygame.draw.circle(image, color, antenna_2, 8, width=4)
    pygame.draw.circle(image, color, antenna_3, 8, width=4)
    pygame.draw.circle(image, color, antenna_4, 8, width=4)


def return_cluster_border(cluster_letter):
    cluster_borders = []
    if cluster_letter == "A": cluster_borders = [0, int(np.floor(926 / 3)), int(np.floor(618 / 2)), 0]
    if cluster_letter == "B": cluster_borders = [0, int(np.floor(926 * (2 / 3))), int(np.floor(618 / 2)), int(np.ceil(926 / 3))]
    if cluster_letter == "C": cluster_borders = [0, 925, int(np.floor(618 / 2)), int(np.ceil(926 * (2 / 3)))]
    if cluster_letter == "D": cluster_borders = [int(np.ceil(618 / 2)), int(np.floor(926 / 3)), 617, 0]
    if cluster_letter == "E": cluster_borders = [int(np.ceil(618 / 2)), int(np.floor(926 * (2 / 3))), 617, int(np.ceil(926 / 3))]
    if cluster_letter == "F": cluster_borders = [int(np.ceil(618 / 2)), 925, 617, int(np.ceil(926 * (2 / 3)))]
    if cluster_letter == "Env": cluster_borders = [0, 925, 617, 0]

    return cluster_borders


def create_datasets(cluster_letter, ppnn_hyperparams, utils_dict):
    train_dataset = []
    test_dataset = []
    cluster_borders = return_cluster_border(cluster_letter)
    dataset = CsvDataset(utils_dict["dataset_path"], utils_dict["scaler_path"],
                         transform=ppnn_hyperparams["composed_transform"], sample=ppnn_hyperparams["samples_for_sampling"],
                         normalize=True, create_scaler=True)
    train_idx = dataset.return_train_index()
    for i in range(0, train_idx):
        train_dataset.append(dataset[i])
    train_dataset_cluster = []
    random.seed(0)
    random.shuffle(train_dataset)
    train_dataset_ground_truth = split_dataset_clusters(train_dataset, cluster_borders)
    df_train_dataset_gt = pd.DataFrame(columns=["x", "y", "RSSI2", "RSSI3", "RSSI4", "RSSI6"])
    for i in range(len(train_dataset_ground_truth)):
        coordinates = train_dataset_ground_truth[i][1][0].tolist()
        RSSIValues = train_dataset_ground_truth[i][0][0].tolist()
        coordinates.extend(RSSIValues)
        df_train_dataset_gt.loc[i] = coordinates
    df_train_dataset_gt.to_csv('NetsImages/Datasets/GroundTruth/train_dataset_g_t_cluster{}.csv'.format(cluster_letter),
                            index=False, header=False)
    tmp_idx = 0
    num_training_points_cluster = 0
    while num_training_points_cluster < ppnn_hyperparams["max_num_training_points_cluster"]:
        x_coordinate = train_dataset[tmp_idx][1][0][0].item()
        y_coordinate = train_dataset[tmp_idx][1][0][1].item()
        if ((x_coordinate <= int(cluster_borders[1])) and (y_coordinate <= int(cluster_borders[2])) and
                (x_coordinate >= int(cluster_borders[3])) and (y_coordinate >= int(cluster_borders[0]))):
            train_dataset_cluster.append(train_dataset[tmp_idx])
            num_training_points_cluster += 1
        else:
            test_dataset.append(train_dataset[tmp_idx])
        tmp_idx += 1

    for i in range(tmp_idx, train_idx):
        test_dataset.append(train_dataset[i])
    for i in range(train_idx, len(dataset)):
        test_dataset.append(dataset[i])
    train_dataset = train_dataset_cluster
    df_train_dataset = pd.DataFrame(columns=["x", "y", "RSSI2", "RSSI3", "RSSI4", "RSSI6"])
    for i in range(len(train_dataset)):
        coordinates = train_dataset[i][1][0].tolist()
        RSSIValues = train_dataset[i][0][0].tolist()
        coordinates.extend(RSSIValues)
        df_train_dataset.loc[i] = coordinates

    df_train_dataset.to_csv('NetsImages/Datasets/TrainPPNN/train_dataset_cluster{}_iter_0.csv'.format(cluster_letter),
                            index=False, header=False)

    return train_dataset, test_dataset


def apply_heatmap_on_cluster(super_imposed_img, img, cluster_letter):
    cluster_borders = return_cluster_border(cluster_letter)
    cropped_heatmap = super_imposed_img[cluster_borders[0]:cluster_borders[2], cluster_borders[3]:cluster_borders[1]].copy()
    new_super_imposed_img = img.copy()
    new_super_imposed_img[cluster_borders[0]:cluster_borders[2], cluster_borders[3]:cluster_borders[1]] = cropped_heatmap

    return new_super_imposed_img

# DRQN related code modified by https://mlpeschl.com/post/tiny_adrqn/
class DRQN(nn.Module):
    def __init__(self, n_actions, state_size, embedding_size):
        super(DRQN, self).__init__()
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.embedder = nn.Linear(n_actions, embedding_size)
        self.obs_layer = nn.Linear(state_size, 16)
        self.obs_layer2 = nn.Linear(16, 32)
        self.lstm = nn.LSTM(input_size=32 + embedding_size, hidden_size=128, batch_first=True)
        self.out_layer = nn.Linear(128, n_actions)

    def forward(self, observation, action, hidden=None):
        action_embedded = self.embedder(action)
        observation = F.relu(self.obs_layer(observation))
        observation = F.relu(self.obs_layer2(observation))
        lstm_input = torch.cat([observation, action_embedded], dim=-1)
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(lstm_input, hidden)
        else:
            lstm_out, hidden_out = self.lstm(lstm_input)

        q_values = self.out_layer(lstm_out)
        return q_values, hidden_out


class ReplayMemoryLstm():
    def __init__(self, max_storage, sample_length):
        self.max_storage = max_storage
        self.sample_length = sample_length
        self.counter = -1
        self.filled = -1
        self.storage = [0 for i in range(max_storage)]

    def write_tuple(self, aoarod):
        if self.counter < self.max_storage - 1:
            self.counter += 1
        if self.filled < self.max_storage:
            self.filled += 1
        else:
            self.counter = 0
        self.storage[self.counter] = aoarod

    def __len__(self):
        return self.filled

    def sample(self, batch_size):
        seq_len = self.sample_length
        last_actions = []
        last_observations = []
        actions = []
        rewards = []
        observations = []
        dones = []

        for i in range(batch_size):
            if self.filled - seq_len < 0:
                raise Exception("Reduce seq_len or increase exploration at start.")
            start_idx = np.random.randint(self.filled - seq_len)
            last_act, last_obs, act, rew, obs, done = zip(*self.storage[start_idx:start_idx + seq_len])
            last_actions.append(list(last_act))
            last_observations.append(last_obs)
            actions.append(list(act))
            rewards.append(list(rew))
            observations.append(list(obs))
            dones.append(list(done))

        return torch.tensor(last_actions), torch.tensor(last_observations, dtype=torch.float32), \
               torch.tensor(actions), torch.tensor(rewards).float(), \
               torch.tensor(observations, dtype=torch.float32), torch.tensor(dones)


def select_action_lstm(net, state, action, hidden, steps_done, ACTION_SELECTION, temperature, EPS_END=0, EPS_DECAY=600, EPS_START=1):
    # Choose the softmax policy
    if ACTION_SELECTION == 2:
        if temperature < 0:
            raise Exception('The temperature value must be >= 0 ')

        if temperature == 0:
            return select_action_lstm(net, state, action, hidden, 0, 0, 0, EPS_END=0)  # greedy choice

        with torch.no_grad():
            net.eval()
            net_out, hidden = net(state, action, hidden)

        temperature = max(temperature, 1e-8)  # set a minimum to the temperature for numerical stability
        softmax_out = nn.functional.softmax(net_out[0][0] / temperature, dim=0).cpu().numpy()
        all_possible_actions = np.arange(0, softmax_out.shape[-1])
        action = np.random.choice(all_possible_actions, p=softmax_out)
        return torch.tensor([action], device=device, dtype=torch.long), hidden
    else:
        if ACTION_SELECTION == 1:
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        else:
            epsilon = EPS_END
        with torch.no_grad():
            net.eval()
            net_out, hidden = net(state, action, hidden)

        best_action = int(net_out.argmax())
        action_space_dim = net_out.shape[-1]
        if random.random() < epsilon:
            non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
            action = random.choice(non_optimal_actions)
        else:
            action = best_action
        return torch.tensor([[action]], device=device, dtype=torch.long), hidden

