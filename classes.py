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
import collections
import itertools
import statistics


pygame.init()
font = pygame.font.SysFont('arial', 25)
Point = namedtuple('Point', 'x, y')
device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Direction(Enum):
    RIGHT_S = 1
    LEFT_S = 2
    UP_S = 3
    DOWN_S = 4


class SnakeGameAI:

    def __init__(self, cluster_letter, err_cluster_image, coord, localization_model,
                 scaler_path, starting_point=None, random_test=False, max_steps=40, w=926, h=618):
        """
        Initialize the environment.

        Parameters
        ----------
        cluster_letter: str
                Letter identifying the cluster considered
        err_cluster_image: pygame.image
                Heatmap of the error on the cluster, used to show the agent in the environment
        coord: list
                List of lists, where each list contains the coordinates and the RSSI values
        localization_model: Net
                Pre-trained neural network
        scaler_path: str
                Path where the standard scaler is saved
        random_test: bool
                True if the environment is used for the random-action policy
        max_steps: int, optional
                Maximum number of steps of the agent inside the environment (default is 40)
        w: int, optional
                Width of the image representing the environment (default is 926)
        h: int, optional
                Height of the image representing the environment (default is 618)
        """
        self.snake = []
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
        self.random_test = random_test


    def reset(self, test=False, pt=None):
        """
        Reset the environment.

        test: bool, optional
                Used to specify if the initialization of the agent is done randomly during training or
                    deterministically during test. (default is False)
        pt: Point, optional
                Used to specify the starting point of the reset deterministically. If None, the agent
                starts every episode from the center of the cluster. (default is None)
        """
        self.prediction_errors = [0]
        self.last_actions = deque(maxlen=10)
        self.last_errors = deque(maxlen=3) #deque(maxlen=5)
        self.direction = Direction.RIGHT_S
        #####################################################################################
        ######################################################################################
        ################# CONTROLLAAAAAA SE PUO ESSRE TOLTO #####################################################
        #self.head = Point(int((self.clusterBorders[1] + self.clusterBorders[3]) / 2),
        #                  int((self.clusterBorders[0] + self.clusterBorders[2]) / 2))
        if pt:
            print("pt: ", pt)
            self.head = Point(int(pt[1][0][0].item()), int(pt[1][0][1].item()))
            print("head: ", self.head)
        else:
            #if test:
            self.head = Point(int((self.clusterBorders[1] + self.clusterBorders[3]) / 2),
                              int((self.clusterBorders[0] + self.clusterBorders[2]) / 2))
            #else:  # randomically initialize the point
            #    rand_x = random.randint(int(self.clusterBorders[3]), int(self.clusterBorders[1]))
            #    rand_y = random.randint(int(self.clusterBorders[0]), int(self.clusterBorders[2]))
            #    self.head = Point(int(rand_x), int(rand_y))
        # Reset the snake to the head
        self.snake = [self.head]
        self.iter = 0
        # List containing all the visited places of the agent
        self.places = []
        state_after_reset = self.get_state()
        return state_after_reset

    def return_snake(self):
        return self.snake

    def return_places(self):
        return self.places

    def save_display(self, path):
        pygame.image.save(self.display, path)

    def play_step(self, action, speed=30):  # , episode_length, cluster_letter,
        """
        Execute the step of the agent, thus move one step in the environment.

        Parameter
        ---------
        action: torch.tensor
                Action of the agent
        speed: int
                The agent won't go faster than speed frames per second

        Returns
        ------
        The tuple (next_state, reward, done)
        """

        # First save the action in the list of old actions (it could be included in the state, to represent the history)
        #self.last_actions.append(action.item())
        self.last_actions.append(action)
        self.iter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move(action)  # update the head
        self.snake.insert(0, self.head)  # insert the head in the first position of the list
        self.places.append(self.head)
        reward = 0
        game_over = False
        # Compute the error between predicted position and true one
        instant_error = self.compute_error(self.head)
        self.last_errors.append(instant_error)
        self.prediction_errors.append(instant_error)
        next_state = self.get_state()

        if self.is_collision(self.head):  # Check if the head did not collide
            game_over = True
            #print('Borders')
            reward = -1000 #-1000
            return next_state, torch.tensor([reward], dtype=torch.float32), game_over
        # Check if the episode lasted for too many steps
        if self.iter > self.max_steps:
            game_over = True
            return next_state, torch.tensor([reward], dtype=torch.float32), game_over
        # Penalize if the agent is in a location where it was already passed
        # (excluding the last location, where it is now)
        weight = 1
        for i in range(len(self.places) - 1):
            if self.head == self.places[i]:
                reward -= 120 * weight   # 120
                weight += 1
        if weight > 2:
            return next_state, torch.tensor([reward], dtype=torch.float32), game_over

        #print("instant_error: ", instant_error)
        #print("np.sum(self.prediction_errors): ", np.sum(self.prediction_errors))
        #print("np.mean(self.prediction_errors): ", np.mean(self.prediction_errors))
        reward += 1 * instant_error + np.sum(self.prediction_errors)/1000

        self._update_ui(self.err_cluster_image)  # , episode_length, cluster_letter)
        # This set the velocity of the agent to "speed" -> It won't go faster than "speed" frames per second
        self.clock.tick(speed)
        return next_state, torch.tensor([reward], dtype=torch.float32), game_over

    def get_RSSI(self, point):
        """
        Obtain the values of the RSSIs of one location in the building.

        Parameters
        ----------
        point: Point
                Point of the environment of which we want to have the RSSI values

        Returns
        -------
        RSSIs_df : pd.DataFrame
                Dataframe containing the RSSI values
        """

        ind = point.x + point.y * 926
        RSSIs = self.coord[ind]
        RSSIs_df = pd.DataFrame(columns=["RSSI1", "RSSI2", "RSSI3", "RSSI4", "RSSI5", "RSSI6"])
        RSSIs_df.loc[0] = RSSIs
        RSSIs_df = RSSIs_df.drop(["RSSI1", "RSSI5"], axis=1).copy()  # Remove 2 extra columns present in original dataset
        return RSSIs_df

    def compute_error(self, pt):
        """
        Computes the error between prediction of the NN for localization and the true point.
        Note: The creation of the dataframe is needed so that the standard scaler does not return the warning that
              the features have no name (while in the training there was a name because they were in a dataframe).
              In addition, it is useful to delete the columns of the RSSIs not used.

        Parameter
        ---------
        pt: Point
                Point from which you obtain the prediction

        Returns
        -------
        error: float
                The MSE between prediction and ground truth
        """

        if pt.x + pt.y * 926 < self.w * self.h:
            RSSIs_df = self.get_RSSI(pt)
        else:
            return 0

        with torch.no_grad():
            # Obtain the prediction of the coordinate, given the RSSIs
            sc_x = pickle.load(open(self.scaler_path, 'rb'))
            data_new = torch.tensor(sc_x.transform(RSSIs_df), dtype=torch.float32)
            data_new = data_new.to(device)
            output = self.localization_model(data_new)
            # Compute the SE between the real point and the predicted one
            error = torch.sqrt((output[0][0] - pt.x) ** 2 + (output[0][1] - pt.y) ** 2)
            error = error.cpu().detach()
        return error

    def is_collision(self, pt):
        """
        Identifies a collision with the cluster borders.

        Parameters
        ----------
        pt: Point
                Point where the agent is

        Returns
        -------
        True if there is a collision
        """

        if ((pt.x > self.clusterBorders[1]) or (pt.x < self.clusterBorders[3]) or
                (pt.y > self.clusterBorders[2]) or (pt.y < self.clusterBorders[0])):
            return True
        else:
            return False

    def _update_ui(self, err_cluster_image):  # , episode_length, cluster_letter, block_size=11):
        """
        Update the user interface drawing the agent past and present locations

        Parameters
        ----------
        err_cluster_image : pygame.image
                Image representing the environment
        block_size: int
                Size of the rectangle with which the agent is represented in the image
        """

        # rgb colors
        WHITE = (255, 255, 255)
        RED = (200, 0, 0)
        CUSTOM = (200, 100, 200)
        BLUE1 = (0, 0, 255)
        BLUE2 = (0, 100, 255)
        BLACK = (0, 0, 0)
        self.display.blit(err_cluster_image, (0, 0))  # Draw one image onto one other
        draw_cluster_borders(self.display)
        for pt in self.snake:
            #pygame.draw.rect(self.display, CUSTOM,
            #                 pygame.Rect(pt.x - block_size / 2, pt.y - block_size / 2, block_size, block_size))
            #pygame.draw.rect(self.display, BLACK,
            #                 pygame.Rect(pt.x - block_size / 2, pt.y - block_size / 2, block_size, block_size), width=2)
            pygame.draw.circle(self.display, CUSTOM, (pt.x, pt.y), 8)
            pygame.draw.circle(self.display, BLACK, (pt.x, pt.y), 8, 2)
        pygame.display.flip()  # Update
        #pygame.image.save(self.display, "screenshot_cluster{}_step{}.jpeg".format(cluster_letter, episode_length))

    def _move(self, action, action_step_length=25):
        """
        Update the head of the snake after the movement (the action).
        Acions list:
        0 -> Straight
        1 -> Right
        2 -> Left

        Parameters
        ----------
        action: torch.tensor
                Action of the agent (it is an array with binary values indicating the direction)
        action_step_length: int, optional
                Length of the step (default is 25)
        """

        #action = action[0]
        clock_wise1 = [Direction.RIGHT_S, Direction.DOWN_S, Direction.LEFT_S, Direction.UP_S]  # Short actions
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
        # Translates the head by the correct quantity
        if self.direction == Direction.RIGHT_S:
            x += action_step_length
        elif self.direction == Direction.LEFT_S:
            x -= action_step_length
        elif self.direction == Direction.DOWN_S:
            y += action_step_length
        elif self.direction == Direction.UP_S:
            y -= action_step_length

        if self.random_test:
            if self.is_collision(Point(x, y)):
                pass
            else:
                self.head = Point(x, y)
        else:
            self.head = Point(x, y)

    def get_state(self, action_step_length=25):
        """
        Obtain the state where the agent is. Recall that the state is composed by 4 main groups ->
        1. Dangers (S,R,L)
        2. Already measured points (S,R,L)
        3. Errors of the neighbors (S,R,L)
        4. History of previous errors (5 previous locations)

        Parameters
        ----------
        action_step_length: int, optional
                Length of the step (default is 25)

        Returns
        -------
        state: list
                State
        """

        head = self.snake[0]
        # Here computes the neighbors to the head of the snake in all the directions
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
        # Now find the values of the errors straight, right and left, thus changing the reference frame and considering
        # it relative to the agent
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
            #################################################################### DANGER
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
            (dir_r and point_r1 in self.snake) or
            (dir_l and point_l1 in self.snake) or
            (dir_u and point_u1 in self.snake) or
            (dir_d and point_d1 in self.snake),

            # Already measured right (recall the direction is in absolute terms!)
            (dir_r and point_d1 in self.snake) or
            (dir_l and point_u1 in self.snake) or
            (dir_u and point_r1 in self.snake) or
            (dir_d and point_l1 in self.snake),

            # Already measured left
            (dir_r and point_u1 in self.snake) or
            (dir_l and point_d1 in self.snake) or
            (dir_u and point_l1 in self.snake) or
            (dir_d and point_r1 in self.snake),
            ########################################################################################################
            # Err value
            err_straight,
            err_right,
            err_left,
        ]
        # Append to the state the history of the prediction errors of the last 5 locations where the agent walked
        errors_state_list = [0] * 3  # 5
        for i in range(len(list(self.last_errors))):
            errors_state_list[i] = self.last_errors[i]
        max_tmp = max(errors_state_list)
        if max_tmp > 0:
            errors_state_list_normalized = [el / max_tmp for el in errors_state_list]
        else:
            errors_state_list_normalized = errors_state_list

        state.extend(errors_state_list_normalized)

        state.append(np.sum(self.prediction_errors)/1000)  ## PIU O MENO NORMALIZZATA
        #print("np.sum(self.prediction_errors)/100: ", np.sum(self.prediction_errors)/100)
        #print("np.mean(self.snake): ", np.mean(self.snake))

        return torch.tensor([state], dtype=torch.float32)


class CsvDataset(Dataset):

    def __init__(self, csv_file, scaler_path, transform=None, sample=0, normalize=True, reduced=False,
                 create_scaler=True):
        """
        Load the dataset reading a csv file.

        Parameters
        ----------
        csv_file : str
                The path of the dataset saved in csv format
        scaler_path : str
                Path where the scaler is saved/loaded
        transform : torchvision.transforms, optional
                The composed transformation that must be applied to the samples of the dataset (default is None)
        sample : int, optional
                The sampling rate, used to create the dataset sampling the original one, with the purpose of extracting
                 the training dataset as a grid from the environment (default is 0, i.e. no sampling)
        normalize : bool, optional
                It is equal to True if you want to normalize the dataset (default is True)
        reduced : bool, optional
                It is True if you are using the 4 standard columns instead of the 6 present in the dataset (default is
                False). It is False if the dataset loaded contains 6 columns and thus the 4 default columns are
                extracted in this method
        create_scaler : bool, optional
                It is True if you want to create the standard scaler, False if you want to load one already created
                (default is True)
        """

        self.transform = transform
        if reduced:
            columns_names = ["x", "y", "RSSI2", "RSSI3", "RSSI4", "RSSI6"]
            df = pd.read_csv(csv_file, names=columns_names)
        else:
            # Read the file from csv to pd
            columns_names = ["x", "y", "RSSI1", "RSSI2", "RSSI3", "RSSI4", "RSSI5", "RSSI6"]
            df = pd.read_csv(csv_file, names=columns_names)
            columns2remove = ["RSSI1", "RSSI5"]
            df = df.drop(columns2remove, axis=1)
        df = df.drop_duplicates()
        self.train_index = 0
        # Sampling is needed to split training dataset and test one. The two are then stacked one above the other.
        if sample:
            df1 = df[(df['x'] % sample == 0) & (df['y'] % sample == 0)].copy()
            df2 = df[(df['x'] % sample != 0) | (df['y'] % sample != 0)].copy()
            df = pd.concat([df1, df2])
            df = df.reset_index(drop=True)
            self.train_index = df1.shape[0]
        else:
            self.train_index = df.shape[0]
        # Get the features and the labels of the dataset
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
        # The length of the dataset is simply the length of the self.data list
        return len(self.X)

    def __getitem__(self, idx):
        sample = (self.X.iloc[idx], self.y.iloc[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def return_train_index(self):
        return self.train_index


class ToTensor(object):
    """Convert samples to Tensors."""
    def __call__(self, sample):
        x, y = sample
        return (torch.tensor([x]).float(),
                torch.tensor([y]).float())


class Net(nn.Module):
    def __init__(self, Ni, num_hid_nodes, No, activation_name, dropout_vec):
        """
        Initialize a neural network.

        Parameters
        ----------
        Ni : int
                Number of input nodes
        num_hid_nodes : list
                List containing the number of hidden nodes for each hidden layer
        No : int
                Number of output nodes.
        activation_name : str
                Name of the activation function used inside the neural network
        dropout_vec : list
                List of dropout probabilities for the hidden layers
        """

        super(Net, self).__init__()
        self.activation_name = activation_name
        activation = getattr(nn, self.activation_name)()
        # Obtain a list with all the dimensions of the layers of the network
        dimension = []
        dimension.append(Ni)
        dimension.extend(num_hid_nodes)
        dimension.append(No)
        linear_layers = [nn.Linear(in_features=dimension[counter],
                                   out_features=dimension[counter + 1])
                         for counter in range(len(dimension) - 1)]
        # Initialization of the weights
        if (self.activation_name != "Sigmoid" and self.activation_name != "Tanh"):
            [torch.nn.init.kaiming_normal_(layer.weight) for layer in linear_layers]
        layers = []
        layers.append(linear_layers[0])
        for i in range(1, len(linear_layers)):
            layers.append(activation)
            layers.append(nn.Dropout(dropout_vec[i - 1]))
            layers.append(linear_layers[i])
        # the * operator to expand the list into positional arguments, because takes
        # as argument or a sequence or an ordered dictionary
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
    """
    This function extract from "dataset" the samples inside the constraints given by "cluster_borders"

    Parameters
    ----------
    dataset : list
            Dataset from which the sub-dataset must be extracted
    cluster_borders : list
            List of 4 extreme points of the area considered (of the environment)

    Returns
    -------
    dataset_cluster : Subset of the dataset containing all the samples inside cluster_borders
    """

    dataset_cluster = []
    for sample in dataset:
        #print("sample.item(): ", sample)
        x_coordinate = sample[1][0][0].item()
        y_coordinate = sample[1][0][1].item()
        if ((x_coordinate < cluster_borders[1]) and (x_coordinate > cluster_borders[3]) and
                (y_coordinate < cluster_borders[2]) and (y_coordinate > cluster_borders[0])):
            dataset_cluster.append(sample)
    #print("dataset finished")

    return dataset_cluster


def draw_cluster_borders(image):
    """
    Draw the cluster boarders on the image to achieve a better visualization.

    Parameter
    ---------
    image : pygame.display
            Image representing the environment of the agent where to draw the cluster borders
    """
    Point = namedtuple('Point', 'x, y')
    antenna_1 = Point(324, 213)  # up left
    antenna_2 = Point(714, 213)  # up right
    antenna_3 = Point(335, 443)  # bottom left
    antenna_4 = Point(747, 455)

    #color = (255, 255, 255)
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
    """
    Given the letter of the cluster considered, it returns the borders of such area. [U,R,B,L]

    Parameter
    ---------
    cluster_letter : str
            Letter identifying the cluster

    Returns
    -------
    cluster_borders : list
            Borders of the area considered
    """
    cluster_borders = []
    if cluster_letter == "A": cluster_borders = [0, int(np.floor(926 / 3)), int(np.floor(618 / 2)), 0]
    if cluster_letter == "B": cluster_borders = [0, int(np.floor(926 * (2 / 3))), int(np.floor(618 / 2)), int(np.ceil(926 / 3))]
    if cluster_letter == "C": cluster_borders = [0, 925, int(np.floor(618 / 2)), int(np.ceil(926 * (2 / 3)))]
    if cluster_letter == "D": cluster_borders = [int(np.ceil(618 / 2)), int(np.floor(926 / 3)), 617, 0]
    if cluster_letter == "E": cluster_borders = [int(np.ceil(618 / 2)), int(np.floor(926 * (2 / 3))), 617, int(np.ceil(926 / 3))]
    if cluster_letter == "F": cluster_borders = [int(np.ceil(618 / 2)), 925, 617, int(np.ceil(926 * (2 / 3)))]
    if cluster_letter == "Env": cluster_borders = [0, 925, 617, 0]

    return cluster_borders


def create_datasets(cluster_letter, lnn_hyperparams, utils_dict):
    """
    Create the datasets needed.

    Parameters
    ----------
    dataset_path : str
            Path of the dataset containing all the samples associating the position to the RSSI values.
    scaler_path : str
            Path where the scaler will be saved
    composed_transform : torchvision.transforms
            Composed transform that will be applied to the samples of the dataset
    samples_for_sampling : int
            Distance between samples in the grid obtained when sampling the dataset to create the training one
    cluster_letter : str
            Letter representing the area of the environment selected
    max_num_training_points_cluster : int
            Maximum number of points in the training dataset of the considered cluster

    Returns
    -------
    train_dataset : list
            Training dataset obtained as a grid from the environment
    test_dataset : list
            Test dataset
    dataset : CsvDataset
            Dataset containing all the points before the split
    """

    train_dataset = []
    test_dataset = []
    cluster_borders = return_cluster_border(cluster_letter)
    dataset = CsvDataset(utils_dict["dataset_path"], utils_dict["scaler_path"],
                         transform=lnn_hyperparams["composed_transform"], sample=lnn_hyperparams["samples_for_sampling"],
                         normalize=True, create_scaler=True)
    train_idx = dataset.return_train_index()
    # Create the training dataset, inserting the first "train_idx" samples of dataset, as they are the points
    # extracted as grid from the environment
    for i in range(0, train_idx):
        train_dataset.append(dataset[i])
    # Now we want to extract the training dataset of the considered cluster from the whole training dataset
    # Note that this dataset will have a length equal to "max_num_training_points_cluster"
    train_dataset_cluster = []
    # To obtain reproducible results
    random.seed(0)
    random.shuffle(train_dataset)
    # Save the train dataset of the whole cluster (which will be used as ground truth for the performance study)
    train_dataset_ground_truth = split_dataset_clusters(train_dataset, cluster_borders)
    df_train_dataset_gt = pd.DataFrame(columns=["x", "y", "RSSI2", "RSSI3", "RSSI4", "RSSI6"])
    for i in range(len(train_dataset_ground_truth)):
        coordinates = train_dataset_ground_truth[i][1][0].tolist()
        RSSIValues = train_dataset_ground_truth[i][0][0].tolist()
        coordinates.extend(RSSIValues)
        df_train_dataset_gt.loc[i] = coordinates
    df_train_dataset_gt.to_csv('NetsImages/Datasets/GroundTruth/train_dataset_g_t_cluster{}.csv'.format(cluster_letter),
                            index=False, header=False)
    # Index of the points in the cluster that will be added to the training dataset
    tmp_idx = 0
    # This counter keeps track of the number of points in the training dataset. When it is higher than
    # "max_num_training_points_cluster" -> it will stop the while loop
    num_training_points_cluster = 0
    while num_training_points_cluster < lnn_hyperparams["max_num_training_points_cluster"]:
        x_coordinate = train_dataset[tmp_idx][1][0][0].item()
        y_coordinate = train_dataset[tmp_idx][1][0][1].item()
        if ((x_coordinate <= int(cluster_borders[1])) and (y_coordinate <= int(cluster_borders[2])) and
                (x_coordinate >= int(cluster_borders[3])) and (y_coordinate >= int(cluster_borders[0]))):
            # The sample in tmp_idx belongs to the cluster identified by cluster_letter
            train_dataset_cluster.append(train_dataset[tmp_idx])
            num_training_points_cluster += 1
        else:
            test_dataset.append(train_dataset[tmp_idx])
        tmp_idx += 1

    # Insert in the test dataset all the remaining samples of the training dataset
    for i in range(tmp_idx, train_idx):
        test_dataset.append(train_dataset[i])
    # And then those of the test dataset
    for i in range(train_idx, len(dataset)):
        test_dataset.append(dataset[i])
    # rename the dataset of the cluster as the training dataset
    train_dataset = train_dataset_cluster
    df_train_dataset = pd.DataFrame(columns=["x", "y", "RSSI2", "RSSI3", "RSSI4", "RSSI6"])
    #df_test_dataset = pd.DataFrame(columns=["x", "y", "RSSI2", "RSSI3", "RSSI4", "RSSI6"])
    # Save the training dataset of the considered cluster. Then it will be updated with the samples collected from
    # the reinforcement learning algorithm
    for i in range(len(train_dataset)):
        coordinates = train_dataset[i][1][0].tolist()
        RSSIValues = train_dataset[i][0][0].tolist()
        coordinates.extend(RSSIValues)
        df_train_dataset.loc[i] = coordinates

    df_train_dataset.to_csv('NetsImages/Datasets/TrainLNN/train_dataset_cluster{}_iter_0.csv'.format(cluster_letter),
                            index=False, header=False)

    return train_dataset, test_dataset


def apply_heatmap_on_cluster(super_imposed_img, img, cluster_letter):
    """
    Apply the heatmap only over the cluster considered, to leave the layout untouched for the remaining part of
    the image.

    Parameters
    ----------
    super_imposed_img : cv2 image
            Heatmap of the whole environment containing the prediciton error
    img : cv2 image
            Layout of the environment
    cluster_letter : str
            Letter representing the cluster selected in the environment
    """
    cluster_borders = return_cluster_border(cluster_letter)  # [U, R, B, L]
    cropped_heatmap = super_imposed_img[cluster_borders[0]:cluster_borders[2], cluster_borders[3]:cluster_borders[1]].copy()
    new_super_imposed_img = img.copy()
    new_super_imposed_img[cluster_borders[0]:cluster_borders[2], cluster_borders[3]:cluster_borders[1]] = cropped_heatmap

    return new_super_imposed_img


class ReplayMemory(object):

    def __init__(self, capacity):
        """
        Initialize the replay memory as a deque with a certain capacity.
        When the maximum capacity has been reached, the old elements are discarded and
        substituted with the new ones.

        Parameter
        ---------
        capacity : int
                Maximum capacity of the replay memory.
        """

        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        # Here len(self) calls the method __len__ which returns the length of the memory
        batch_size = min(batch_size, len(self))
        return random.sample(self.memory, batch_size)  # Randomly select "batch_size" samples

    def __len__(self):
        return len(self.memory)  # Return the number of samples currently stored in the memory


class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        """
        Initialize the deep Q network as a fully connected one,
        to use the state of the environment as input to the neural network.

        Parameters
        ----------
        state_space_dim : int
                Dimension of the state space.
        action_space_dim : int
                Dimension of the action space.
        """

        super().__init__()
        # Define the generic block which will compose the neural network.
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.Tanh())
            return layers

        self.linear = nn.Sequential(
            *block(state_space_dim, 256),
            *block(256, 256),
            nn.Linear(256, action_space_dim)
        )

    def forward(self, x):
        x = x.to(device)
        return self.linear(x)


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
        # Takes observations with shape (batch_size, seq_len, state_size)
        # Takes one_hot actions with shape (batch_size, seq_len, n_actions)
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

    def act(self, observation, last_action, epsilon, hidden=None):
        q_values, hidden_out = self.forward(observation, last_action, hidden)
        if np.random.uniform() > epsilon:
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.n_actions)

        return action, hidden_out

class ReplayMemory_v2(object):

    def __init__(self, capacity, seq_len):
        """
        Initialize the replay memory as a deque with a certain capacity.
        When the maximum capacity has been reached, the old elements are discarded and
        substituted with the new ones.

        Parameter
        ---------
        capacity : int
                Maximum capacity of the replay memory.
        """
        self.seq_len = seq_len
        self.memory = deque(maxlen=capacity)

    def push(self, last_action, last_state, reward, action, state, done):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((last_action, last_state, reward, action, state, done))

    def sample(self, batch_size):
        # Here len(self) calls the method __len__ which returns the length of the memory
        batch_size = min(batch_size, len(self))
        transitions = []
        for i in range(batch_size):
            if len(self) - self.seq_len < 0:
                raise Exception("Reduce seq_len or increase exploration at start.")
            random_idx = np.random.randint(len(self) - self.seq_len)
            print("random_idx: ", random_idx)
            print("self.seq_len: ", self.seq_len)
            #transitions.append(self.memory[random_idx:random_idx + self.seq_len])
            transitions.append(collections.deque(itertools.islice(self.memory, random_idx, random_idx+self.seq_len)))
        #return random.sample(self.memory, batch_size)  # Randomly select "batch_size" samples
        return transitions

    def __len__(self):
        return len(self.memory)  # Return the number of samples currently stored in the memory


class ExpBuffer():
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
        # Returns sizes of (batch_size, seq_len, *) depending on action/observation/return/done
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
            # print(self.filled)
            # print(start_idx)
            last_act, last_obs, act, rew, obs, done = zip(*self.storage[start_idx:start_idx + seq_len])
            last_actions.append(list(last_act))
            #print("last_obs: ", last_obs)
            #print("last_obs.numpy(): ", last_obs.numpy())
            last_observations.append(last_obs)
            actions.append(list(act))
            rewards.append(list(rew))
            observations.append(list(obs))
            dones.append(list(done))

        return torch.tensor(last_actions), torch.tensor(last_observations, dtype=torch.float32), \
               torch.tensor(actions), torch.tensor(rewards).float(), \
               torch.tensor(observations, dtype=torch.float32), torch.tensor(dones)



def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
    """
    Function that updates policy network.

    Parameters
    ----------
    policy_net : DQN
            Policy network that must be updated.
    target_net : DQN
            Target network used to compute the target update.
    replay_mem : ReplayMemory
            Replay memory.
    gamma : float
            Parameter to weight the future estimate of the return.
    optimizer : torch.optim
            Optimizer of the policy_net.
    loss_fn : torch.nn
            Loss function used.
    batch_size : int
            Batch size.
    """

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
    # Sample a transition tuple from the replay memory
    transitions = replay_mem.sample(batch_size)
    # puts together in the same dimension all the states, al the actions and so on
    # Like this:  [('a', 1), ('b', 2), ('c', 3), ('d', 4)] -> (['a', 'b', 'c', 'd'], [1, 2, 3, 4])
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a_t)
    policy_net.train()
    #print("state_batch: ", state_batch)
    #print("action_batch: ", action_batch)
    state_action_values = policy_net(state_batch).gather(1, action_batch)  # MA STO CALCOLANDO SUL MOMENTO, NON DOVREI INVECE SALVARE STEP DOPO STEP IL VALORE DELL'ACTION VALUE FUCNTION?
    # Now we have to compute the value of r + max_a[Q(s_{t+1}, a_{t+1})]
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        target_net.eval()
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(dim=1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    # Compute Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Clipping the gradients
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()



def select_action_lstm(net, state, action, hidden, steps_done, ACTION_SELECTION, temperature, EPS_END=0, EPS_DECAY=600, EPS_START=1):
    """
    ACTION_SELECTION is:
        - 0 -> fixed epsilon greedy algorithm
        - 1 -> epsilon greedy with decay
        - 2 -> softmax.

        Note: The state is already a tensor.

        Parameters
        ----------
        net : DQN
                Policy net.
        state : list
                State.
        steps_done : int
                Number of episodes used inside the training, used to compute
                    the decay for the epsilon greedy policy with decay.
        ACTION_SELECTION : int
                Select the type of policy that is used.
        temperature : float
                Temperature parameter for the softmax policy.
        EPS_END : float
                Epsilon parameter for the epsilon greedy policies.
        EPS_DECAY : float
                Constant which tunes the decay curve for the epsilon greedy policy with decay.
        EPS_START: float
                Epsilon parameter for the epsilon greedy policies.

        Returns
        -------
        action : int
                Action chosen from the selected policy.
    """

    # Choose the softmax policy
    if ACTION_SELECTION == 2:
        if temperature < 0:
            raise Exception('The temperature value must be >= 0 ')

        # If the temperature is 0, just select the greedy action using the eps-greedy policy with (fixed) epsilon = 0
        if temperature == 0:
            # The parameters "steps_done" and "temperature" are not used in the epsilon greedy policy
            return select_action_lstm(net, state, action, hidden, 0, 0, 0, EPS_END=0)  # greedy choice

        # Evaluate the network output from the current state -> Expected return from all the actions
        with torch.no_grad():
            net.eval()
            net_out, hidden = net(state, action, hidden)

        #print("net_out.shape: ", net_out.shape)
        #print("net_out.shape: ", net_out)
        # Apply softmax
        temperature = max(temperature, 1e-8)  # set a minimum to the temperature for numerical stability
        softmax_out = nn.functional.softmax(net_out[0][0] / temperature, dim=0).cpu().numpy()  # prima di LSTM era solo net_out[0]
        #print("softmax_out: ", softmax_out)
        # Sample the action using softmax output as pdf
        all_possible_actions = np.arange(0, softmax_out.shape[-1])
        # this samples a random element from "all_possible_actions" with the probability distribution p
        action = np.random.choice(all_possible_actions, p=softmax_out)
        #print("action: ", action)

        return torch.tensor([action], device=device, dtype=torch.long), hidden

    else:
        # Selects an action according to epsilon greedy policy. The probability of
        # choosing a random action will start at EPS_START and will decay exponentially
        # towards EPS_END.
        # This implies exploration at the beginning and more greedy actions at the end of the training.
        if ACTION_SELECTION == 1:  # Epsilon greedy policy with decay
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        else:  # epsilon greedy policy
            epsilon = EPS_END
        with torch.no_grad():
            net.eval()
            net_out, hidden = net(state, action, hidden)

        best_action = int(net_out.argmax())
        # Get the number of possible actions
        action_space_dim = net_out.shape[-1]
        if random.random() < epsilon:
            # List of non-optimal actions, removing the best action
            non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
            # Select randomly
            action = random.choice(non_optimal_actions)
        else:
            # Select best action
            action = best_action
        return torch.tensor([[action]], device=device, dtype=torch.long), hidden


def select_action(net, state, steps_done, ACTION_SELECTION, temperature, EPS_END=0, EPS_DECAY=600, EPS_START=1):
    """
    ACTION_SELECTION is:
        - 0 -> fixed epsilon greedy algorithm
        - 1 -> epsilon greedy with decay
        - 2 -> softmax.

        Note: The state is already a tensor.

        Parameters
        ----------
        net : DQN
                Policy net.
        state : list
                State.
        steps_done : int
                Number of episodes used inside the training, used to compute
                    the decay for the epsilon greedy policy with decay.
        ACTION_SELECTION : int
                Select the type of policy that is used.
        temperature : float
                Temperature parameter for the softmax policy.
        EPS_END : float
                Epsilon parameter for the epsilon greedy policies.
        EPS_DECAY : float
                Constant which tunes the decay curve for the epsilon greedy policy with decay.
        EPS_START: float
                Epsilon parameter for the epsilon greedy policies.

        Returns
        -------
        action : int
                Action chosen from the selected policy.
    """

    # Choose the softmax policy
    if ACTION_SELECTION == 2:
        if temperature < 0:
            raise Exception('The temperature value must be >= 0 ')

        # If the temperature is 0, just select the greedy action using the eps-greedy policy with (fixed) epsilon = 0
        if temperature == 0:
            # The parameters "steps_done" and "temperature" are not used in the epsilon greedy policy
            return select_action(net, state, 0, 0, 0, EPS_END=0)  # greedy choice

        # Evaluate the network output from the current state -> Expected return from all the actions
        with torch.no_grad():
            net.eval()
            net_out = net(state)

        #print("net_out.shape: ", net_out.shape)
        #print("net_out.shape: ", net_out)
        # Apply softmax
        temperature = max(temperature, 1e-8)  # set a minimum to the temperature for numerical stability
        softmax_out = nn.functional.softmax(net_out[0] / temperature, dim=0).cpu().numpy()  # prima di LSTM era solo net_out[0]
        #print("softmax_out: ", softmax_out)
        # Sample the action using softmax output as pdf
        all_possible_actions = np.arange(0, softmax_out.shape[-1])
        # this samples a random element from "all_possible_actions" with the probability distribution p
        action = np.random.choice(all_possible_actions, p=softmax_out)
        #print("action: ", action)

        return torch.tensor([[action]], device=device, dtype=torch.long)

    else:
        # Selects an action according to epsilon greedy policy. The probability of
        # choosing a random action will start at EPS_START and will decay exponentially
        # towards EPS_END.
        # This implies exploration at the beginning and more greedy actions at the end of the training.
        if ACTION_SELECTION == 1:  # Epsilon greedy policy with decay
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        else:  # epsilon greedy policy
            epsilon = EPS_END
        with torch.no_grad():
            net.eval()
            net_out = net(state)

        best_action = int(net_out.argmax())
        # Get the number of possible actions
        action_space_dim = net_out.shape[-1]
        if random.random() < epsilon:
            # List of non-optimal actions, removing the best action
            non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
            # Select randomly
            action = random.choice(non_optimal_actions)
        else:
            # Select best action
            action = best_action
        return torch.tensor([[action]], device=device, dtype=torch.long)
