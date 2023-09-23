import json
import cv2
import torchvision.transforms as transforms
from classes import *
from utils import *


def create_train_test_dataset_dict(lnn_hyperparams, utils_dict):
    print("Creating datasets...")
    test_dataset_dict = {}
    for cluster_letter in utils_dict["cluster_letters"]:
        print("cluster letter ", cluster_letter)
        train_dataset, test_dataset = create_datasets(cluster_letter, lnn_hyperparams, utils_dict)
        cluster_borders = return_cluster_border(cluster_letter)
        test_dataset_cluster = split_dataset_clusters(test_dataset, cluster_borders)
        test_dataset_dict[cluster_letter] = test_dataset_cluster
    return test_dataset_dict


def pretraining(lnn_hyperparams, utils_dict, sam_length, emb_size, bs, pl,):
    print("Pretraining...")
    for cluster_letter in utils_dict["cluster_letters"]:
        print("Training the model for letter {}...".format(cluster_letter))
        train_dataset = CsvDataset(
            "NetsImages/Datasets/TrainLNN/train_dataset_cluster{}_iter_0.csv".format(cluster_letter),
            utils_dict["scaler_path"],
            transform=lnn_hyperparams["composed_transform"], sample=0, normalize=False, reduced=True,
            create_scaler=False)
        train_dataset_list = from_dataframe_to_dataset(train_dataset)
        random.seed(0)
        random.shuffle(train_dataset_list)
        train_loader = DataLoader(dataset=train_dataset_list, batch_size=lnn_hyperparams["batch_size"], shuffle=False,
                                  num_workers=0)
        model = train_lnn(train_loader, cluster_letter, lnn_hyperparams)
        model.save(file_name='net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth'.format(cluster_letter, sam_length, emb_size, bs, pl, 0))


def train_lnn(train_loader, cluster_letter, lnn_hyperparams, after_dqn=False):
    '''
    This section is used to train the neural network for localization.
    In particular, this is used to create the pre-trained neural network LNN (trained with 10 points).
    '''
    torch.manual_seed(0)
    model = Net(lnn_hyperparams["Ni"], lnn_hyperparams["num_hid_nodes"], lnn_hyperparams["No"],
                lnn_hyperparams["activation_name"], lnn_hyperparams["dropout_vec"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lnn_hyperparams["learning_rate"])
    if after_dqn:
        if cluster_letter=="C":
            num_epochs = 3000  # 10k for cluster letter C
        else:
            num_epochs = 3000
    else:
        num_epochs = 500
    for epoch in range(num_epochs):
        if ((epoch + 1) % 1000 == 0): print("EPOCH: ", epoch + 1)
        for i, (RSSI, pos) in enumerate(train_loader):
            RSSI = RSSI.to(device)
            pos = pos.to(device)
            outputs = model(RSSI)
            loss = criterion(outputs, pos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach()
    return model


def test_lnn(test_dataset_dict, train_cluster_dict, lnn_hyperparams, utils_dict, k, sam_length, emb_size, bs, pl,):
    '''
    This section is used for the test of the neural network for localization. The section is also needed to
    draw the heatmaps showing the error values over the test dataset of the pre-trained neural networks.
    '''
    print("Testing the pre-trained LNN...")
    for cluster_letter in utils_dict["cluster_letters"]:
        if train_cluster_dict[cluster_letter]:
            cluster_borders = return_cluster_border(cluster_letter)
            print("Testing the model for letter {}...".format(cluster_letter))
            img = cv2.imread('NetsImages/FloorImages/Layout.png', cv2.IMREAD_COLOR)
            img = cv2.resize(img, (926, 618), interpolation=cv2.INTER_AREA)
            # Create empty images where the prediction error will be written
            empty_img = np.zeros((618, 926), np.uint8)
            empty_img_tmp = np.zeros((618, 926), np.longlong)
            test_loader = DataLoader(dataset=test_dataset_dict[cluster_letter], batch_size=1, shuffle=False, num_workers=0)
            torch.manual_seed(0)
            model = Net(lnn_hyperparams["Ni"], lnn_hyperparams["num_hid_nodes"], lnn_hyperparams["No"],
                        lnn_hyperparams["activation_name"], lnn_hyperparams["dropout_vec"])
            print("load model: NetsImages/RegModels/net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth".format(cluster_letter, sam_length, emb_size, bs, pl, k))
            model.load_state_dict(
                torch.load('NetsImages/RegModels/net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth'.format(cluster_letter, sam_length, emb_size, bs, pl, k)))
            model.eval()
            # This list contains the MSE of every point of the test dataset
            mse_array = []
            # This list contains the MSE of every point of the test dataset for the considered cluster
            mse_array_cluster = []
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for i, (RSSI, pos) in enumerate(test_loader):
                    RSSI = RSSI.to(device)
                    pos = pos.to(device)
                    coord_x = int(pos[0][0][0].item())
                    coord_y = int(pos[0][0][1].item())
                    if (coord_x < cluster_borders[1]) and (coord_y < cluster_borders[2]) and (coord_x > cluster_borders[3]) and \
                            (coord_y > cluster_borders[0]):
                        outputs = model(RSSI)
                        mse = torch.sqrt((outputs[0][0][0] - pos[0][0][0]) ** 2 + (outputs[0][0][1] - pos[0][0][1]) ** 2)
                        mse = mse.cpu()
                        mse_array.append(mse)
                        # Insert the error of the points inside the cluster considered
                        mse_array_cluster.append(mse.item())
                        empty_img_tmp[coord_y, coord_x] = mse.item()
                    n_samples = len(mse_array)
                    n_correct += (mse < 15).sum().item()
                #acc = 100.0 * n_correct / n_samples
                #print(f'Accuracy over the 6 areas = {acc}')
            empty_img = create_cluster_heatmap(empty_img, empty_img_tmp, mse_array_cluster)
            # Blur the image with the MSE heatmap
            blur = cv2.GaussianBlur(empty_img, (3, 3), 1)
            # Then change the colormap
            heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
            # Superimpose the heatmap with the prediction error and the layout of the environment
            super_imposed_img = cv2.addWeighted(heatmap_img, 0.7, img, 0.3, 0)
            super_imposed_img = apply_heatmap_on_cluster(super_imposed_img, img, cluster_letter)
            cv2.imwrite('NetsImages/super_imposed_img{}_iter_{}.jpg'.format(cluster_letter, k), super_imposed_img)


def train_dqn_lstm(lnn_hyperparams, utils_dict, dqn_hyperparams, k, sam_length=None, emb_size=None, bs=None, pl=None):
    """ Train the DQN """
    # Size image environment
    for GENERALIZATION in utils_dict["generalization_vec"]:
        err_cluster_image = pygame.image.load('NetsImages/super_imposed_img{}_iter_{}.jpg'.format(dqn_hyperparams["cluster_letter_training_dqn"], k))
        coord = np.loadtxt('NetsImages/Datasets/RSSIDataframe.csv', delimiter=',', dtype=np.float32, skiprows=0)[:, 2:]
        scaler_path = 'NetsImages/scalers/scaler.pkl'
        localization_model = Net(lnn_hyperparams["Ni"], lnn_hyperparams["num_hid_nodes"], lnn_hyperparams["No"],
                                 activation_name=lnn_hyperparams["activation_name"],
                                 dropout_vec=lnn_hyperparams["dropout_vec"])
        localization_model.load_state_dict(torch.load(
                'NetsImages/RegModels/net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth'.format(dqn_hyperparams["cluster_letter_training_dqn"], sam_length, emb_size, bs, pl, k)))

        localization_model.eval()
        if (dqn_hyperparams["cluster_letter_training_dqn"] == "C"):  # and (GENERALIZATION is False):
            env = SnakeGameAI(dqn_hyperparams["cluster_letter_training_dqn"], max_steps=80, err_cluster_image=err_cluster_image, coord=coord,  # 80
                              localization_model=localization_model, scaler_path=scaler_path,
                              random_test=utils_dict["random_action_test"])
        else:
            env = SnakeGameAI(dqn_hyperparams["cluster_letter_training_dqn"], max_steps=40, err_cluster_image=err_cluster_image, coord=coord,
                              localization_model=localization_model, scaler_path=scaler_path,
                              random_test=utils_dict["random_action_test"])

        # Get the shapes of the state space (observation_space) and action space (action_space)
        state_space_dim = env.observation_space_dim
        action_space_dim = env.action_space_dim

        # Set random seeds
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        # Set the learning curve
        exp_decay = np.exp(-np.log(dqn_hyperparams["initial_value"]) / dqn_hyperparams["num_iterations"] * 1)
        exploration_profile = [dqn_hyperparams["initial_value"] * (exp_decay ** i) for i in range(dqn_hyperparams["num_iterations"])]
        # Initialize the replay memory

        replay_buffer_size = dqn_hyperparams["replay_memory_capacity"]
        #sample_length = 10  # 3, 5 is good
        replay_buffer = ExpBuffer(replay_buffer_size, sam_length)
        #embedding_size = 16
        policy_net_drqn = DRQN(action_space_dim, state_space_dim, emb_size)
        target_net_drqn = DRQN(action_space_dim, state_space_dim, emb_size)
        target_net_drqn.load_state_dict(policy_net_drqn.state_dict())
        optimizer = torch.optim.Adam(policy_net_drqn.parameters(), lr=dqn_hyperparams["lr"])
        EXPLORE = 100  # 300
        #batch_size = 64
        gamma = 0.9


        # Real training part
        for episode_num, tau in enumerate(exploration_profile):
            print("Episode number: ", episode_num)
            last_observation = env.reset(test=True)
            last_action = torch.tensor(0)
            hidden = None
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                # Choose the action following the policy
                action, hidden = select_action_lstm(policy_net_drqn, last_observation.float().view(1, 1, -1),
                                       F.one_hot(last_action, action_space_dim).view(1, 1, -1).float(),
                                       hidden, episode_num, dqn_hyperparams["policy_type"], tau)
                    #print("action: ", action)
                observation, reward, done = env.play_step(action)
                replay_buffer.write_tuple((last_action, last_observation[0].numpy(), action, reward, observation[0].numpy(), done))
                last_action = action
                last_observation = observation

                # Updating Network
                #if episode_num > EXPLORE:
                # Sample a batch of action/observation/reward sequences
                if len(replay_buffer) > dqn_hyperparams["min_samples_for_training"]:
                    last_actions, last_observations, actions, rewards, observations, dones = replay_buffer.sample(bs)
                    # Pass the sequence of last observations and actions through the network
                    q_values, _ = policy_net_drqn.forward(last_observations, F.one_hot(last_actions, action_space_dim).float())
                    # Get the q_values for the executed actions in the respective observations
                    q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
                    # Query the target network for Q value predictions
                    predicted_q_values, _ = target_net_drqn.forward(observations, F.one_hot(actions, action_space_dim).float())
                    # Compute Q update target
                    target_values = rewards + (gamma * (1 - dones.float()) * torch.max(predicted_q_values, dim=-1)[0])
                    # Updating network parameters
                    optimizer.zero_grad()
                    loss = torch.nn.MSELoss()(q_values, target_values.detach())
                    loss.backward()
                    optimizer.step()
                if done:
                    break
            # Update the target network
            if episode_num%10==0:
                target_net_drqn.load_state_dict(policy_net_drqn.state_dict())
            torch.save(target_net_drqn.state_dict(),
                       'NetsImages/DQN/target_net_{}_samLength_{}_embSize{}_bs_{}_pl_{}.pth'.format(
                           dqn_hyperparams["cluster_letter_training_dqn"], sam_length, emb_size, bs, pl))


def test_dqn_lstm(train_cluster_dict, lnn_hyperparams, utils_dict, dqn_hyperparams, k, sam_length, emb_size, bs, pl):
    """ Test the DQN, obtaining the optimal path """
    for cluster_letter in utils_dict["cluster_letters"]:
        if train_cluster_dict[cluster_letter]:
            env_w = 926
            env_h = 618
            err_cluster_image = pygame.image.load('NetsImages/super_imposed_img{}_iter_{}.jpg'.format(cluster_letter, k))
            localization_model = Net(lnn_hyperparams["Ni"], lnn_hyperparams["num_hid_nodes"], lnn_hyperparams["No"],
                                     activation_name=lnn_hyperparams["activation_name"],
                                     dropout_vec=lnn_hyperparams["dropout_vec"])
            print("load LNN NetsImages/RegModels/net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth".format(cluster_letter, sam_length, emb_size, bs, pl, k))
            localization_model.load_state_dict(torch.load(
                'NetsImages/RegModels/net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth'.format(cluster_letter, sam_length, emb_size, bs, pl, k)))  # PERCHE SERVE STA RETE NELL ENVIRONMENT??
            localization_model.eval()
            coord = np.loadtxt(
                'NetsImages/Datasets/RSSIDataframe.csv',
                delimiter=',', dtype=np.float32, skiprows=0)[:, 2:]
            scaler_path = 'NetsImages/scalers/scaler.pkl'   ### QUESTO C'É ANCHE DENTRO UTILS DICT
            if cluster_letter == "C":  # ### PRIMA DEL 19-09-2023 ERA A 80 LA LUNGHEZZA QUI E 40 SOTTO
                env = SnakeGameAI(cluster_letter, max_steps=pl, err_cluster_image=err_cluster_image, coord=coord,
                                  localization_model=localization_model, scaler_path=scaler_path, starting_point=dqn_hyperparams["starting_point_dqn_test"][cluster_letter],
                                  random_test=utils_dict["random_action_test"])
            else:
                env = SnakeGameAI(cluster_letter, max_steps=pl, err_cluster_image=err_cluster_image, coord=coord,
                                  localization_model=localization_model, scaler_path=scaler_path, starting_point=dqn_hyperparams["starting_point_dqn_test"][cluster_letter],
                                  random_test=utils_dict["random_action_test"])
            # Get the shapes of the state space (observation_space) and action space (action_space)
            state_space_dim = env.observation_space_dim
            action_space_dim = env.action_space_dim
            # Load the weights of the DQN obtained
            #policy_net = DQN(state_space_dim, action_space_dim).to(device)
            #target_net = DQN(state_space_dim, action_space_dim).to(device)
            #embedding_size = 16
            policy_net_drqn = DRQN(action_space_dim, state_space_dim, emb_size)
            target_net_drqn = DRQN(action_space_dim, state_space_dim, emb_size)
            #if cluster_letter=="C":
            #policy_net.load_state_dict(torch.load('NetsImages/DQN/target_net_{}.pth'.format(dqn_hyperparams["cluster_letter_training_dqn"])))
            #target_net.load_state_dict(torch.load('NetsImages/DQN/target_net_{}.pth'.format(dqn_hyperparams["cluster_letter_training_dqn"])))
            policy_net_drqn.load_state_dict(torch.load('NetsImages/DQN/target_net_{}_samLength_{}_embSize{}_bs_{}_pl_{}.pth'.format(
                           dqn_hyperparams["cluster_letter_training_dqn"], sam_length, emb_size, bs, pl)))
            target_net_drqn.load_state_dict(torch.load('NetsImages/DQN/target_net_{}_samLength_{}_embSize{}_bs_{}_pl_{}.pth'.format(
                           dqn_hyperparams["cluster_letter_training_dqn"], sam_length, emb_size, bs, pl)))
            #else:
            #policy_net.load_state_dict(torch.load('NetsImages/DQN/target_net_generalization.pth'))
            #target_net.load_state_dict(torch.load('NetsImages/DQN/target_net_generalization.pth'))
            #policy_net.eval()
            #target_net.eval()
            policy_net_drqn.eval()
            target_net_drqn.eval()

            #run_test(env, utils_dict["random_action_test"], cluster_letter, policy_net, dqn_hyperparams)  ## NON SO SE METTENDOLO QUA DENTRO L ENVIRONMENT DOPO COMUNQUE LO VEDE MODIFICATO O NO (FACENDO RETURN_SNAKE)
            run_test_lstm(env, utils_dict["random_action_test"], cluster_letter, policy_net_drqn, dqn_hyperparams,
                          k, sam_length, emb_size, bs, pl)
            save_optimal_path(env, env_w, env_h, cluster_letter, coord, lnn_hyperparams, k)


def train_dqn(lnn_hyperparams, utils_dict, dqn_hyperparams, k, sam_length, emb_size, bs, pl):
    """ Train the DQN """
    # Size image environment
    for GENERALIZATION in utils_dict["generalization_vec"]:
        err_cluster_image = pygame.image.load(
            'NetsImages/super_imposed_img{}_iter_{}.jpg'.format(dqn_hyperparams["cluster_letter_training_dqn"], k))
        coord = np.loadtxt('NetsImages/Datasets/RSSIDataframe.csv', delimiter=',', dtype=np.float32, skiprows=0)[:, 2:]
        scaler_path = 'NetsImages/scalers/scaler.pkl'
        localization_model = Net(lnn_hyperparams["Ni"], lnn_hyperparams["num_hid_nodes"], lnn_hyperparams["No"],
                                 activation_name=lnn_hyperparams["activation_name"],
                                 dropout_vec=lnn_hyperparams["dropout_vec"])
        localization_model.load_state_dict(torch.load(
            'NetsImages/RegModels/net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth'.format(dqn_hyperparams["cluster_letter_training_dqn"],
                                                                                                sam_length, emb_size, bs, pl, k)))

        localization_model.eval()
        if (dqn_hyperparams["cluster_letter_training_dqn"] == "C"):  # and (GENERALIZATION is False):
            env = SnakeGameAI(dqn_hyperparams["cluster_letter_training_dqn"], max_steps=80,
                              err_cluster_image=err_cluster_image, coord=coord,  # 80
                              localization_model=localization_model, scaler_path=scaler_path,
                              random_test=utils_dict["random_action_test"])
        else:
            env = SnakeGameAI(dqn_hyperparams["cluster_letter_training_dqn"], max_steps=40,
                              err_cluster_image=err_cluster_image, coord=coord,
                              localization_model=localization_model, scaler_path=scaler_path,
                              random_test=utils_dict["random_action_test"])

        # Get the shapes of the state space (observation_space) and action space (action_space)
        state_space_dim = env.observation_space_dim
        action_space_dim = env.action_space_dim

        # Set random seeds
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        # Set the learning curve
        exp_decay = np.exp(-np.log(dqn_hyperparams["initial_value"]) / dqn_hyperparams["num_iterations"] * 1)
        exploration_profile = [dqn_hyperparams["initial_value"] * (exp_decay ** i) for i in
                               range(dqn_hyperparams["num_iterations"])]
        # Initialize the replay memory

        replay_mem = ReplayMemory(dqn_hyperparams["replay_memory_capacity"])
        policy_net = DQN(state_space_dim, action_space_dim).to(device)
        target_net = DQN(state_space_dim, action_space_dim).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=dqn_hyperparams["lr"])
        loss_fn = nn.SmoothL1Loss()

        # Real training part
        for episode_num, tau in enumerate(exploration_profile):
            print("Episode number: ", episode_num)

            state = env.reset(test=True)
            #print("first state: ", state)
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action = select_action(policy_net, state, episode_num, dqn_hyperparams["policy_type"], tau)
                next_state, reward, done = env.play_step(action)
                replay_mem.push(state, action, next_state, reward)
                #print("state: ", state)
                #print("action: ", action)
                #print("reward: ", reward)
                #print("next_state: ", next_state)

                # Update the network
                if len(replay_mem) > dqn_hyperparams["min_samples_for_training"]:
                    update_step(policy_net, target_net, replay_mem, dqn_hyperparams["gamma"], optimizer, loss_fn, dqn_hyperparams["batch_size"])
                ## Set the current state for the next iteration
                state = next_state
                episode_reward += reward.item()

            # Update the target network every "target_net_update_steps" episodes
            if episode_num % dqn_hyperparams["target_net_update_steps"] == 0:
                target_net.load_state_dict(policy_net.state_dict())
                if GENERALIZATION:
                    torch.save(target_net.state_dict(), 'NetsImages/DQN/target_net_generalization.pth')
                else:
                    torch.save(target_net.state_dict(), 'NetsImages/DQN/target_net_{}.pth'.format(dqn_hyperparams["cluster_letter_training_dqn"]))


def test_dqn(train_cluster_dict, lnn_hyperparams, utils_dict, dqn_hyperparams, k):
    """ Test the DQN, obtaining the optimal path """
    for cluster_letter in utils_dict["cluster_letters"]:
        if train_cluster_dict[cluster_letter]:
            env_w = 926
            env_h = 618
            err_cluster_image = pygame.image.load('NetsImages/super_imposed_img{}_iter_{}.jpg'.format(cluster_letter, k))
            localization_model = Net(lnn_hyperparams["Ni"], lnn_hyperparams["num_hid_nodes"], lnn_hyperparams["No"],
                                     activation_name=lnn_hyperparams["activation_name"],
                                     dropout_vec=lnn_hyperparams["dropout_vec"])
            print("load LNN NetsImages/RegModels/net_{}_pts_{}_iter_{}.pth".format(lnn_hyperparams["max_num_training_points_cluster"], cluster_letter, k))
            localization_model.load_state_dict(torch.load(
                'NetsImages/RegModels/net_{}_pts_{}_iter_{}.pth'.format(lnn_hyperparams["max_num_training_points_cluster"], cluster_letter, k)))  # PERCHE SERVE STA RETE NELL ENVIRONMENT??
            localization_model.eval()
            coord = np.loadtxt(
                'NetsImages/Datasets/RSSIDataframe.csv',
                delimiter=',', dtype=np.float32, skiprows=0)[:, 2:]
            scaler_path = 'NetsImages/scalers/scaler.pkl'   ### QUESTO C'É ANCHE DENTRO UTILS DICT
            if cluster_letter == "C":
                env = SnakeGameAI(cluster_letter, max_steps=80, err_cluster_image=err_cluster_image, coord=coord,
                                  localization_model=localization_model, scaler_path=scaler_path, starting_point=dqn_hyperparams["starting_point_dqn_test"][cluster_letter],
                                  random_test=utils_dict["random_action_test"])
            else:
                env = SnakeGameAI(cluster_letter, max_steps=40, err_cluster_image=err_cluster_image, coord=coord,
                                  localization_model=localization_model, scaler_path=scaler_path, starting_point=dqn_hyperparams["starting_point_dqn_test"][cluster_letter],
                                  random_test=utils_dict["random_action_test"])
            # Get the shapes of the state space (observation_space) and action space (action_space)
            state_space_dim = env.observation_space_dim
            action_space_dim = env.action_space_dim
            # Load the weights of the DQN obtained
            policy_net = DQN(state_space_dim, action_space_dim).to(device)
            target_net = DQN(state_space_dim, action_space_dim).to(device)

            #if cluster_letter=="C":
            policy_net.load_state_dict(torch.load('NetsImages/DQN/target_net_{}.pth'.format(dqn_hyperparams["cluster_letter_training_dqn"])))
            target_net.load_state_dict(torch.load('NetsImages/DQN/target_net_{}.pth'.format(dqn_hyperparams["cluster_letter_training_dqn"])))
            #else:
            #policy_net.load_state_dict(torch.load('NetsImages/DQN/target_net_generalization.pth'))
            #target_net.load_state_dict(torch.load('NetsImages/DQN/target_net_generalization.pth'))
            policy_net.eval()
            target_net.eval()

            run_test(env, utils_dict["random_action_test"], cluster_letter, policy_net, dqn_hyperparams)  ## NON SO SE METTENDOLO QUA DENTRO L ENVIRONMENT DOPO COMUNQUE LO VEDE MODIFICATO O NO (FACENDO RETURN_SNAKE)
            save_optimal_path(env, env_w, env_h, cluster_letter, coord, lnn_hyperparams, k)


def train_lnn_after_dqn(train_cluster_dict, lnn_hyperparams, utils_dict, k, sam_length, emb_size, bs, pl):
    '''
    This section re-trains the pre-trained neural network, by using the dataset collected (as path) by the agent of
    the reinforcement learning algorithm.
    '''
    print("train_lnn_after_dqn")
    for cluster_letter in utils_dict["cluster_letters"]:
        if train_cluster_dict[cluster_letter]:
            random.seed(0)
            if utils_dict["RL"]:
                print("load dataset NetsImages/Datasets/TrainLNN/train_dataset_cluster{}_iter_{}.csv".format(cluster_letter, k+1))
                train_RL_dataset = CsvDataset(
                    "NetsImages/Datasets/TrainLNN/train_dataset_cluster{}_iter_{}.csv".format(cluster_letter, k+1), utils_dict["scaler_path"],
                    transform=lnn_hyperparams["composed_transform"], sample=0, normalize=False, reduced=True, create_scaler=False)
                print("Retraining the model for letter {}...".format(cluster_letter))
                train_dataset_reinforcement = from_dataframe_to_dataset(train_RL_dataset)
                random.seed(0)
                random.shuffle(train_dataset_reinforcement)
                train_loader = DataLoader(dataset=train_dataset_reinforcement, batch_size=1, shuffle=False,  # PERCHE BATCH SIZE A 1 PER IL TRAIING? VA CAMBIATO!!
                                          num_workers=0)
                #with open('NetsImages/accuraciesComparison/numSamples_samLength{}_embSize{}_bs{}_pl{}_k{}_cluster_{}.txt'
                #                  .format(sam_length, emb_size, bs, pl, k, cluster_letter), 'w') as f:  ## SCEGLIERE NOME METODO AUTOMATICAMENTE!!!!
                #    print("len(train_RL_dataset): ", len(train_RL_dataset))
                #    f.write(json.dumps(len(train_RL_dataset)))
            elif utils_dict["GT"]:
                print("Train dataset ground truth for letter {}...".format(cluster_letter))
                train_dataset_gt = CsvDataset(
                    'NetsImages/Datasets/GroundTruth/train_dataset_g_t_cluster{}.csv'.format(cluster_letter), utils_dict["scaler_path"],
                    transform=lnn_hyperparams["composed_transform"], sample=0, normalize=False, reduced=True, create_scaler=False)
                train_dataset_gt_tmp = []
                for i in range(len(train_dataset_gt)):
                    train_dataset_gt_tmp.append(train_dataset_gt[i])
                random.seed(0)
                random.shuffle(train_dataset_gt_tmp)
                train_loader = DataLoader(dataset=train_dataset_gt_tmp, batch_size=1, shuffle=False, num_workers=0)

            model = train_lnn(train_loader, cluster_letter, lnn_hyperparams, after_dqn=True)
            print("save net NetsImages/RegModels/net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth".format(cluster_letter, sam_length, emb_size, bs, pl, k+1))
            model.save(file_name='net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth'.format(cluster_letter, sam_length, emb_size, bs, pl, k+1))


def test_lnn_after_dqn(test_dataset_dict, train_cluster_dict, lnn_hyperparams, utils_dict, k, sam_length, emb_size, bs, pl):
    '''
    Test the re-trained neural network over the dataset of the cluster identified by cluster_letter
    '''
    for cluster_letter in utils_dict["cluster_letters"]:
        if train_cluster_dict[cluster_letter]:
            print("Test accuracy for letter: ", cluster_letter)
            random.seed(0)
            test_dataset_cluster = test_dataset_dict[cluster_letter]
            test_loader = DataLoader(dataset=test_dataset_cluster, batch_size=1, shuffle=False, num_workers=0)
            model = Net(lnn_hyperparams["Ni"], lnn_hyperparams["num_hid_nodes"], lnn_hyperparams["No"],
                        lnn_hyperparams["activation_name"], lnn_hyperparams["dropout_vec"])
            model.load_state_dict(torch.load('NetsImages/RegModels/net_{}_samLength_{}_embSize{}_bs_{}_pl_{}_iter_{}.pth'.format(cluster_letter, sam_length, emb_size, bs, pl, k)))
            model.eval()
            # rmse is the Euclidean distance in pixels, not the real RMSE
            rmse_m_vec = []
            rmse_array = []
            rmse_not_accured = []
            with torch.no_grad():
                n_correct = 0
                n_correct_2 = 0
                n_correct_3 = 0
                for i, (RSSI, pos) in enumerate(test_loader):
                    RSSI = RSSI.to(device)
                    pos = pos.to(device)
                    outputs = model(RSSI)
                    rmse = torch.sqrt((outputs[0][0][0] - pos[0][0][0]) ** 2 + (outputs[0][0][1] - pos[0][0][1]) ** 2)
                    rmse = rmse.cpu()
                    rmse_m_vec.append(rmse.item() * 30 / 926)  # Convert from px to m
                    rmse_array.append(rmse)
                    n_correct += (rmse <= 15).sum().item()
                    n_correct_2 += (rmse <= 20).sum().item()
                    n_correct_3 += (rmse <= 25).sum().item()
                    if rmse > 15: rmse_not_accured.append(rmse)
                n_samples = len(rmse_array)
                print("n_samples: ", n_samples)
                acc = 100.0 * n_correct / n_samples
                acc2 = 100.0 * n_correct_2 / n_samples
                acc3 = 100.0 * n_correct_3 / n_samples
                print(f'Accuracy of cluster after training = {acc3} %')
                #print(f'Accuracy2 of cluster after training = {acc2} %')
                #print(f'Accuracy3 of cluster after training = {acc3} %')
                #print("RMSE: ", np.mean(rmse_array))
                #print("AED: ", np.mean(rmse_m_vec))
                #print("RMSE of not accured points: ", np.mean(rmse_not_accured))

                #with open('NetsImages/accuraciesComparison/cdf_IFSP_{}_iter_{}.txt'.format(cluster_letter, k), 'w') as f:   ## SCEGLIERE NOME METODO AUTOMATICAMENTE!!!!
                #    f.write(json.dumps(rmse_m_vec))
                with open('NetsImages/accuraciesComparison/acc_samLength{}_embSize{}_bs{}_pl{}_k{}_cluster_{}.txt'.format(sam_length, emb_size, bs, pl, k, cluster_letter), 'w') as f:   ## SCEGLIERE NOME METODO AUTOMATICAMENTE!!!!
                    f.write(json.dumps(acc3))
                #with open('NetsImages/accuraciesComparison/numSamples_samLength{}_embSize{}_bs{}_pl{}_k{}_cluster_{}.txt'.format(sam_length, emb_size, bs, pl, k, cluster_letter), 'w') as f:   ## SCEGLIERE NOME METODO AUTOMATICAMENTE!!!!
                #    f.write(json.dumps(acc3))
                #if acc3 >= 70:
                #    train_cluster_dict[cluster_letter] = False

