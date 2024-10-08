import json
import cv2
from utils import *


def create_train_test_dataset_dict(ppnn_hyperparams, utils_dict):
    print("Creating datasets...")
    test_dataset_dict = {}
    for cluster_letter in utils_dict["cluster_letters"]:
        print("cluster letter ", cluster_letter)
        train_dataset, test_dataset = create_datasets(cluster_letter, ppnn_hyperparams, utils_dict)
        cluster_borders = return_cluster_border(cluster_letter)
        test_dataset_cluster = split_dataset_clusters(test_dataset, cluster_borders)
        test_dataset_dict[cluster_letter] = test_dataset_cluster
    return test_dataset_dict


def pretraining(ppnn_hyperparams, utils_dict):
    print("Pretraining...")
    for cluster_letter in utils_dict["cluster_letters"]:
        print("Training the model for cluster {}...".format(cluster_letter))
        train_dataset = CsvDataset(
            "NetsImages/Datasets/TrainPPNN/train_dataset_cluster{}_iter_0.csv".format(cluster_letter),
            utils_dict["scaler_path"],
            transform=ppnn_hyperparams["composed_transform"], sample=0, normalize=False, reduced=True,
            create_scaler=False)
        train_dataset_list = from_dataframe_to_dataset(train_dataset)
        random.seed(0)
        random.shuffle(train_dataset_list)
        train_loader = DataLoader(dataset=train_dataset_list, batch_size=ppnn_hyperparams["batch_size"], shuffle=False,
                                  num_workers=0)
        model = train_ppnn(train_loader, ppnn_hyperparams)
        model.save(file_name='net_{}_iter_{}.pth'.format(cluster_letter, 0))


def train_ppnn(train_loader, ppnn_hyperparams, after_drqn=False):
    torch.manual_seed(0)
    model = Net(ppnn_hyperparams["Ni"], ppnn_hyperparams["num_hid_nodes"], ppnn_hyperparams["No"],
                ppnn_hyperparams["activation_name"], ppnn_hyperparams["dropout_vec"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ppnn_hyperparams["learning_rate"])
    if after_drqn:
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
    return model


def test_ppnn(test_dataset_dict, train_cluster_dict, ppnn_hyperparams, utils_dict, k):
    print("Testing the PPNN...")
    for cluster_letter in utils_dict["cluster_letters"]:
        if train_cluster_dict[cluster_letter]:
            cluster_borders = return_cluster_border(cluster_letter)
            print("Testing the model for cluster {}...".format(cluster_letter))
            img = cv2.imread('NetsImages/FloorImages/Layout.png', cv2.IMREAD_COLOR)
            img = cv2.resize(img, (926, 618), interpolation=cv2.INTER_AREA)
            # Create empty images where the prediction error will be written
            empty_img = np.zeros((618, 926), np.uint8)
            empty_img_tmp = np.zeros((618, 926), np.longlong)
            test_loader = DataLoader(dataset=test_dataset_dict[cluster_letter], batch_size=1, shuffle=False, num_workers=0)
            torch.manual_seed(0)
            model = Net(ppnn_hyperparams["Ni"], ppnn_hyperparams["num_hid_nodes"], ppnn_hyperparams["No"],
                        ppnn_hyperparams["activation_name"], ppnn_hyperparams["dropout_vec"])
            print("load model: NetsImages/RegModels/net_{}_iter_{}.pth".format(cluster_letter, k))
            model.load_state_dict(
                torch.load('NetsImages/RegModels/net_{}_iter_{}.pth'.format(cluster_letter, k)))
            model.eval()
            mse_array = []
            mse_array_cluster = []
            with torch.no_grad():
                n_correct = 0
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
                        mse_array_cluster.append(mse.item())
                        empty_img_tmp[coord_y, coord_x] = mse.item()

                    n_correct += (mse < 15).sum().item()  ### QUESTO IN TEORIA NON SERVE, MEGLIO CANCELLARLO PERCHE SOTTO CE L HO SIMILE MA CON <25
            empty_img = create_cluster_heatmap(empty_img, empty_img_tmp, mse_array_cluster)
            # Blur the image with the MSE heatmap
            blur = cv2.GaussianBlur(empty_img, (3, 3), 1)
            # Then change the colormap
            heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
            # Superimpose the heatmap with the prediction error and the layout of the environment
            super_imposed_img = cv2.addWeighted(heatmap_img, 0.7, img, 0.3, 0)
            super_imposed_img = apply_heatmap_on_cluster(super_imposed_img, img, cluster_letter)
            cv2.imwrite('NetsImages/super_imposed_img{}_iter_{}.jpg'.format(cluster_letter, k), super_imposed_img)


def train_drqn_lstm(ppnn_hyperparams, drqn_hyperparams, k):
    # Size image environment
    err_cluster_image = pygame.image.load('NetsImages/super_imposed_img{}_iter_{}.jpg'.format(drqn_hyperparams["cluster_letter_training_drqn"], k))
    coord = np.loadtxt('NetsImages/Datasets/RSSIDataframe.csv', delimiter=',', dtype=np.float32, skiprows=0)[:, 2:]
    scaler_path = 'NetsImages/scalers/scaler.pkl'
    localization_model = Net(ppnn_hyperparams["Ni"], ppnn_hyperparams["num_hid_nodes"], ppnn_hyperparams["No"],
                             activation_name=ppnn_hyperparams["activation_name"],
                             dropout_vec=ppnn_hyperparams["dropout_vec"])
    localization_model.load_state_dict(torch.load(
            'NetsImages/RegModels/net_{}_iter_{}.pth'.format(drqn_hyperparams["cluster_letter_training_drqn"], k)))

    localization_model.eval()
    env = SamplingOperator(drqn_hyperparams["cluster_letter_training_drqn"], max_steps=80, err_cluster_image=err_cluster_image, coord=coord,
                      localization_model=localization_model, scaler_path=scaler_path)

    # Get the shapes of the state space (observation_space) and action space (action_space)
    state_space_dim = env.observation_space_dim
    action_space_dim = env.action_space_dim

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # Set the learning curve
    exp_decay = np.exp(-np.log(drqn_hyperparams["initial_value"]) / drqn_hyperparams["num_iterations"] * 1)
    exploration_profile = [drqn_hyperparams["initial_value"] * (exp_decay ** i) for i in range(drqn_hyperparams["num_iterations"])]

    replay_buffer_size = drqn_hyperparams["replay_memory_capacity"]
    replay_buffer = ReplayMemoryLstm(replay_buffer_size, drqn_hyperparams["sample_length"])
    policy_net_drqn = DRQN(action_space_dim, state_space_dim, drqn_hyperparams["embedding_size"])
    target_net_drqn = DRQN(action_space_dim, state_space_dim, drqn_hyperparams["embedding_size"])

    target_net_drqn.load_state_dict(policy_net_drqn.state_dict())
    optimizer = torch.optim.Adam(policy_net_drqn.parameters(), lr=drqn_hyperparams["lr"])
    gamma = 0.9

    for episode_num, tau in enumerate(exploration_profile):
        print("Episode number: ", episode_num)
        last_observation = env.reset()
        last_action = torch.tensor(0)
        hidden = None
        done = False

        while not done:
            # Choose the action following the policy
            action, hidden = select_action_lstm(policy_net_drqn, last_observation.float().view(1, 1, -1),
                                   F.one_hot(last_action, action_space_dim).view(1, 1, -1).float(),
                                   hidden, episode_num, drqn_hyperparams["policy_type"], tau)
            observation, reward, done = env.play_step(action)
            replay_buffer.write_tuple((last_action, last_observation[0].numpy(), action, reward, observation[0].numpy(), done))
            last_action = action
            last_observation = observation

            # Updating Policy Network
            if len(replay_buffer) > drqn_hyperparams["min_samples_for_training"]:
                last_actions, last_observations, actions, rewards, observations, dones = replay_buffer.sample(drqn_hyperparams["batch_size"])
                q_values, _ = policy_net_drqn.forward(last_observations, F.one_hot(last_actions, action_space_dim).float())
                q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
                predicted_q_values, _ = target_net_drqn.forward(observations, F.one_hot(actions, action_space_dim).float())
                target_values = rewards + (gamma * (1 - dones.float()) * torch.max(predicted_q_values, dim=-1)[0])
                optimizer.zero_grad()
                loss = torch.nn.MSELoss()(q_values, target_values.detach())
                loss.backward()
                optimizer.step()
            if done:
                break
        # Update the target network
        if episode_num % 10 == 0:
            target_net_drqn.load_state_dict(policy_net_drqn.state_dict())
        torch.save(target_net_drqn.state_dict(),
                   'NetsImages/DRQN/target_net_{}.pth'.format(drqn_hyperparams["cluster_letter_training_drqn"]))


def test_drqn_lstm(train_cluster_dict, ppnn_hyperparams, utils_dict, drqn_hyperparams, k):
    for cluster_letter in utils_dict["cluster_letters"]:
        if train_cluster_dict[cluster_letter]:
            env_w = 926
            env_h = 618
            err_cluster_image = pygame.image.load('NetsImages/super_imposed_img{}_iter_{}.jpg'.format(cluster_letter, k))
            localization_model = Net(ppnn_hyperparams["Ni"], ppnn_hyperparams["num_hid_nodes"], ppnn_hyperparams["No"],
                                     activation_name=ppnn_hyperparams["activation_name"],
                                     dropout_vec=ppnn_hyperparams["dropout_vec"])
            print("load PPNN NetsImages/RegModels/net_{}_iter_{}.pth".format(cluster_letter, k))
            localization_model.load_state_dict(torch.load('NetsImages/RegModels/net_{}_iter_{}.pth'.format(cluster_letter, k)))
            localization_model.eval()
            coord = np.loadtxt('NetsImages/Datasets/RSSIDataframe.csv', delimiter=',', dtype=np.float32, skiprows=0)[:, 2:]

            env = SamplingOperator(cluster_letter, max_steps=drqn_hyperparams["path_length"], err_cluster_image=err_cluster_image, coord=coord,
                              localization_model=localization_model, scaler_path=utils_dict["scaler_path"])
            # Get the shapes of the state space (observation_space) and action space (action_space)
            state_space_dim = env.observation_space_dim
            action_space_dim = env.action_space_dim
            # Load the weights of the DRQN obtained
            policy_net_drqn = DRQN(action_space_dim, state_space_dim, drqn_hyperparams["embedding_size"])
            target_net_drqn = DRQN(action_space_dim, state_space_dim, drqn_hyperparams["embedding_size"])

            policy_net_drqn.load_state_dict(torch.load('NetsImages/DRQN/target_net_{}.pth'.format(
                           drqn_hyperparams["cluster_letter_training_drqn"])))
            target_net_drqn.load_state_dict(torch.load('NetsImages/DRQN/target_net_{}.pth'.format(
                           drqn_hyperparams["cluster_letter_training_drqn"])))

            policy_net_drqn.eval()
            target_net_drqn.eval()

            run_test_lstm(env, cluster_letter, policy_net_drqn, k)
            save_optimal_path(env, env_w, env_h, cluster_letter, coord, k)


def train_ppnn_after_drqn(train_cluster_dict, ppnn_hyperparams, utils_dict, k):
    print("train_ppnn_after_drqn")
    for cluster_letter in utils_dict["cluster_letters"]:
        if train_cluster_dict[cluster_letter]:
            print("load dataset NetsImages/Datasets/TrainPPNN/train_dataset_cluster{}_iter_{}.csv".format(cluster_letter, k+1))
            train_RL_dataset = CsvDataset(
                "NetsImages/Datasets/TrainPPNN/train_dataset_cluster{}_iter_{}.csv".format(cluster_letter, k+1), utils_dict["scaler_path"],
                transform=ppnn_hyperparams["composed_transform"], sample=0, normalize=False, reduced=True, create_scaler=False)
            print("Retraining the model for letter {}...".format(cluster_letter))
            train_dataset_reinforcement = from_dataframe_to_dataset(train_RL_dataset)
            random.seed(0)
            random.shuffle(train_dataset_reinforcement)
            train_loader = DataLoader(dataset=train_dataset_reinforcement, batch_size=1, shuffle=False, num_workers=0)

            model = train_ppnn(train_loader, ppnn_hyperparams, after_drqn=True)
            print("save net NetsImages/RegModels/net_{}_iter_{}.pth".format(cluster_letter, k+1))
            model.save(file_name='net_{}_iter_{}.pth'.format(cluster_letter, k+1))


def test_ppnn_after_drqn(test_dataset_dict, train_cluster_dict, ppnn_hyperparams, utils_dict, drqn_hyperparams, k):
    for cluster_letter in utils_dict["cluster_letters"]:
        if train_cluster_dict[cluster_letter]:
            print("Test accuracy for letter: ", cluster_letter)
            random.seed(0)
            test_dataset_cluster = test_dataset_dict[cluster_letter]
            test_loader = DataLoader(dataset=test_dataset_cluster, batch_size=1, shuffle=False, num_workers=0)
            model = Net(ppnn_hyperparams["Ni"], ppnn_hyperparams["num_hid_nodes"], ppnn_hyperparams["No"],
                        ppnn_hyperparams["activation_name"], ppnn_hyperparams["dropout_vec"])
            model.load_state_dict(torch.load('NetsImages/RegModels/net_{}_iter_{}.pth'.format(cluster_letter, k, drqn_hyperparams["path_length"])))
            model.eval()

            ed_m_vec = []
            ed_array = []
            with torch.no_grad():
                n_correct = 0
                for i, (RSSI, pos) in enumerate(test_loader):
                    RSSI = RSSI.to(device)
                    pos = pos.to(device)
                    outputs = model(RSSI)
                    ed = torch.sqrt((outputs[0][0][0] - pos[0][0][0]) ** 2 + (outputs[0][0][1] - pos[0][0][1]) ** 2)
                    ed = ed.cpu()
                    ed_m_vec.append(ed.item() * 30 / 926)  # Convert from px to m
                    ed_array.append(ed)
                    n_correct += (ed <= 25).sum().item()
                n_samples = len(ed_array)
                print("n_samples: ", n_samples)
                acc = 100.0 * n_correct / n_samples
                print(f'Accuracy of cluster after training = {acc} %')
                with open('NetsImages/accuraciesComparison/acc_k{}_cluster_{}_pathLength{}.txt'.format(k, cluster_letter, drqn_hyperparams["path_length"]), 'w') as f:
                    f.write(json.dumps(acc))


