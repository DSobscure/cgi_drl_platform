import torch
import numpy as np
import importlib

class EncoderTrainer():
    def __init__(self, config):
        EncoderModel = getattr(importlib.import_module(config["model_define_path"]), "EncoderModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network_settings = config["network_settings"]
        self.hierarchy_count = config["hierarchy_count"]
        self.encoder_model = EncoderModel(network_settings)
        self.encoder_model = torch.nn.DataParallel(self.encoder_model)
        self.encoder_model.to(self.device)

        params_to_optimize = [
            {'params': [param for name, param in self.encoder_model.named_parameters() if 'vq_embedding' in name], 'weight_decay': 0},
            {'params': [param for name, param in self.encoder_model.named_parameters() if 'vq_embedding' not in name]}
        ]
        self.optimizer = torch.optim.Adam(params_to_optimize, weight_decay=0.0001)
        self.distance_loss_fn = torch.nn.MSELoss()
        self.vq_beta = config["vq_beta"]
        # self.vq_mean_beta = config["vq_mean_beta"]
        self.board_augmentation = config.get("board_augmentation", 0)

    def _parse_state_code(self, k, hierarchy_index):      
        return "{}|{}".format(hierarchy_index, ",".join(k))

    def get_discrite_states(self, observations):
        self.encoder_model.eval()
        batch_size = len(observations)
        codes = []
        for i_hierarchy in range(self.hierarchy_count):
            codes.append([])

        hierarchy_usages = np.random.uniform(0, 1, [batch_size, self.hierarchy_count-1])
        hierarchy_usages = torch.tensor(hierarchy_usages, dtype=torch.float32).to(self.device)
        observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
            
        states = self.encoder_model(observations, hierarchy_usages)["k"]

        for i_hierarchy in range(self.hierarchy_count):
            for k in states[i_hierarchy].cpu().detach().numpy().astype(str):
                codes[i_hierarchy].append(self._parse_state_code(k, i_hierarchy))
        return codes

    def update(self, observations, actions, values, loss_scale, learning_rate):
        batch_size = len(loss_scale.to('cpu').detach().numpy())

        self.encoder_model.train()
        hierarchy_usages = np.random.uniform(0, 1, [batch_size, self.hierarchy_count-1])
        hierarchy_usages = torch.tensor(hierarchy_usages, dtype=torch.float32).to(self.device)

        for i in range(len(self.optimizer.param_groups)): 
            self.optimizer.param_groups[i]['lr'] = learning_rate

        self.optimizer.zero_grad()

        network_output = self.encoder_model(observations, hierarchy_usages)

        predictions, targets = torch.argmax(network_output["policy"], dim=-1).to('cpu').detach().numpy(), torch.argmax(actions[:, 0], dim=-1).to('cpu').detach().numpy()
        correct_count = (predictions == targets).sum()

        policy_loss = torch.nn.functional.cross_entropy(network_output["policy"], actions[:, 0])
        value_loss = (torch.nn.functional.mse_loss(network_output["value"], values[:, 0], reduction='none') * loss_scale).mean()
        # value_loss = self.distance_loss_fn(network_output["reconstruction"], observations)

        vq_loss = 0
        commit_loss = 0
        # vq_mean_loss = 0
        hierarchy_count = 0
        for i_hierarchy in range(self.hierarchy_count):
            hierarchy_count += 1
            vq_loss += self.distance_loss_fn(network_output["z_e"][i_hierarchy].detach(),network_output["z_q"][i_hierarchy])
            commit_loss += self.distance_loss_fn(network_output["z_e"][i_hierarchy], network_output["z_q"][i_hierarchy].detach())
            # vq_mean_loss += self.distance_loss_fn(network_output["embedding_mean"][i_hierarchy], network_output["z_e"][i_hierarchy].detach())

        vq_loss /= hierarchy_count
        commit_loss /= hierarchy_count
        # vq_mean_loss /= hierarchy_count
      
        loss = self.policy_coefficient * policy_loss + self.value_coefficient * value_loss + vq_loss + self.vq_beta * commit_loss # + self.vq_mean_beta * vq_mean_loss
        loss.backward()
        self.optimizer.step()

        used_state_count = np.zeros(self.hierarchy_count)
        for i_hierarchy in range(self.hierarchy_count):
            states = network_output["k"][i_hierarchy].cpu().detach().numpy().astype(str)
            state_set = set()
            for k in states:
                state_set.add(self._parse_state_code(k, i_hierarchy))
            used_state_count[i_hierarchy] = len(state_set)
        
        return policy_loss.item(), value_loss.item(), vq_loss.item(), 0, correct_count, batch_size, used_state_count

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        torch.save(self.encoder_model.state_dict(), f"{path}/model_{time_step}.ckpt")
        print(f"Model saved in file: {path}/model_{time_step}.ckpt")

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        print("Model starts loading.")
        self.encoder_model.load_state_dict(torch.load(path))
        print("Model restored.")