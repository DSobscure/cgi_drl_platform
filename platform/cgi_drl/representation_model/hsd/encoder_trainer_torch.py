import torch
import numpy as np
import importlib

class EncoderTrainer():
    def __init__(self, config):
        EncoderModel = getattr(importlib.import_module(config["model_define_path"]), "EncoderModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network_settings = config["network_settings"]
        self.hierarchy_count = config["hierarchy_count"]
        self.encoder_model = EncoderModel(network_settings).to(self.device)
        self.observation_prodiver = config["observation_prodiver"]
        self.compressed_observation_prodiver = config["compressed_observation_prodiver"]
        self.hierarchy_usage_provider = config["hierarchy_usage_provider"]
        self.action_prodiver = config["action_prodiver"]
        self.optimizer = torch.optim.Adam(self.encoder_model.parameters())
        self.distance_loss_fn = torch.nn.HuberLoss()
        self.vq_beta = config["vq_beta"]
        self.vq_mean_beta = config["vq_mean_beta"]

    def get_compressed_observation_and_policy(self, observations, hierarchy_usages):
        self.encoder_model.eval()
        observation_inputs = self.observation_prodiver(observations, self.device)
        hierarchy_usages = self.hierarchy_usage_provider(hierarchy_usages, self.device)

        _, _, reconstructed_observation, policy = self.encoder_model(observation_inputs, hierarchy_usages)
        actions = torch.distributions.Categorical(logits=policy).sample()
        for key in reconstructed_observation:
            reconstructed_observation[key] = reconstructed_observation[key].cpu().detach().numpy()
        return reconstructed_observation, policy.cpu().detach().numpy(), actions.cpu().detach().numpy()

    def get_discrite_states(self, observations):
        self.encoder_model.eval()
        observation_inputs = self.observation_prodiver(observations, self.device)
        return self.encoder_model.state_forward(observation_inputs)

    def update(self, observations, compressed_observations, actions, batch_size, learning_rate):
        self.encoder_model.train()
        observation_inputs = self.observation_prodiver(observations, self.device)
        compressed_observation_inputs = self.compressed_observation_prodiver(compressed_observations, self.device)
        hierarchy_usages = self.hierarchy_usage_provider(np.random.uniform(0, 1, [batch_size, self.hierarchy_count - 1]), self.device)
        actions = self.action_prodiver(actions, self.device)

        for i in range(len(self.optimizer.param_groups)): 
            self.optimizer.param_groups[i]['lr'] = learning_rate

        self.optimizer.zero_grad()

        vq_embedding_means = []
        for i_hierarchy in range(self.hierarchy_count):
            vq_embedding_mean = torch.mean(self.encoder_model.vq_embeddings[i_hierarchy].weight, dim=0) 
            vq_embedding_mean = vq_embedding_mean.unsqueeze(0).expand(batch_size, -1)
            vq_embedding_means.append(vq_embedding_mean)

        z_e_q_pairs, discrete_states, reconstructed_observation, policy = self.encoder_model(observation_inputs, hierarchy_usages)
        
        policy_loss = torch.nn.functional.cross_entropy(policy, actions)
        compression_loss = 0
        for key in reconstructed_observation:
            compression_loss += self.distance_loss_fn(reconstructed_observation[key], compressed_observation_inputs[key])
        compression_loss /= len(reconstructed_observation)

        vq_loss = 0
        commit_loss = 0
        vq_mean_loss = 0
        for i_hierarchy in range(self.hierarchy_count):
            vq_loss += self.distance_loss_fn(z_e_q_pairs[i_hierarchy][0].detach(), z_e_q_pairs[i_hierarchy][1])
            commit_loss += self.distance_loss_fn(z_e_q_pairs[i_hierarchy][0], z_e_q_pairs[i_hierarchy][1].detach())
            extend_vq_embedding_mean = vq_embedding_means[i_hierarchy]
            if i_hierarchy == 0:
                latent_shape = z_e_q_pairs[i_hierarchy][0].size()
                vq_mean_loss += self.distance_loss_fn(vq_embedding_means[i_hierarchy].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, latent_shape[-2], latent_shape[-1]), z_e_q_pairs[i_hierarchy][0].detach())
            else:
                latent_shape = z_e_q_pairs[i_hierarchy][0].size()
                vq_mean_loss += self.distance_loss_fn(vq_embedding_means[i_hierarchy].unsqueeze(-1).expand(-1, -1, latent_shape[-1]), z_e_q_pairs[i_hierarchy][0].detach())

        loss = policy_loss + compression_loss + vq_loss + self.vq_beta * commit_loss + self.vq_mean_beta * vq_mean_loss
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), compression_loss.item(), vq_loss.item(), vq_mean_loss.item()

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        torch.save(self.encoder_model.state_dict(), f"{path}/model_{time_step}.ckpt")
        print(f"Model saved in file: {path}/model_{time_step}.ckpt")

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        self.encoder_model.load_state_dict(torch.load(path))
        print("Model restored.")