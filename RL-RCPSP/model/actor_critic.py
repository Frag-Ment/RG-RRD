import torch.nn as nn
from model.mlp import MLPActor
from model.mlp import MLPCritic
import torch.nn.functional as F
from model.graphcnn import GraphCNN
from torch.distributions.categorical import Categorical
import torch
import numpy as np


class Agent(nn.Module):
    def __init__(self,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        super(Agent, self).__init__()

        self.device = device
        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2 + 4, hidden_dim_actor, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim + 4, hidden_dim_critic, 1).to(device)

    #
    def get_value(self, state):

        adj = torch.FloatTensor(state[0]).to(self.device)
        # 除以10以加快训练速度
        x = torch.FloatTensor(state[1]).to(self.device) / 10
        resource = torch.FloatTensor(state[2]).to(self.device) / 10
        num_nodes = adj.shape[0]
        graph_pool = torch.full(size=[1, num_nodes], fill_value=1 / num_nodes, dtype=torch.float32, device=self.device)
        h_pooled, _ = self.feature_extract(x=x,
                                           adj=adj,
                                           graph_pool=graph_pool)

        # resource = resource / 10
        h_pooled_concat_reource = torch.cat((h_pooled, resource.unsqueeze(0)), dim=-1)

        value = self.critic(h_pooled_concat_reource)
        return value

    def get_action_and_value(self,
                state,
                action = None,
                greedy_test = False
                ):
        adj = torch.FloatTensor(state[0]).to(self.device)
        x = torch.FloatTensor(state[1]).to(self.device) / 10
        resource = torch.FloatTensor(state[2]).to(self.device) / 10
        runable_nodes_idx = torch.LongTensor(state[3]).to(self.device)
        num_nodes = adj.shape[0]

        graph_pool = torch.full(size=[1, num_nodes], fill_value=1 / num_nodes, dtype=torch.float32, device=self.device)

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 adj=adj,
                                                 graph_pool=graph_pool)



        runable_nodes_features = torch.index_select(h_nodes, 0, runable_nodes_idx)
        # runable_nodes_features = h_nodes[runable_nodes_idx]

        h_pooled_repeated = h_pooled.expand_as(runable_nodes_features)
        resource_repeated = resource.repeat(runable_nodes_idx.shape[0], 1)

        concateFea = torch.cat((runable_nodes_features, h_pooled_repeated, resource_repeated), dim=-1)

        candidate_scores = self.actor(concateFea)
        pi = F.softmax(candidate_scores, dim=0)

        h_pooled_concat_reource = torch.cat((h_pooled, resource.unsqueeze(0)), dim=-1)
        value = self.critic(h_pooled_concat_reource)

        pi = torch.squeeze(pi, dim=1)
        probs = Categorical(pi)
        if action is None and greedy_test is False:

            sample_idx = probs.sample()
            action = runable_nodes_idx[sample_idx]
            # 还是需要log_prob的
            log_prob = probs.log_prob(sample_idx)

        if action is None and greedy_test is True:
            _, sample_idx = pi.squeeze().max(0)
            action = runable_nodes_idx[sample_idx]
            log_prob = False

        # 需要找到和输入action对应的那个log_prob并输出

        else:
            action_idx = torch.where(runable_nodes_idx == action)
            log_prob = probs.log_prob(action_idx[0])

        return action, log_prob, probs.entropy(), value

