import torch.nn as nn
from model.mlp import MLPActor
from model.mlp import MLPCritic
import torch.nn.functional as F
from model.graphcnn import GraphCNN
from torch.distributions.categorical import Categorical
import torch


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
    def get_value(self, state, graph_pool):
        adj = torch.FloatTensor(state[0]).to(self.device)
        x = torch.FloatTensor(state[1]).to(self.device) / 10
        resource = torch.FloatTensor(state[2]).to(self.device) / 10
        h_pooled, _ = self.feature_extract(x=x,
                                           adj=adj,
                                           graph_pool=graph_pool)

        h_pooled_concat_reource = torch.cat((h_pooled, resource.unsqueeze(0)), dim=-1)

        value = self.critic(h_pooled_concat_reource)
        return value

    def get_pi_and_value(self,
                adj_mat,
                fea_mat,
                resource,
                runable_nodes_idx,
                graph_pool,
                ):
        adj = torch.FloatTensor(adj_mat).to(self.device)
        x = torch.FloatTensor(fea_mat).to(self.device) / 10
        resource = torch.FloatTensor(resource).to(self.device) / 10
        runable_nodes_idx = torch.LongTensor(runable_nodes_idx).to(self.device)

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 adj=adj,
                                                 graph_pool=graph_pool)


        # 不知道这样直接用索引取出tensor是否正确？
        # 打算改成mask的方式
        runable_nodes_features = torch.index_select(h_nodes, 0, runable_nodes_idx)
        # runable_nodes_features = h_nodes[runable_nodes_idx]

        h_pooled_repeated = h_pooled.expand_as(runable_nodes_features)
        resource_repeated = resource.repeat(runable_nodes_idx.shape[0], 1)

        concateFea = torch.cat((runable_nodes_features, h_pooled_repeated, resource_repeated), dim=-1)

        candidate_scores = self.actor(concateFea)
        pi = F.softmax(candidate_scores, dim=0)

        h_pooled_concat_reource = torch.cat((h_pooled, resource.unsqueeze(0)), dim=-1)
        value = self.critic(h_pooled_concat_reource)
        value = torch.squeeze(value)

        pi = torch.squeeze(pi, dim=1)
        # probs = Categorical(pi)

        return pi, value

        # if action is None and test is None:
        #
        #     sample_idx = probs.sample()
        #     action = runable_nodes_idx[sample_idx]
        #     # 还是需要log_prob的
        #     log_prob = probs.log_prob(sample_idx)
        #
        # if action is None and test is True:
        #     _, sample_idx = pi.squeeze().max(0)
        #     action = runable_nodes_idx[sample_idx]
        #     log_prob = False
        #
        # # 从这里开始写，昨天头晕的厉害，写不下去了
        # # 需要找到和输入action对应的那个log_prob并输出
        #
        # else:
        #     action_idx = torch.where(runable_nodes_idx == action)
        #     log_prob = probs.log_prob(action_idx[0])



