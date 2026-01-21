import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from model.graphcnn import GraphCNN
from model.mlp import MLPActor, MLPCritic


class ProjectSelector(nn.Module):
    """
    Stage 1: Select which project to schedule from
    """
    def __init__(self, project_feature_dim, hidden_dim, device):
        super(ProjectSelector, self).__init__()
        self.device = device

        # Project feature encoder
        self.project_encoder = nn.Sequential(
            nn.Linear(project_feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # Resource encoder
        self.resource_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )

        # Attention-based project scorer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, project_features, shared_resources, active_mask):
        """
        Args:
            project_features: [num_projects, project_feature_dim]
            shared_resources: [4]
            active_mask: [num_projects] - 1 for active, 0 for completed
        Returns:
            project_scores: [num_projects]
        """
        # Encode projects
        h_projects = self.project_encoder(project_features)  # [num_projects, hidden_dim]

        # Encode resources
        h_resources = self.resource_encoder(shared_resources)  # [hidden_dim//2]

        # Broadcast resources to all projects
        h_resources_expanded = h_resources.unsqueeze(0).expand(h_projects.size(0), -1)

        # Concatenate
        combined = torch.cat([h_projects, h_resources_expanded], dim=-1)

        # Compute scores
        scores = self.attention(combined).squeeze(-1)  # [num_projects]

        # Mask inactive projects (set to very negative value)
        scores = scores.masked_fill(active_mask == 0, -1e9)

        return scores


class TaskModeSelector(nn.Module):
    """
    Stage 2: Select task and mode within selected project
    """
    def __init__(self, num_gnn_layers, gnn_hidden_dim, mlp_hidden_dim,
                 node_feature_dim, num_mlp_layers, device):
        super(TaskModeSelector, self).__init__()
        self.device = device

        # GNN for task graph
        self.gnn = GraphCNN(
            num_layers=num_gnn_layers,
            num_mlp_layers=num_mlp_layers,
            input_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            learn_eps=False,
            neighbor_pooling_type='sum',
            device=device
        )

        # Mode encoder
        self.mode_encoder = nn.Sequential(
            nn.Linear(5, mlp_hidden_dim),  # [duration, r1, r2, r3, r4]
            nn.LeakyReLU(0.2),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2)
        )

        # Task-mode scorer
        # Input: [task_embedding, global_graph, resources, mode_features]
        input_dim = gnn_hidden_dim * 2 + 4 + mlp_hidden_dim // 2
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )

    def forward(self, adj_mat, node_features, shared_resources, feasible_task_modes):
        """
        Args:
            adj_mat: [num_nodes, num_nodes]
            node_features: [num_nodes, node_feature_dim]
            shared_resources: [4]
            feasible_task_modes: List of (task_idx, mode_id, mode_features)
        Returns:
            scores: [num_feasible_actions]
        """
        if len(feasible_task_modes) == 0:
            return torch.tensor([], device=self.device)

        # Extract graph features
        num_nodes = adj_mat.size(0)
        graph_pool = torch.full(
            size=[1, num_nodes],
            fill_value=1 / num_nodes,
            dtype=torch.float32,
            device=self.device
        )

        h_pooled, h_nodes = self.gnn(
            x=node_features,
            adj=adj_mat,
            graph_pool=graph_pool
        )  # h_pooled: [1, hidden], h_nodes: [num_nodes, hidden]

        # Prepare features for each feasible action
        scores = []

        for task_idx, mode_id, mode_features in feasible_task_modes:
            # Task embedding
            h_task = h_nodes[task_idx]  # [hidden_dim]

            # Mode embedding
            h_mode = self.mode_encoder(mode_features)  # [mlp_hidden//2]

            # Combine features
            combined = torch.cat([
                h_task,
                h_pooled.squeeze(0),
                shared_resources,
                h_mode
            ])  # [gnn_hidden*2 + 4 + mlp_hidden//2]

            # Compute score
            score = self.scorer(combined)
            scores.append(score)

        scores = torch.cat(scores)  # [num_feasible_actions]

        return scores


class HierarchicalAgent(nn.Module):
    """
    Hierarchical agent for MRCMPSP
    Two-stage decision:
    1. Project selector
    2. Task-mode selector
    """
    def __init__(self,
                 # Project selector params
                 project_feature_dim=8,
                 project_hidden_dim=64,
                 # Task-mode selector params
                 num_gnn_layers=3,
                 gnn_hidden_dim=64,
                 mlp_hidden_dim=32,
                 node_feature_dim=9,
                 num_mlp_layers=2,
                 # Critic params
                 critic_hidden_dim=64,
                 device='cuda:0'):

        super(HierarchicalAgent, self).__init__()
        self.device = device

        # Stage 1: Project selector
        self.project_selector = ProjectSelector(
            project_feature_dim, project_hidden_dim, device
        ).to(device)

        # Stage 2: Task-mode selector
        self.task_mode_selector = TaskModeSelector(
            num_gnn_layers, gnn_hidden_dim, mlp_hidden_dim,
            node_feature_dim, num_mlp_layers, device
        ).to(device)

        # Critic (global value estimator)
        # Input: average project features + shared resources
        critic_input_dim = project_feature_dim + 4
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, critic_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(critic_hidden_dim, critic_hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(critic_hidden_dim // 2, 1)
        ).to(device)

    def get_value(self, state):
        """
        Estimate value of current state
        Args:
            state: MRCMPSP state
        Returns:
            value: Scalar value estimate
        """
        project_features = torch.FloatTensor(state.project_features).to(self.device) / 10
        shared_resources = torch.FloatTensor(state.shared_resources).to(self.device) / 10

        # Global state: average over active projects + resources
        active_proj_features = project_features[state.active_projects]

        if len(active_proj_features) > 0:
            avg_proj_features = active_proj_features.mean(dim=0)
        else:
            avg_proj_features = torch.zeros(project_features.size(1), device=self.device)

        global_features = torch.cat([avg_proj_features, shared_resources])
        value = self.critic(global_features)

        return value

    def get_action_and_value(self, state, action=None, greedy=False):
        """
        Get action (hierarchical) and value estimate
        Args:
            state: MRCMPSP state
            action: If provided, compute log prob of this action
            greedy: If True, select argmax instead of sampling
        Returns:
            action: (project_idx, task_idx, mode_id)
            log_prob: Log probability of action
            entropy: Policy entropy
            value: Value estimate
        """
        project_features = torch.FloatTensor(state.project_features).to(self.device) / 10
        shared_resources = torch.FloatTensor(state.shared_resources).to(self.device) / 10

        # Create active mask
        active_mask = torch.zeros(len(project_features), device=self.device)
        active_mask[state.active_projects] = 1.0

        # Get value
        value = self.get_value(state)

        # If no feasible actions, return None
        if len(state.feasible_actions) == 0:
            return None, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), value

        # Stage 1: Select project
        project_scores = self.project_selector(
            project_features, shared_resources, active_mask
        )
        project_probs = F.softmax(project_scores, dim=0)
        project_dist = Categorical(project_probs)

        # Stage 2: For each project, get task-mode scores
        # Aggregate into single action space
        all_action_scores = []
        action_to_project = []

        for proj_idx in state.active_projects:
            # Get task-mode pairs for this project
            proj_actions = [
                (task_idx, mode_id)
                for (p_idx, task_idx, mode_id) in state.feasible_actions
                if p_idx == proj_idx
            ]

            if len(proj_actions) == 0:
                continue

            # Prepare features
            adj_mat = torch.FloatTensor(state.project_graphs[proj_idx]).to(self.device)
            node_features = torch.FloatTensor(state.project_node_features[proj_idx]).to(self.device) / 10

            # Get mode features for each action
            project = state.project_node_features[proj_idx]
            feasible_with_features = []

            for task_idx, mode_id in proj_actions:
                # Mode features: [duration, r1, r2, r3, r4]
                # We need to get this from the node
                # For now, use average from node features (approximation)
                mode_features = torch.FloatTensor([
                    project[task_idx][0],  # duration
                    project[task_idx][1],  # r1
                    project[task_idx][2],  # r2
                    project[task_idx][3],  # r3
                    project[task_idx][4]   # r4
                ]).to(self.device) / 10

                feasible_with_features.append((task_idx, mode_id, mode_features))

            # Get task-mode scores
            task_mode_scores = self.task_mode_selector(
                adj_mat, node_features, shared_resources, feasible_with_features
            )

            # Weight by project probability
            weighted_scores = task_mode_scores + torch.log(project_probs[proj_idx] + 1e-10)

            all_action_scores.append(weighted_scores)
            action_to_project.extend([(proj_idx, t, m) for t, m in proj_actions])

        # Combine all scores
        all_action_scores = torch.cat(all_action_scores)
        action_probs = F.softmax(all_action_scores, dim=0)
        action_dist = Categorical(action_probs)

        # Select action
        if action is None:
            if greedy:
                action_idx = torch.argmax(action_probs)
            else:
                action_idx = action_dist.sample()

            selected_action = action_to_project[action_idx.item()]
            log_prob = action_dist.log_prob(action_idx)
        else:
            # Find index of provided action
            action_idx = action_to_project.index(action)
            action_idx_tensor = torch.tensor(action_idx, device=self.device)
            log_prob = action_dist.log_prob(action_idx_tensor)
            selected_action = action

        entropy = action_dist.entropy()

        return selected_action, log_prob, entropy, value


    def get_action(self, state, greedy=False):
        """
        Get action without value (for inference)
        """
        action, _, _, _ = self.get_action_and_value(state, greedy=greedy)
        return action
