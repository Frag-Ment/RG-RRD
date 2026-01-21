import copy
import numpy as np
from mrcmpsp_simulator.multimode_node import MultiModeNode

class MultiModeJobdag(object):
    """
    Represents a single project with multi-mode tasks
    """
    def __init__(self, adj_mat, nodes_information, project_id=0, priority=1.0, deadline=float('inf')):
        """
        Args:
            adj_mat: Adjacency matrix (n x n)
            nodes_information: List of node data, each containing:
                              {'modes': [{'duration': d, 'resources': [r1,r2,r3,r4]}, ...]}
            project_id: Unique identifier for this project
            priority: Project priority weight
            deadline: Project deadline
        """
        self.project_id = project_id
        self.priority = priority
        self.deadline = deadline

        self.nodes = []
        self.original_adj_mat = copy.deepcopy(adj_mat)

        # Remove dummy start/end nodes if they exist (indices 0 and -1)
        temp = np.delete(self.original_adj_mat, [0, -1], axis=0)
        temp = np.delete(temp, [0, -1], axis=1)
        self.adj_mat = copy.deepcopy(temp)

        self.reversed_adj_mat = copy.deepcopy(self.adj_mat.T)
        self.adj_mat_line_shape = self.adj_mat.shape[0]

        # Status
        self.nodes_all_done = False
        self.num_nodes = self.adj_mat_line_shape
        self.done_nodes = []

        # Create multi-mode nodes
        for idx in range(self.num_nodes):
            node_info = nodes_information[idx]

            # Ensure modes are in correct format
            if 'modes' in node_info:
                modes = node_info['modes']
            else:
                # Fallback: create single mode from legacy format
                modes = [{
                    'duration': node_info[0],
                    'resources': [node_info[1], node_info[2], node_info[3], node_info[4]],
                    'mode_id': 0
                }]

            new_node = MultiModeNode(idx=idx, modes=modes)
            self.nodes.append(new_node)

        # Set parent and child relationships
        for node_idx, node in enumerate(self.nodes):
            node.child_nodes = [
                idx for idx, var in enumerate(self.adj_mat[node.idx])
                if var == 1 and idx != node_idx
            ]
            node.parent_nodes = [
                idx for idx, var in enumerate(self.reversed_adj_mat[node.idx])
                if var == 1 and idx != node_idx
            ]

    def get_runnable_nodes(self):
        """
        Get indices of nodes that can be executed (all predecessors completed/assigned)
        """
        runnable = []
        for node in self.nodes:
            if node.condition == 'not_start':
                # Check if all parents are assigned or done
                if set(node.parent_nodes) == set(node.unlocked_parent_nodes):
                    runnable.append(node.idx)

        return runnable

    def get_initial_runnable_nodes(self):
        """Get nodes with no predecessors (root nodes)"""
        return [node.idx for node in self.nodes if len(node.parent_nodes) == 0]

    def get_feature_matrix(self):
        """
        Get feature matrix for all nodes
        Returns:
            numpy array [num_nodes, feature_dim]
            Features: [avg_duration, avg_r1, avg_r2, avg_r3, avg_r4,
                      is_running, is_done, num_modes, selected_mode_id]
        """
        features = []

        for node in self.nodes:
            # Compute average across all modes
            all_modes = node.get_mode_features()
            avg_duration = np.mean(all_modes[:, 0])
            avg_resources = np.mean(all_modes[:, 1:5], axis=0)

            # Status encoding
            is_running = 1.0 if node.condition == 'running' else 0.0
            is_done = 1.0 if node.condition == 'done' else 0.0

            # Mode information
            num_modes = float(node.num_modes)
            selected_mode = float(node.selected_mode_id) if node.selected_mode_id is not None else -1.0

            feature = np.array([
                avg_duration, avg_resources[0], avg_resources[1],
                avg_resources[2], avg_resources[3],
                is_running, is_done, num_modes, selected_mode
            ], dtype=np.float32)

            features.append(feature)

        return np.array(features)

    def assign_task_with_mode(self, task_idx, mode_id):
        """
        Assign mode to a task
        Args:
            task_idx: Index of task to assign
            mode_id: Mode to use for execution
        """
        node = self.nodes[task_idx]
        node.select_mode(mode_id)
        node.condition = 'running'

        # Unlock children
        for child_idx in node.child_nodes:
            self.nodes[child_idx].unlocked_parent_nodes.append(task_idx)

    def complete_task(self, task_idx):
        """Mark a task as completed"""
        node = self.nodes[task_idx]
        node.condition = 'done'
        self.done_nodes.append(task_idx)

        # Update children's completed parent list
        for child_idx in node.child_nodes:
            self.nodes[child_idx].completed_parent_nodes.append(task_idx)

    def is_complete(self):
        """Check if all tasks in this project are done"""
        return len(self.done_nodes) == self.num_nodes

    def get_completion_ratio(self):
        """Get project completion percentage"""
        return len(self.done_nodes) / self.num_nodes

    def get_critical_path_length(self):
        """
        Estimate critical path length using minimum duration modes
        Returns rough estimate of remaining time
        """
        if self.is_complete():
            return 0

        # Simple topological traversal
        longest_path = [0] * self.num_nodes

        # Process nodes in topological order
        for node_idx in range(self.num_nodes):
            node = self.nodes[node_idx]

            if node.condition == 'done':
                continue

            # Get minimum duration across all modes
            min_duration = min(mode['duration'] for mode in node.modes)

            # Update based on predecessors
            for parent_idx in node.parent_nodes:
                longest_path[node_idx] = max(
                    longest_path[node_idx],
                    longest_path[parent_idx] + min_duration
                )

            if len(node.parent_nodes) == 0:
                longest_path[node_idx] = min_duration

        return max(longest_path)
