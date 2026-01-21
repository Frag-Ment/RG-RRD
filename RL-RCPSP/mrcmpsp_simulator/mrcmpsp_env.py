import copy
import numpy as np
from collections import namedtuple
from mrcmpsp_simulator.multiproject_manager import MultiProjectManager
from mrcmpsp_simulator.mrcmpsp_executor import MRCMPSPExecutor

class MRCMPSPEnvironment(object):
    """
    Environment for Multi-Resource-Constrained Multi-Project Scheduling Problem
    with hierarchical action space: (project_id, task_id, mode_id)
    """

    def __init__(self):
        self.executor = None
        self.project_manager = None
        self.last_decision_time = 0

        # State tuple definition
        self.StateType = namedtuple('MRCMPSPState', [
            'project_features',      # [num_projects, project_feature_dim]
            'project_graphs',        # List of adjacency matrices
            'project_node_features', # List of feature matrices
            'shared_resources',      # [4] global resource vector
            'feasible_actions',      # List of (proj, task, mode) tuples
            'active_projects'        # List of active project indices
        ])

    def reset(self, project_data_list, shared_resources):
        """
        Reset environment with new problem instance
        Args:
            project_data_list: List of project specifications
            shared_resources: Global resource pool [r1, r2, r3, r4]
        Returns:
            Initial state
        """
        # Initialize project manager
        self.project_manager = MultiProjectManager(project_data_list, shared_resources)

        # Initialize executor
        self.executor = MRCMPSPExecutor(self.project_manager)

        # Reset time
        self.last_decision_time = 0

        # Return initial state
        return self._get_state()

    def step(self, action):
        """
        Execute one scheduling action
        Args:
            action: Tuple (project_idx, task_idx, mode_id)
        Returns:
            (next_state, reward, done)
        """
        project_idx, task_idx, mode_id = action

        # Validate action
        assert (project_idx, task_idx, mode_id) in self.executor.get_feasible_actions(), \
            f"Invalid action: {action}"

        # Wait until task can be started
        while not self.executor.can_assign_task(project_idx, task_idx, mode_id):
            self.executor.advance_time()

        # Assign task
        self.executor.assign_task(project_idx, task_idx, mode_id)

        # Calculate reward (negative time elapsed since last decision)
        reward = -(self.executor.walltime - self.last_decision_time)
        self.last_decision_time = self.executor.walltime

        # Check if all projects complete
        done = self.executor.is_all_complete()

        if done:
            # Final step: let all running tasks finish
            while len(self.executor.running_tasks) > 0:
                self.executor.advance_time()

            # Add final time penalty
            reward -= (self.executor.walltime - self.last_decision_time)

        # Get next state
        next_state = self._get_state()

        return next_state, reward, done

    def _get_state(self):
        """
        Construct state representation
        """
        exec_state = self.executor.get_state()

        # Get feasible actions
        feasible_actions = self.executor.get_feasible_actions()

        # Build project graphs and features
        project_graphs = []
        project_node_features = []

        for proj_state in exec_state['projects']:
            project_graphs.append(proj_state['adj_mat'])
            project_node_features.append(proj_state['feature_mat'])

        state = self.StateType(
            project_features=exec_state['project_features'],
            project_graphs=project_graphs,
            project_node_features=project_node_features,
            shared_resources=exec_state['shared_resources'],
            feasible_actions=feasible_actions,
            active_projects=self.project_manager.get_active_projects()
        )

        return state

    def get_info(self):
        """Get additional information about current state"""
        return {
            'walltime': self.executor.walltime,
            'completed_tasks': self.executor.project_manager.completed_tasks,
            'total_tasks': self.executor.project_manager.total_tasks,
            'num_active_projects': len(self.project_manager.get_active_projects()),
            'resource_utilization': 1.0 - (
                self.project_manager.shared_resources /
                self.project_manager.original_shared_resources
            )
        }


class MRCMPSPHierarchicalEnvironment(MRCMPSPEnvironment):
    """
    MRCMPSP Environment with hierarchical action space
    Splits action selection into two stages:
    1. Select project
    2. Select task + mode
    """

    def __init__(self):
        super().__init__()
        self.selected_project = None

    def select_project(self, project_idx):
        """
        Stage 1: Select which project to schedule from
        Args:
            project_idx: Index of project to select
        """
        assert project_idx in self.project_manager.get_active_projects(), \
            f"Project {project_idx} is not active"

        self.selected_project = project_idx

    def step_with_project(self, task_idx, mode_id):
        """
        Stage 2: Select task and mode within selected project
        Args:
            task_idx: Task to schedule
            mode_id: Mode to use
        Returns:
            (next_state, reward, done)
        """
        assert self.selected_project is not None, "Must select project first"

        action = (self.selected_project, task_idx, mode_id)
        return self.step(action)

    def get_project_state(self, project_idx):
        """
        Get state for a specific project (for hierarchical decision)
        """
        project = self.project_manager.projects[project_idx]

        return {
            'adj_mat': project.adj_mat,
            'feature_mat': project.get_feature_matrix(),
            'runnable_nodes': project.get_runnable_nodes(),
            'shared_resources': self.project_manager.shared_resources.copy()
        }

    def get_feasible_task_modes(self, project_idx):
        """
        Get feasible (task, mode) pairs for a given project
        Returns:
            List of (task_idx, mode_id) tuples
        """
        feasible = []
        project = self.project_manager.projects[project_idx]
        runnable_tasks = project.get_runnable_nodes()

        for task_idx in runnable_tasks:
            node = project.nodes[task_idx]

            for mode_id in range(node.num_modes):
                if self.executor.can_assign_task(project_idx, task_idx, mode_id):
                    feasible.append((task_idx, mode_id))

        return feasible
