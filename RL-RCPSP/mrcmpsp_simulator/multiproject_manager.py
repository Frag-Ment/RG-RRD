import copy
import numpy as np
from mrcmpsp_simulator.multimode_jobdag import MultiModeJobdag

class MultiProjectManager(object):
    """
    Manages multiple projects with shared resources in MRCMPSP
    """
    def __init__(self, project_data_list, shared_resources):
        """
        Args:
            project_data_list: List of project data, each containing:
                               {'adj_mat': ..., 'nodes_info': ..., 'project_id': ...,
                                'priority': ..., 'deadline': ...}
            shared_resources: Global resource pool [r1, r2, r3, r4]
        """
        self.num_projects = len(project_data_list)
        self.projects = []
        self.project_ids = []

        # Initialize projects
        for proj_data in project_data_list:
            project = MultiModeJobdag(
                adj_mat=proj_data['adj_mat'],
                nodes_information=proj_data['nodes_info'],
                project_id=proj_data.get('project_id', len(self.projects)),
                priority=proj_data.get('priority', 1.0),
                deadline=proj_data.get('deadline', float('inf'))
            )
            self.projects.append(project)
            self.project_ids.append(project.project_id)

        # Shared resource pool
        self.original_shared_resources = np.array(shared_resources, dtype=np.float32)
        self.shared_resources = np.array(shared_resources, dtype=np.float32)

        # Tracking
        self.completed_projects = []
        self.active_projects = list(range(self.num_projects))

        # Statistics
        self.total_tasks = sum(proj.num_nodes for proj in self.projects)
        self.completed_tasks = 0

    def get_active_projects(self):
        """Get list of projects that are not yet completed"""
        return [i for i in range(self.num_projects) if i not in self.completed_projects]

    def get_project_features(self):
        """
        Get feature representation for each project
        Returns:
            List of feature vectors for each project
        """
        features = []
        for i, project in enumerate(self.projects):
            if i in self.completed_projects:
                feature = np.zeros(8, dtype=np.float32)
            else:
                completion_ratio = len(project.done_nodes) / project.num_nodes
                remaining_tasks = project.num_nodes - len(project.done_nodes)
                avg_resource_demand = self._estimate_resource_demand(project)

                feature = np.array([
                    1.0,  # Active indicator
                    completion_ratio,
                    remaining_tasks,
                    project.priority,
                    project.deadline,
                    avg_resource_demand[0],
                    avg_resource_demand[1],
                    avg_resource_demand[2] + avg_resource_demand[3]
                ], dtype=np.float32)

            features.append(feature)

        return np.array(features)

    def _estimate_resource_demand(self, project):
        """Estimate average resource demand for remaining tasks"""
        uncompleted_nodes = [node for node in project.nodes if node.condition != 'done']

        if len(uncompleted_nodes) == 0:
            return np.zeros(4)

        total_demand = np.zeros(4)
        for node in uncompleted_nodes:
            # Use minimum resource mode as estimate
            min_resources = node.modes[0]['resources']
            for mode in node.modes:
                if sum(mode['resources']) < sum(min_resources):
                    min_resources = mode['resources']
            total_demand += np.array(min_resources)

        return total_demand / len(uncompleted_nodes)

    def get_runnable_tasks_all_projects(self):
        """
        Get all runnable tasks across all active projects
        Returns:
            List of tuples: [(project_idx, task_idx, runnable_modes), ...]
        """
        all_runnable = []

        for proj_idx in self.get_active_projects():
            project = self.projects[proj_idx]
            runnable_tasks = project.get_runnable_nodes()

            for task_idx in runnable_tasks:
                node = project.nodes[task_idx]
                # Check which modes are feasible given current resources
                feasible_modes = self._get_feasible_modes(node)

                if len(feasible_modes) > 0:
                    all_runnable.append((proj_idx, task_idx, feasible_modes))

        return all_runnable

    def _get_feasible_modes(self, node):
        """Get list of mode IDs that are feasible given current resources"""
        feasible = []
        for mode_id, mode in enumerate(node.modes):
            if all(mode['resources'][i] <= self.shared_resources[i] for i in range(4)):
                feasible.append(mode_id)
        return feasible

    def is_all_projects_complete(self):
        """Check if all projects are completed"""
        return len(self.completed_projects) == self.num_projects

    def mark_project_complete(self, project_idx):
        """Mark a project as complete"""
        if project_idx not in self.completed_projects:
            self.completed_projects.append(project_idx)

    def update_resources(self, delta_resources):
        """Update shared resource availability"""
        self.shared_resources += np.array(delta_resources)

    def get_state_summary(self):
        """Get summary statistics of current state"""
        return {
            'active_projects': len(self.get_active_projects()),
            'completed_projects': len(self.completed_projects),
            'completed_tasks': self.completed_tasks,
            'total_tasks': self.total_tasks,
            'resource_utilization': 1.0 - (self.shared_resources / self.original_shared_resources)
        }
