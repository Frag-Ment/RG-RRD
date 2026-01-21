import copy
import numpy as np
from rcpsp_simulator.task import Task

class MRCMPSPExecutor(object):
    """
    Executes MRCMPSP schedules with multi-project and multi-mode support
    """
    def __init__(self, project_manager):
        """
        Args:
            project_manager: MultiProjectManager instance
        """
        self.project_manager = project_manager

        # Time tracking
        self.walltime = 0

        # Task execution tracking
        self.running_tasks = []  # List of (project_id, task, start_time)
        self.done_tasks = []

        # Resource tracking
        self.resource_history = []

        # Action sequence for replay/analysis
        self.action_sequence = []

    def assign_task(self, project_idx, task_idx, mode_id):
        """
        Assign a task with specific mode to execution
        Args:
            project_idx: Which project
            task_idx: Which task within project
            mode_id: Which mode to use
        """
        project = self.project_manager.projects[project_idx]
        node = project.nodes[task_idx]

        # Select mode
        node.select_mode(mode_id)

        # Create task
        new_task = Task(task_idx, node, self.walltime)
        new_task.project_id = project_idx
        new_task.mode_id = mode_id

        self.running_tasks.append({
            'project_id': project_idx,
            'task': new_task,
            'start_time': self.walltime,
            'node': node
        })

        # Update project state
        project.assign_task_with_mode(task_idx, mode_id)

        # Consume resources
        self.project_manager.update_resources(-np.array(node.modes[mode_id]['resources']))

        # Record action
        self.action_sequence.append((project_idx, task_idx, mode_id, self.walltime))

    def advance_time(self):
        """
        Advance time by 1 unit and update task statuses
        Returns:
            List of completed tasks: [(project_idx, task_idx), ...]
        """
        self.walltime += 1
        completed_this_step = []

        wait_remove = []

        for task_info in self.running_tasks:
            task = task_info['task']
            node = task_info['node']
            project_idx = task_info['project_id']

            # Decrement remaining time
            task.have_run_time += 1
            node.rest_time -= 1

            # Check if completed
            if node.rest_time <= 0:
                wait_remove.append(task_info)
                completed_this_step.append((project_idx, task.node_idx))

        # Complete tasks
        for task_info in wait_remove:
            self._complete_task(task_info)

        return completed_this_step

    def _complete_task(self, task_info):
        """Internal method to complete a task"""
        task = task_info['task']
        node = task_info['node']
        project_idx = task_info['project_id']
        project = self.project_manager.projects[project_idx]

        # Remove from running
        self.running_tasks.remove(task_info)

        # Mark as done
        task.finish_time = self.walltime
        task.during_time = task.finish_time - task.start_time
        node.condition = 'done'

        # Add to done list
        self.done_tasks.append(task_info)

        # Return resources
        self.project_manager.update_resources(
            np.array(node.modes[node.selected_mode_id]['resources'])
        )

        # Update project completion tracking
        project.complete_task(task.node_idx)
        self.project_manager.completed_tasks += 1

        # Check if project is complete
        if project.is_complete():
            self.project_manager.mark_project_complete(project_idx)

    def can_assign_task(self, project_idx, task_idx, mode_id):
        """
        Check if a task can be assigned with given mode
        Returns:
            bool: True if assignable
        """
        project = self.project_manager.projects[project_idx]
        node = project.nodes[task_idx]
        mode = node.modes[mode_id]

        # Check resources
        if not all(mode['resources'][i] <= self.project_manager.shared_resources[i]
                   for i in range(4)):
            return False

        # Check precedence
        if set(node.parent_nodes) != set(node.completed_parent_nodes):
            return False

        return True

    def get_feasible_actions(self):
        """
        Get all feasible (project, task, mode) actions
        Returns:
            List of tuples: [(project_idx, task_idx, mode_id), ...]
        """
        feasible = []

        for proj_idx in self.project_manager.get_active_projects():
            project = self.project_manager.projects[proj_idx]
            runnable_tasks = project.get_runnable_nodes()

            for task_idx in runnable_tasks:
                node = project.nodes[task_idx]

                # Check each mode
                for mode_id in range(node.num_modes):
                    if self.can_assign_task(proj_idx, task_idx, mode_id):
                        feasible.append((proj_idx, task_idx, mode_id))

        return feasible

    def get_state(self):
        """
        Get current state representation
        Returns:
            Dictionary with state information
        """
        state = {
            'walltime': self.walltime,
            'shared_resources': self.project_manager.shared_resources.copy(),
            'project_features': self.project_manager.get_project_features(),
            'num_running_tasks': len(self.running_tasks),
            'completed_projects': len(self.project_manager.completed_projects),
            'active_projects': len(self.project_manager.get_active_projects())
        }

        # Add per-project information
        state['projects'] = []
        for project in self.project_manager.projects:
            proj_state = {
                'adj_mat': project.adj_mat,
                'feature_mat': project.get_feature_matrix(),
                'runnable_nodes': project.get_runnable_nodes(),
                'completion_ratio': project.get_completion_ratio(),
                'is_complete': project.is_complete()
            }
            state['projects'].append(proj_state)

        return state

    def is_all_complete(self):
        """Check if all projects are complete"""
        return self.project_manager.is_all_projects_complete()

    def get_makespan(self):
        """Get total elapsed time"""
        return self.walltime

    def get_statistics(self):
        """Get execution statistics"""
        return {
            'makespan': self.walltime,
            'total_tasks': len(self.done_tasks),
            'completed_projects': len(self.project_manager.completed_projects),
            'action_sequence_length': len(self.action_sequence),
            'avg_resource_utilization': self._compute_avg_utilization()
        }

    def _compute_avg_utilization(self):
        """Compute average resource utilization"""
        if len(self.resource_history) == 0:
            return 0.0

        total_util = sum(
            1.0 - (res / self.project_manager.original_shared_resources)
            for res in self.resource_history
        )
        return np.mean(total_util)
