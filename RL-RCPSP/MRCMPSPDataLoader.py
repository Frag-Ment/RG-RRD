import numpy as np
import random
import pickle


class MRCMPSPDataLoader(object):
    """
    Data loader for MRCMPSP instances
    Generates synthetic multi-project, multi-mode RCPSP instances
    """

    def __init__(self, num_instances=100, num_projects=3, tasks_per_project=20,
                 num_modes=3, shared_resources=None, seed=42):
        """
        Args:
            num_instances: Number of problem instances to generate
            num_projects: Number of projects per instance
            tasks_per_project: Number of tasks per project (excluding dummy start/end)
            num_modes: Number of execution modes per task
            shared_resources: Global resource pool [r1, r2, r3, r4]
            seed: Random seed
        """
        self.num_instances = num_instances
        self.num_projects = num_projects
        self.tasks_per_project = tasks_per_project
        self.num_modes = num_modes
        self.seed = seed

        if shared_resources is None:
            # Default: scale resources with number of projects
            self.shared_resources = np.array([12 * num_projects, 12 * num_projects,
                                               12 * num_projects, 12 * num_projects])
        else:
            self.shared_resources = np.array(shared_resources)

        # Generate instances
        random.seed(seed)
        np.random.seed(seed)
        self.instances = self._generate_instances()

        # Iteration tracking
        self.current_idx = 0

    def _generate_instances(self):
        """Generate synthetic MRCMPSP instances"""
        instances = []

        for inst_id in range(self.num_instances):
            # Generate multiple projects
            projects = []

            for proj_id in range(self.num_projects):
                project_data = self._generate_single_project(proj_id)
                projects.append(project_data)

            instance = {
                'projects': projects,
                'shared_resources': self.shared_resources.copy(),
                'instance_id': inst_id
            }

            instances.append(instance)

        return instances

    def _generate_single_project(self, project_id):
        """
        Generate a single project with multi-mode tasks
        Returns:
            Dictionary with project data
        """
        n_tasks = self.tasks_per_project

        # Generate adjacency matrix (DAG structure)
        # Include dummy start (0) and end (n_tasks+1)
        adj_mat = self._generate_dag_adjacency(n_tasks + 2)

        # Generate multi-mode task information
        nodes_info = []

        for task_idx in range(1, n_tasks + 1):  # Skip dummy start
            modes = self._generate_task_modes(task_idx)
            nodes_info.append({'modes': modes})

        # Add dummy tasks
        dummy_mode = [{
            'duration': 0,
            'resources': [0, 0, 0, 0],
            'mode_id': 0
        }]

        nodes_info = [{'modes': dummy_mode}] + nodes_info + [{'modes': dummy_mode}]

        # Project metadata
        priority = np.random.uniform(0.5, 1.5)
        deadline = np.random.uniform(80, 150) * self.tasks_per_project / 20

        return {
            'adj_mat': adj_mat,
            'nodes_info': nodes_info,
            'project_id': project_id,
            'priority': priority,
            'deadline': deadline
        }

    def _generate_dag_adjacency(self, n_nodes):
        """
        Generate random DAG adjacency matrix
        Args:
            n_nodes: Total nodes (including dummy start and end)
        Returns:
            adjacency matrix [n_nodes, n_nodes]
        """
        adj = np.zeros((n_nodes, n_nodes), dtype=np.int32)

        # Ensure proper DAG structure
        # Connect start node (0) to some initial tasks
        num_initial = min(3, n_nodes - 2)
        for i in range(1, num_initial + 1):
            adj[0, i] = 1

        # Add random precedence relationships
        for i in range(1, n_nodes - 1):
            # Each task can have 0-3 successors
            num_successors = np.random.randint(0, min(4, n_nodes - i))

            if num_successors > 0:
                successors = np.random.choice(
                    range(i + 1, n_nodes),
                    size=min(num_successors, n_nodes - i - 1),
                    replace=False
                )
                for j in successors:
                    adj[i, j] = 1

        # Ensure all tasks connect to end node
        for i in range(1, n_nodes - 1):
            has_successor = np.any(adj[i, 1:-1])
            if not has_successor:
                adj[i, n_nodes - 1] = 1

        # Ensure end node has at least one predecessor
        if not np.any(adj[:-1, n_nodes - 1]):
            adj[n_nodes - 2, n_nodes - 1] = 1

        return adj

    def _generate_task_modes(self, task_idx):
        """
        Generate multiple execution modes for a task
        Time-resource trade-off: faster modes require more resources
        Returns:
            List of mode dictionaries
        """
        modes = []

        # Base duration
        base_duration = np.random.randint(3, 12)

        for mode_id in range(self.num_modes):
            # Mode 0: Normal (base duration, base resources)
            # Mode 1: Fast (less duration, more resources)
            # Mode 2: Slow (more duration, less resources)

            if mode_id == 0:
                # Normal mode
                duration = base_duration
                resource_factor = 1.0
            elif mode_id == 1:
                # Fast mode (30% faster, 50% more resources)
                duration = max(1, int(base_duration * 0.7))
                resource_factor = 1.5
            else:
                # Slow mode (30% slower, 30% fewer resources)
                duration = int(base_duration * 1.3)
                resource_factor = 0.7

            # Generate resources
            resources = [
                max(1, int(np.random.randint(1, 5) * resource_factor)),
                max(1, int(np.random.randint(1, 5) * resource_factor)),
                max(0, int(np.random.randint(0, 4) * resource_factor)),
                max(0, int(np.random.randint(0, 4) * resource_factor))
            ]

            mode = {
                'duration': duration,
                'resources': resources,
                'mode_id': mode_id
            }

            modes.append(mode)

        return modes

    def next_instance(self):
        """
        Get next instance
        Returns:
            instance dictionary
        """
        instance = self.instances[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.num_instances
        return instance

    def read_instance(self, idx):
        """
        Get specific instance by index
        """
        return self.instances[idx % self.num_instances]

    def reset(self):
        """Reset to first instance"""
        self.current_idx = 0

    def save(self, filepath):
        """Save generated instances to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.instances, f)
        print(f"Saved {len(self.instances)} instances to {filepath}")

    def load(self, filepath):
        """Load instances from file"""
        with open(filepath, 'rb') as f:
            self.instances = pickle.load(f)
        self.num_instances = len(self.instances)
        print(f"Loaded {self.num_instances} instances from {filepath}")

    def __len__(self):
        return self.num_instances

    def get_statistics(self):
        """Get statistics about the generated instances"""
        total_tasks = sum(
            len(proj['nodes_info']) - 2  # Exclude dummy nodes
            for inst in self.instances
            for proj in inst['projects']
        )

        avg_tasks_per_project = total_tasks / (self.num_instances * self.num_projects)

        return {
            'num_instances': self.num_instances,
            'num_projects_per_instance': self.num_projects,
            'avg_tasks_per_project': avg_tasks_per_project,
            'num_modes_per_task': self.num_modes,
            'shared_resources': self.shared_resources.tolist()
        }


# Convenience function to create dataloaders for training/validation/testing
def create_mrcmpsp_dataloaders(num_projects=3, tasks_per_project=20, num_modes=3):
    """
    Create train/val/test dataloaders
    """
    train_loader = MRCMPSPDataLoader(
        num_instances=200,
        num_projects=num_projects,
        tasks_per_project=tasks_per_project,
        num_modes=num_modes,
        seed=42
    )

    val_loader = MRCMPSPDataLoader(
        num_instances=30,
        num_projects=num_projects,
        tasks_per_project=tasks_per_project,
        num_modes=num_modes,
        seed=100
    )

    test_loader = MRCMPSPDataLoader(
        num_instances=30,
        num_projects=num_projects,
        tasks_per_project=tasks_per_project,
        num_modes=num_modes,
        seed=200
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loader
    print("Testing MRCMPSP Data Loader...")

    loader = MRCMPSPDataLoader(
        num_instances=10,
        num_projects=2,
        tasks_per_project=10,
        num_modes=3
    )

    print(f"\nDataset statistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test loading instance
    print("\nTesting instance loading...")
    instance = loader.next_instance()
    print(f"  Instance ID: {instance['instance_id']}")
    print(f"  Number of projects: {len(instance['projects'])}")
    print(f"  Shared resources: {instance['shared_resources']}")

    for proj in instance['projects']:
        print(f"\n  Project {proj['project_id']}:")
        print(f"    Tasks: {len(proj['nodes_info']) - 2}")
        print(f"    Priority: {proj['priority']:.2f}")
        print(f"    Deadline: {proj['deadline']:.1f}")
        print(f"    Adjacency matrix shape: {proj['adj_mat'].shape}")

    print("\n✓ Data loader test passed!")
