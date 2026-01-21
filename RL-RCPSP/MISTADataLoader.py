"""
MISTA Benchmark Data Loader for MRCMPSP
Supports:
- Multi-mode RCPSP instances (PSPLib .mm format)
- Multi-project instances (A, B, X sets)
"""
import os
import numpy as np
import re
from pathlib import Path


class MISTADataLoader:
    """
    Load MISTA benchmark instances for MRCMPSP
    """

    def __init__(self, data_root='Mista'):
        """
        Args:
            data_root: Path to Mista directory
        """
        self.data_root = Path(data_root)
        self.single_project_cache = {}  # Cache for parsed single-project files

    def parse_mm_file(self, filepath):
        """
        Parse PSPLib .mm format file (multi-mode RCPSP)
        Returns:
            dict with 'adj_mat', 'nodes_info', 'resources'
        """
        filepath = Path(filepath)

        # Check cache
        if str(filepath) in self.single_project_cache:
            return self.single_project_cache[str(filepath)]

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse header
        num_jobs = None
        num_resources_renewable = 0
        num_resources_nonrenewable = 0

        for line in lines:
            if 'jobs (incl. supersource/sink' in line:
                num_jobs = int(line.split(':')[1].strip())
            elif '- renewable' in line:
                num_resources_renewable = int(line.split(':')[1].split()[0])
            elif '- nonrenewable' in line:
                num_resources_nonrenewable = int(line.split(':')[1].split()[0])

        # Parse precedence relations
        prec_start = None
        for i, line in enumerate(lines):
            if 'PRECEDENCE RELATIONS:' in line:
                prec_start = i + 2  # Skip header line
                break

        # Build adjacency matrix
        adj_mat = np.zeros((num_jobs, num_jobs), dtype=np.int32)
        job_modes = {}  # job_id -> num_modes

        i = prec_start
        while i < len(lines) and not lines[i].startswith('***'):
            parts = lines[i].strip().split()
            if len(parts) >= 3:
                job_id = int(parts[0])
                num_modes = int(parts[1])
                num_successors = int(parts[2])

                job_modes[job_id] = num_modes

                if num_successors > 0:
                    successors = [int(x) for x in parts[3:3+num_successors]]
                    for succ in successors:
                        adj_mat[job_id - 1, succ - 1] = 1

            i += 1

        # Parse requests/durations (modes)
        req_start = None
        for i, line in enumerate(lines):
            if 'REQUESTS/DURATIONS:' in line:
                req_start = i + 2  # Skip header
                break

        # Extract mode information for each job
        nodes_info = []
        i = req_start
        current_job = None
        current_modes = []

        while i < len(lines) and not lines[i].startswith('***'):
            parts = lines[i].strip().split()

            if len(parts) >= 5:  # Valid mode line
                job_or_mode = int(parts[0])

                if job_or_mode <= num_jobs:
                    # This is a job ID (first line for that job)
                    if current_job is not None and len(current_modes) > 0:
                        nodes_info.append({'modes': current_modes})

                    current_job = job_or_mode
                    current_modes = []

                    # Parse this mode
                    mode_id = int(parts[1])
                    duration = int(parts[2])
                    resources = [int(parts[j]) for j in range(3, 3 + num_resources_renewable + num_resources_nonrenewable)]

                    # Separate renewable and non-renewable (we'll focus on renewable for now)
                    renewable = resources[:num_resources_renewable]

                    # Pad to 4 resources if needed
                    while len(renewable) < 4:
                        renewable.append(0)
                    renewable = renewable[:4]  # Take first 4

                    current_modes.append({
                        'duration': duration,
                        'resources': renewable,
                        'mode_id': mode_id - 1  # 0-indexed
                    })
                else:
                    # Continuation of modes for current job
                    mode_id = int(parts[0])
                    duration = int(parts[1])
                    resources = [int(parts[j]) for j in range(2, 2 + num_resources_renewable + num_resources_nonrenewable)]

                    renewable = resources[:num_resources_renewable]
                    while len(renewable) < 4:
                        renewable.append(0)
                    renewable = renewable[:4]

                    current_modes.append({
                        'duration': duration,
                        'resources': renewable,
                        'mode_id': len(current_modes)
                    })

            i += 1

        # Add last job
        if current_job is not None and len(current_modes) > 0:
            nodes_info.append({'modes': current_modes})

        # Parse resource availabilities
        for i, line in enumerate(lines):
            if 'RESOURCEAVAILABILITIES:' in line:
                resource_line = lines[i + 2].strip().split()
                renewable_resources = [int(x) for x in resource_line[:num_resources_renewable]]

                # Pad to 4
                while len(renewable_resources) < 4:
                    renewable_resources.append(0)
                resources = renewable_resources[:4]
                break

        result = {
            'adj_mat': adj_mat,
            'nodes_info': nodes_info,
            'resources': np.array(resources, dtype=np.float32),
            'num_jobs': num_jobs
        }

        # Cache result
        self.single_project_cache[str(filepath)] = result

        return result

    def load_multi_project_instance(self, filepath):
        """
        Load multi-project instance from A/, B/, or X/ directory
        Returns:
            dict with 'projects', 'shared_resources'
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        num_projects = int(lines[0])
        global_resource = int(lines[2])

        projects = []
        idx = 3

        for proj_id in range(num_projects):
            # Read project file path
            project_file = lines[idx]
            idx += 1

            # Read resource value (we'll use this as a multiplier or priority)
            resource_val = int(lines[idx])
            idx += 1

            # Read due date
            due_date = int(lines[idx])
            idx += 1

            # Load the actual project data
            project_filepath = self.data_root / 'MRCPSP' / project_file

            if not project_filepath.exists():
                print(f"Warning: Project file not found: {project_filepath}")
                continue

            project_data = self.parse_mm_file(project_filepath)

            # Add project metadata
            project_data['project_id'] = proj_id
            project_data['deadline'] = float(due_date)
            project_data['priority'] = resource_val / global_resource  # Normalize as priority

            projects.append(project_data)

        # Global shared resources (use max of individual project resources)
        if len(projects) > 0:
            max_resources = np.max([p['resources'] for p in projects], axis=0)
            shared_resources = max_resources * (global_resource / 10.0)  # Scale appropriately
        else:
            shared_resources = np.array([global_resource] * 4, dtype=np.float32)

        return {
            'projects': projects,
            'shared_resources': shared_resources,
            'instance_file': str(filepath.name)
        }

    def load_set(self, set_name='A'):
        """
        Load all instances from a set (A, B, or X)
        Returns:
            List of instances
        """
        set_dir = self.data_root / set_name

        if not set_dir.exists():
            raise FileNotFoundError(f"Set directory not found: {set_dir}")

        instances = []
        instance_files = sorted(set_dir.glob('*.txt'))

        print(f"Loading {len(instance_files)} instances from set {set_name}...")

        for filepath in instance_files:
            try:
                instance = self.load_multi_project_instance(filepath)
                instance['set'] = set_name
                instances.append(instance)
            except Exception as e:
                print(f"Error loading {filepath.name}: {e}")
                continue

        print(f"Successfully loaded {len(instances)} instances from set {set_name}")

        return instances

    def load_single_project_set(self, size=10):
        """
        Load single-project multi-mode instances (j10, j20, etc.)
        Args:
            size: 10, 20, or 30
        Returns:
            List of single-project instances
        """
        project_dir = self.data_root / 'MRCPSP' / f'j{size}.mm'

        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")

        instances = []
        instance_files = sorted(project_dir.glob('*.mm'))

        print(f"Loading {len(instance_files)} j{size} instances...")

        for filepath in instance_files[:100]:  # Limit to first 100
            try:
                project_data = self.parse_mm_file(filepath)

                # Wrap as single-project instance
                instance = {
                    'projects': [project_data],
                    'shared_resources': project_data['resources'],
                    'instance_file': filepath.name,
                    'set': f'j{size}'
                }
                instances.append(instance)

            except Exception as e:
                print(f"Error loading {filepath.name}: {e}")
                continue

        print(f"Successfully loaded {len(instances)} j{size} instances")

        return instances

    def get_statistics(self, instances):
        """Get statistics about loaded instances"""
        if len(instances) == 0:
            return {}

        num_projects_list = [len(inst['projects']) for inst in instances]
        num_tasks_list = []
        num_modes_list = []

        for inst in instances:
            for proj in inst['projects']:
                num_tasks_list.append(proj['num_jobs'] - 2)  # Exclude dummy nodes
                for node_info in proj['nodes_info']:
                    if 'modes' in node_info:
                        num_modes_list.append(len(node_info['modes']))

        return {
            'num_instances': len(instances),
            'projects_per_instance': {
                'min': min(num_projects_list),
                'max': max(num_projects_list),
                'mean': np.mean(num_projects_list)
            },
            'tasks_per_project': {
                'min': min(num_tasks_list) if num_tasks_list else 0,
                'max': max(num_tasks_list) if num_tasks_list else 0,
                'mean': np.mean(num_tasks_list) if num_tasks_list else 0
            },
            'modes_per_task': {
                'min': min(num_modes_list) if num_modes_list else 0,
                'max': max(num_modes_list) if num_modes_list else 0,
                'mean': np.mean(num_modes_list) if num_modes_list else 0
            }
        }


def create_mista_dataloaders(data_root='Mista', train_set='A', val_set='B', test_set='X'):
    """
    Create train/val/test dataloaders from MISTA benchmark
    """
    loader = MISTADataLoader(data_root)

    # Load sets
    try:
        train_instances = loader.load_set(train_set)
    except FileNotFoundError:
        print(f"Warning: {train_set} set not found, using j10 instead")
        train_instances = loader.load_single_project_set(10)

    try:
        val_instances = loader.load_set(val_set)
    except FileNotFoundError:
        print(f"Warning: {val_set} set not found")
        val_instances = []

    try:
        test_instances = loader.load_set(test_set)
    except FileNotFoundError:
        print(f"Warning: {test_set} set not found")
        test_instances = []

    return train_instances, val_instances, test_instances


# Test the loader
if __name__ == "__main__":
    print("="*60)
    print("Testing MISTA Data Loader")
    print("="*60)

    loader = MISTADataLoader(data_root='Mista')

    # Test single-project loading
    print("\n1. Testing single-project file parsing...")
    project_file = 'Mista/MRCPSP/j10.mm/j1010_1.mm'
    project_data = loader.parse_mm_file(project_file)

    print(f"   Loaded: {project_file}")
    print(f"   Jobs: {project_data['num_jobs']}")
    print(f"   Adjacency shape: {project_data['adj_mat'].shape}")
    print(f"   Tasks with modes: {len(project_data['nodes_info'])}")
    print(f"   Resources: {project_data['resources']}")
    print(f"   First task modes: {project_data['nodes_info'][0]['modes'][0]}")

    # Test multi-project loading
    print("\n2. Testing multi-project instance loading...")
    instance = loader.load_multi_project_instance('Mista/A/A-1.txt')

    print(f"   Loaded: A-1.txt")
    print(f"   Number of projects: {len(instance['projects'])}")
    print(f"   Shared resources: {instance['shared_resources']}")

    for i, proj in enumerate(instance['projects']):
        print(f"   Project {i}: {proj['num_jobs']-2} tasks, "
              f"deadline={proj['deadline']}, priority={proj['priority']:.2f}")

    # Test loading entire set
    print("\n3. Testing set loading...")
    instances_A = loader.load_set('A')

    stats = loader.get_statistics(instances_A)
    print(f"\n   Set A Statistics:")
    for key, value in stats.items():
        print(f"     {key}: {value}")

    print("\n✓ MISTA Data Loader Test Complete!")
