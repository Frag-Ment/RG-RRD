import numpy as np

# Multi-mode node for MRCMPSP
# Each task can be executed in different modes with different duration-resource trade-offs
class MultiModeNode(object):
    def __init__(self, idx, modes):
        """
        Initialize a multi-mode node
        Args:
            idx: Node index
            modes: List of mode dictionaries, each containing:
                   {'duration': int, 'resources': [r1, r2, r3, r4], 'mode_id': int}
        """
        self.idx = idx
        self.modes = modes  # List of available modes
        self.num_modes = len(modes)

        # Selected mode (will be set during scheduling)
        self.selected_mode = None
        self.selected_mode_id = None

        # Current execution properties (based on selected mode)
        self.task_duration = None
        self.rest_time = None
        self.resource1 = None
        self.resource2 = None
        self.resource3 = None
        self.resource4 = None

        # Precedence relationships
        self.parent_nodes = []
        self.child_nodes = []
        self.descendant_nodes = []

        # Execution tracking
        self.completed_parent_nodes = []
        self.unlocked_parent_nodes = []

        # Status: 'running', 'done', 'not_start'
        self.condition = 'not_start'
        self.if_could_run = None
        self.if_need_check = None

    def select_mode(self, mode_id):
        """
        Select execution mode for this task
        Args:
            mode_id: Index of the mode to select (0 to num_modes-1)
        """
        assert 0 <= mode_id < self.num_modes, f"Invalid mode_id: {mode_id}"

        self.selected_mode_id = mode_id
        self.selected_mode = self.modes[mode_id]

        # Set execution properties based on selected mode
        self.task_duration = self.selected_mode['duration']
        self.rest_time = self.task_duration
        self.resource1 = self.selected_mode['resources'][0]
        self.resource2 = self.selected_mode['resources'][1]
        self.resource3 = self.selected_mode['resources'][2]
        self.resource4 = self.selected_mode['resources'][3]

    def get_mode_features(self):
        """
        Get feature representation of all modes
        Returns:
            numpy array of shape [num_modes, 5] with [duration, r1, r2, r3, r4]
        """
        mode_features = []
        for mode in self.modes:
            feature = [mode['duration']] + mode['resources']
            mode_features.append(feature)
        return np.array(mode_features, dtype=np.float32)

    def get_mode_efficiency_score(self, mode_id):
        """
        Calculate efficiency score for a mode
        Higher score = more resource-intensive but faster
        """
        mode = self.modes[mode_id]
        resource_intensity = sum(mode['resources']) / (mode['duration'] + 1e-6)
        return resource_intensity
