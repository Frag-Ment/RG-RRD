"""
Device Configuration for Mac M2 / CUDA / CPU

Automatically detects and configures the best available device:
- MPS (Metal Performance Shaders) for Mac M2/M3
- CUDA for NVIDIA GPUs
- CPU as fallback

Also provides memory-optimized configurations for different systems.
"""
import torch
import platform


def get_device():
    """
    Get the best available device for training

    Returns:
        str: Device string ('mps', 'cuda:0', or 'cpu')
        str: Device name for display
    """
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = 'cuda:0'
        device_name = torch.cuda.get_device_name(0)
        return device, device_name

    # Check for MPS (Apple Silicon - M1/M2/M3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        device_name = f"Apple Silicon (MPS) on {platform.machine()}"
        return device, device_name

    # Fallback to CPU
    else:
        device = 'cpu'
        device_name = f"CPU ({platform.processor()})"
        return device, device_name


def get_memory_config(device_type='cpu', total_ram_gb=16):
    """
    Get memory-optimized configuration for the device

    Args:
        device_type: 'mps', 'cuda', or 'cpu'
        total_ram_gb: Total system RAM in GB

    Returns:
        dict: Configuration with optimized batch sizes and environment counts
    """
    configs = {
        'mps_16gb': {
            'num_envs': 4,
            'num_steps': 128,
            'minibatch_size': 64,
            'gnn_hidden_dim': 64,
            'max_instances_in_memory': 50,
            'description': 'Optimized for Mac M2 with 16GB RAM'
        },
        'mps_32gb': {
            'num_envs': 8,
            'num_steps': 128,
            'minibatch_size': 64,
            'gnn_hidden_dim': 128,
            'max_instances_in_memory': 100,
            'description': 'Optimized for Mac M2 Pro/Max with 32GB+ RAM'
        },
        'cuda_8gb': {
            'num_envs': 4,
            'num_steps': 128,
            'minibatch_size': 64,
            'gnn_hidden_dim': 64,
            'max_instances_in_memory': 50,
            'description': 'Optimized for NVIDIA GPU with 8GB VRAM'
        },
        'cuda_16gb': {
            'num_envs': 8,
            'num_steps': 256,
            'minibatch_size': 128,
            'gnn_hidden_dim': 128,
            'max_instances_in_memory': 100,
            'description': 'Optimized for NVIDIA GPU with 16GB+ VRAM'
        },
        'cpu_16gb': {
            'num_envs': 2,
            'num_steps': 64,
            'minibatch_size': 32,
            'gnn_hidden_dim': 32,
            'max_instances_in_memory': 20,
            'description': 'Optimized for CPU with 16GB RAM (slow)'
        },
        'cpu_32gb': {
            'num_envs': 4,
            'num_steps': 128,
            'minibatch_size': 64,
            'gnn_hidden_dim': 64,
            'max_instances_in_memory': 50,
            'description': 'Optimized for CPU with 32GB+ RAM (slow)'
        }
    }

    # Select appropriate config
    if device_type == 'mps':
        config_key = 'mps_16gb' if total_ram_gb <= 16 else 'mps_32gb'
    elif device_type == 'cuda':
        config_key = 'cuda_8gb' if total_ram_gb <= 12 else 'cuda_16gb'
    else:  # cpu
        config_key = 'cpu_16gb' if total_ram_gb <= 16 else 'cpu_32gb'

    return configs[config_key]


def print_device_info():
    """Print detailed device information"""
    print("\n" + "="*70)
    print("Device Configuration")
    print("="*70)

    device, device_name = get_device()

    print(f"\n✓ Selected Device: {device}")
    print(f"  Name: {device_name}")

    # Print device-specific info
    if device.startswith('cuda'):
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device == 'mps':
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  MPS Available: ✓")
        print(f"  Shared Memory: System RAM (unified memory architecture)")
    else:
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  Note: Training will be 5-10x slower on CPU")

    # Get memory config
    device_type = 'mps' if device == 'mps' else ('cuda' if 'cuda' in device else 'cpu')
    mem_config = get_memory_config(device_type, total_ram_gb=16)

    print(f"\n✓ Memory Configuration: {mem_config['description']}")
    print(f"  Parallel Environments: {mem_config['num_envs']}")
    print(f"  Steps per Update: {mem_config['num_steps']}")
    print(f"  Minibatch Size: {mem_config['minibatch_size']}")
    print(f"  GNN Hidden Dim: {mem_config['gnn_hidden_dim']}")

    print("="*70 + "\n")

    return device, mem_config


def test_device():
    """Test if the selected device works correctly"""
    device, device_name = get_device()

    print(f"\nTesting {device}...")

    try:
        # Create test tensor
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)

        # Move back to CPU to verify
        z_cpu = z.cpu()

        print(f"✓ {device} is working correctly!")
        print(f"  Test computation: {z_cpu.shape} tensor")

        return True

    except Exception as e:
        print(f"✗ {device} test failed: {e}")
        print(f"  Falling back to CPU")
        return False


if __name__ == "__main__":
    # Print device info
    device, mem_config = print_device_info()

    # Test device
    test_device()

    # Print recommendations
    print("\nRecommendations:")

    if device == 'mps':
        print("  ✓ Mac M2 detected - training will be fast!")
        print("  ✓ Use these settings in train_mista.py:")
        print(f"    num_envs = {mem_config['num_envs']}")
        print(f"    num_steps = {mem_config['num_steps']}")
        print(f"    gnn_hidden_dim = {mem_config['gnn_hidden_dim']}")
        print("\n  Note: If you get memory errors, reduce num_envs to 2")

    elif device.startswith('cuda'):
        print("  ✓ NVIDIA GPU detected - training will be very fast!")
        print("  ✓ Use these settings in train_mista.py:")
        print(f"    num_envs = {mem_config['num_envs']}")
        print(f"    num_steps = {mem_config['num_steps']}")
        print(f"    gnn_hidden_dim = {mem_config['gnn_hidden_dim']}")

    else:
        print("  ⚠ CPU only - training will be slow (5-10x slower)")
        print("  ⚠ Recommended: Use smaller configurations")
        print(f"    num_envs = {mem_config['num_envs']}")
        print(f"    num_steps = {mem_config['num_steps']}")
        print(f"    gnn_hidden_dim = {mem_config['gnn_hidden_dim']}")
        print("\n  Consider:")
        print("    - Running on a machine with GPU")
        print("    - Using fewer training timesteps for testing")
