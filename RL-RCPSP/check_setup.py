"""
Check if MRCMPSP + MISTA setup is ready for training
Run this before starting training to verify everything works
"""
import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ⚠ Warning: Python 3.8+ recommended, you have {version.major}.{version.minor}")
        return False
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")

    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'tqdm': 'tqdm'
    }

    optional = {
        'tensorboard': 'TensorBoard (for monitoring)'
    }

    all_good = True

    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - REQUIRED")
            print(f"    Install: pip install {module}")
            all_good = False

    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - Optional but recommended")
            print(f"    Install: pip install {module}")

    return all_good


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA/GPU...")

    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available: {device_count} device(s)")
            print(f"    Device 0: {device_name}")
            return True
        else:
            print(f"  ⚠ CUDA not available - will use CPU")
            print(f"    Training will be slower (~10x)")
            return False
    except ImportError:
        print(f"  ✗ Cannot check CUDA (torch not installed)")
        return False


def check_mista_data():
    """Check MISTA data availability"""
    print("\nChecking MISTA data...")

    mista_dir = Path('Mista')

    if not mista_dir.exists():
        print(f"  ✗ Mista directory not found!")
        print(f"    Expected location: {mista_dir.absolute()}")
        print(f"    Make sure MISTA data is in the correct location")
        return False

    # Check subdirectories
    required_dirs = ['A', 'B', 'MRCPSP']
    all_present = True

    for dirname in required_dirs:
        subdir = mista_dir / dirname
        if subdir.exists():
            if dirname == 'MRCPSP':
                # Count instances
                j10_dir = subdir / 'j10.mm'
                if j10_dir.exists():
                    count = len(list(j10_dir.glob('*.mm')))
                    print(f"  ✓ {dirname}/ ({count} j10 instances)")
                else:
                    print(f"  ✓ {dirname}/ (exists)")
            else:
                count = len(list(subdir.glob('*.txt')))
                print(f"  ✓ {dirname}/ ({count} instances)")
        else:
            print(f"  ✗ {dirname}/ missing")
            all_present = False

    return all_present


def check_implementation_files():
    """Check if implementation files are present"""
    print("\nChecking implementation files...")

    required_files = {
        'MISTADataLoader.py': 'MISTA data loader',
        'model/hierarchical_agent.py': 'Hierarchical agent',
        'mrcmpsp_simulator/mrcmpsp_env.py': 'MRCMPSP environment',
        'train_mista.py': 'Training script',
        'evaluate_mista.py': 'Evaluation script'
    }

    all_present = True

    for filepath, description in required_files.items():
        if Path(filepath).exists():
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} missing: {filepath}")
            all_present = False

    return all_present


def test_data_loading():
    """Test if data can be loaded"""
    print("\nTesting data loading...")

    try:
        from MISTADataLoader import MISTADataLoader

        loader = MISTADataLoader(data_root='Mista')

        # Try loading Set A
        try:
            instances_A = loader.load_set('A')
            print(f"  ✓ Loaded {len(instances_A)} instances from Set A")

            # Test parsing first instance
            if len(instances_A) > 0:
                first = instances_A[0]
                num_projects = len(first['projects'])
                print(f"    First instance: {num_projects} projects")
                return True
            else:
                print(f"  ✗ Set A is empty")
                return False

        except Exception as e:
            print(f"  ✗ Error loading Set A: {e}")
            return False

    except ImportError as e:
        print(f"  ✗ Cannot import MISTADataLoader: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_environment():
    """Test if environment can be created"""
    print("\nTesting environment...")

    try:
        from mrcmpsp_simulator.mrcmpsp_env import MRCMPSPEnvironment
        from MISTADataLoader import MISTADataLoader

        loader = MISTADataLoader(data_root='Mista')
        instances = loader.load_set('A')

        if len(instances) == 0:
            print(f"  ✗ No instances to test")
            return False

        instance = instances[0]

        env = MRCMPSPEnvironment()
        state = env.reset(instance['projects'], instance['shared_resources'])

        print(f"  ✓ Environment created successfully")
        print(f"    Active projects: {len(state.active_projects)}")
        print(f"    Feasible actions: {len(state.feasible_actions)}")

        return True

    except ImportError as e:
        print(f"  ✗ Cannot import environment: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent():
    """Test if agent can be created"""
    print("\nTesting agent...")

    try:
        import torch
        from model.hierarchical_agent import HierarchicalAgent

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        agent = HierarchicalAgent(
            project_feature_dim=8,
            project_hidden_dim=32,
            num_gnn_layers=2,
            gnn_hidden_dim=32,
            mlp_hidden_dim=16,
            node_feature_dim=9,
            device=device
        ).to(device)

        param_count = sum(p.numel() for p in agent.parameters())
        print(f"  ✓ Agent created successfully")
        print(f"    Parameters: {param_count:,}")
        print(f"    Device: {device}")

        return True

    except ImportError as e:
        print(f"  ✗ Cannot import agent: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print("="*70)
    print("MRCMPSP + MISTA Setup Verification")
    print("="*70)

    results = {
        'Python version': check_python_version(),
        'Dependencies': check_dependencies(),
        'CUDA/GPU': check_cuda(),
        'MISTA data': check_mista_data(),
        'Implementation files': check_implementation_files(),
        'Data loading': test_data_loading(),
        'Environment': test_environment(),
        'Agent': test_agent()
    }

    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    critical_checks = ['Python version', 'Dependencies', 'MISTA data', 'Implementation files']
    optional_checks = ['CUDA/GPU']
    functional_checks = ['Data loading', 'Environment', 'Agent']

    all_critical_pass = all(results[check] for check in critical_checks)
    all_functional_pass = all(results[check] for check in functional_checks)

    for check, passed in results.items():
        status = "✓" if passed else "✗"
        importance = ""

        if check in critical_checks:
            importance = " [REQUIRED]"
        elif check in optional_checks:
            importance = " [Optional]"
        elif check in functional_checks:
            importance = " [Functional]"

        print(f"  {status} {check}{importance}")

    print("\n" + "="*70)

    if all_critical_pass and all_functional_pass:
        print("✓ All checks passed!")
        print("\n Ready to train:")
        print("   python train_mista.py")
        print("\n Monitor training:")
        print("   tensorboard --logdir=runs")
        return 0
    elif all_critical_pass:
        print("⚠ Critical checks passed, but some functional tests failed")
        print("  Check error messages above and fix issues before training")
        return 1
    else:
        print("✗ Some critical checks failed")
        print("  Please fix the issues above before proceeding")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
