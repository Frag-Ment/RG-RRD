"""
Debug MISTA Data Loading Issues

This script will help identify exactly where the "list index out of range" error is occurring.
"""
import traceback
import sys
from pathlib import Path

def test_file_loading():
    """Test loading a single MISTA file with detailed error reporting"""

    # Import after path setup
    from MISTADataLoader import MISTADataLoader

    print("="*70)
    print("MISTA Data Loading Debug")
    print("="*70)

    loader = MISTADataLoader('Mista')

    # Test on A-1.txt
    test_file = Path('Mista/A/A-1.txt')

    print(f"\n1. Testing file: {test_file}")
    print("-"*70)

    # Show file contents
    print("\nFile contents:")
    with open(test_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(f"  Line {i}: {repr(line)}")

    print(f"\nTotal lines: {len(lines)}")

    # Try to load
    print("\n2. Attempting to load...")
    print("-"*70)

    try:
        instance = loader.load_multi_project_instance(test_file)
        print(f"✓ SUCCESS! Loaded {len(instance['projects'])} projects")
        print(f"  Shared resources: {instance['shared_resources']}")
        for i, proj in enumerate(instance['projects']):
            print(f"  Project {i}: {proj.get('num_jobs', 'unknown')} jobs")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        print(f"\nError type: {type(e).__name__}")
        print("\nFull traceback:")
        print("-"*70)
        traceback.print_exc()
        print("-"*70)

        # Try to identify the issue
        print("\n3. Diagnostic Information:")
        print("-"*70)

        # Get the traceback
        tb = sys.exc_info()[2]
        while tb.tb_next:
            tb = tb.tb_next

        frame = tb.tb_frame
        print(f"Error occurred in: {frame.f_code.co_filename}")
        print(f"Function: {frame.f_code.co_name}")
        print(f"Line number: {tb.tb_lineno}")
        print(f"\nLocal variables at error:")
        for var_name, var_value in frame.f_locals.items():
            if not var_name.startswith('__'):
                try:
                    print(f"  {var_name} = {repr(var_value)[:100]}")
                except:
                    print(f"  {var_name} = <cannot display>")

        return False

    return True

if __name__ == "__main__":
    success = test_file_loading()

    print("\n" + "="*70)
    if success:
        print("✓ Data loading works correctly!")
        print("\nYou can now run: python check_setup.py")
    else:
        print("✗ Data loading failed - see error details above")
        print("\nPlease report the full output above for further debugging")
    print("="*70)
