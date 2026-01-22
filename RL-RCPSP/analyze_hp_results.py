"""
Analyze and Visualize Hyperparameter Tuning Results

Usage:
    python analyze_hp_results.py
    python analyze_hp_results.py --results_file hp_tuning_results/hp_search_random_123456.json
    python analyze_hp_results.py --plot  # Generate plots (requires matplotlib)
"""
import json
import argparse
from pathlib import Path
import numpy as np


def load_results(results_file=None):
    """Load hyperparameter tuning results"""

    if results_file is None:
        # Find most recent results file
        results_dir = Path('hp_tuning_results')

        if not results_dir.exists():
            print("✗ No results directory found!")
            print("  Run hyperparameter tuning first: python tune_hyperparameters.py")
            return None

        result_files = list(results_dir.glob('hp_search_*.json'))

        if len(result_files) == 0:
            print("✗ No results files found!")
            print("  Run hyperparameter tuning first: python tune_hyperparameters.py")
            return None

        # Get most recent
        results_file = max(result_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading most recent results: {results_file}")

    with open(results_file, 'r') as f:
        data = json.load(f)

    return data


def print_summary(results):
    """Print summary of hyperparameter tuning results"""

    print("\n" + "="*70)
    print("Hyperparameter Tuning Results Summary")
    print("="*70)

    print(f"\nSearch Type: {results['search_type']}")
    print(f"Configurations Tested: {results['n_configs']}")
    print(f"Timesteps per Config: {results['timesteps_per_config']:,}")

    all_results = results['all_results']

    # Filter valid results
    valid = [r for r in all_results if r['val_makespan_mean'] != float('inf')]
    failed = len(all_results) - len(valid)

    print(f"\nResults:")
    print(f"  ✓ Successful: {len(valid)}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")

    if len(valid) == 0:
        print("\n✗ No valid results to analyze!")
        return

    # Statistics
    makespans = [r['val_makespan_mean'] for r in valid]

    print(f"\nValidation Makespan Statistics:")
    print(f"  Best: {min(makespans):.2f}")
    print(f"  Worst: {max(makespans):.2f}")
    print(f"  Mean: {np.mean(makespans):.2f}")
    print(f"  Std: {np.std(makespans):.2f}")
    print(f"  Improvement (best over worst): {(max(makespans) - min(makespans)) / max(makespans) * 100:.1f}%")

    # Top configurations
    sorted_results = sorted(valid, key=lambda x: x['val_makespan_mean'])

    print(f"\n" + "="*70)
    print(f"Top 5 Configurations")
    print("="*70)

    for i, result in enumerate(sorted_results[:5]):
        print(f"\n{'#'*70}")
        print(f"Rank {i+1}: Makespan = {result['val_makespan_mean']:.2f} ± {result['val_makespan_std']:.2f}")
        print(f"{'#'*70}")

        config = result['config']

        print(f"\nArchitecture:")
        print(f"  GNN layers: {config['gnn_layers']}")
        print(f"  GNN hidden dim: {config['gnn_hidden_dim']}")
        print(f"  MLP hidden dim: {config['mlp_hidden_dim']}")
        print(f"  Project hidden dim: {config['project_hidden_dim']}")

        print(f"\nTraining:")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Clip coef: {config['clip_coef']}")
        print(f"  Entropy coef: {config['ent_coef']}")
        print(f"  Value coef: {config['vf_coef']}")

        print(f"\nPPO:")
        print(f"  Num envs: {config['num_envs']}")
        print(f"  Num steps: {config['num_steps']}")
        print(f"  Minibatch size: {config['minibatch_size']}")
        print(f"  Update epochs: {config['update_epochs']}")

        print(f"\nMetrics:")
        print(f"  Success rate: {result.get('val_success_rate', 0):.1f}%")
        print(f"  Num parameters: {result.get('num_parameters', 0):,}")
        print(f"  Model path: {result.get('model_path', 'N/A')}")


def analyze_parameter_importance(results):
    """Analyze which hyperparameters matter most"""

    print("\n" + "="*70)
    print("Parameter Importance Analysis")
    print("="*70)

    all_results = results['all_results']
    valid = [r for r in all_results if r['val_makespan_mean'] != float('inf')]

    if len(valid) < 3:
        print("\n⚠ Need at least 3 valid results for analysis")
        return

    # Collect data
    param_values = {}

    for result in valid:
        config = result['config']
        makespan = result['val_makespan_mean']

        for param, value in config.items():
            if param not in param_values:
                param_values[param] = {}

            if value not in param_values[param]:
                param_values[param][value] = []

            param_values[param][value].append(makespan)

    # Analyze each parameter
    print("\nParameter Impact (lower makespan is better):")
    print("-" * 70)

    param_impacts = []

    for param, values_dict in sorted(param_values.items()):
        if len(values_dict) < 2:
            continue  # Skip if only one value tested

        # Calculate average makespan for each value
        value_avgs = {}
        for value, makespans in values_dict.items():
            value_avgs[value] = np.mean(makespans)

        # Range (max - min)
        best_value = min(value_avgs, key=value_avgs.get)
        worst_value = max(value_avgs, key=value_avgs.get)
        impact = value_avgs[worst_value] - value_avgs[best_value]
        impact_pct = impact / value_avgs[worst_value] * 100

        param_impacts.append((param, impact, impact_pct, best_value, value_avgs))

    # Sort by impact
    param_impacts.sort(key=lambda x: x[1], reverse=True)

    for param, impact, impact_pct, best_value, value_avgs in param_impacts:
        print(f"\n{param}:")
        print(f"  Impact: {impact:.2f} ({impact_pct:.1f}%)")
        print(f"  Best value: {best_value} (avg makespan: {value_avgs[best_value]:.2f})")
        print(f"  All values:")
        for value, avg in sorted(value_avgs.items(), key=lambda x: x[1]):
            print(f"    {value}: {avg:.2f}")

    # Recommendations
    print("\n" + "="*70)
    print("Recommendations")
    print("="*70)

    print("\nMost Important Parameters (largest impact):")
    for i, (param, impact, impact_pct, best_value, _) in enumerate(param_impacts[:3]):
        print(f"  {i+1}. {param}: Use {best_value} (impact: {impact_pct:.1f}%)")

    if len(param_impacts) > 3:
        print("\nLess Important Parameters (smaller impact):")
        for param, impact, impact_pct, best_value, _ in param_impacts[3:]:
            print(f"  - {param}: {impact_pct:.1f}% impact")


def generate_best_config_file(results, output_file='best_config_analyzed.json'):
    """Generate a file with the best configuration and analysis"""

    all_results = results['all_results']
    valid = [r for r in all_results if r['val_makespan_mean'] != float('inf')]

    if len(valid) == 0:
        print("\n✗ No valid results to save")
        return

    best_result = min(valid, key=lambda x: x['val_makespan_mean'])

    output = {
        'best_config': best_result['config'],
        'performance': {
            'val_makespan_mean': best_result['val_makespan_mean'],
            'val_makespan_std': best_result.get('val_makespan_std', 0),
            'val_success_rate': best_result.get('val_success_rate', 0),
            'num_parameters': best_result.get('num_parameters', 0)
        },
        'model_path': best_result.get('model_path', 'N/A'),
        'search_info': {
            'search_type': results['search_type'],
            'n_configs_tested': results['n_configs'],
            'timesteps_per_config': results['timesteps_per_config']
        }
    }

    output_path = Path('hp_tuning_results') / output_file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Best configuration saved to: {output_path}")


def plot_results(results):
    """Generate visualization plots (requires matplotlib)"""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠ matplotlib not installed - skipping plots")
        print("  Install with: pip install matplotlib")
        return

    all_results = results['all_results']
    valid = [r for r in all_results if r['val_makespan_mean'] != float('inf')]

    if len(valid) < 2:
        print("\n⚠ Need at least 2 valid results for plotting")
        return

    print("\nGenerating plots...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Tuning Results Analysis', fontsize=16, fontweight='bold')

    # 1. Makespan distribution
    makespans = [r['val_makespan_mean'] for r in valid]
    axes[0, 0].hist(makespans, bins=min(20, len(valid)), color='skyblue', edgecolor='black')
    axes[0, 0].axvline(min(makespans), color='green', linestyle='--', linewidth=2, label='Best')
    axes[0, 0].axvline(np.mean(makespans), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].set_xlabel('Validation Makespan', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Validation Makespan', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Configuration ranking
    sorted_valid = sorted(valid, key=lambda x: x['val_makespan_mean'])
    ranks = list(range(1, len(sorted_valid) + 1))
    makespans_sorted = [r['val_makespan_mean'] for r in sorted_valid]
    axes[0, 1].plot(ranks, makespans_sorted, marker='o', color='coral', linewidth=2)
    axes[0, 1].scatter([1], [makespans_sorted[0]], color='green', s=200, zorder=5, label='Best')
    axes[0, 1].set_xlabel('Configuration Rank', fontsize=12)
    axes[0, 1].set_ylabel('Validation Makespan', fontsize=12)
    axes[0, 1].set_title('Configuration Performance Ranking', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Learning rate impact
    lr_values = {}
    for r in valid:
        lr = r['config']['learning_rate']
        if lr not in lr_values:
            lr_values[lr] = []
        lr_values[lr].append(r['val_makespan_mean'])

    lr_sorted = sorted(lr_values.keys())
    lr_means = [np.mean(lr_values[lr]) for lr in lr_sorted]
    lr_stds = [np.std(lr_values[lr]) for lr in lr_sorted]

    axes[1, 0].errorbar([str(lr) for lr in lr_sorted], lr_means, yerr=lr_stds,
                       marker='o', capsize=5, capthick=2, linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Learning Rate', fontsize=12)
    axes[1, 0].set_ylabel('Avg Validation Makespan', fontsize=12)
    axes[1, 0].set_title('Learning Rate Impact', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. GNN hidden dim impact
    gnn_values = {}
    for r in valid:
        gnn_dim = r['config']['gnn_hidden_dim']
        if gnn_dim not in gnn_values:
            gnn_values[gnn_dim] = []
        gnn_values[gnn_dim].append(r['val_makespan_mean'])

    gnn_sorted = sorted(gnn_values.keys())
    gnn_means = [np.mean(gnn_values[dim]) for dim in gnn_sorted]
    gnn_stds = [np.std(gnn_values[dim]) for dim in gnn_sorted]

    axes[1, 1].errorbar([str(dim) for dim in gnn_sorted], gnn_means, yerr=gnn_stds,
                       marker='s', capsize=5, capthick=2, linewidth=2, color='teal')
    axes[1, 1].set_xlabel('GNN Hidden Dimension', fontsize=12)
    axes[1, 1].set_ylabel('Avg Validation Makespan', fontsize=12)
    axes[1, 1].set_title('GNN Hidden Dimension Impact', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path('hp_tuning_results')
    output_dir.mkdir(exist_ok=True)
    plot_file = output_dir / 'hp_analysis_plots.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plots saved to: {plot_file}")

    # Show plot
    plt.show()


def export_to_csv(results, output_file='hp_results.csv'):
    """Export results to CSV for external analysis"""

    import csv

    all_results = results['all_results']
    valid = [r for r in all_results if r['val_makespan_mean'] != float('inf')]

    if len(valid) == 0:
        print("\n✗ No valid results to export")
        return

    output_path = Path('hp_tuning_results') / output_file

    # Get all config keys
    config_keys = list(valid[0]['config'].keys())

    # CSV header
    header = config_keys + [
        'val_makespan_mean', 'val_makespan_std',
        'val_success_rate', 'num_parameters',
        'train_loss_mean', 'train_reward_mean'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for result in valid:
            row = result['config'].copy()
            row['val_makespan_mean'] = result['val_makespan_mean']
            row['val_makespan_std'] = result.get('val_makespan_std', 0)
            row['val_success_rate'] = result.get('val_success_rate', 0)
            row['num_parameters'] = result.get('num_parameters', 0)
            row['train_loss_mean'] = result.get('train_loss_mean', 0)
            row['train_reward_mean'] = result.get('train_reward_mean', 0)

            writer.writerow(row)

    print(f"\n✓ Results exported to CSV: {output_path}")
    print(f"  You can open this in Excel, Google Sheets, or analyze with pandas")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze hyperparameter tuning results'
    )
    parser.add_argument('--results_file', type=str, default=None,
                       help='Path to results JSON file (default: most recent)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots (requires matplotlib)')
    parser.add_argument('--csv', action='store_true',
                       help='Export results to CSV')

    args = parser.parse_args()

    # Load results
    results = load_results(args.results_file)

    if results is None:
        return

    # Print summary
    print_summary(results)

    # Analyze parameter importance
    analyze_parameter_importance(results)

    # Generate best config file
    generate_best_config_file(results)

    # Generate plots if requested
    if args.plot:
        plot_results(results)

    # Export to CSV if requested
    if args.csv:
        export_to_csv(results)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review top configurations above")
    print("  2. Copy best config to train_mista.py")
    print("  3. Run full training: python train_mista.py")
    print("  4. Evaluate: python evaluate_mista.py")

    if not args.plot:
        print("\nTip: Run with --plot flag to generate visualizations:")
        print("  python analyze_hp_results.py --plot")

    if not args.csv:
        print("\nTip: Run with --csv flag to export results for external analysis:")
        print("  python analyze_hp_results.py --csv")


if __name__ == "__main__":
    main()
