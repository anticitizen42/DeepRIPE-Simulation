#!/usr/bin/env python3
"""
vortex_parameter_scanner.py - Automated parameter scanning for DV/RIPE field simulations

This script systematically explores the parameter space of the complex field model
to identify regions where vortex formation occurs spontaneously. For each parameter
combination, it:
  1. Initializes the field with random noise
  2. Evolves the system for a specified number of steps
  3. Analyzes vortex formation and properties
  4. Records data and continues to the next parameter set

Results are saved to disk for later detailed analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import json
import time
from datetime import datetime
import argparse
from itertools import product
from multiprocessing import Pool, cpu_count


def initialize_field(grid_size, initialization='random', seed=None):
    """
    Initialize the complex field.
    
    Parameters:
        grid_size: Tuple of (nx, ny) grid dimensions
        initialization: 'random', 'vortex_pair', or 'uniform'
        seed: Random seed for reproducibility
    
    Returns:
        Complex numpy array representing the field
    """
    if seed is not None:
        np.random.seed(seed)
    
    nx, ny = grid_size
    field = np.zeros((nx, ny), dtype=complex)
    
    if initialization == 'random':
        # Random noise initialization to allow spontaneous formation
        real_part = np.random.normal(0, 0.1, (nx, ny))
        imag_part = np.random.normal(0, 0.1, (nx, ny))
        field = real_part + 1j * imag_part
    
    elif initialization == 'vortex_pair':
        # Single vortex-antivortex pair as initial condition
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # First vortex
        r1 = np.sqrt((X - 0.3)**2 + (Y - 0)**2)
        theta1 = np.arctan2(Y, X - 0.3)
        
        # Second vortex (oppositely charged)
        r2 = np.sqrt((X + 0.3)**2 + (Y - 0)**2)
        theta2 = np.arctan2(Y, X + 0.3)
        
        field = np.exp(-r1) * np.exp(1j * theta1) + np.exp(-r2) * np.exp(-1j * theta2)
    
    elif initialization == 'uniform':
        # Uniform field with small random perturbations
        field = np.ones((nx, ny)) + np.random.normal(0, 0.01, (nx, ny)) + \
                1j * np.random.normal(0, 0.01, (nx, ny))
    
    # Normalize to have unit average amplitude
    field = field / np.sqrt(np.mean(np.abs(field)**2))
    
    return field


def evolve_field(field, dt, parameters):
    """
    Evolve the complex field using the complex Ginzburg-Landau equation.
    
    Parameters:
        field: Complex field array
        dt: Time step
        parameters: Dictionary of physical parameters
    
    Returns:
        Updated field
    """
    # Extract parameters
    alpha = parameters.get('alpha', 1.0)    # Linear coefficient
    beta = parameters.get('beta', 1.0)      # Nonlinearity strength
    gamma = parameters.get('gamma', 0.5)    # Dissipation
    D = parameters.get('D', 0.1)            # Diffusion coefficient
    
    # Compute Laplacian (using periodic boundary conditions)
    laplacian = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) + 
                 np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 
                 4 * field)
    
    # Compute field magnitude
    magnitude_squared = np.abs(field)**2
    
    # Complex Ginzburg-Landau dynamics
    # dψ/dt = α*ψ - β*|ψ|²*ψ + D*∇²ψ - γ*ψ
    field_new = field + dt * (alpha * field - beta * magnitude_squared * field + 
                              D * laplacian - gamma * field)
    
    return field_new


def detect_vortices(field):
    """
    Detect vortices by analyzing the phase winding.
    
    Parameters:
        field: Complex field array
    
    Returns:
        positions: List of (x, y) vortex positions
        charges: List of topological charges (+1 or -1)
    """
    phase = np.angle(field)
    
    # Compute phase gradients
    dx_phase = np.angle(np.exp(1j * (np.roll(phase, -1, axis=1) - phase)))
    dy_phase = np.angle(np.exp(1j * (np.roll(phase, -1, axis=0) - phase)))
    
    # Calculate phase winding around each plaquette
    winding = np.zeros_like(phase)
    
    for i in range(field.shape[0]-1):
        for j in range(field.shape[1]-1):
            # Sum phase differences around a plaquette
            plaquette_sum = dx_phase[i, j] + \
                           dy_phase[i, j+1] - \
                           dx_phase[i+1, j] - \
                           dy_phase[i, j]
            
            # Normalize to get winding number
            winding[i, j] = np.round(plaquette_sum / (2 * np.pi))
    
    # Find vortex locations
    vortex_pos = np.where(np.abs(winding) > 0.5)
    positions = list(zip(vortex_pos[0], vortex_pos[1]))
    charges = [int(winding[pos]) for pos in positions]
    
    return positions, charges


def analyze_vortex_properties(field, positions, charges):
    """
    Analyze properties of detected vortices.
    
    Parameters:
        field: Complex field array
        positions: List of (x, y) vortex positions
        charges: List of topological charges
    
    Returns:
        properties: Dictionary of vortex properties
    """
    if not positions:
        return {"count": 0, "positive_count": 0, "negative_count": 0, 
                "mean_energy": 0, "stability_metric": 0}
    
    properties = {
        "count": len(positions),
        "positive_count": sum(1 for c in charges if c > 0),
        "negative_count": sum(1 for c in charges if c < 0),
    }
    
    # Calculate energy density around vortices
    energy_densities = []
    for x, y in positions:
        # Extract a small region around the vortex
        x_min, x_max = max(0, x-5), min(field.shape[0], x+6)
        y_min, y_max = max(0, y-5), min(field.shape[1], y+6)
        region = field[x_min:x_max, y_min:y_max]
        
        # Calculate gradients
        dx = np.gradient(region, axis=0)
        dy = np.gradient(region, axis=1)
        
        # Energy density from gradients
        energy_density = np.mean(np.abs(dx)**2 + np.abs(dy)**2)
        energy_densities.append(energy_density)
    
    properties["mean_energy"] = np.mean(energy_densities) if energy_densities else 0
    
    # Calculate a simple stability metric - how close are vortex-antivortex pairs?
    # If all vortices are paired with antivortices, the field is typically more stable
    stability = 0
    if properties["positive_count"] > 0 and properties["negative_count"] > 0:
        pos_positions = [positions[i] for i, c in enumerate(charges) if c > 0]
        neg_positions = [positions[i] for i, c in enumerate(charges) if c < 0]
        
        # Calculate minimum distances between opposite-charge vortices
        min_distances = []
        for pos in pos_positions:
            distances = [np.sqrt((pos[0]-neg[0])**2 + (pos[1]-neg[1])**2) for neg in neg_positions]
            min_distances.append(min(distances) if distances else field.shape[0])
        
        # Stability is higher when vortices are paired (closer together)
        avg_min_distance = np.mean(min_distances)
        stability = 1.0 / (1.0 + avg_min_distance / field.shape[0])
    
    properties["stability_metric"] = stability
    
    return properties


def run_simulation(parameters, grid_size=(64, 64), steps=500, dt=0.01, seed=None, log_interval=100):
    """
    Run a simulation with the given parameters.
    
    Parameters:
        parameters: Dictionary of physical parameters
        grid_size: Tuple of grid dimensions
        steps: Number of simulation steps
        dt: Time step
        seed: Random seed for initialization
        log_interval: How often to log progress
    
    Returns:
        results: Dictionary containing simulation results
    """
    # Initialize field
    field = initialize_field(grid_size, initialization='random', seed=seed)
    
    # Track metrics over time
    vortex_counts = []
    positive_counts = []
    negative_counts = []
    energy_values = []
    stability_values = []
    
    # Simulate the field evolution
    for step in range(steps):
        field = evolve_field(field, dt, parameters)
        
        # Log progress and collect data at intervals
        if step % log_interval == 0 or step == steps - 1:
            positions, charges = detect_vortices(field)
            vortex_props = analyze_vortex_properties(field, positions, charges)
            
            vortex_counts.append(vortex_props["count"])
            positive_counts.append(vortex_props["positive_count"])
            negative_counts.append(vortex_props["negative_count"])
            energy_values.append(np.mean(np.abs(field)**2))
            stability_values.append(vortex_props["stability_metric"])
            
            if step % (log_interval * 10) == 0:
                print(f"Step {step}/{steps} - Parameters: {parameters} - Vortices: {vortex_props['count']}")
    
    # Final analysis
    positions, charges = detect_vortices(field)
    final_props = analyze_vortex_properties(field, positions, charges)
    
    # Compute some aggregate statistics
    results = {
        "parameters": parameters.copy(),
        "final_state": {
            "vortex_count": final_props["count"],
            "positive_count": final_props["positive_count"],
            "negative_count": final_props["negative_count"],
            "mean_energy": final_props["mean_energy"],
            "stability": final_props["stability_metric"]
        },
        "time_series": {
            "vortex_counts": vortex_counts,
            "positive_counts": positive_counts,
            "negative_counts": negative_counts,
            "energy_values": energy_values,
            "stability_values": stability_values
        },
        "metadata": {
            "grid_size": grid_size,
            "steps": steps,
            "dt": dt,
            "seed": seed
        }
    }
    
    # Also compute derived metrics for parameter scanning
    # Higher scores mean more interesting vortex dynamics
    vortex_formation_score = np.mean(vortex_counts)
    vortex_stability_score = np.mean(stability_values)
    
    # We want stable vortices that persist, not temporary fluctuations
    vortex_persistence = final_props["count"] / (np.mean(vortex_counts) + 1e-6)
    
    results["scores"] = {
        "vortex_formation": vortex_formation_score,
        "vortex_stability": vortex_stability_score,
        "vortex_persistence": vortex_persistence,
        # Combined score favoring parameter sets that produce stable, persistent vortices
        "combined_score": vortex_formation_score * vortex_stability_score * vortex_persistence
    }
    
    # Save the final field state (smaller format for storage)
    results["final_field"] = {
        "real": np.real(field).tolist(),
        "imag": np.imag(field).tolist()
    }
    
    return results


def process_parameter_set(params_info):
    """
    Process a single parameter set - used with multiprocessing.
    
    Parameters:
        params_info: Tuple of (param_index, parameters, config)
    
    Returns:
        Tuple of (param_index, results)
    """
    param_index, parameters, config = params_info
    
    print(f"Starting simulation for parameter set {param_index}: {parameters}")
    
    # Extract simulation config
    grid_size = config["grid_size"]
    steps = config["steps"]
    dt = config["dt"]
    seed = config["seed"] + param_index if config["seed"] is not None else None
    
    # Run the simulation
    results = run_simulation(
        parameters=parameters,
        grid_size=grid_size,
        steps=steps,
        dt=dt,
        seed=seed,
        log_interval=config["log_interval"]
    )
    
    print(f"Completed parameter set {param_index}. Score: {results['scores']['combined_score']:.4f}")
    return param_index, results


def run_parameter_scan(param_ranges, config, output_dir="results", parallel=True):
    """
    Run a parameter scan over the specified ranges.
    
    Parameters:
        param_ranges: Dictionary mapping parameter names to lists of values
        config: Dictionary with simulation configuration
        output_dir: Directory to save results
        parallel: Whether to use parallel processing
    
    Returns:
        all_results: Dictionary mapping parameter indices to results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all parameter combinations
    param_names = sorted(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    param_combinations = list(product(*param_values))
    
    print(f"Starting parameter scan with {len(param_combinations)} combinations")
    print(f"Parameter space: {param_ranges}")
    
    # Create parameter dictionaries
    parameter_dicts = []
    for combo in param_combinations:
        params = {name: value for name, value in zip(param_names, combo)}
        parameter_dicts.append(params)
    
    # Prepare task list for parallel processing
    tasks = [(i, params, config) for i, params in enumerate(parameter_dicts)]
    
    start_time = time.time()
    
    # Run simulations (parallel or sequential)
    all_results = {}
    if parallel and len(tasks) > 1:
        # Use multiprocessing for parallel execution
        num_processes = min(cpu_count(), len(tasks))
        print(f"Running in parallel with {num_processes} processes")
        
        with Pool(processes=num_processes) as pool:
            for i, result in pool.imap_unordered(process_parameter_set, tasks):
                all_results[i] = result
                
                # Save individual result to file
                result_file = os.path.join(output_dir, f"result_{i:04d}.json")
                with open(result_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    json_result = json.dumps(result, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                    f.write(json_result)
    else:
        print("Running sequentially")
        for param_info in tasks:
            i, result = process_parameter_set(param_info)
            all_results[i] = result
            
            # Save individual result to file
            result_file = os.path.join(output_dir, f"result_{i:04d}.json")
            with open(result_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_result = json.dumps(result, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                f.write(json_result)
    
    elapsed_time = time.time() - start_time
    print(f"Parameter scan completed in {elapsed_time:.2f} seconds")
    
    # Save parameter space information
    param_space_file = os.path.join(output_dir, "parameter_space.json")
    with open(param_space_file, 'w') as f:
        param_space_info = {
            "param_ranges": param_ranges,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            "num_combinations": len(param_combinations)
        }
        json.dump(param_space_info, f, indent=2)
    
    # Generate summary visualization
    generate_summary_visualization(all_results, param_ranges, output_dir)
    
    return all_results


def generate_summary_visualization(results, param_ranges, output_dir):
    """
    Generate summary visualizations of the parameter scan results.
    
    Parameters:
        results: Dictionary of simulation results
        param_ranges: Parameter ranges used for the scan
        output_dir: Directory to save visualizations
    """
    # Extract parameter values and scores
    param_indices = sorted(results.keys())
    scores = [results[i]["scores"]["combined_score"] for i in param_indices]
    vortex_counts = [results[i]["final_state"]["vortex_count"] for i in param_indices]
    
    # Create a color map for scores
    max_score = max(scores) if scores else 1.0
    normalized_scores = [score / max_score for score in scores]
    
    # Identify the best parameter sets
    top_indices = sorted(param_indices, key=lambda i: results[i]["scores"]["combined_score"], reverse=True)[:5]
    
    # Get parameter names
    param_names = sorted(param_ranges.keys())
    
    # Create a scatter plot matrix if we have multiple parameters
    if len(param_names) >= 2:
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, n_params, figsize=(n_params*3, n_params*3))
        
        # If only two parameters, axes isn't a 2D array
        if n_params == 2:
            axes = np.array([[axes]])
        
        for i, param_i in enumerate(param_names):
            for j, param_j in enumerate(param_names):
                if i == j:
                    # Histogram on diagonal
                    if len(results) > 1:
                        param_values = [list(results[idx]["parameters"].values())[i] for idx in param_indices]
                        axes[i, j].hist(param_values, bins=min(10, len(set(param_values))))
                        axes[i, j].set_title(param_i)
                else:
                    # Scatter plot off-diagonal
                    x_values = [results[idx]["parameters"][param_j] for idx in param_indices]
                    y_values = [results[idx]["parameters"][param_i] for idx in param_indices]
                    scatter = axes[i, j].scatter(x_values, y_values, c=normalized_scores, 
                                               cmap='viridis', s=50, alpha=0.7)
                    
                    # Highlight top results
                    for idx in top_indices:
                        axes[i, j].scatter(results[idx]["parameters"][param_j], 
                                         results[idx]["parameters"][param_i], 
                                         s=200, facecolors='none', edgecolors='red')
                    
                    axes[i, j].set_xlabel(param_j)
                    axes[i, j].set_ylabel(param_i)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parameter_scatter_matrix.png"), dpi=150)
        plt.close()
    
    # Create a bar chart of the top parameter sets
    if top_indices:
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_data = [results[i]["scores"]["combined_score"] for i in top_indices]
        x_labels = [", ".join([f"{p}={results[i]['parameters'][p]:.2f}" for p in param_names]) for i in top_indices]
        
        bars = ax.bar(range(len(top_indices)), bar_data)
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel("Combined Score")
        ax.set_title("Top Parameter Combinations")
        
        # Annotate vortex counts
        for i, bar in enumerate(bars):
            vortex_count = results[top_indices[i]]["final_state"]["vortex_count"]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 * max(bar_data),
                   f"{vortex_count} vortices", ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_parameters.png"), dpi=150)
        plt.close()
    
    # Create a heatmap for 2D parameter spaces
    if len(param_names) == 2:
        param1, param2 = param_names
        param1_values = sorted(set([results[i]["parameters"][param1] for i in param_indices]))
        param2_values = sorted(set([results[i]["parameters"][param2] for i in param_indices]))
        
        score_matrix = np.zeros((len(param1_values), len(param2_values)))
        vortex_matrix = np.zeros((len(param1_values), len(param2_values)))
        
        for i in param_indices:
            p1_idx = param1_values.index(results[i]["parameters"][param1])
            p2_idx = param2_values.index(results[i]["parameters"][param2])
            score_matrix[p1_idx, p2_idx] = results[i]["scores"]["combined_score"]
            vortex_matrix[p1_idx, p2_idx] = results[i]["final_state"]["vortex_count"]
        
        # Plot score heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(score_matrix, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(im, ax=ax, label="Combined Score")
        
        # Set tick labels
        ax.set_xticks(np.arange(len(param2_values)))
        ax.set_yticks(np.arange(len(param1_values)))
        ax.set_xticklabels([f"{x:.2f}" for x in param2_values])
        ax.set_yticklabels([f"{x:.2f}" for x in param1_values])
        
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)
        ax.set_title(f"Parameter Space Heatmap (Combined Score)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_score.png"), dpi=150)
        plt.close()
        
        # Plot vortex count heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(vortex_matrix, cmap='plasma', origin='lower', aspect='auto')
        plt.colorbar(im, ax=ax, label="Vortex Count")
        
        # Set tick labels
        ax.set_xticks(np.arange(len(param2_values)))
        ax.set_yticks(np.arange(len(param1_values)))
        ax.set_xticklabels([f"{x:.2f}" for x in param2_values])
        ax.set_yticklabels([f"{x:.2f}" for x in param1_values])
        
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)
        ax.set_title(f"Parameter Space Heatmap (Vortex Count)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_vortices.png"), dpi=150)
        plt.close()


def create_visualization_panel(results, output_dir, num_to_visualize=5):
    """
    Create visualization panels for the top results.
    
    Parameters:
        results: Dictionary of simulation results
        output_dir: Directory to save visualizations
        num_to_visualize: Number of top results to visualize
    """
    # Identify top results based on combined score
    top_indices = sorted(results.keys(), 
                         key=lambda i: results[i]["scores"]["combined_score"], 
                         reverse=True)[:num_to_visualize]
    
    os.makedirs(os.path.join(output_dir, "field_visualizations"), exist_ok=True)
    
    for idx in top_indices:
        result = results[idx]
        
        # Reconstruct the final field
        real_part = np.array(result["final_field"]["real"])
        imag_part = np.array(result["final_field"]["imag"])
        field = real_part + 1j * imag_part
        
        # Detect vortices
        positions, charges = detect_vortices(field)
        
        # Create a visualization panel
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Field amplitude
        amplitude_plot = axes[0].imshow(np.abs(field), cmap='viridis', 
                                      interpolation='nearest', origin='lower')
        axes[0].set_title('Field Amplitude')
        plt.colorbar(amplitude_plot, ax=axes[0])
        
        # Field phase with vortices marked
        phase_plot = axes[1].imshow(np.angle(field), cmap='hsv', 
                                  interpolation='nearest', origin='lower', 
                                  vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Field Phase with Vortices')
        plt.colorbar(phase_plot, ax=axes[1])
        
        # Plot vortex positions
        for pos, charge in zip(positions, charges):
            color = 'r' if charge > 0 else 'b'
            marker = 'o' if charge > 0 else 'x'
            axes[1].plot(pos[1], pos[0], marker=marker, color=color, markersize=8, markeredgewidth=2)
        
        # Plot time series data
        time_steps = np.arange(0, len(result["time_series"]["vortex_counts"]))
        axes[2].plot(time_steps, result["time_series"]["vortex_counts"], 'b-', label='Total Vortices')
        axes[2].plot(time_steps, result["time_series"]["positive_counts"], 'r-', label='Positive Charge')
        axes[2].plot(time_steps, result["time_series"]["negative_counts"], 'g-', label='Negative Charge')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Vortex Count Evolution')
        axes[2].legend()
        
        # Add parameter information
        param_text = "\n".join([f"{key} = {value:.4f}" for key, value in result["parameters"].items()])
        score_text = f"Score: {result['scores']['combined_score']:.4f}\nVortices: {result['final_state']['vortex_count']}"
        fig.text(0.01, 0.01, param_text, fontsize=10, verticalalignment='bottom')
        fig.text(0.01, 0.95, score_text, fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "field_visualizations", f"result_{idx:04d}.png"), dpi=150)
        plt.close()


def main():
    """Main function to set up and run the parameter scan."""
    parser = argparse.ArgumentParser(description='Run a parameter scan for vortex formation')
    
    # Basic simulation parameters
    parser.add_argument('--grid_size', type=int, default=64, help='Grid size (N×N)')
    parser.add_argument('--steps', type=int, default=500, help='Simulation steps per parameter set')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--output_dir', type=str, default='vortex_scan_results', 
                      help='Directory to save results')
    parser.add_argument('--log_interval', type=int, default=50, 
                      help='Interval for logging and analysis')
    
    # Parameter scan options
    parser.add_argument('--alpha_min', type=float, default=0.5, help='Minimum alpha value')
    parser.add_argument('--alpha_max', type=float, default=1.5, help='Maximum alpha value')
    parser.add_argument('--alpha_steps', type=int, default=5, help='Number of alpha steps')
    
    parser.add_argument('--beta_min', type=float, default=0.5, help='Minimum beta value')
    parser.add_argument('--beta_max', type=float, default=1.5, help='Maximum beta value')
    parser.add_argument('--beta_steps', type=int, default=5, help='Number of beta steps')
    
    parser.add_argument('--gamma_min', type=float, default=0.1, help='Minimum gamma value')
    parser.add_argument('--gamma_max', type=float, default=0.9, help='Maximum gamma value')
    parser.add_argument('--gamma_steps', type=int, default=5, help='Number of gamma steps')
    
    parser.add_argument('--D_min', type=float, default=0.05, help='Minimum D value')
    parser.add_argument('--D_max', type=float, default=0.2, help='Maximum D value')
    parser.add_argument('--D_steps', type=int, default=4, help='Number of D steps')
    
    parser.add_argument('--no_parallel', action='store_true', 
                      help='Disable parallel processing')
    
    args = parser.parse_args()
    
    # Create parameter ranges
    param_ranges = {
        'alpha': np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps),
        'beta': np.linspace(args.beta_min, args.beta_max, args.beta_steps),
        'gamma': np.linspace(args.gamma_min, args.gamma_max, args.gamma_steps),
        'D': np.linspace(args.D_min, args.D_max, args.D_steps)
    }
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # Configuration for simulations
    config = {
        "grid_size": (args.grid_size, args.grid_size),
        "steps": args.steps,
        "dt": args.dt,
        "seed": args.seed,
        "log_interval": args.log_interval
    }
    
    print(f"Starting parameter scan with the following ranges:")
    for param, values in param_ranges.items():
        print(f"  {param}: {min(values):.4f} to {max(values):.4f} ({len(values)} steps)")
    
    # Run the parameter scan
    results = run_parameter_scan(
        param_ranges=param_ranges,
        config=config,
        output_dir=output_dir,
        parallel=not args.no_parallel
    )
    
    # Create visualizations for top results
    create_visualization_panel(results, output_dir)
    
    print(f"Parameter scan complete. Results saved to {output_dir}")
    
    # Print information about the top 3 results
    top_indices = sorted(results.keys(), 
                        key=lambda i: results[i]["scores"]["combined_score"], 
                        reverse=True)[:3]
    
    print("\nTop parameter sets:")
    for i, idx in enumerate(top_indices):
        result = results[idx]
        param_str = ", ".join([f"{key}={value:.4f}" for key, value in result["parameters"].items()])
        vortex_count = result["final_state"]["vortex_count"]
        score = result["scores"]["combined_score"]
        print(f"{i+1}. Parameters: {param_str}")
        print(f"   Vortex count: {vortex_count}, Score: {score:.4f}")
    
    return results


if __name__ == '__main__':
    main()