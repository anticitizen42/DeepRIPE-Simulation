# src/diagnostics.py

import numpy as np

# Try to import PyOpenCL to check GPU usage.
try:
    import pyopencl as cl
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def report_gpu_usage():
    """
    Reports whether GPU acceleration is available and prints out GPU device information.
    """
    if not GPU_AVAILABLE:
        print("PyOpenCL is not available. GPU acceleration is disabled.")
        return

    try:
        # Create a context and get device information.
        platforms = cl.get_platforms()
        if platforms:
            # Use the first platform and first device.
            platform = platforms[0]
            devices = platform.get_devices()
            if devices:
                device = devices[0]
                print("GPU acceleration is enabled.")
                print(f"Platform: {platform.name}, Vendor: {platform.vendor}")
                print(f"Device: {device.name}")
            else:
                print("No GPU devices found on the first platform.")
        else:
            print("No OpenCL platforms found.")
    except Exception as e:
        print(f"Error reporting GPU usage: {e}")

def compute_spin(field, dx):
    """
    Compute the topological winding (raw spin) from a central 2D slice of the field.
    Supports 4D fields (N0, N1, Ny, Nz) or 5D fields (e.g. a field strength tensor).
    """
    if len(field.shape) == 4:
        N0, N1, Ny, Nz = field.shape
        slice_field = field[N0 // 2, N1 // 2, :, :]
    elif len(field.shape) == 5:
        _, _, Nx, Ny, Nz = field.shape
        slice_field = field[1, 2, :, :]
    else:
        raise ValueError(f"Unexpected field shape for compute_spin: {field.shape}")
    
    phase = np.angle(slice_field)
    winding = 0.0
    # Top edge (left-to-right)
    for j in range(0, Nz - 1):
        diff = np.angle(np.exp(1j * (phase[0, j+1] - phase[0, j])))
        winding += diff
    # Right edge (top-to-bottom)
    for i in range(0, Ny - 1):
        diff = np.angle(np.exp(1j * (phase[i+1, Nz-1] - phase[i, Nz-1])))
        winding += diff
    # Bottom edge (right-to-left)
    for j in range(Nz - 1, 0, -1):
        diff = np.angle(np.exp(1j * (phase[Ny-1, j-1] - phase[Ny-1, j])))
        winding += diff
    # Left edge (bottom-to-top)
    for i in range(Ny - 1, 0, -1):
        diff = np.angle(np.exp(1j * (phase[i-1, 0] - phase[i, 0])))
        winding += diff
    
    return winding / (2 * np.pi)

def compute_charge(A, dx):
    """
    Compute the net charge from the gauge field A.
    (Placeholder: currently returns -1.)
    """
    return -1.0

def compute_gravity_indentation(Phi, dx):
    """
    Compute a gravitational indentation proxy for mass.
    (Placeholder: returns 1.)
    """
    return 1.0

def compute_energy_density(phi, dx):
    """
    Compute a rough energy density for a scalar field phi as the mean square of its gradients.
    """
    grad_y = np.gradient(phi, dx, axis=2)
    grad_z = np.gradient(phi, dx, axis=3)
    energy_density = np.mean(np.abs(grad_y)**2 + np.abs(grad_z)**2)
    return energy_density

def output_diagnostics_csv(snapshots, filename="simulation_diagnostics.csv", max_rows=100):
    """
    Write simulation diagnostics to a CSV file with columns:
      time, phi1_amp_mean, phi1_amp_std, phi1_phase_mean, phi2_phase_mean, spin, charge, grav_indentation.
    Downsamples snapshots to keep the CSV file size compact.
    """
    import csv
    total_snapshots = len(snapshots)
    step = max(1, total_snapshots // max_rows)
    selected_snapshots = snapshots[::step]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["time", "phi1_amp_mean", "phi1_amp_std", "phi1_phase_mean", "phi2_phase_mean", "spin", "charge", "grav_indentation"]
        writer.writerow(header)

        for snapshot in selected_snapshots:
            tau, phi1, phi2, A = snapshot
            amp1 = np.abs(phi1)
            amp2 = np.abs(phi2)
            phase1 = np.angle(phi1)
            phase2 = np.angle(phi2)
            phi1_amp_mean = np.mean(amp1)
            phi1_amp_std = np.std(amp1)
            phi1_phase_mean = np.mean(phase1)
            phi2_phase_mean = np.mean(phase2)
            spin = compute_spin(phi1, dx)
            charge = compute_charge(A, dx)
            grav_indentation = 1.0  # Placeholder
            row = [tau, phi1_amp_mean, phi1_amp_std, phi1_phase_mean, phi2_phase_mean, spin, charge, grav_indentation]
            writer.writerow(row)
    print(f"Diagnostics CSV written to {filename}")

if __name__ == "__main__":
    print("GPU Usage Report:")
    report_gpu_usage()
    # For testing: create dummy fields.
    shape = (16, 32, 64, 64)
    phi1 = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    phi2 = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    A = np.random.randn(4, 32, 64, 64) + 1j * np.random.randn(4, 32, 64, 64)
    dump_str = "Detailed Diagnostics Report\n===========================\n\n"
    N0, N1, Ny, Nz = shape
    central_phi1 = phi1[N0//2, N1//2, :, :]
    amp1 = np.abs(central_phi1)
    phase1 = np.angle(central_phi1)
    dump_str += f"phi1 Amplitude: min = {np.min(amp1):.4e}, max = {np.max(amp1):.4e}, mean = {np.mean(amp1):.4e}, std = {np.std(amp1):.4e}\n"
    dump_str += f"phi1 Phase: min = {np.min(phase1):.4e}, max = {np.max(phase1):.4e}, mean = {np.mean(phase1):.4e}, std = {np.std(phase1):.4e}\n"
    energy = compute_energy_density(phi1, dx=0.1)
    dump_str += f"Energy Density: {energy:.4e}\n"
    print(dump_str)
