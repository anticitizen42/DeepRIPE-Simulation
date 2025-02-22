# src/diagnostics.py

import numpy as np

def compute_spin(field, dx):
    """
    Compute the topological winding (interpreted as "spin") from a 2D slice of a field.
    
    If the input field has 4 dimensions (N0, N1, Ny, Nz), we extract the central slice.
    If the input field has 5 dimensions (e.g., a field strength tensor with shape (4,4,Nx,Ny,Nz)),
    we select a representative component (here F[1,2]) and extract its central slice.
    
    The winding is computed by summing the phase differences along the boundary of the slice,
    then dividing by 2π.
    """
    if len(field.shape) == 4:
        N0, N1, Ny, Nz = field.shape
        slice_field = field[N0 // 2, N1 // 2, :, :]
    elif len(field.shape) == 5:
        # Assume field has shape (4, 4, Nx, Ny, Nz)
        _, _, Nx, Ny, Nz = field.shape
        # Use component [1,2] as a representative slice.
        slice_field = field[1, 2, :, :]
    else:
        raise ValueError("Unexpected field shape for compute_spin: {}".format(field.shape))
    
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
    (Placeholder: In this simulation, we assume the net charge is fixed at -1.)
    """
    return -1.0

def compute_gravity_indentation(Phi, dx):
    """
    Compute a gravitational indentation proxy for mass.
    (Placeholder: Returns a constant value.)
    """
    return 1.0

def output_diagnostics_csv(snapshots, filename="simulation_diagnostics.csv", max_rows=100):
    """
    Write simulation diagnostics to a CSV file with columns:
      time, phi1_amp_mean, phi1_amp_std, phi1_phase_mean, phi2_phase_mean, spin, charge, grav_indentation.
      
    If the number of snapshots exceeds max_rows, we downsample by taking every nth snapshot.
    This ensures the output CSV remains under roughly 20 KB.
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
            spin = compute_spin(phi1, 0.1)  # dx passed as 0.1 (or use actual dx)
            charge = compute_charge(A, 0.1)
            grav_indentation = 1.0  # placeholder
            row = [tau, phi1_amp_mean, phi1_amp_std, phi1_phase_mean, phi2_phase_mean, spin, charge, grav_indentation]
            writer.writerow(row)

if __name__ == "__main__":
    # For testing: create a few dummy snapshots and write diagnostics to CSV.
    snapshots = []
    for i in range(50):
        tau = i * 0.01
        shape = (4, 8, 16, 16)
        # For testing, we generate a simple vortex-like pattern in phi1 and phi2.
        phi1 = np.ones(shape, dtype=np.complex128) * np.exp(1j * np.linspace(-3.14, 3.14, num=shape[2]).reshape(1,1,shape[2],1))
        phi2 = phi1.copy()
        A = np.zeros((4, 8, 16, 16), dtype=np.complex128)
        snapshots.append((tau, phi1, phi2, A))
    
    output_diagnostics_csv(snapshots)
    print("Diagnostics CSV written.")
