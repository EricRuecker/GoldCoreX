import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, mesolve

# === Constants ===
hbar = 1.0545718e-34
eV = 1.60218e-19
base_delta_E_eV = 2.40
qubit_count = 100000
tlist = np.linspace(0, 6e-13, 10000)  # ~1 minute evolution scale in simulation units

# === Simulation Parameters ===
pulse_jitter = 0.05
amplitude_drift = 0.02
detection_noise_sigma = 0.015
apply_refresh = True

# === Solver options as dictionary (for QuTiP 5+) ===
solver_options = {"nsteps": 400000}

# === Run Simulation ===
flip_data = []
final_zs = []

for q in range(qubit_count):
    try:
        delta_E = base_delta_E_eV + np.random.uniform(-0.01, 0.01)
        jitter_factor = 1 + np.random.uniform(-pulse_jitter, pulse_jitter)
        tlist_jittered = tlist * jitter_factor
        drift_factor = 1 + np.random.uniform(-amplitude_drift, amplitude_drift)
        omega = delta_E * eV / hbar * drift_factor

        psi0 = basis(2, 0)
        H = 0.5 * omega * sigmax()
        result = mesolve(H, psi0, tlist_jittered, [], [sigmax(), sigmay(), sigmaz()], options=solver_options)
        x, y, z = result.expect

        if apply_refresh:
            theta = np.pi / 2
            y_new = y[-1] * np.cos(theta) - z[-1] * np.sin(theta)
            z_new = y[-1] * np.sin(theta) + z[-1] * np.cos(theta)
            z_final = z_new
        else:
            z_final = z[-1]

        z_final += np.random.normal(0, detection_noise_sigma)
        z_final = np.clip(z_final, -1, 1)

        vec_length = np.sqrt(x[-1]**2 + y[-1]**2 + z[-1]**2)
        print(f"Qubit {q+1} final vector length: {vec_length:.3f}")

        flip_data.append((x, y, z, delta_E))
        final_zs.append(z_final)

    except Exception as e:
        print(f"❌ Qubit {q+1} failed: {e}")

print(f"✅ Qubits processed: {len(final_zs)}")