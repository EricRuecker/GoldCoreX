import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, destroy

# === Constants ===
hbar = 1.0545718e-34
eV = 1.60218e-19
base_delta_E_eV = 2.40
qubit_count = 100
tlist = np.linspace(0, 2e-13, 500)  # longer run with more steps

# === Noise Parameters ===
pulse_jitter = 0.30  # extreme jitter
amplitude_drift = 0.30  # extreme drift
detection_noise_sigma = 0.05  # stronger measurement noise
apply_refresh = True

# === Physical decoherence (realistic)
T1 = 1e-13  # amplitude damping time (fast decay)
T2 = 5e-14  # dephasing time (short coherence)

# === Collapse operators for Lindblad noise ===
c_ops = []
if T1 > 0.0:
    c_ops.append(np.sqrt(1 / T1) * destroy(2))
if T2 > 0.0:
    c_ops.append(np.sqrt(1 / T2) * sigmaz())

# === Solver options ===
solver_options = {"nsteps": 10000}

# === Simulation ===
flip_data = []
final_zs = []

for q in range(qubit_count):
    try:
        delta_E = base_delta_E_eV + np.random.uniform(-0.02, 0.02)
        jitter_factor = 1 + np.random.uniform(-pulse_jitter, pulse_jitter)
        tlist_jittered = tlist * jitter_factor
        drift_factor = 1 + np.random.uniform(-amplitude_drift, amplitude_drift)
        omega = delta_E * eV / hbar * drift_factor

        psi0 = basis(2, 0)
        H = 0.5 * omega * sigmax()
        result = mesolve(H, psi0, tlist_jittered, c_ops=c_ops, e_ops=[sigmax(), sigmay(), sigmaz()], options=solver_options)
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
        print(f"Qubit {q+1:3d} | Final Z: {z_final:+.3f} | Vector length: {vec_length:.3f}")

        flip_data.append((x, y, z, delta_E))
        final_zs.append(z_final)

    except Exception as e:
        print(f"❌ Qubit {q+1} failed: {e}")

print(f"✅ Realistic noisy sim complete. Qubits processed: {len(final_zs)}")