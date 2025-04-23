import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, Bloch, Qobj

# === Constants
hbar = 1.0545718e-34
eV = 1.60218e-19
base_delta_E_eV = 2.40
qubit_count = 50
tlist = np.linspace(0, 2e-14, 250)

# === Simulation Parameters
pulse_jitter = 0.05  # up to ±5% timing variation
amplitude_drift = 0.02  # ±2% amplitude drift
decoherence_rate = 0.015  # decay rate applied to expectation values
detection_noise_sigma = 0.015  # Gaussian noise added to final measurement
apply_refresh = True  # Toggle refresh correction

# === Simulate 50 gold atom flips with realistic noise

flip_data = []
final_zs = []

for q in range(qubit_count):
    try:
        # === Energy variation
        delta_E = base_delta_E_eV + np.random.uniform(-0.01, 0.01)
        
        # === Timing jitter
        jitter_factor = 1 + np.random.uniform(-pulse_jitter, pulse_jitter)
        tlist_jittered = tlist * jitter_factor

        # === Amplitude drift
        drift_factor = 1 + np.random.uniform(-amplitude_drift, amplitude_drift)
        omega = delta_E * eV / hbar * drift_factor

        # === Hamiltonian and solve
        psi0 = basis(2, 0)
        H = 0.5 * omega * sigmax()
        result = mesolve(H, psi0, tlist_jittered, [], e_ops=[sigmax(), sigmay(), sigmaz()])
        x, y, z = result.expect

        # === Decoherence
        decay = np.exp(-decoherence_rate * np.arange(len(tlist)))
        x *= decay
        y *= decay
        z *= decay

        # === Refresh correction
        if apply_refresh:
            theta = np.pi / 2
            y_new = y[-1] * np.cos(theta) - z[-1] * np.sin(theta)
            z_new = y[-1] * np.sin(theta) + z[-1] * np.cos(theta)
            z_final = z_new
        else:
            z_final = z[-1]

        # === Detection noise
        z_final += np.random.normal(0, detection_noise_sigma)
        z_final = np.clip(z_final, -1, 1)

        # === Store
        flip_data.append((x, y, z, delta_E))
        final_zs.append(z_final)

    except Exception as e:
        print(f"❌ Qubit {q+1} failed: {e}")


print(f"✅ Qubits processed: {len(final_zs)}")

# === Layout for 50 Bloch spheres
cols = 10
rows = 5
fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))
gs = gridspec.GridSpec(rows, cols)
blochs = []

for i in range(qubit_count):
    ax = fig.add_subplot(gs[i // cols, i % cols], projection='3d')
    bloch = Bloch(fig=fig, axes=ax)
    bloch.vector_color = ['g']
    bloch.point_color = ['b']
    bloch.point_marker = ['o']
    bloch.ylims = [-1, 1]
    bloch.xlims = [-1, 1]
    bloch.zlims = [-1, 1]
    bloch.title = f"Qubit {i+1}\nFinal Z: {final_zs[i]:.2f}"
    blochs.append(bloch)

# === Animation update
def update_all(n):
    for i, bloch in enumerate(blochs):
        x, y, z, _ = flip_data[i]
        bloch.clear()
        bloch.add_vectors([x[n], y[n], z[n]])
        bloch.make_sphere()
    return fig,

ani = animation.FuncAnimation(fig, update_all, frames=len(tlist), blit=False)
ani.save("goldcorex_50qubit_realistic_flip.mp4", fps=30, dpi=120)
print("✅ Saved: goldcorex_50qubit_realistic_flip.mp4")
