import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, expect

# === Constants ===
hbar = 1.0545718e-34  # Planck constant (J·s)
eV = 1.60218e-19      # Electron volt to joules

# === Compressed orbital energy gap (increased slightly by pressure) ===
delta_E_eV = 2.40     # Increased from 2.33 eV → orbital compression effect
omega = delta_E_eV * eV / hbar  # Angular frequency (rad/s)

# === Time range (same as original)
tlist = np.linspace(0, 2e-14, 500)  # 0 → 20 fs

# === Initial state
psi0 = basis(2, 0)

# === Hamiltonian: constant RX drive with compressed frequency
H = 0.5 * omega * sigmax()

# === Simulate all 5 GoldCoreX qubits independently
results = []
for i in range(5):
    result = mesolve(H, psi0, tlist, [], [sigmax(), sigmay(), sigmaz()])
    results.append(result)

# === Plotting Bloch trajectories
fig = plt.figure(figsize=(12, 8))
for i, result in enumerate(results):
    x, y, z = result.expect
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    ax.plot(x, y, z, label=f'Qubit {i + 1}', lw=2)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='r', label='Final')

    # Reference Bloch sphere
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x_s = np.cos(u) * np.sin(v)
    y_s = np.sin(u) * np.sin(v)
    z_s = np.cos(v)
    ax.plot_wireframe(x_s, y_s, z_s, color="lightgray", alpha=0.2)

    ax.set_title(f"Qubit {i+1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

fig.suptitle("GoldCoreX 5-Qubit Flip — Orbital Compression Model", fontsize=16)
plt.tight_layout()
plt.show()
