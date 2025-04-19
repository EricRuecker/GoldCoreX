import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, Bloch, expect
from matplotlib import animation
from scipy.constants import hbar, e, h

# --------------------------------------------
# Gold Atom: 5d → 6s driven by 2.3 eV green photon
# --------------------------------------------
E_eV = 2.3  # photon energy
omega_0 = 2 * np.pi * (E_eV * e) / h  # transition frequency in rad/s

# Dipole interaction
E0 = 1e8  # V/m (strong field for flipping)
mu = 1.2e-29  # dipole moment (C·m)
rabi_peak = (mu * E0) / hbar  # Rabi frequency in rad/s

# Time array
t_max = 400e-15  # 400 fs
num_points = 3000
t = np.linspace(0, t_max, num_points)

# Gaussian envelope Ω(t)
pulse_center = t_max / 2
pulse_width = 80e-15
envelope = rabi_peak * np.exp(-((t - pulse_center)**2) / (2 * pulse_width**2))
A_interp = lambda t_val: np.interp(t_val, t, envelope)

# Full drive function (no RWA)
def drive_amplitude(t_val, args=None):
    return float(A_interp(t_val) * np.cos(omega_0 * t_val))

# Hamiltonian: H = (ω₀/2)σz + Ω(t)·cos(ωt)·σx
H0 = 0.5 * omega_0 * sigmaz()
H1 = [sigmax(), drive_amplitude]
H = [H0, H1]

# Initial state: |0⟩ (5d)
psi0 = basis(2, 0)

# Solve time evolution
result = mesolve(H, psi0, t)

# Compute Bloch vector components
x_vals = [expect(sigmax(), state) for state in result.states]
y_vals = [expect(sigmay(), state) for state in result.states]
z_vals = [expect(sigmaz(), state) for state in result.states]

# --------------------------------------------
# Diagnostic plots
# --------------------------------------------
plt.figure(figsize=(10, 4))

# Drive field
plt.subplot(1, 2, 1)
plt.plot(t * 1e15, envelope * np.cos(omega_0 * t))
plt.title("Drive Field: Ω(t)·cos(ωt)")
plt.xlabel("Time (fs)")
plt.ylabel("Amplitude")

# Z axis
plt.subplot(1, 2, 2)
plt.plot(t * 1e15, z_vals)
plt.title("Z-axis (Quantum State Flip)")
plt.xlabel("Time (fs)")
plt.ylabel("Z (⟨σ_z⟩)")

plt.tight_layout()
plt.show()

# --------------------------------------------
# Bloch sphere animation
# --------------------------------------------
fig = plt.figure(figsize=(6, 6))
plt.tight_layout()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
b = Bloch(fig=fig, axes=ax)

def update_bloch(n):
    b.clear()
    b.vector_color = ['r']
    b.point_color = ['b']
    b.add_vectors([x_vals[n], y_vals[n], z_vals[n]])
    if n > 0:
        b.add_points([x_vals[:n], y_vals[:n], z_vals[:n]])
    b.make_sphere()
    return b.fig,

ani = animation.FuncAnimation(fig, update_bloch, frames=len(t), blit=False)
ani.save("gold_atom_green_photon_flip_success.mp4", fps=10, dpi=150)
print("✅ Saved animation: gold_atom_green_photon_flip_success.mp4")
