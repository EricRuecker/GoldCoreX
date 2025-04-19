import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, Bloch, expect
from matplotlib import animation
from scipy.constants import hbar, e, h

# --------------------------------------------
# Gold Atomic Transition Parameters
# --------------------------------------------
E_eV = 5.11  # 6s -> 6p transition in eV
omega_0 = 2 * np.pi * (E_eV * e) / h  # transition freq in rad/s

# Assume we're driving ON resonance: Δ = 0
delta = 0

# Rabi frequency from E-field × dipole moment
E0 = 1e6  # V/m (laser field)
mu = 1.2e-29  # C·m (dipole moment estimate)
rabi_max = (mu * E0) / hbar  # in rad/s

# Time settings
t_max = 100e-12  # 100 ps
num_points = 400
t = np.linspace(0, t_max, num_points)

# Gaussian pulse envelope Ω(t)
pulse_center = t_max / 2
pulse_width = 10e-12  # 10 ps
Omega_t = rabi_max * np.exp(-((t - pulse_center)**2) / (2 * pulse_width**2))

# Interpolated Ω(t) for mesolve
A_interp = lambda t_val: np.interp(t_val, t, Omega_t)

# Time-dependent Hamiltonian (RWA): H = Δ/2 σ_z + Ω(t)/2 σ_x
H0 = 0.5 * delta * sigmaz()
H1 = [0.5 * sigmax(), A_interp]
H = [H0, H1]

# Initial state |0⟩ = ground (6s)
psi0 = basis(2, 0)

# Solve dynamics
result = mesolve(H, psi0, t)

# Extract Bloch components
x_vals = [expect(sigmax(), state) for state in result.states]
y_vals = [expect(sigmay(), state) for state in result.states]
z_vals = [expect(sigmaz(), state) for state in result.states]

# --------------------------------------------
# Bloch Sphere Animation
# --------------------------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
b = Bloch(fig=fig, axes=ax)

def update_bloch(n):
    b.clear()
    b.add_vectors([x_vals[n], y_vals[n], z_vals[n]])
    if n > 0:
        b.add_points([x_vals[:n], y_vals[:n], z_vals[:n]])
    b.make_sphere()
    return b.fig,

ani = animation.FuncAnimation(fig, update_bloch, frames=len(t), blit=False)
ani.save("gold_atom_rwa_bloch_flip.mp4", fps=30, dpi=150)
print("✅ Gold atom RWA simulation saved: gold_atom_rwa_bloch_flip.mp4")
