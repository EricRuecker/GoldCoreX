import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, expect

# === Parameters ===
omega_0 = 2 * np.pi * 15.0
t_max = 10
num_points = 300
t = np.linspace(0, t_max, num_points)

# === Pulse ===
pulse_center = 5.0
pulse_width = 0.7
pulse_amplitude = 6.0
omega_drive = omega_0

gaussian_pulse = pulse_amplitude * np.exp(-((t - pulse_center)**2) / (2 * pulse_width**2))
A_interp = lambda tau: np.interp(tau, t, gaussian_pulse)
def drive(tau, args=None):
    return A_interp(tau) * np.cos(omega_drive * tau)

# === Hamiltonian ===
H0 = 0.5 * omega_0 * sigmaz()
H1 = [sigmax(), drive]
H = [H0, H1]

# === Initial State ===
psi0 = basis(2, 0)

# === Solve Time Evolution ===
result = mesolve(H, psi0, t)

# === Bloch Coordinates ===
x_vals = [expect(sigmax(), psi) for psi in result.states]
y_vals = [expect(sigmay(), psi) for psi in result.states]
z_vals = [expect(sigmaz(), psi) for psi in result.states]

# === Manual Matplotlib Plot ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals, label="GoldCoreX Bloch Path", lw=2)
ax.scatter([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], color='r', label='Final State', s=50)

# Sphere boundary for aesthetics
u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
x_sphere = np.cos(u)*np.sin(v)
y_sphere = np.sin(u)*np.sin(v)
z_sphere = np.cos(v)
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="lightgray", alpha=0.3)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("GoldCoreX Qubit Flip via THz Pulse")
ax.legend()
plt.tight_layout()
plt.show()
