import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmaz, mesolve
from scipy.interpolate import interp1d

# ----------------------------
# Constants and Parameters
# ----------------------------
hbar = 1.055e-34  # Reduced Planck's constant (J·s)
e = 1.602e-19     # Charge (C)

E_ev = 3.5
E_joule = E_ev * e
omega_0 = E_joule / hbar
omega_0_thz = omega_0 / 1e12

# Time array (ps)
t_max = 6
num_points = 2000
t = np.linspace(0, t_max, num_points)

# ----------------------------
# Rabi Pulse Parameters (cleaned)
# ----------------------------
pulse_center = 3.0             # pulse center (ps)
pulse_width = 2.0              # wider smoother pulse
pulse_amplitude = 2 * np.pi * 5000  # 5,000 THz coupling

envelope = pulse_amplitude * np.exp(-((t - pulse_center)**2) / (2 * pulse_width**2))
A_interp = interp1d(t, envelope, kind='cubic', fill_value="extrapolate")

def rabi_envelope(t_val, args=None):
    return float(A_interp(t_val))

# ----------------------------
# Hamiltonian and Initial State
# ----------------------------
H0 = 0.5 * 2 * np.pi * omega_0_thz * sigmaz()
H1 = [sigmax(), rabi_envelope]
H = [H0, H1]

psi0 = basis(2, 0)

# Solve evolution
result = mesolve(H, psi0, t, e_ops=[])

# Extract state probabilities
P0 = [abs(state.overlap(basis(2, 0)))**2 for state in result.states]
P1_raw = [abs(state.overlap(basis(2, 1)))**2 for state in result.states]

# Optional smoothing
def smooth(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

P1 = smooth(P1_raw, window_size=10)

# ----------------------------
# Plot results
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, P0, label="P(|0⟩)", linestyle='--', alpha=0.8)
plt.plot(t, P1, label="P(|1⟩)", linewidth=2)
plt.xlabel("Time (ps)")
plt.ylabel("Probability")
plt.title("Cleaned 5d Qubit Flip: Ω = 5,000 THz, Width = 2 ps, E = 3.5 eV")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
