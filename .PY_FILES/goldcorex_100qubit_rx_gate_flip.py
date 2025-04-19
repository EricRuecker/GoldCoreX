import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmaz, sigmax, expect

# === Manually define RX gate (for QuTiP < 5.0)
def rx(theta):
    return (-1j * theta / 2 * sigmax()).expm()

# === Simulation setup
num_qubits = 100
psi0 = basis(2, 0)  # Start in |0⟩

z_final_values = []

# === Flip all 100 qubits with RX(π ± jitter)
for i in range(num_qubits):
    jitter = np.random.normal(loc=1.0, scale=0.01)   # ±1% angle variation
    angle = np.pi * jitter                           # RX(π ± jitter)
    final_state = rx(angle) * psi0                   # Apply RX(θ)|0⟩
    z_final = expect(sigmaz(), final_state)          # Measure Z
    z_final_values.append(z_final)

# === Histogram of final Z projections
plt.figure(figsize=(9, 6))
plt.hist(z_final_values, bins=30, color='indigo', edgecolor='black')
plt.title("GoldCoreX 100-Qubit RX(π) Flip — Ideal Gate Driven")
plt.xlabel("Final Bloch Z Value")
plt.ylabel("Qubit Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Fidelity report
z_array = np.array(z_final_values)
perfect = np.sum(z_array < -0.90)
print("\n✅ RX Gate Flip Summary:")
print(f"Mean Final Z: {z_array.mean():.4f}")
print(f"Std Dev:      {z_array.std():.4f}")
print(f"{perfect} / 100 qubits reached >90% flip fidelity.")
