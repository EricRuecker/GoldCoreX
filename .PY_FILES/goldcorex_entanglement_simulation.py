import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.qip.operations import cnot, hadamard_transform  # ✅ Fix for missing gates

# -------------------------------
# GoldCoreX Entanglement Example
# -------------------------------

# 1. Define basis states for 2 qubits
zero = basis(2, 0)
one = basis(2, 1)
q0 = tensor(zero, zero)  # Start in |00⟩

# 2. Define quantum gates
H = hadamard_transform()  # Hadamard gate (use built-in now)
CNOT = cnot()             # Controlled-NOT gate

# 3. Apply Hadamard to qubit 0, then CNOT to entangle
H_on_q0 = tensor(H, qeye(2))
entangled_state = CNOT * H_on_q0 * q0

# 4. Show the resulting state
print("Entangled Bell State (|Φ+⟩):")
print(entangled_state)

# 5. Calculate density matrix
rho = entangled_state.proj()

# 6. Partial trace to observe reduced state of each qubit
rho_A = rho.ptrace(0)
rho_B = rho.ptrace(1)

# 7. Entanglement check: Von Neumann entropy
S_A = entropy_vn(rho_A)
S_B = entropy_vn(rho_B)
print(f"\nEntropy Qubit A: {S_A:.4f}")
print(f"Entropy Qubit B: {S_B:.4f}")

# 8. Visualize Bloch spheres of reduced states
b0 = Bloch()
b1 = Bloch()
b0.add_states(rho_A)
b1.add_states(rho_B)

b0.title = "GoldCoreX Qubit A (Reduced State)"
b1.title = "GoldCoreX Qubit B (Reduced State)"
b0.show()
b1.show()

# 9. Visualize entangled state probabilities
fig, ax = plt.subplots()
labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
probs = np.abs(entangled_state.full())**2
ax.bar(labels, probs.flatten())
ax.set_ylabel("Probability")
ax.set_title("GoldCoreX Entangled State Probabilities")
plt.show()


