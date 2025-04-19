from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_vector
from math import pi
import matplotlib.pyplot as plt
import numpy as np

# Build GoldCoreX single-qubit flip circuit
qc = QuantumCircuit(1)
qc.rx(pi, 0)  # Simulate THz pulse that flips |0⟩ → |1⟩

# Get final statevector
state = Statevector.from_instruction(qc)

# Convert to Bloch vector manually
# Bloch: x = Re(α*β), y = Im(α*β), z = |α|² - |β|²
alpha, beta = state.data
x = 2 * np.real(np.conj(alpha) * beta)
y = 2 * np.imag(np.conj(alpha) * beta)
z = np.abs(alpha)**2 - np.abs(beta)**2

# Plot Bloch vector
plot_bloch_vector([x, y, z], title="GoldCoreX Flip Result")
plt.show()
