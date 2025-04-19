from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# === Constants ===
hbar = 1.0545718e-34
eV = 1.60218e-19
delta_E_eV = 2.33  # green photon energy
omega = delta_E_eV * eV / hbar

# === System Parameters ===
num_qubits = 10
center = num_qubits // 2
tlist = np.linspace(0, 2e-14, 500)  # same as perfect flip

# === Initial state: center in |0⟩
psi_list = [basis(2, 0) for _ in range(num_qubits)]
psi0 = tensor(psi_list)

# === Operators
sx_list, sm_list = [], []
for i in range(num_qubits):
    op_x = [qeye(2)] * num_qubits
    op_m = [qeye(2)] * num_qubits
    op_x[i] = sigmax()
    op_m[i] = destroy(2)
    sx_list.append(tensor(op_x))
    sm_list.append(tensor(op_m))

# === Hamiltonian: Apply flip to center qubit
H = 0.5 * omega * sx_list[center]

# === Weak Decoherence
gamma = 0.0  # perfect flip: zero decoherence
c_ops = []
for i in range(num_qubits):
    c_ops.append(np.sqrt(gamma) * sm_list[i])
    c_ops.append(np.sqrt(gamma / 2) * sm_list[i].dag() * sm_list[i])

# === Observable: center qubit population in |1⟩
proj1 = basis(2, 1) * basis(2, 1).dag()
obs_list = [qeye(2)] * num_qubits
obs_list[center] = proj1
e_ops = [tensor(obs_list)]

# === Solve
print("🧠 Simulating GoldCoreX 10-qubit perfect flip...")
result = mesolve(H, psi0, tlist, c_ops, e_ops)
print("✅ Simulation complete.")

# === Plot
plt.plot(tlist, result.expect[0], label="Center Qubit |1⟩ Population", color='darkgreen')
plt.xlabel("Time (s)")
plt.ylabel("Population")
plt.title("GoldCoreX Perfect Flip (10 Qubits, Green Photon Drive)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
