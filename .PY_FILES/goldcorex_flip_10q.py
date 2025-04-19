from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
num_qubits = 10
tlist = np.linspace(0, 1, 200)

# === INITIAL STATE ===
psi_list = [basis(2, 0) for _ in range(num_qubits)]
psi_list[num_qubits // 2] = basis(2, 1)  # center qubit in |1>
psi0 = tensor(psi_list)

# === OPERATORS ===
sx_list = []
sm_list = []
for i in range(num_qubits):
    op_list_x = [qeye(2) for _ in range(num_qubits)]
    op_list_m = [qeye(2) for _ in range(num_qubits)]
    op_list_x[i] = sigmax()
    op_list_m[i] = destroy(2)
    sx_list.append(tensor(op_list_x))
    sm_list.append(tensor(op_list_m))

# === HAMILTONIAN ===
H = 0.5 * sx_list[num_qubits // 2]  # drive center qubit

# === DECOHERENCE ===
gamma = 0.05
c_ops = []
for i in range(num_qubits):
    c_ops.append(np.sqrt(gamma) * sm_list[i])                         # amplitude damping
    c_ops.append(np.sqrt(gamma / 2) * sm_list[i].dag() * sm_list[i]) # dephasing

# === OBSERVABLE ===
proj1 = basis(2, 1) * basis(2, 1).dag()
obs_list = [qeye(2) for _ in range(num_qubits)]
obs_list[num_qubits // 2] = proj1
e_ops = [tensor(obs_list)]

# === SOLVE ===
print("🧠 Running GoldCoreX 10-qubit fast simulation...")
result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=e_ops)
print("✅ Done.")

# === PLOT ===
plt.figure(figsize=(6, 4))
plt.plot(tlist, np.real(result.expect[0]), label='|1⟩ population (center qubit)', color='darkorange')
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("GoldCoreX Fast Flip (10 Qubits)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
