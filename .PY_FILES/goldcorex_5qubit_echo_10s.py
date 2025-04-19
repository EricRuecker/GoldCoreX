from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
num_qubits = 5
t_end = 10.0
t_steps = 1000
tlist = np.linspace(0, t_end, t_steps)
omega = 2 * np.pi  # 1 flip per second

# === Initial state
psi0 = tensor([basis(2, 0) for _ in range(num_qubits)])

# === Operators
sx_list, sm_list, sz_list = [], [], []
for i in range(num_qubits):
    ops_x = [qeye(2)] * num_qubits
    ops_m = [qeye(2)] * num_qubits
    ops_z = [qeye(2)] * num_qubits
    ops_x[i] = sigmax()
    ops_m[i] = destroy(2)
    ops_z[i] = sigmaz()
    sx_list.append(tensor(ops_x))
    sm_list.append(tensor(ops_m))
    sz_list.append(tensor(ops_z))

# === Hamiltonian: global X drive
H_base = 0.5 * omega * sum(sx_list)

# === Periodic π Z-pulses every 50 ms
def pi_pulse(t, args):
    return 1.0 if int(t * 1000) % 100 == 0 else 0.0

H_total = [H_base] + [[sz, pi_pulse] for sz in sz_list]  # add Z rotation pulses

# === Collapse ops: decoherence
gamma = 0.1
c_ops = []
for sm in sm_list:
    c_ops.append(np.sqrt(gamma) * sm)                            # T1
    c_ops.append(np.sqrt(gamma / 2) * sm.dag() * sm)            # T2

# === Observables: |1⟩ population
proj1 = basis(2, 1) * basis(2, 1).dag()
e_ops = []
for i in range(num_qubits):
    op = [qeye(2)] * num_qubits
    op[i] = proj1
    e_ops.append(tensor(op))

# === Solve
print("🧠 Running GoldCoreX 5-qubit echo stabilization for 10s...")
result = mesolve(H_total, psi0, tlist, c_ops=c_ops, e_ops=e_ops)
print("✅ Simulation complete.")

# === Plot
plt.figure(figsize=(10, 6))
for i in range(num_qubits):
    plt.plot(tlist, result.expect[i], label=f"Qubit {i}")

plt.xlabel("Time (s)")
plt.ylabel("Population in |1⟩")
plt.title("GoldCoreX 5-Qubit Flip with π Pulses (Hahn Echo Style, 10s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
