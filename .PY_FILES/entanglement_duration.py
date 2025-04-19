import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# -------------------------------
# GoldCoreX Entanglement Duration Simulation
# -------------------------------

# 1. Define basis states and Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
zero = basis(2, 0)
one = basis(2, 1)
bell_phi_plus = (tensor(zero, zero) + tensor(one, one)).unit()

# 2. Initial density matrix
rho0 = bell_phi_plus.proj()  # dims: [[2,2],[2,2]]

# 3. Time axis (0 to 10 μs, 100 steps)
times = np.linspace(0, 10, 100)

# 4. No Hamiltonian evolution (still define in composite space)
H = Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]])

# 5. Collapse operators — now stay in [[2,2],[2,2]] space
gamma_phi = 0.2  # phase damping rate
gamma_amp = 0.1  # amplitude damping rate
sm = destroy(2)

L1 = np.sqrt(gamma_phi) * tensor(sigmaz(), qeye(2))  # phase noise on qubit A
L2 = np.sqrt(gamma_phi) * tensor(qeye(2), sigmaz())  # phase noise on qubit B
L3 = np.sqrt(gamma_amp) * tensor(sm, qeye(2))        # amplitude on qubit A
L4 = np.sqrt(gamma_amp) * tensor(qeye(2), sm)        # amplitude on qubit B

c_ops = [L1, L2, L3, L4]

# 6. Solve master equation
result = mesolve(H, rho0, times, c_ops)

# 7. Measure entanglement over time
concurrences = []
entropies_A = []

for rho_t in result.states:
    rho_A = rho_t.ptrace(0)
    entropies_A.append(entropy_vn(rho_A))
    conc = concurrence(rho_t)
    concurrences.append(np.real_if_close(conc))

# 8. Plotting
plt.figure(figsize=(10, 5))
plt.plot(times, concurrences, label="Concurrence (Entanglement)")
plt.plot(times, entropies_A, label="Von Neumann Entropy (Qubit A)")
plt.xlabel("Time (μs)")
plt.ylabel("Entanglement Measures")
plt.title("GoldCoreX Entanglement Duration under Decoherence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


