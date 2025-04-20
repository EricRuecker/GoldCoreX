
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.qip.operations import rx

# === Manual Concurrence and Fidelity Functions ===
def concurrence(rho):
    sy = Qobj([[0, -1j], [1j, 0]])
    Y = tensor(sy, sy)
    rho_tilde = Y * rho.conj() * Y
    R = (rho * rho_tilde).sqrtm()
    evals = np.real(np.sort(R.eigenenergies()))
    return max(0, evals[-1] - sum(evals[:-1]))

def fidelity(target_state, rho):
    if target_state.isket:
        target_state = target_state * target_state.dag()
    return np.real((target_state.sqrtm() * rho * target_state.sqrtm()).sqrtm().tr() ** 2)

# === Constants ===
hbar = 1.0545718e-34
eV = 1.60218e-19
delta_E_eV = 2.40
omega = delta_E_eV * eV / hbar
t_total = 2e-14
steps = 200
tlist = np.linspace(0, t_total, steps)

# === Operators ===
sx, sy, sz = sigmax(), sigmay(), sigmaz()
I = qeye(2)

# === Bell State (|00> + |11>) / sqrt(2)
bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()

# === GoldCoreX Hamiltonian ===
H = 0.5 * omega * (tensor(sx, I) + tensor(I, sx))

# === Collapse Operators for decoherence
T1, T2 = 10e-3, 5e-3
gamma_relax = 1.0 / T1
gamma_deph = 1.0 / T2
c_ops = [
    np.sqrt(gamma_relax) * tensor(destroy(2), I),
    np.sqrt(gamma_relax) * tensor(I, destroy(2)),
    np.sqrt(gamma_deph) * tensor(sz, I),
    np.sqrt(gamma_deph) * tensor(I, sz)
]

# === Photon Loss
loss_prob = 0.02
if np.random.rand() < loss_prob:
    print("⚠️ Photon loss occurred — entanglement may degrade.")
    c_ops.append(tensor(qeye(2), destroy(2)))

# === Refresh Cycle Definition
def apply_refresh(state):
    refresh_gate = rx(np.pi)
    refreshed = tensor(refresh_gate, refresh_gate) * state
    return refreshed.unit()

# === Run Pre-Refresh Dynamics
result = mesolve(H, bell, tlist, c_ops)
states = result.states

# === Midpoint Refresh
midpoint = len(states) // 2
refreshed_state = apply_refresh(states[midpoint])

# === Post-Refresh Dynamics
result_post = mesolve(H, refreshed_state, tlist, c_ops)
states_post = result_post.states

# === Calculate Concurrence and Fidelity
concurrences = [concurrence(rho) for rho in states[:midpoint]]
fidelities = [fidelity(bell, rho) for rho in states[:midpoint]]
concurrences += [concurrence(rho) for rho in states_post]
fidelities += [fidelity(bell, rho) for rho in states_post]

# === Full time axis
full_t = np.concatenate([tlist[:midpoint], tlist])

# === Plotting
plt.figure(figsize=(10, 5))
plt.plot(full_t, concurrences, label="Concurrence")
plt.plot(full_t, fidelities, label="Fidelity to |Phi+>")
plt.axvline(x=tlist[midpoint], color='gray', linestyle='--', label="🔁 Refresh")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("GoldCoreX: Immortal Entanglement via Refresh Cycle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("immortal_entanglement_result.png", dpi=200)
plt.show()

print("✅ Saved: immortal_entanglement_result.png")
