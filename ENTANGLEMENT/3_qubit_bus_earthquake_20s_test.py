import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.qip.operations import rx, rz

def fidelity(target_state, rho):
    if target_state.isket:
        target_state = target_state * target_state.dag()
    return np.real((target_state.sqrtm() * rho * target_state.sqrtm()).sqrtm().tr() ** 2)

# Constants
hbar = 1.0545718e-34
eV = 1.60218e-19
delta_E_eV = 2.40
omega_base = delta_E_eV * eV / hbar
t_total = 2e-13  # 20 seconds in sim-time
steps = 1000
tlist = np.linspace(0, t_total, steps)

# Operators
sx, sz, I = sigmax(), sigmaz(), qeye(2)

# GHZ Initial State
initial_state = (tensor(basis(2,0),basis(2,0),basis(2,0),basis(2,0)) +
                 tensor(basis(2,1),basis(2,1),basis(2,1),basis(2,1))).unit()

# Collapse Operators (extra decoherence on Q0)
T1, T2 = 10e-3, 5e-3
gamma_relax = 1.0 / T1
gamma_deph = 1.0 / T2
c_ops = []

for i in range(4):
    ops_r = [I]*4
    ops_r[i] = destroy(2)
    decay_factor = 3 if i == 0 else 1
    c_ops.append(np.sqrt(decay_factor * gamma_relax) * tensor(ops_r))

    ops_d = [I]*4
    ops_d[i] = sz
    c_ops.append(np.sqrt(decay_factor * gamma_deph) * tensor(ops_d))

# Refresh Function (with noise)
def apply_refresh(state):
    delta = np.pi + np.random.uniform(-0.15, 0.15)
    refresh_gate = rx(delta)
    refreshed = tensor(refresh_gate, I, I, I) * state

    rand_q = np.random.choice([1, 2, 3])
    theta = np.random.uniform(-0.2, 0.2)
    rz_gate = rz(theta)
    gates = [I, I, I, I]
    gates[rand_q] = rz_gate
    return tensor(*gates) * refreshed.unit()

# Simulation Run
refresh_interval = int(steps / 10)
all_fidelities = []

omega_drift = omega_base + np.random.uniform(-0.01, 0.01) * eV / hbar
H = 0.5 * omega_drift * (
    tensor(sx, I, I, I) +
    tensor(I, sx, I, I) +
    tensor(I, I, sx, I) +
    tensor(I, I, I, sx)
)
result = mesolve(H, initial_state, tlist, c_ops)
states = result.states

for i in range(1, 11):
    jitter = np.random.randint(-3, 4)
    refresh_time = min(refresh_interval * i + jitter, len(states) - 1)
    refreshed_state = apply_refresh(states[refresh_time])

    omega_shifted = omega_base + np.random.uniform(-0.02, 0.02) * eV / hbar
    H_shifted = 0.5 * omega_shifted * (
        tensor(sx, I, I, I) +
        tensor(I, sx, I, I) +
        tensor(I, I, sx, I) +
        tensor(I, I, I, sx)
    )
    result_post = mesolve(H_shifted, refreshed_state, tlist, c_ops)
    states_post = result_post.states

    fidelities = [fidelity(initial_state, rho) for rho in states[:refresh_time]]
    fidelities += [fidelity(initial_state, rho) for rho in states_post]
    all_fidelities.append(fidelities[:len(tlist)])

# Plot
plt.figure(figsize=(10, 5))
for i, fid in enumerate(all_fidelities):
    plt.plot(tlist, fid, label=f'Fidelity after refresh {i+1}')

plt.axvline(x=tlist[refresh_interval], color='gray', linestyle='--', label="🔁 First Refresh")
plt.xlabel("Time (s)")
plt.ylabel("Fidelity to 4-Qubit GHZ")
plt.title("GoldCoreX: Immortal Bus Extended Run (20s Sim Time)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig("goldcorex_immortal_bus_extended_20s.png", dpi=200)
plt.show()