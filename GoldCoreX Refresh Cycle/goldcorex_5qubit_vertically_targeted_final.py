import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, Bloch
from qutip.solver import Options

# === Constants
hbar = 1.0545718e-34
eV = 1.60218e-19
delta_E_eV = 2.40
omega = delta_E_eV * eV / hbar

# === Time settings
steps = 160
t_flip = np.linspace(0, 2e-14, steps)
t_refresh = np.linspace(0, 7e-14, steps)

# === Solver options
opts = Options(store_states=True)

# === Initial state
psi0 = basis(2, 0)

# === Vertical photon targeting simulation
def photon_targeting_success():
    return np.random.rand() < 0.98

# === Flip and refresh simulation for each qubit
qubit_data = []
final_z_values = []

for i in range(5):
    if not photon_targeting_success():
        qubit_data.append(None)
        final_z_values.append(None)
        continue

    jitter_std = 0.0015 * omega
    omega_jittered = omega + np.random.normal(0, jitter_std)

    H_flip = 0.5 * omega_jittered * sigmax()
    H_refresh = 0.85 * omega * sigmax()
    H_correction = 0.2 * omega * sigmax()

    result1 = mesolve(H_flip, psi0, t_flip, [], [sigmax(), sigmay(), sigmaz()], options=opts)
    psi_after1 = result1.states[-1]

    result2 = mesolve(H_refresh, psi_after1, t_refresh, [], [sigmax(), sigmay(), sigmaz()], options=opts)
    psi_after2 = result2.states[-1]

    result3 = mesolve(H_refresh, psi_after2, t_refresh, [], [sigmax(), sigmay(), sigmaz()], options=opts)
    psi_after3 = result3.states[-1]

    result4 = mesolve(H_correction, psi_after3, t_refresh, [], [sigmax(), sigmay(), sigmaz()], options=opts)

    x = np.concatenate((result1.expect[0], result2.expect[0], result3.expect[0], result4.expect[0]))
    y = np.concatenate((result1.expect[1], result2.expect[1], result3.expect[1], result4.expect[1]))
    z = np.concatenate((result1.expect[2], result2.expect[2], result3.expect[2], result4.expect[2]))

    qubit_data.append((x, y, z))
    final_z_values.append(z[-1])

# === Visualization
fig = plt.figure(figsize=(15, 4))
axes = [fig.add_subplot(1, 5, i + 1, projection='3d') for i in range(5)]
blochs = []

for i, ax in enumerate(axes):
    b = Bloch(fig=fig, axes=ax)
    b.vector_color = ['g']
    b.point_color = ['b']
    b.add_points([[0], [0], [-1]], meth='s')  # target state
    ax.set_title(f"Qubit {i+1}")
    blochs.append(b)

def update(frame):
    for i, b in enumerate(blochs):
        b.clear()
        data = qubit_data[i]
        if data:
            x, y, z = data
            b.add_points([[x[frame]], [y[frame]], [z[frame]]])
            b.add_vectors([x[frame], y[frame], z[frame]])
        b.add_points([[0], [0], [-1]], meth='s')
        b.render()

ani = FuncAnimation(fig, update, frames=len(t_flip) + 3 * len(t_refresh), interval=100)
ani.save("goldcorex_5qubit_vertically_targeted_final.mp4", fps=10, dpi=150)

# === Final readout
for i, z in enumerate(final_z_values):
    label = f"Qubit {i+1}"
    if z is None:
        print(f"{label}: ❌ Missed target (photon misalignment)")
    elif z < -0.90:
        print(f"{label}: ✅ Refreshed (Z = {z:.3f})")
    else:
        print(f"{label}: ⚠️ Incomplete (Z = {z:.3f})")
