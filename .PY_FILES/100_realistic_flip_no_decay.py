import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, Bloch

# === Constants
hbar = 1.0545718e-34
eV = 1.60218e-19
base_delta_E_eV = 2.40
qubit_count = 100
tlist = np.linspace(0, 1.2e-14, 150)  # Shorter time, fast overnight render

# === Simulation Parameters
pulse_jitter = 0.05
amplitude_drift = 0.02
decoherence_rate = 0.015
detection_noise_sigma = 0.015
apply_refresh = True

# === Run Simulation
flip_data = []
final_zs = []

for q in range(qubit_count):
    try:
        delta_E = base_delta_E_eV + np.random.uniform(-0.01, 0.01)
        jitter_factor = 1 + np.random.uniform(-pulse_jitter, pulse_jitter)
        tlist_jittered = tlist * jitter_factor
        drift_factor = 1 + np.random.uniform(-amplitude_drift, amplitude_drift)
        omega = delta_E * eV / hbar * drift_factor

        psi0 = basis(2, 0)
        H = 0.5 * omega * sigmax()
        result = mesolve(H, psi0, tlist_jittered, [], [sigmax(), sigmay(), sigmaz()])
        x, y, z = result.expect

        decay = np.exp(-decoherence_rate * np.arange(len(tlist)))
     

        if apply_refresh:
            theta = np.pi / 2
            y_new = y[-1] * np.cos(theta) - z[-1] * np.sin(theta)
            z_new = y[-1] * np.sin(theta) + z[-1] * np.cos(theta)
            z_final = z_new
        else:
            z_final = z[-1]

        z_final += np.random.normal(0, detection_noise_sigma)
        z_final = np.clip(z_final, -1, 1)

        flip_data.append((x, y, z, delta_E))
        final_zs.append(z_final)

    except Exception as e:
        print(f"❌ Qubit {q+1} failed: {e}")

print(f"✅ Qubits processed: {len(final_zs)}")

# === Layout
cols = 10
rows = 10
fig = plt.figure(figsize=(cols * 2.0, rows * 2.0))
gs = gridspec.GridSpec(rows, cols)
blochs = []

for i in range(qubit_count):
    ax = fig.add_subplot(gs[i // cols, i % cols], projection='3d')
    bloch = Bloch(fig=fig, axes=ax)
    bloch.vector_color = ['g']
    bloch.point_color = ['b']
    bloch.point_marker = ['o']
    bloch.ylims = [-1, 1]
    bloch.xlims = [-1, 1]
    bloch.zlims = [-1, 1]
    bloch.title = f"Qubit {i+1}\nFinal Z: {final_zs[i]:.2f}"
    blochs.append(bloch)

# === Animate
def update_all(n):
    for i, bloch in enumerate(blochs):
        x, y, z, _ = flip_data[i]
        bloch.clear()
        bloch.add_vectors([x[n], y[n], z[n]])
        bloch.make_sphere()
    return fig,

ani = animation.FuncAnimation(fig, update_all, frames=len(tlist), blit=False)
ani.save("goldcorex_100qubit_realistic_flip.mp4", fps=30, dpi=100)
print("✅ Saved: goldcorex_100qubit_realistic_flip.mp4")
