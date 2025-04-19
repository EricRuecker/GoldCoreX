from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Constants ===
hbar = 1.0545718e-34
eV = 1.60218e-19
delta_E_eV = 2.33
omega = delta_E_eV * eV / hbar

# === Initial state: 5d
psi0 = basis(2, 0)

# === Time list
tlist = np.linspace(0, 2e-14, 500)

# === Hamiltonian: green photon pulse
H = 0.5 * omega * sigmax()

# === Solve time evolution
result = mesolve(H, psi0, tlist, [], [sigmax(), sigmay(), sigmaz()])
x, y, z = result.expect

# === Set up Bloch sphere manually
fig = plt.figure(figsize=(6, 6))
b = Bloch(fig=fig)
b.vector_color = ['r']
b.point_color = ['b']
b.point_marker = ['o']

# === Animation update
def animate(i):
    b.clear()
    b.add_vectors([x[i], y[i], z[i]])
    if i > 0:
        b.add_points([x[:i], y[:i], z[:i]])
    b.render()  # Render only after data is added
    return b.fig,



# === Create and save animation
ani = animation.FuncAnimation(fig, animate, frames=len(tlist), interval=50, blit=False)
ani.save("gold_atom_5d_to_6s_flip.mp4", fps=30, dpi=150)

print("✅ MP4 saved successfully.")
