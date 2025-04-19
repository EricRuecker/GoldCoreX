import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, Bloch

# === Constants
hbar = 1.0545718e-34
eV = 1.60218e-19

# === Energy gap with orbital compression
delta_E_eV = 2.40
omega = delta_E_eV * eV / hbar

# === Time range
tlist = np.linspace(0, 2e-14, 500)

# === Initial state and Hamiltonian
psi0 = basis(2, 0)
H = 0.5 * omega * sigmax()

# === Solve time evolution
result = mesolve(H, psi0, tlist, [], [sigmax(), sigmay(), sigmaz()])
x, y, z = result.expect

# === Create Bloch sphere
fig = plt.figure(figsize=(6, 6))
bloch = Bloch(fig=fig)
bloch.vector_color = ['g']
bloch.point_color = ['b']
bloch.point_marker = ['o']
bloch.ylims = [-1, 1]
bloch.xlims = [-1, 1]
bloch.zlims = [-1, 1]

def update_bloch(n):
    bloch.clear()
    bloch.add_vectors([x[n], y[n], z[n]])
    pts = np.array([x[:n+1], y[:n+1], z[:n+1]])  # shape (3, N)
    bloch.add_points(pts)
    bloch.make_sphere()
    return bloch.fig,

ani = animation.FuncAnimation(fig, update_bloch, frames=len(tlist), blit=False)

# === Save animation
ani.save("gold_atom_5d_to_6s_flip.mp4", fps=30, dpi=200)

print("✅ MP4 saved as: gold_atom_5d_to_6s_flip.mp4")
