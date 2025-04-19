from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Simulation parameters
N = 100
T = 100e-6
dt = T / N
tlist = np.linspace(0, T, N)
refresh_interval = 5
refresh_start = 20  # ⏳ Delay refresh until entanglement has begun forming

# === Initial state: |10⟩ — qubit A excited, B ground
psi0 = tensor(basis(2, 1), basis(2, 0))
rho = ket2dm(psi0)

# === Target Bell state (for refresh stabilization)
target_bell = (tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
rho_target = ket2dm(target_bell)

# === Operators
sx = sigmax()
sz = sigmaz()
sm = destroy(2)
I = qeye(2)

# === Decoherence
gamma_dephase = 5e3
gamma_relax = 2e3
c_ops = [
    np.sqrt(gamma_dephase) * tensor(sz, I),
    np.sqrt(gamma_dephase) * tensor(I, sz),
    np.sqrt(gamma_relax) * tensor(sm, I),
    np.sqrt(gamma_relax) * tensor(I, sm),
]

# === No Hamiltonian
H = 0 * tensor(sx, sx)

# === Bloch vector trackers
vecs_A = []
vecs_B = []

for i in range(N):
    # Evolve
    result = mesolve(H, rho, [0, dt], c_ops, [])
    rho = result.states[-1]

    # Refresh (only after refresh_start)
    if i >= refresh_start and i % refresh_interval == 0:
        rho = 0.8 * rho + 0.2 * rho_target

    # Bloch vectors
    rho_A = rho.ptrace(0)
    rho_B = rho.ptrace(1)
    vecs_A.append([expect(pauli, rho_A) for pauli in [sx, sigmay(), sz]])
    vecs_B.append([expect(pauli, rho_B) for pauli in [sx, sigmay(), sz]])

# === Animation setup
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
bloch_A = Bloch(fig=fig, axes=ax1)
bloch_B = Bloch(fig=fig, axes=ax2)
bloch_A.point_color = ['blue']
bloch_B.point_color = ['green']
bloch_A.vector_color = ['r']
bloch_B.vector_color = ['r']

def animate(i):
    bloch_A.clear()
    bloch_B.clear()
    bloch_A.add_vectors(vecs_A[i])
    bloch_B.add_vectors(vecs_B[i])
    if i > 0:
        bloch_A.add_points(np.array(vecs_A[:i+1]).T)
        bloch_B.add_points(np.array(vecs_B[:i+1]).T)
    bloch_A.make_sphere()
    bloch_B.make_sphere()
    ax1.set_title("Qubit A (Gold Atom A)")
    ax2.set_title("Qubit B (Gold Atom B)")
    return fig,

ani = animation.FuncAnimation(fig, animate, frames=N, interval=80, blit=False)
ani.save("goldcorex_bloch_entanglement_formation.mp4", fps=20, dpi=150)

print("✅ Bloch animation saved: goldcorex_bloch_entanglement_formation.mp4")
