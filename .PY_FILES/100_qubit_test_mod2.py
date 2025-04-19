import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmaz, mesolve

# === Constants ===
hbar = 1.0545718e-34
eV = 1.60218e-19
delta_E_eV = 2.40
omega = delta_E_eV * eV / hbar

tlist = np.linspace(0, 2e-14, 500)
psi0 = basis(2, 0)

# === New Parameters
detection_efficiency = 0.95  # 5% loss
jitter_std = 0.01            # 1% pulse strength variation

z_final_values = []

for i in range(100):
    jitter = np.random.normal(loc=1.0, scale=jitter_std)
    omega_i = omega * jitter
    H = 0.5 * omega_i * sigmax()

    result = mesolve(H, psi0, tlist, [], [sigmaz()])
    z = result.expect[0][-1]

    if np.random.rand() < detection_efficiency:
        z_final_values.append(z)

# === Plot
plt.figure(figsize=(9, 6))
plt.hist(z_final_values, bins=30, color='darkorange', edgecolor='black')
plt.title("GoldCoreX 100-Qubit Flip — Detection Loss + Jitter")
plt.xlabel("Final Bloch Z Value (Detected Only)")
plt.ylabel("Qubit Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Summary
perfect = np.sum(np.array(z_final_values) < -0.90)
print(f"✅ Perfect flips: {perfect} / 100")
print(f"📉 Undetected (loss): {100 - len(z_final_values)}")
