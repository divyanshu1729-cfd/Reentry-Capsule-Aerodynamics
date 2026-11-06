import numpy as np
import matplotlib.pyplot as plt

from taylor_maccoll import find_shock_angle

# Cone HALF angles (deg)
theta_c = np.arange(5,41,2)

theta_s = []

print()
print("="*60)
print("THETA-BETA VALIDATION")
print("="*60)
print()

print(f"{'Cone':>8} {'Shock':>10}")
print("-"*22)

for tc in theta_c:

    beta = find_shock_angle(
        mach=5.0,
        theta_c_deg=tc
    )

    theta_s.append(beta)

    print(f"{tc:8.2f} {beta:10.4f}")

plt.figure(figsize=(6,6))

plt.plot(
    theta_c,
    theta_s,
    '-o',
    linewidth=2,
    markersize=4
)

plt.xlim(0,40)
plt.ylim(10,55)

plt.xticks(np.arange(0,41,5))
plt.yticks(np.arange(10,56,5))

plt.xlabel("Cone half-angle θc (deg)")
plt.ylabel("Shock half-angle θs (deg)")

plt.grid(True)

plt.tight_layout()
