import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv

# Create theta array from -pi to pi
theta = np.linspace(-np.pi, np.pi, 1000)

# MexicFunc1 = (Exp[kap*Cos[theta]]*(Cos[theta] - kap*Sin[theta]^2))/(2*Pi*BesselI[1, kap]) /. kap -> 1
kap1 = 1
mexic_func1 = (np.exp(kap1 * np.cos(theta)) * (np.cos(theta) - kap1 * np.sin(theta)**2)) / \
              (2 * np.pi * iv(1, kap1))

# MexicFunc2 = (Exp[kap*Cos[theta]]*(Cos[theta] - kap*Sin[theta]^2))/(2*Pi*BesselI[1, kap]) /. kap -> 2
kap2 = 2
mexic_func2 = (np.exp(kap2 * np.cos(theta)) * (np.cos(theta) - kap2 * np.sin(theta)**2)) / \
              (2 * np.pi * iv(1, kap2))

# MexicFunc3 = (Exp[kap*Cos[theta]]*(Cos[theta] - kap*Sin[theta]^2))/(2*Pi*BesselI[1, kap]) /. kap -> 4
kap3 = 4
mexic_func3 = (np.exp(kap3 * np.cos(theta)) * (np.cos(theta) - kap3 * np.sin(theta)**2)) / \
              (2 * np.pi * iv(1, kap3))

# ThreeMexicanHats = Plot[{MexicFunc1, MexicFunc2, MexicFunc3}, {theta, -Pi, Pi}, PlotRange -> All]
plt.figure(figsize=(10, 6))
plt.plot(theta, mexic_func1, label=f'? = {kap1}')
plt.plot(theta, mexic_func2, label=f'? = {kap2}')
plt.plot(theta, mexic_func3, label=f'? = {kap3}')
plt.xlabel('?')
plt.ylabel('Amplitude')
plt.title('Three Mexican Hats')
plt.legend()
plt.grid(True)
plt.xlim(-np.pi, np.pi)
plt.show()

# Table[(BesselI[n, kap]*n^2)/(2*Pi*kap*BesselI[1, kap]) + 0.0, {n, 1, 10}]
# Note: Since 'kap' is undefined in the original table expression, we compute for the three kappa values used above
def compute_bessel_table(kap):
    n_values = np.arange(1, 11)
    table_values = (iv(n_values, kap) * n_values**2) / (2 * np.pi * kap * iv(1, kap))
    return table_values

print("Table values for ? = 1:")
print(compute_bessel_table(1))
print("\nTable values for ? = 2:")
print(compute_bessel_table(2))
print("\nTable values for ? = 4:")
print(compute_bessel_table(4))

# Plot[BesselI[1, kap], {kap, 0, 4}]
kap_range = np.linspace(0, 4, 1000)
bessel_values = iv(1, kap_range)

plt.figure(figsize=(10, 6))
plt.plot(kap_range, bessel_values, 'b-', linewidth=2)
plt.xlabel('?')
plt.ylabel('I?(?)')
plt.title('Modified Bessel Function of the First Kind (Order 1)')
plt.grid(True)
plt.xlim(0, 4)
plt.ylim(0, None)
plt.show()