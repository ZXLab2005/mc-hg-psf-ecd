import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('spec.dat')
energy, absorption = data[:, 0], data[:, 1]

# 0–7 eV 筛选
mask = (energy >= 0) & (energy <= 7)
energy_cut, absorption_cut = energy[mask], absorption[mask]

# —— 保存到文件供 Origin 使用 ——
out = np.column_stack((energy_cut, absorption_cut))
np.savetxt('spec_0-7eV.csv',
           out,
           fmt='%.6f',
           delimiter=',',
           header='Energy(eV),Absorption',
           comments='')

# —— 可选：同时绘图验证 ——
plt.plot(energy_cut, absorption_cut, color='green')
plt.xlabel('Energy (eV)')
plt.ylabel('Absorption Intensity')
plt.title('Photoabsorption Spectrum (0–7 eV)')
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

