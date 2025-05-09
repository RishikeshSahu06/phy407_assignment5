import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

data = np.loadtxt('ising_data.txt', skiprows=1)

temperature = data[:, 0]
energy = data[:, 1]
energy_error = data[:, 2]
magnetization = data[:, 3]
magnetization_error = data[:, 4]
specific_heat = data[:, 5]
susceptibility = data[:, 6]

idx = np.argsort(temperature)
temperature = temperature[idx]
energy = energy[idx]
energy_error = energy_error[idx]
magnetization = magnetization[idx]
magnetization_error = magnetization_error[idx]
specific_heat = specific_heat[idx]
susceptibility = susceptibility[idx]

plt.figure(figsize=(15, 12))
gs = GridSpec(2, 2)

# Plot 1: Energy per spin vs Temperature
ax1 = plt.subplot(gs[0, 0])
ax1.errorbar(temperature, energy, yerr=energy_error, fmt='o-', color='blue', capsize=3, markersize=4)
ax1.set_xlabel('Temperature ($T/J$)', fontsize=12)
ax1.set_ylabel('Energy per spin ($E/N$)', fontsize=12)
ax1.set_title('(i) Energy per spin vs Temperature', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Absolute Magnetization per spin vs Temperature
ax2 = plt.subplot(gs[0, 1])
ax2.errorbar(temperature, magnetization, yerr=magnetization_error, fmt='o-', color='red', capsize=3, markersize=4)
ax2.set_xlabel('Temperature ($T/J$)', fontsize=12)
ax2.set_ylabel('|Magnetization| per spin ($|M|/N$)', fontsize=12)
ax2.set_title('(ii) Absolute Magnetization per spin vs Temperature', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

# Plot 3: Specific Heat per spin vs Temperature
ax3 = plt.subplot(gs[1, 0])
ax3.plot(temperature, specific_heat, 'o-', color='green', markersize=4)
ax3.set_xlabel('Temperature ($T/J$)', fontsize=12)
ax3.set_ylabel('Specific Heat per spin ($C_v/N$)', fontsize=12)
ax3.set_title('(iii) Specific Heat per spin vs Temperature', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.7)

# Plot 4: Magnetic Susceptibility per spin vs Temperature
ax4 = plt.subplot(gs[1, 1])
ax4.plot(temperature, susceptibility, 'o-', color='purple', markersize=4)
ax4.set_xlabel('Temperature ($T/J$)', fontsize=12)
ax4.set_ylabel('Magnetic Susceptibility per spin ($\\chi/N$)', fontsize=12)
ax4.set_title('(iv) Magnetic Susceptibility per spin vs Temperature', fontsize=14)
ax4.grid(True, linestyle='--', alpha=0.7)

# Finding the critical temperature from the peaks of specific heat and susceptibility
specific_heat_peak_idx = np.argmax(specific_heat)
susceptibility_peak_idx = np.argmax(susceptibility)
Tc_specific_heat = temperature[specific_heat_peak_idx]
Tc_susceptibility = temperature[susceptibility_peak_idx]

# Annotate the critical temperature
for ax in [ax1, ax2, ax3, ax4]:
    ax.axvline(x=Tc_specific_heat, color='green', linestyle='--', alpha=0.5, 
              label=f'$T_c$ (from $C_v$) ≈ {Tc_specific_heat:.3f}')
    ax.axvline(x=Tc_susceptibility, color='purple', linestyle=':', alpha=0.5,
              label=f'$T_c$ (from $\\chi$) ≈ {Tc_susceptibility:.3f}')
    ax.legend(loc='best', fontsize=10)

# Finding critical temperature from magnetization (where it drops most rapidly)
mag_derivative = np.gradient(magnetization, temperature)
mag_derivative_idx = np.argmin(mag_derivative)  # Most negative slope
Tc_magnetization = temperature[mag_derivative_idx]

plt.figtext(0.5, 0.01, 
           f"(v) Estimated critical temperature ($T_c$):\n"
           f"From specific heat peak: $T_c$ ≈ {Tc_specific_heat:.3f} $J/k_B$\n"
           f"From susceptibility peak: $T_c$ ≈ {Tc_susceptibility:.3f} $J/k_B$\n"
           f"From magnetization inflection point: $T_c$ ≈ {Tc_magnetization:.3f} $J/k_B$\n"
           f"Theoretical value for infinite 2D Ising model: $T_c$ ≈ 2.269 $J/k_B$",
           ha='center', fontsize=12, bbox=dict(facecolor='lightyellow', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.suptitle('2D Ising Model Simulation (L=32, Metropolis Algorithm)', fontsize=16)
plt.savefig('ising_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print estimated critical temperature
print(f"Estimated critical temperatures:")
print(f"From specific heat peak: Tc ≈ {Tc_specific_heat:.4f} J/kB")
print(f"From susceptibility peak: Tc ≈ {Tc_susceptibility:.4f} J/kB")
print(f"From magnetization inflection point: Tc ≈ {Tc_magnetization:.4f} J/kB")
print(f"Theoretical value for infinite 2D Ising model: Tc ≈ 2.269 J/kB")