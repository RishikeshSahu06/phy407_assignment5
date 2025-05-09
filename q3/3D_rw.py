import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

def random_walk_3d(N):
    """
    Performing a 3D random walk of N steps.
    Returns the squared end-to-end distance.
    """
    x, y, z = 0, 0, 0  
    
    for i in range(N):
        direction = np.random.randint(1, 7)
        
        if direction == 1:
            x += 1  # move right
        elif direction == 2:
            x -= 1  # move left
        elif direction == 3:
            y += 1  # move up
        elif direction == 4:
            y -= 1  # move down
        elif direction == 5:
            z += 1  # move forward
        else:  # direction == 6
            z -= 1  # move backward
    
    R2 = x**2 + y**2 + z**2
    
    return R2

def simulate_walks(N_steps, num_walks=1000):
    """
    Performing multiple random walks of length N_steps.
    Returns average R² and its standard deviation.
    """
    start_time = time.time()
    print(f"Simulating {num_walks} random walks of length {N_steps}...")
    
    R2_values = np.zeros(num_walks)
    
    for i in range(num_walks):
        R2_values[i] = random_walk_3d(N_steps)
        
        if (i+1) % (num_walks // 10) == 0 or i == num_walks - 1:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{num_walks} walks ({elapsed:.1f} seconds)")
    
    avg_R2 = np.mean(R2_values)
    std_R2 = np.std(R2_values) / np.sqrt(num_walks)  
    
    print(f"  Average R²: {avg_R2:.4f} ± {std_R2:.4f}")
    return avg_R2, std_R2

def power_law(x, a, nu):
    """Power law function: a * x^(nu)"""
    return a * x**(nu)

N_values = [10, 20, 50, 100, 200, 500, 1000]
num_simulations = 1000 

avg_R2 = np.zeros(len(N_values))
std_R2 = np.zeros(len(N_values))

for i, N in enumerate(N_values):
    avg_R2[i], std_R2[i] = simulate_walks(N, num_simulations)

# R² ~ N^(nu)
params, cov = curve_fit(power_law, N_values, avg_R2, p0=[2, 1], sigma=std_R2)
a_fit, nu_fit = params
nu_err = np.sqrt(cov[1, 1])

print("\nResults:")
print(f"Fitted parameters: a = {a_fit:.4f}, ν = {nu_fit:.4f} ± {nu_err:.4f}")
print(f"Theoretical value for 3D: ν = 1.0 (random walk)")

plt.figure(figsize=(10, 6))
plt.errorbar(N_values, avg_R2, yerr=std_R2, fmt='o', capsize=5, label='Simulation data')

N_range = np.logspace(np.log10(min(N_values)), np.log10(max(N_values)), 100)
plt.plot(N_range, power_law(N_range, a_fit, nu_fit), 'r-', 
         label=f'Fit: $R^2 \\propto N^{{\\nu}}$ with $\\nu = {nu_fit:.4f} \\pm {nu_err:.4f}$')

plt.plot(N_range, N_range, 'g--', label='Theoretical: $R^2 \\propto N$ ($\\nu = 1.0$)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of steps (N)')
plt.ylabel('Mean squared end-to-end distance $\\langle R^2(N) \\rangle$')
plt.title('3D Random Walk: Mean Squared End-to-End Distance vs. Number of Steps')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig('random_walk_3d.png', dpi=300)
plt.show()

print("\nSimulation data:")
print("N\t<R²>\t\tStd Error")
for i, N in enumerate(N_values):
    print(f"{N}\t{avg_R2[i]:.4f}\t\t{std_R2[i]:.4f}")