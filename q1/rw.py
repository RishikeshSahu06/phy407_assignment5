import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def random_walk_2d(N):
    """
    Perform a 2D random walk of N steps.
    Returns the squared end-to-end distance.
    """
    x, y = 0, 0  # start at origin
    
    for i in range(N):
        # select random integer between 1 and 4
        direction = np.random.randint(1, 5)
        
        # move according to direction
        if direction == 1:
            x += 1  # move right
        elif direction == 2:
            x -= 1  # move left
        elif direction == 3:
            y += 1  # move up
        else:  # direction == 4
            y -= 1  # move down
    
    # Compute squared end-to-end distance
    R2 = x**2 + y**2
    
    return R2

def simulate_walks(N_steps, num_walks=1000):
    """
    Perform multiple random walks of length N_steps.
    Returns average R² and its standard deviation.
    """
    R2_values = np.zeros(num_walks)
    
    for i in range(num_walks):
        R2_values[i] = random_walk_2d(N_steps)
    
    avg_R2 = np.mean(R2_values)
    std_R2 = np.std(R2_values) / np.sqrt(num_walks)  # Standard error
    
    return avg_R2, std_R2

def power_law(x, a, nu):
    """Power law function: a * x^nu"""
    return a * x**nu

# List of walk lengths to simulate
N_values = [10, 20, 50, 100, 200, 500, 1000]
num_simulations = 1000  # Number of walks per length

# Arrays to store results
avg_R2 = np.zeros(len(N_values))
std_R2 = np.zeros(len(N_values))

# Run simulations for each walk length
for i, N in enumerate(N_values):
    print(f"Simulating {num_simulations} walks of length {N}...")
    avg_R2[i], std_R2[i] = simulate_walks(N, num_simulations)

# Fit power law to determine exponent
# R² ~ N^(2*nu), so we fit to find nu
def fit_function(N, a, nu):
    return a * N**(nu)

params, cov = curve_fit(fit_function, N_values, avg_R2, p0=[1, 0.5], sigma=std_R2)
a_fit, nu_fit = params
nu_err = np.sqrt(cov[1, 1])

print("\nResults:")
print(f"Fitted parameters: a = {a_fit:.4f}, ν = {nu_fit:.4f} ± {nu_err:.4f}")
print(f"Theoretical value for 2D: ν = 1.0 (random walk)")

# Create plot with error bars on logarithmic scale
plt.figure(figsize=(10, 6))
plt.errorbar(N_values, avg_R2, yerr=std_R2, fmt='o', capsize=5, label='Simulation data')

# Add fitted power law
N_range = np.logspace(np.log10(min(N_values)), np.log10(max(N_values)), 100)
plt.plot(N_range, fit_function(N_range, a_fit, nu_fit), 'r-', 
         label=f'Fit: $R^2 \\propto N^{{\\nu}}$ with $\\nu = {nu_fit:.4f} \\pm {nu_err:.4f}$')

# Add theoretical power law
plt.plot(N_range, N_range, 'g--', label='Theoretical: $R^2 \\propto N$ ($\\nu = 1.0$)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of steps (N)')
plt.ylabel('Mean squared end-to-end distance $\\langle R^2(N) \\rangle$')
plt.title('2D Random Walk: Mean Squared End-to-End Distance vs. Number of Steps')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig('random_walk_2d.png', dpi=300)
plt.show()

# Print the actual data
print("\nSimulation data:")
print("N\t<R²>\t\tStd Error")
for i, N in enumerate(N_values):
    print(f"{N}\t{avg_R2[i]:.4f}\t\t{std_R2[i]:.4f}")