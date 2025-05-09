import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

def self_avoiding_random_walk_3d(N, max_attempts=1000):
    """
    Performing a 3D self-avoiding random walk of N steps.
    Returns the squared end-to-end distance or None if trapped.
    
    Parameters:
    N: Number of steps
    max_attempts: Maximum number of attempts to find a valid walk
    """
    # Initializing grid to track visited sites (0=vacant, 1=visited)
    grid_size = 2*N + 1
    grid_center = N
    
    for attempt in range(max_attempts):
        grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)
        
        x, y, z = grid_center, grid_center, grid_center
        grid[z, y, x] = 1  # Marking starting position as visited
        
        i = 0
        while i < N:
            # Possible moves (±1 in x, y, or z direction)
            possible_moves = []
            
            if x + 1 < grid_size and grid[z, y, x + 1] == 0:  # +x direction
                possible_moves.append((1, 0, 0))
            if x - 1 >= 0 and grid[z, y, x - 1] == 0:  # -x direction
                possible_moves.append((-1, 0, 0))
            if y + 1 < grid_size and grid[z, y + 1, x] == 0:  # +y direction
                possible_moves.append((0, 1, 0))
            if y - 1 >= 0 and grid[z, y - 1, x] == 0:  # -y direction
                possible_moves.append((0, -1, 0))
            if z + 1 < grid_size and grid[z + 1, y, x] == 0:  # +z direction
                possible_moves.append((0, 0, 1))
            if z - 1 >= 0 and grid[z - 1, y, x] == 0:  # -z direction
                possible_moves.append((0, 0, -1))
            
            if not possible_moves:
                break
            
            dx, dy, dz = possible_moves[np.random.randint(0, len(possible_moves))]
            x += dx
            y += dy
            z += dz
            
            # Marking new position as visited
            grid[z, y, x] = 1
            i += 1
        
        if i == N:
            dx = x - grid_center
            dy = y - grid_center
            dz = z - grid_center
            R2 = dx**2 + dy**2 + dz**2
            return R2
    
    print(f"Failed to complete walk of length {N} after {max_attempts} attempts.")
    return None

def simulate_sarw(N_steps, num_walks=100, max_attempts_per_walk=1000):
    """
    Performing multiple self-avoiding random walks of length N_steps.
    Returns average R² and its standard deviation.
    """
    print(f"Simulating {num_walks} self-avoiding walks of length {N_steps}...")
    start_time = time.time()
    
    R2_values = []
    successful_walks = 0
    
    for i in range(num_walks):
        R2 = self_avoiding_random_walk_3d(N_steps, max_attempts=max_attempts_per_walk)
        if R2 is not None:
            R2_values.append(R2)
            successful_walks += 1
        
        if (i+1) % 10 == 0 or i == num_walks - 1:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{num_walks} attempts, {successful_walks} successful ({elapsed:.1f} seconds)")
    
    if not R2_values:
        return None, None
    
    R2_values = np.array(R2_values)
    avg_R2 = np.mean(R2_values)
    std_R2 = np.std(R2_values) / np.sqrt(len(R2_values))  # Standard error
    
    print(f"  Completed {len(R2_values)} valid walks in {elapsed:.1f} seconds")
    print(f"  Average R²: {avg_R2:.4f} ± {std_R2:.4f}")
    
    return avg_R2, std_R2

def power_law(x, a, nu):
    """Power law function: a * x^(nu)"""
    return a * x**(nu)


N_values = [5, 10, 15, 20, 25, 30]
num_simulations = 100  

def get_num_walks(N):
    if N <= 10:
        return 200
    elif N <= 20:
        return 100
    elif N <= 30:
        return 50
    else:
        return 30

avg_R2 = []
std_R2 = []
valid_N = []

for N in N_values:
    num_walks = get_num_walks(N)
    avg, std = simulate_sarw(N, num_walks=num_walks)
    
    if avg is not None:
        avg_R2.append(avg)
        std_R2.append(std)
        valid_N.append(N)

avg_R2 = np.array(avg_R2)
std_R2 = np.array(std_R2)
valid_N = np.array(valid_N)

# R² ~ N^(nu)
params, cov = curve_fit(power_law, valid_N, avg_R2, p0=[2, 1.176], sigma=std_R2)
a_fit, nu_fit = params
nu_err = np.sqrt(cov[1, 1])

print("\nResults:")
print(f"Fitted parameters: a = {a_fit:.4f}, ν = {nu_fit:.4f} ± {nu_err:.4f}")
print(f"Theoretical value for 3D SARW: ν = 1.176")

# Create plot with error bars on logarithmic scale
plt.figure(figsize=(10, 6))
plt.errorbar(valid_N, avg_R2, yerr=std_R2, fmt='o', capsize=5, label='Simulation data')

# Add fitted power law
N_range = np.logspace(np.log10(min(valid_N)), np.log10(max(valid_N)), 100)
plt.plot(N_range, power_law(N_range, a_fit, nu_fit), 'r-', 
         label=f'Fit: $R^2 \\propto N^{{\\nu}}$ with $\\nu = {nu_fit:.4f} \\pm {nu_err:.4f}$')

# Add theoretical power law
plt.plot(N_range, power_law(N_range, a_fit/a_fit, 0.588), 'g--', 
         label='Theoretical: $R^2 \\propto N^{1.176}$ ($\\nu = 1.176$)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of steps (N)')
plt.ylabel('Mean squared end-to-end distance $\\langle R^2(N) \\rangle$')
plt.title('3D Self-Avoiding Random Walk: Mean Squared End-to-End Distance vs. Number of Steps')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig('self_avoiding_random_walk_3d.png', dpi=300)
plt.show()

# Print the actual data
print("\nSimulation data:")
print("N\t<R²>\t\tStd Error")
for i, N in enumerate(valid_N):
    print(f"{N}\t{avg_R2[i]:.4f}\t\t{std_R2[i]:.4f}")