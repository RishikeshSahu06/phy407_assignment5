#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>

const int L = 32;              // Lattice size (L x L)
const int N = L * L;           // Total number of spins
const double J = 1.0;          // Exchange interaction (scaled value)
const double kB = 1.0;         // Boltzmann constant (scaled value)

// For Metropolis algorithm
const int EQUILIBRATION_STEPS = 10000;   // Equilibration steps
const int MEASUREMENT_STEPS = 50000;     // Measurement steps
const int SAMPLING_INTERVAL = 10;        // Interval between measurements

class IsingModel {
private:
    std::vector<int> spins;                // Spin configuration
    std::mt19937 rng;                      // Random number generator
    std::uniform_real_distribution<double> dist;  // Uniform distribution [0,1]
    std::uniform_int_distribution<int> lattice_dist; // Uniform distribution [0,L-1]
    
    double temperature;                   // Current temperature
    
    double energy_sum = 0.0;
    double energy_squared_sum = 0.0;
    double magnetization_sum = 0.0;
    double abs_magnetization_sum = 0.0;
    double magnetization_squared_sum = 0.0;
    int num_measurements = 0;

public:
    IsingModel() : 
        spins(N, 1),  // Initialize all spins up (+1)
        dist(0.0, 1.0),
        lattice_dist(0, L-1) {
        
        std::random_device rd;
        rng.seed(rd());
    }
    
    int get_index(int x, int y) const {
        return (x + L) % L + ((y + L) % L) * L;
    }
    
    double calculate_spin_energy(int x, int y) const {
        int spin = spins[get_index(x, y)];
        int neighbor_sum = spins[get_index(x+1, y)] + 
                          spins[get_index(x-1, y)] + 
                          spins[get_index(x, y+1)] + 
                          spins[get_index(x, y-1)];
        
        return -J * spin * neighbor_sum;
    }
    
    double calculate_total_energy() const {
        double energy = 0.0;
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                energy += calculate_spin_energy(x, y);
            }
        }
        return energy / 2.0;
    }
    
    double calculate_magnetization() const {
        double magnetization = 0.0;
        for (int i = 0; i < N; ++i) {
            magnetization += spins[i];
        }
        return magnetization;
    }
    
    void monte_carlo_step() {
        for (int i = 0; i < N; ++i) {
            // Select random site
            int x = lattice_dist(rng);
            int y = lattice_dist(rng);
            int idx = get_index(x, y);
            
            double dE = -2.0 * calculate_spin_energy(x, y);
            
            // Metropolis acceptance criterion
            if (dE <= 0.0 || dist(rng) < exp(-dE / (kB * temperature))) {
                spins[idx] *= -1;  // Flip the spin
            }
        }
    }
    
    void run_simulation(double temp) {
        temperature = temp;
        
        energy_sum = 0.0;
        energy_squared_sum = 0.0;
        magnetization_sum = 0.0;
        abs_magnetization_sum = 0.0;
        magnetization_squared_sum = 0.0;
        num_measurements = 0;
        
        for (int i = 0; i < EQUILIBRATION_STEPS; ++i) {
            monte_carlo_step();
        }
        
        for (int i = 0; i < MEASUREMENT_STEPS; ++i) {
            monte_carlo_step();
            
            if (i % SAMPLING_INTERVAL == 0) {
                double energy = calculate_total_energy();
                double magnetization = calculate_magnetization();
                
                energy_sum += energy;
                energy_squared_sum += energy * energy;
                magnetization_sum += magnetization;
                abs_magnetization_sum += fabs(magnetization);
                magnetization_squared_sum += magnetization * magnetization;
                
                num_measurements++;
            }
        }
    }
    
    double get_energy_per_spin() const {
        return energy_sum / (num_measurements * N);
    }
    
    double get_energy_squared_per_spin() const {
        return energy_squared_sum / (num_measurements * N * N);
    }
    
    double get_abs_magnetization_per_spin() const {
        return abs_magnetization_sum / (num_measurements * N);
    }
    
    double get_magnetization_per_spin() const {
        return magnetization_sum / (num_measurements * N);
    }
    
    double get_magnetization_squared_per_spin() const {
        return magnetization_squared_sum / (num_measurements * N * N);
    }
    
    double get_specific_heat_per_spin() const {
        double energy_mean = energy_sum / num_measurements;
        double energy_squared_mean = energy_squared_sum / num_measurements;
        return (energy_squared_mean - energy_mean * energy_mean) / (N * temperature * temperature);
    }
    
    double get_magnetic_susceptibility_per_spin() const {
        double magnetization_squared_mean = magnetization_squared_sum / num_measurements;
        double magnetization_mean = magnetization_sum / num_measurements;
        return (magnetization_squared_mean - magnetization_mean * magnetization_mean) / (N * temperature);
    }
    
    double get_energy_error() const {
        double energy_mean = energy_sum / num_measurements;
        double variance = energy_squared_sum / num_measurements - energy_mean * energy_mean;
        return sqrt(variance / num_measurements) / N;
    }
    
    double get_magnetization_error() const {
        double magnetization_mean = magnetization_sum / num_measurements;
        double magnetization_squared_mean = magnetization_squared_sum / num_measurements;
        double variance = magnetization_squared_mean - magnetization_mean * magnetization_mean;
        return sqrt(variance / num_measurements) / N;
    }
};

int main() {
    IsingModel model;
    std::ofstream output_file("ising_data.txt");
    
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file." << std::endl;
        return 1;
    }
    
    output_file << "# Temperature, Energy per spin, Energy error, |Magnetization| per spin, |Magnetization| error, "
                << "Specific heat per spin, Magnetic susceptibility per spin" << std::endl;
    
    double T_min = 0.1;
    double T_max = 3.0;
    double T_step = 0.1;
    
    // Additional points near the critical temperature (approx 2.27 for 2D Ising)
    double T_crit_min = 2.0;
    double T_crit_max = 2.5;
    double T_crit_step = 0.02;
    
    std::cout << "Starting simulation..." << std::endl;
    
    for (double T = T_min; T <= T_max; T += T_step) {
        std::cout << "Running at temperature T = " << T << std::endl;
        model.run_simulation(T);
        
        output_file << std::fixed << std::setprecision(6)
                   << T << " "
                   << model.get_energy_per_spin() << " "
                   << model.get_energy_error() << " "
                   << model.get_abs_magnetization_per_spin() << " "
                   << model.get_magnetization_error() << " "
                   << model.get_specific_heat_per_spin() << " "
                   << model.get_magnetic_susceptibility_per_spin() << std::endl;
    }
    
    for (double T = T_crit_min; T <= T_crit_max; T += T_crit_step) {
        if (fmod(T - T_min, T_step) < 1e-6) continue;
        
        std::cout << "Running at temperature T = " << T << std::endl;
        model.run_simulation(T);
        
        output_file << std::fixed << std::setprecision(6)
                   << T << " "
                   << model.get_energy_per_spin() << " "
                   << model.get_energy_error() << " "
                   << model.get_abs_magnetization_per_spin() << " "
                   << model.get_magnetization_error() << " "
                   << model.get_specific_heat_per_spin() << " "
                   << model.get_magnetic_susceptibility_per_spin() << std::endl;
    }
    
    output_file.close();
    std::cout << "Simulation completed. Data saved to ising_data.txt" << std::endl;
    
    return 0;
}