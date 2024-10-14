import numpy as np

def initialize_bats(pop_size, dim):
    return np.random.rand(pop_size, dim)

def update_position(position, velocity):
    return position + velocity

def bat_algorithm(objective_function, pop_size=10, max_iterations=100, loudness=0.5, pulse_rate=0.5, dim=10):
    bats = initialize_bats(pop_size, dim)
    velocities = np.zeros((pop_size, dim))

    # Evaluate initial fitness
    fitness = np.apply_along_axis(objective_function, 1, bats)
    best_index = np.argmin(fitness)
    best_solution = bats[best_index].copy()

    for iteration in range(max_iterations):
        current_loudness = loudness * (1 - np.exp(-pulse_rate * iteration))

        for i in range(pop_size):
            frequency = np.random.uniform(0, 1)  # Randomly generate frequency
            velocities[i] += frequency * (bats[i] - best_solution)
            bats[i] = update_position(bats[i], velocities[i])

            # Randomly adjust the position based on loudness
            if np.random.rand() < current_loudness:
                bats[i] = best_solution + 0.001 * np.random.randn(dim)

        new_fitness = np.apply_along_axis(objective_function, 1, bats)

        # Update best solution if found new better fitness
        new_best_index = np.argmin(new_fitness)
        if new_fitness[new_best_index] < fitness[best_index]:
            best_solution = bats[new_best_index].copy()
            best_index = new_best_index
            fitness[best_index] = new_fitness[new_best_index]

    return best_solution, fitness[best_index]
