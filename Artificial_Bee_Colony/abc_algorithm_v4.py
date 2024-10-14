import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Load the dataset
data = pd.read_csv("medium.csv")
X = data["X"].values
Y = data["Y"].values
data["City"] = np.arange(1, len(data) + 1)
data.to_csv("medium.csv")
cities = data["City"].values


# Function to calculate the Euclidean distance between two cities
def euclidean_distance(city1, city2):
    return np.sqrt((X[city2] - X[city1]) ** 2 + (Y[city2] - Y[city1]) ** 2)


# Objective function for TSP: Total distance of the path
def objective_function(path):
    total_distance = 0
    for i in range(1, len(path)):
        total_distance += euclidean_distance(path[i - 1], path[i])
    # Add the distance to return to the start
    total_distance += euclidean_distance(path[-1], path[0])
    return total_distance


# Generate a random path (initial solution)
def random_solution():
    path = np.arange(len(X))
    np.random.shuffle(path)
    return path


# Mutate the path (swap two cities)
def mutate_solution(solution):
    new_solution = solution.copy()
    i, j = np.random.choice(len(new_solution), 2, replace=False)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution


# Artificial Bee Colony (ABC) Algorithm for TSP
def ABC_algorithm():
    num_bees = 20  # Population size
    max_iterations = 100  # Maximum number of iterations
    limit = 50  # Limit for abandonment

    # Initialize population and fitness
    population = [random_solution() for _ in range(num_bees)]
    fitness = [objective_function(sol) for sol in population]
    trial_counter = [0] * num_bees  # Counter for scout bee phase

    for iteration in range(max_iterations):
        # Employed bee phase
        for i in range(num_bees):
            mutant = mutate_solution(population[i])
            if objective_function(mutant) < fitness[i]:
                population[i] = mutant
                fitness[i] = objective_function(mutant)
                trial_counter[i] = 0  # Reset trial counter for improved solution
            else:
                trial_counter[i] += 1

        # Calculate selection probability based on fitness (for onlooker bees)
        fitness_sum = sum(fitness)
        probabilities = [f / fitness_sum for f in fitness]

        # Onlooker bee phase
        for _ in range(num_bees):
            selected = np.random.choice(num_bees, p=probabilities)
            mutant = mutate_solution(population[selected])
            if objective_function(mutant) < fitness[selected]:
                population[selected] = mutant
                fitness[selected] = objective_function(mutant)
                trial_counter[selected] = 0

        # Scout bee phase
        for i in range(num_bees):
            if trial_counter[i] > limit:  # Abandon solution if limit is exceeded
                population[i] = random_solution()
                fitness[i] = objective_function(population[i])
                trial_counter[i] = 0

        # Print progress
        best_solution = population[np.argmin(fitness)]
        print(f"Iteration {iteration + 1}, Best Fitness: {min(fitness)}")

    # Return the best solution found
    best_index = np.argmin(fitness)
    return population[best_index], min(fitness)


# Run the ABC algorithm and set the timer
start_time = datetime.datetime.now()
best_solution, best_fitness = ABC_algorithm()
end_time = datetime.datetime.now()
execution_time = (end_time - start_time).total_seconds()

print(
    "Best Path (city indices):",
    best_solution,
    "\nBest Fitness (total distance):",
    best_fitness,
    "\nABC Execution Time:",
    execution_time,
    "seconds",
)

# Plot the best path
plt.figure(figsize=(10, 6))
for i in range(len(best_solution)):
    start_city = best_solution[i]
    end_city = best_solution[(i + 1) % len(best_solution)]
    plt.plot([X[start_city], X[end_city]], [Y[start_city], Y[end_city]], "bo-")
    plt.text(X[start_city], Y[start_city], cities[start_city])

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Optimized Travel Path Between Cities")
plt.show()
