import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from tqdm.auto import tqdm

# Reinforcement Learning Model to predict PSO parameters
class RLpredict(nn.Module):
    def __init__(self, input_dim):
        super(RLpredict, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softplus()  # Ensure positive outputs for w, c1, c2
        )

    def forward(self, x):
        return self.model(x)

# Helper Functions
def calculate_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))

def swap_operator(route1, route2):
    swaps = []
    for i in range(len(route1)):
        if route1[i] != route2[i]:
            j = route1.index(route2[i])
            swaps.append((i, j))
            route1[i], route1[j] = route1[j], route1[i]
    return swaps

def apply_swaps(route, swaps):
    new_route = route[:]
    for i, j in swaps:
        new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def crossover_elimination(route, distance_matrix):
    best_route = route[:]
    best_distance = calculate_distance(route, distance_matrix)
    for i in range(len(route) - 1):
        for j in range(i + 2, len(route)):
            new_route = route[:]
            new_route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
            new_distance = calculate_distance(new_route, distance_matrix)
            if new_distance < best_distance:
                best_route, best_distance = new_route, new_distance
    return best_route

def pso_tsp(distance_matrix, model, num_particles=10, max_iter=500):
    num_cities = len(distance_matrix)

    # Initialize particles
    particles = [np.random.permutation(num_cities).tolist() for _ in range(num_particles)]
    velocities = [[] for _ in range(num_particles)]

    fitness = [calculate_distance(p, distance_matrix) for p in particles]

    personal_best = particles[:]
    personal_best_fitness = fitness[:]
    global_best = particles[np.argmin(fitness)]
    global_best_fitness = min(fitness)

    global_best_collector = []

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    w_values, c1_values, c2_values = [], [], []

    for iteration in tqdm(range(max_iter)):
        # Input: Global best distance and current iteration
        model_input = torch.tensor([global_best_fitness, iteration / max_iter], dtype=torch.float32)
        w, c1, c2 = model(model_input).detach().numpy()

        w_values.append(w)
        c1_values.append(c1)
        c2_values.append(c2)

        for i in range(num_particles):
            v_personal = swap_operator(particles[i][:], personal_best[i][:])
            v_global = swap_operator(particles[i][:], global_best[:])

            # random.seed(42)
            new_velocity = (
                random.sample(velocities[i], min(len(velocities[i]), int(w * len(velocities[i])))) +
                random.sample(v_personal, min(len(v_personal), int(c1 * len(v_personal)))) +
                random.sample(v_global, min(len(v_global), int(c2 * len(v_global))))
            )

            velocities[i] = new_velocity
            particles[i] = apply_swaps(particles[i], velocities[i])

            current_fitness = calculate_distance(particles[i], distance_matrix)

            if current_fitness < personal_best_fitness[i]:
                personal_best[i] = particles[i][:]
                personal_best_fitness[i] = current_fitness

            if current_fitness < global_best_fitness:
                global_best = particles[i][:]
                global_best_fitness = current_fitness

        global_best = crossover_elimination(global_best, distance_matrix)
        global_best_fitness = calculate_distance(global_best, distance_matrix)
        global_best_collector.append(global_best_fitness)

        # Reward: Decrease in global best distance
        reward = -(np.log(global_best_fitness))

        optimizer.zero_grad()
        w, c1, c2 = model(model_input)
        loss = -torch.tensor(reward, dtype=torch.float32, requires_grad=True)
        loss.backward()
        optimizer.step()

    return global_best, global_best_fitness, global_best_collector, w_values, c1_values, c2_values

def calculate_distance_matrix(coords):
    num_nodes = len(coords)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return np.round(distance_matrix).astype(int)

def main():
    coordinates = np.array([
        [565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
        [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
        [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
        [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
        [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
        [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
        [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
        [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
        [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
        [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0],
        [1340.0, 725.0], [1740.0, 245.0]
    ])
    
    coordinates = np.array([
    [0, 0], [6, 24], [17, 19], [25, 19], [35, 17], [43, 26], [50, 32], [52, 43],
    [31, 61], [21, 57], [13, 56], [6, 49], [3, 38], [17, 33], [24, 42], [36, 26],
    [47, 43], [58, 48], [69, 42], [73, 52], [82, 66], [93, 58], [95, 74], [91, 78],
    [80, 79], [67, 81], [63, 71], [56, 64], [49, 57], [40, 60], [32, 64], [21, 72],
    [16, 76], [12, 66], [4, 58], [0, 52], [7, 39], [14, 26], [22, 18], [29, 11],
    [40, 6], [50, 0]
    ])
    

    coordinates = np.array([
        [6734, 1453], [2233, 10], [5530, 1424], [401, 841], [3082, 1644], [7608, 4458], 
        [7573, 3716], [7265, 1268], [6898, 1885], [1112, 2049], [5468, 2606], [5989, 2873], 
        [4706, 2674], [4612, 2035], [6347, 2683], [6107, 669], [7611, 5184], [7462, 3590], 
        [7732, 4723], [5900, 3561], [4483, 3369], [6101, 1110], [5199, 2182], [1633, 2809], 
        [4307, 2322], [675, 1006], [7555, 4819], [7541, 3981], [3177, 756], [7352, 4506], 
        [7545, 2801], [3245, 3305], [6426, 3173], [4608, 1198], [23, 2216], [7248, 3779], 
        [7762, 4595], [7392, 2244], [3484, 2829], [6271, 2135], [4985, 140], [1916, 1569], 
        [7280, 4899], [7509, 3239], [10, 2676], [6807, 2993], [5185, 3258], [3023, 1942]
    ])

    optimal_distance_att48 = 33523



    distance_matrix = calculate_distance_matrix(coordinates)

    model = RLpredict(input_dim=2)

    best_route, best_distance, global_best_collector, w_values, c1_values, c2_values = pso_tsp(distance_matrix, model, max_iter=100)

    print("Best Route:", best_route)
    print("Best Distance:", best_distance)

    
    plt.plot(global_best_collector, label=f"Global Fitness {best_distance}", color="orange")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.4)
    
    best_route_sorted_x, best_route_sorted_y = [], []
    for i in best_route:
        best_route_sorted_x.append(coordinates[i][0])
        best_route_sorted_y.append(coordinates[i][1])
    best_route_sorted_x.append(best_route_sorted_x[0])
    best_route_sorted_y.append(best_route_sorted_y[0])
    plt.figure(figsize=(10, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', label="Cities")
    plt.plot(best_route_sorted_x, best_route_sorted_y, c='red', linestyle='-', linewidth=1, label="Best Route")
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(i + 1), fontsize=8, verticalalignment='bottom')
    plt.title("TSP Solution (Berlin52)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(w_values, label="w", color="blue")
    plt.plot(c1_values, label="c1", color="green")
    plt.plot(c2_values, label="c2", color="red")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.4)

    plt.show()

if __name__ == "__main__":
    main()
