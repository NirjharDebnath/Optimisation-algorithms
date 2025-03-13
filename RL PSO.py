import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.optim as optim
import scipy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# --- Policy Model ---
class RLpredict(nn.Module):
    def __init__(self, input_dim):
        super(RLpredict, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)  # Output: [w, c1, c2] (sums to 1)
        )

    def forward(self, x):
        return self.model(x)

# --- Helper Functions ---
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

def pso_tsp(distance_matrix, model, optimizer, max_iter=400, num_particles=40):
    num_cities = len(distance_matrix)

    particles = [np.random.permutation(num_cities).tolist() for _ in range(num_particles)]
    velocities = [[] for _ in range(num_particles)]

    fitness = [calculate_distance(p, distance_matrix) for p in particles]

    personal_best = particles[:]
    personal_best_fitness = fitness[:]
    global_best = particles[np.argmin(fitness)]
    global_best_fitness = min(fitness)

    global_best_collector = []

    for iteration in tqdm(range(max_iter)):
        input_tensor = torch.tensor([global_best_fitness / 10000.0, iteration / max_iter], dtype=torch.float32)
        w, c1, c2 = model(input_tensor).detach().numpy()

        for i in range(num_particles):
            v_personal = swap_operator(particles[i][:], personal_best[i][:])
            v_global = swap_operator(particles[i][:], global_best[:])

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

        global_best_collector.append(global_best_fitness)

        reward = scipy.special.expit(-global_best_fitness)
        optimizer.zero_grad()
        w, c1, c2 = model(input_tensor)
        log_probs = torch.log(w * c1 * c2)
        loss = reward * log_probs
        loss.backward()
        optimizer.step()

    return global_best, global_best_fitness, global_best_collector

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
    distance_matrix = calculate_distance_matrix(coordinates)

    model = RLpredict(input_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.5)

    best_route, best_distance, global_best_collector = pso_tsp(distance_matrix, model, optimizer, max_iter=1000, num_particles=40)

    print("\nBest Route:", best_route)
    print("Best Distance:", best_distance)

    plt.plot(global_best_collector, label=f"Global Fitness {best_distance}", color="orange")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.4)
    plt.show()

if __name__ == "__main__":
    main()
