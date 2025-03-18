# **🚀 Particle Swarm Optimization (PSO) for Solving the Traveling Salesman Problem (TSP) 🗺️**

## **📌 Overview**
This repository contains an implementation of **Particle Swarm Optimization (PSO)** applied to the **Traveling Salesman Problem (TSP)**. The objective is to find an optimal or near-optimal route that minimizes the total travel distance between a given set of cities. 🏙️✨

### **🔥 Key Features**
✅ **Custom PSO variant** tailored for permutation-based optimization problems like TSP.
✅ **Mathematical foundation** behind the PSO update rules used in this implementation.
✅ **Pre-trained model loading** to use an optimized set of PSO parameters.
✅ **Performance visualization** using convergence plots 📊.

---

## **📍 Problem Formulation: Traveling Salesman Problem (TSP)**
The **TSP** is an **NP-hard** combinatorial optimization problem where a salesman must visit `N` cities exactly once and return to the starting city while minimizing the total distance traveled. 🚗💨

Mathematically, given `N` cities and a distance matrix \( D_{ij} \) (distance between city `i` and `j`), we aim to find a permutation \( \pi \) that minimizes:

$$
F(\pi) = \sum_{i=1}^{N-1} D_{\pi(i), \pi(i+1)} + D_{\pi(N), \pi(1)}
$$

where $\pi(i)$ represents the city at position `i` in the tour.

---

## **⚙️ Particle Swarm Optimization (PSO) Approach**

### **1️⃣ Encoding the TSP Solution in PSO**
Unlike traditional PSO, where solutions are real-valued vectors, TSP solutions are **permutations** of city indices. To handle this:
- Each **particle** represents a permutation (a possible tour) 🧩.
- Particle velocity is defined using a **swap sequence** (a set of city swaps to transform one permutation into another).
- Position updates involve **applying swaps** to the current tour.

### **2️⃣ Velocity and Position Update Rules**
The standard PSO update formulas are modified to work with permutations:

#### **🚀 Velocity Update:**
For a given particle `i`, its velocity (a sequence of swaps) is updated as:

$$
V_i^{(t+1)} = w \cdot V_i^{(t)} + c_1 \cdot r_1 \cdot (P_i^{(t)} \ominus X_i^{(t)}) + c_2 \cdot r_2 \cdot (G^{(t)} \ominus X_i^{(t)})
$$

where:
- $V_i^{(t)}$ is the velocity (swap sequence) at iteration `t`.
- $X_i^{(t)}$ is the current tour (permutation) of the particle.
- $P_i^{(t)}$ is the personal best tour.
- $G^{(t)}$ is the global best tour.
- $\ominus$ represents the **difference operator**, which computes a swap sequence to transform one permutation into another.
- $w$ is the inertia weight ⚖️.
- $c_1, c_2$ are acceleration coefficients 🚀.
- $r_1, r_2 \sim U(0,1)$ are random numbers 🎲.

#### **📌 Position Update:**
The new position (tour) is obtained by applying the swap sequence:

$$
X_i^{(t+1)} = X_i^{(t)} \oplus V_i^{(t+1)}
$$

where $\oplus$ applies the swap sequence to the current permutation.

---

## **📈 Reward Calculation and Optimization in PSO**

### **🎯 Overview**
This section defines a **reward mechanism** for optimizing PSO parameters using a reinforcement learning approach. The reward is based on the improvement in the **global best fitness value**, and it is used to update an optimization model.

### **🔍 Explanation**
- **Reward Calculation:**
  - The reward is computed as the decrease in the global best distance relative to the previous best fitness value.
  - A small value (`1e-6`) is added to the denominator to prevent division by zero.
  
  ```python
  reward = (prev_best_fitness - global_best_fitness) / (prev_best_fitness + 1e-6)
  ```
  - A **positive reward** indicates an improvement in the best solution found by the PSO algorithm. 🏆

- **Reward Collection:**
  - The computed reward is stored in `reward_collector` for later analysis or training stability. 📊
  
  ```python
  reward_collector.append(reward)
  ```

- **Loss Calculation & Optimization:**
  - The loss function is defined as the **negative reward** scaled by the mean of PSO hyperparameters.
  - The optimizer updates the parameters using **gradient descent**.
  
  ```python
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  ```

---

## **🏋️‍♂️ Training & Using the Model**

### **📌 Training the Model**
1️⃣ **Initialize particles** with random tours.
2️⃣ **Compute fitness** (total distance of the tour).
3️⃣ **Update velocities and positions** using the modified PSO update rules.
4️⃣ **Track global and personal bests**.
5️⃣ **Repeat for `N` iterations**.

To train the model, run the following command:
```python
# Run PSO on a randomly generated TSP instance
train_pso()
```

### **📂 Loading and Using a Pre-Trained Model**
A pre-trained model with optimized PSO parameters can be loaded and used for TSP solving:
```python
# Load the trained PSO parameters
load_and_run_pso()
```

---

## **📊 Results & Visualization**
After execution, the algorithm outputs:
- The **best-found TSP route** 🗺️.
- **Convergence plot** showing the decrease in total travel distance over iterations 📉.
- **Graphical representation** of the optimal path found.

Example output:
```
Optimal tour found: [3, 1, 7, 2, 6, 4, 5, 0, 8, 9]
Total distance: 375.4
```

---

## **🚀 Future Improvements**
✨ Experimenting with **adaptive PSO parameters**.
✨ Hybridizing PSO with **local search algorithms** for improved performance.
✨ Applying **deep learning** to guide the PSO process.

---

## **📚 References**
1️⃣ Kennedy, J., & Eberhart, R. (1995). **Particle swarm optimization**. *Proceedings of ICNN'95 - International Conference on Neural Networks*.

2️⃣ Clerc, M. (2010). **Particle Swarm Optimization**. *Wiley-ISTE*.

3️⃣ Shi, X. H., Liang, Y. C., Lee, H. P., Lu, C., & Wang, Q. X. (2007). **Particle swarm optimization-based algorithms for TSP and generalized TSP**. *Information Processing Letters, 103(5), 169-176*.

---

🎉 **Happy Optimizing! 🚀💡**

