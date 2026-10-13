"""Embryo: dynamically growing neural network.
Not pre-built — grows during the bug's lifetime, like a real brain.
Input: visual (25 dims) + olfactory (8 dims)
Output: 8 actions
Growth: new neurons can appear, new connections can form, weights adjust.
"""
import numpy as np


class DynamicBrain:
    """A brain that grows dynamically during the bug's lifetime.
    Uses use-it-or-lose-it principle:
    - Frequently used neurons: kept and strengthened
    - Unused neurons: gradually die (pruned)
    - Overloaded: split new neurons to handle the load
    """
    def __init__(self, seed=0):
        np.random.seed(seed)
        
        self.n_input = 33  # visual (25) + olfactory (8)
        self.n_output = 8
        
        self.neurons = []
        self.neuron_types = []
        self.neuron_usage = []  # how often each neuron is used
        
        self.connections = []
        
        # Start with input and output layers
        for i in range(self.n_input):
            self.neurons.append(0.0)
            self.neuron_types.append('input')
            self.neuron_usage.append(1.0)  # input neurons always used
        
        for i in range(self.n_output):
            self.neurons.append(0.0)
            self.neuron_types.append('output')
            self.neuron_usage.append(1.0)
        
        # Initial connections: input -> output
        for i in range(self.n_input):
            for j in range(self.n_output):
                self.connections.append((i, self.n_input + j, np.random.randn() * 0.3))
        
        self.step_count = 0
        self.total_activity = 0.0

    def forward(self, visual, olfactory):
        """Run the network, return action scores.
        Supports multi-layer: input -> hidden -> output.
        """
        # Reset all neuron activations
        input_signal = np.concatenate([visual, olfactory])
        for i in range(self.n_input):
            self.neurons[i] = input_signal[i]
        
        # Propagate through layers (simple iterative)
        # Run multiple passes to allow signal to flow through hidden layers
        for _ in range(3):
            new_activations = self.neurons.copy()
            for from_idx, to_idx, weight in self.connections:
                if self.neuron_types[from_idx] != 'output':  # output neurons don't send signals
                    new_activations[to_idx] += self.neurons[from_idx] * weight
            # Apply activation function to non-input neurons
            for i in range(len(self.neurons)):
                if self.neuron_types[i] != 'input':
                    self.neurons[i] = np.tanh(new_activations[i])
        
        output = self.neurons[self.n_input:self.n_input + self.n_output]
        
        self.step_count += 1
        
        # Track total activity for load balancing
        current_activity = np.sum(np.abs(self.neurons))
        self.total_activity = 0.9 * self.total_activity + 0.1 * current_activity
        
        # Maybe grow new neuron if overloaded
        if self.step_count % 10 == 0 and len(self.neurons) < 200:
            if self.total_activity > 20:  # overloaded, need more capacity
                self.grow_neuron()
        
        # Prune unused neurons (use-it-or-lose-it)
        if self.step_count % 50 == 0:
            self.prune_neurons()
        
        return output

    def grow_neuron(self):
        """Grow a new hidden neuron to handle load."""
        new_idx = len(self.neurons)
        self.neurons.append(0.0)
        self.neuron_types.append('hidden')
        self.neuron_usage.append(0.1)  # starts with low usage
        
        # Connect from a random input neuron
        input_idx = np.random.randint(self.n_input)
        self.connections.append((input_idx, new_idx, np.random.randn() * 0.3))
        
        # Connect to a random output neuron
        output_idx = self.n_input + np.random.randint(self.n_output)
        self.connections.append((new_idx, output_idx, np.random.randn() * 0.3))

    def prune_neurons(self):
        """Prune unused neurons — use it or lose it."""
        # Decay usage for all neurons
        for i in range(len(self.neurons)):
            if self.neuron_types[i] == 'hidden':
                self.neuron_usage[i] *= 0.9
        
        # Find hidden neurons with very low usage
        to_prune = []
        for i in range(len(self.neurons)):
            if self.neuron_types[i] == 'hidden' and self.neuron_usage[i] < 0.01:
                to_prune.append(i)
        
        # Remove unused neurons (from end to keep indices valid)
        for i in reversed(to_prune):
            # Remove connections to/from this neuron
            self.connections = [(f, t, w) for f, t, w in self.connections if f != i and t != i]
            self.neurons.pop(i)
            self.neuron_types.pop(i)
            self.neuron_usage.pop(i)

    def adjust_weights(self, reward):
        """Adjust weights based on reward (Hebbian learning)."""
        lr = 0.1
        for i, (from_idx, to_idx, weight) in enumerate(self.connections):
            if self.neurons[from_idx] > 0 and self.neurons[to_idx] > 0:
                new_weight = weight + lr * reward * 0.01
                self.connections[i] = (from_idx, to_idx, new_weight)
                # Update usage scores
                if self.neuron_types[from_idx] == 'hidden':
                    self.neuron_usage[from_idx] = min(1.0, self.neuron_usage[from_idx] + 0.05)
                if self.neuron_types[to_idx] == 'hidden':
                    self.neuron_usage[to_idx] = min(1.0, self.neuron_usage[to_idx] + 0.05)


class Bug:
    """A bug with a dynamically growing brain."""
    def __init__(self, seed=0):
        self.brain = DynamicBrain(seed=seed)
        self.pos = np.array([np.random.randint(10, 20), np.random.randint(10, 20)])
        self.energy = 0.8
        self.alive = True
        self.food_eaten = 0

    def decide(self, visual, olfactory):
        out = self.brain.forward(visual, olfactory)
        return np.argmax(out)


class World:
    """Simple world: grid, food, walls, smell."""
    def __init__(self, size=30, seed=0):
        np.random.seed(seed)
        self.size = size
        self.grid = np.zeros((size, size))
        self.food_positions = []

        for _ in range(20):
            r, c = np.random.randint(2, size-2, 2)
            self.grid[r, c] = -1

        for _ in range(5):
            r, c = np.random.randint(0, size, 2)
            while self.grid[r, c] != 0:
                r, c = np.random.randint(0, size, 2)
            self.food_positions.append((r, c))

    def sense_visual(self, bug):
        r, c = bug.pos
        view = np.zeros((5, 5))
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    if (rr, cc) in self.food_positions:
                        view[dr+2, dc+2] = 2
                    elif self.grid[rr, cc] == -1:
                        view[dr+2, dc+2] = -1
        return view.flatten()

    def sense_olfactory(self, bug):
        r, c = bug.pos
        smell = np.zeros(8)
        for fr, fc in self.food_positions:
            dr = fr - r
            dc = fc - c
            if abs(dr) > abs(dc):
                if dr < 0: smell[0] += 1.0 / (abs(dr) + 1)
                else: smell[4] += 1.0 / (abs(dr) + 1)
            else:
                if dc > 0: smell[2] += 1.0 / (abs(dc) + 1)
                else: smell[6] += 1.0 / (abs(dc) + 1)
        return smell

    def step(self, bug, action):
        dr_list = [-1, -1, 0, 1, 1, 1, 0, -1]
        dc_list = [0, 1, 1, 1, 0, -1, -1, -1]
        dr, dc = dr_list[action], dc_list[action]
        nr, nc = int(bug.pos[0] + dr), int(bug.pos[1] + dc)
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != -1:
            bug.pos = np.array([nr, nc])

        ate = False
        for (fr, fc) in self.food_positions:
            if (nr, nc) == (fr, fc):
                bug.food_eaten += 1
                bug.energy = min(1.0, bug.energy + 0.3)
                self.food_positions.remove((fr, fc))
                new_r, new_c = np.random.randint(0, self.size, 2)
                while self.grid[new_r, new_c] != 0 or (new_r, new_c) in self.food_positions:
                    new_r, new_c = np.random.randint(0, self.size, 2)
                self.food_positions.append((new_r, new_c))
                ate = True
                break

        bug.energy -= 0.003
        if bug.energy <= 0:
            bug.alive = False
        return ate


def run_generation(population, steps=2000, seed=0):
    world = World(size=30, seed=seed)
    scores = []
    for i, bug in enumerate(population):
        bug.pos = np.array([np.random.randint(10, 20), np.random.randint(10, 20)])
        bug.energy = 0.8
        bug.alive = True
        bug.food_eaten = 0

        for step in range(steps):
            visual = world.sense_visual(bug)
            olfactory = world.sense_olfactory(bug)
            action = bug.decide(visual, olfactory)
            ate = world.step(bug, action)
            if ate:
                bug.brain.adjust_weights(reward=1.0)
            if not bug.alive:
                break

        score = bug.food_eaten * 10 + step / steps * 2
        scores.append(score)
    return scores


import pickle
import os

save_dir = "saved_brains"
os.makedirs(save_dir, exist_ok=True)


def save_brain(brain, filename):
    """Save network structure and weights."""
    data = {
        'neurons': brain.neurons,
        'neuron_types': brain.neuron_types,
        'neuron_usage': brain.neuron_usage,
        'connections': brain.connections,
        'n_input': brain.n_input,
        'n_output': brain.n_output,
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved to {filename}")


def main():
    np.random.seed(42)
    population = [Bug(seed=i) for i in range(100)]

    for gen in range(100):
        scores = run_generation(population, steps=2000, seed=gen)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        best_score = ranked[0][1]
        best_foods = population[ranked[0][0]].food_eaten
        best_neurons = len(population[ranked[0][0]].brain.neurons)
        
        if (gen + 1) % 10 == 0:
            print(f"Gen {gen+1}: best score={best_score:.1f}, foods eaten={best_foods}, neurons={best_neurons}")
        
        # Save every 100 generations
        if (gen + 1) % 100 == 0:
            save_brain(population[ranked[0][0]].brain, f"{save_dir}/gen_{gen+1}_best.pkl")


if __name__ == "__main__":
    main()
