"""Embryo: random neural networks + evolution.
Don't design behavior — let evolution shape the network from random."""
import numpy as np


class RandomNet:
    """A simple feedforward network, randomly initialized.
    Input: sensory (25 dims)
    Output: action (8 dims)
    """
    def __init__(self, seed=0):
        np.random.seed(seed)
        # Two-layer network: 25 -> 16 -> 8
        self.W1 = np.random.randn(16, 25) * 0.5
        self.b1 = np.random.randn(16) * 0.1
        self.W2 = np.random.randn(8, 16) * 0.5
        self.b2 = np.random.randn(8) * 0.1

    def forward(self, sensory):
        """Run the network, return action scores."""
        h = np.tanh(self.W1 @ sensory + self.b1)
        out = self.W2 @ h + self.b2
        return out

    def mutate(self, rate=0.1, strength=0.3):
        """Random mutation for breeding."""
        if np.random.random() < rate:
            self.W1 += np.random.randn(*self.W1.shape) * strength
        if np.random.random() < rate:
            self.b1 += np.random.randn(*self.b1.shape) * strength
        if np.random.random() < rate:
            self.W2 += np.random.randn(*self.W2.shape) * strength
        if np.random.random() < rate:
            self.b2 += np.random.randn(*self.b2.shape) * strength


class Bug:
    """A bug with a random neural network as its brain."""
    def __init__(self, seed=0, net=None):
        self.net = net if net else RandomNet(seed)
        self.pos = np.array([np.random.randint(10, 20), np.random.randint(10, 20)])
        self.energy = 0.8
        self.alive = True
        self.food_eaten = 0

    def decide(self, sensory):
        """Use the neural network to decide action."""
        out = self.net.forward(sensory)
        return np.argmax(out)


class World:
    """Simple world: grid, food, walls."""
    def __init__(self, size=30, seed=0):
        np.random.seed(seed)
        self.size = size
        self.grid = np.zeros((size, size))
        self.food_positions = []

        # Place walls
        for _ in range(20):
            r, c = np.random.randint(2, size-2, 2)
            self.grid[r, c] = -1

        # Place food
        for _ in range(5):
            r, c = np.random.randint(0, size, 2)
            while self.grid[r, c] != 0:
                r, c = np.random.randint(0, size, 2)
            self.food_positions.append((r, c))

    def sense(self, bug):
        """What does this bug see? 5x5 grid."""
        r, c = bug.pos
        view = np.zeros((5, 5))
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    if (rr, cc) in self.food_positions:
                        view[dr+2, dc+2] = 2  # food
                    elif self.grid[rr, cc] == -1:
                        view[dr+2, dc+2] = -1  # wall
        return view.flatten()

    def step(self, bug, action):
        """Bug moves, returns True if ate food."""
        dr_list = [-1, -1, 0, 1, 1, 1, 0, -1]
        dc_list = [0, 1, 1, 1, 0, -1, -1, -1]
        dr, dc = dr_list[action], dc_list[action]
        nr, nc = int(bug.pos[0] + dr), int(bug.pos[1] + dc)
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != -1:
            bug.pos = np.array([nr, nc])

        # Check food
        ate = False
        for (fr, fc) in self.food_positions:
            if (nr, nc) == (fr, fc):
                bug.food_eaten += 1
                bug.energy = min(1.0, bug.energy + 0.3)
                self.food_positions.remove((fr, fc))
                # Respawn food
                new_r, new_c = np.random.randint(0, self.size, 2)
                while self.grid[new_r, new_c] != 0 or (new_r, new_c) in self.food_positions:
                    new_r, new_c = np.random.randint(0, self.size, 2)
                self.food_positions.append((new_r, new_c))
                ate = True
                break

        bug.energy -= 0.002
        if bug.energy <= 0:
            bug.alive = False
        return ate


def run_generation(population, steps=2000, seed=0):
    """Run one generation, return scores for each bug."""
    world = World(size=30, seed=seed)
    scores = []
    for i, bug in enumerate(population):
        bug.pos = np.array([np.random.randint(10, 20), np.random.randint(10, 20)])
        bug.energy = 0.8
        bug.alive = True
        bug.food_eaten = 0

        for step in range(steps):
            s = world.sense(bug)
            action = bug.decide(s)
            world.step(bug, action)
            if not bug.alive:
                break

        # Score: food eaten + steps survived
        score = bug.food_eaten * 10 + step / steps * 2
        scores.append(score)
    return scores


def main():
    np.random.seed(42)
    # Start with 100 bugs, each with random network
    population = [Bug(seed=i) for i in range(100)]

    for gen in range(10):
        scores = run_generation(population, steps=2000, seed=gen)
        # Sort by score
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        best_score = ranked[0][1]
        best_foods = population[ranked[0][0]].food_eaten
        print(f"Gen {gen+1}: best score={best_score:.1f}, foods eaten={best_foods}")

        # Keep top 3, breed them
        top3 = [population[i].net for i, _ in ranked[:3]]
        next_gen = []
        for net in top3:
            # Keep original
            next_gen.append(Bug(net=net))
            # 33 mutated copies each = 100 total
            for _ in range(33):
                child_net = RandomNet(seed=np.random.randint(10000))
                child_net.W1 = net.W1.copy()
                child_net.b1 = net.b1.copy()
                child_net.W2 = net.W2.copy()
                child_net.b2 = net.b2.copy()
                child_net.mutate()
                next_gen.append(Bug(net=child_net))
        population = next_gen[:100]


if __name__ == "__main__":
    main()
