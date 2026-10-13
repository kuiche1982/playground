"""Embryo: 10 bugs in one world, competing for food.
Each bug has different genes — watch which ones survive."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from embryo.world import World
from embryo.brain import Brain


# 10 bugs with different genes
BUGS = [
    {'smell_weight': 0.8, 'explore_noise': 0.2, 'momentum': 0.5, 'lr': 0.1},   # hunter
    {'smell_weight': 0.3, 'explore_noise': 0.8, 'momentum': 0.1, 'lr': 0.1},   # explorer
    {'smell_weight': 0.5, 'explore_noise': 0.5, 'momentum': 0.3, 'lr': 0.1},   # average
    {'smell_weight': 0.9, 'explore_noise': 0.1, 'momentum': 0.7, 'lr': 0.1},   # aggressive
    {'smell_weight': 0.2, 'explore_noise': 0.9, 'momentum': 0.0, 'lr': 0.1},   # wanderer
    {'smell_weight': 0.6, 'explore_noise': 0.4, 'momentum': 0.4, 'lr': 0.2},   # learner
    {'smell_weight': 0.4, 'explore_noise': 0.6, 'momentum': 0.2, 'lr': 0.05},  # cautious
    {'smell_weight': 0.7, 'explore_noise': 0.3, 'momentum': 0.6, 'lr': 0.1},   # focused
    {'smell_weight': 0.1, 'explore_noise': 1.0, 'momentum': 0.0, 'lr': 0.05},  # clueless
    {'smell_weight': 0.5, 'explore_noise': 0.5, 'momentum': 0.5, 'lr': 0.15},  # balanced
]

# Different colors for each bug
COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'gray', 'yellow']


class MultiBugWorld:
    """World with 10 bugs competing for food."""
    def __init__(self, size=20, n_food=8, n_walls=10, seed=42):
        np.random.seed(seed)
        self.size = size
        self.grid = np.zeros((size, size))
        self.food_positions = []
        self.wall_positions = []
        self.bug_positions = []
        self.bug_energy = []
        self.bug_foods = []
        self.bug_alive = []

        # Place walls
        for _ in range(n_walls):
            r, c = np.random.randint(2, size-2, 2)
            self.grid[r, c] = -1
            self.wall_positions.append((r, c))

        # Place food
        for _ in range(n_food):
            r, c = np.random.randint(0, size, 2)
            while self.grid[r, c] != 0:
                r, c = np.random.randint(0, size, 2)
            self.grid[r, c] = 1
            self.food_positions.append((r, c))

        # Place bugs at random positions
        for _ in range(10):
            r, c = np.random.randint(0, size, 2)
            while self.grid[r, c] != 0:
                r, c = np.random.randint(0, size, 2)
            self.bug_positions.append(np.array([r, c]))
            self.bug_energy.append(1.0)
            self.bug_foods.append(0)
            self.bug_alive.append(True)

    def sense(self, bug_idx):
        """What does this bug see?"""
        r, c = self.bug_positions[bug_idx]
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

    def innate_bias(self, bug_idx):
        """Innate tropism: move towards food, avoid walls."""
        s = self.sense(bug_idx).reshape(5, 5)
        bias = np.zeros(8)
        dr_list = [-1, -1, 0, 1, 1, 1, 0, -1]
        dc_list = [0, 1, 1, 1, 0, -1, -1, -1]
        for action in range(8):
            dr, dc = dr_list[action], dc_list[action]
            # Where would we end up?
            nr, nc = int(self.bug_positions[bug_idx][0] + dr), int(self.bug_positions[bug_idx][1] + dc)
            if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] == -1:
                bias[action] = -2.0  # avoid walls
            elif (nr, nc) in self.food_positions:
                bias[action] += 3.0  # go for food
        return bias

    def step(self, bug_idx, action):
        """Bug moves, returns True if it ate food."""
        if not self.bug_alive[bug_idx]:
            return False
        dr_list = [-1, -1, 0, 1, 1, 1, 0, -1]
        dc_list = [0, 1, 1, 1, 0, -1, -1, -1]
        dr, dc = dr_list[action], dc_list[action]
        nr, nc = int(self.bug_positions[bug_idx][0] + dr), int(self.bug_positions[bug_idx][1] + dc)
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != -1:
            self.bug_positions[bug_idx] = np.array([nr, nc])
        # Check food
        ate = False
        for (fr, fc) in self.food_positions:
            if (nr, nc) == (fr, fc):
                self.bug_foods[bug_idx] += 1
                self.bug_energy[bug_idx] = min(1.0, self.bug_energy[bug_idx] + 0.3)  # not full refill
                # Respawn food somewhere else
                self.food_positions.remove((fr, fc))
                new_r, new_c = np.random.randint(0, self.size, 2)
                while self.grid[new_r, new_c] != 0:
                    new_r, new_c = np.random.randint(0, self.size, 2)
                self.food_positions.append((new_r, new_c))
                ate = True
                break
        # Energy decay — faster now
        self.bug_energy[bug_idx] -= 0.002
        if self.bug_energy[bug_idx] <= 0:
            self.bug_alive[bug_idx] = False
        return ate


# Setup
world = MultiBugWorld(size=20, n_food=2, n_walls=10, seed=42)
brains = []
for i, params in enumerate(BUGS):
    brain = Brain(n_sensors=25, n_actions=8, lr=params['lr'], world_size=20, seed=i)
    brain.momentum_strength = params['momentum']
    brain.explore_noise = params['explore_noise']
    brain.smell_weight = params['smell_weight']
    brains.append(brain)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)
ax1.set_aspect('equal')

# Draw walls
for r, c in world.wall_positions:
    ax1.plot(c, 19-r, 's', color='darkgreen', markersize=10)

# Draw food
food_scatter = ax1.scatter([], [], color='gold', s=100, marker='s')

# Draw bugs
bug_scatters = []
for i in range(10):
    sc = ax1.scatter([], [], color=COLORS[i], s=150, marker='*', label=f'Bug {i}')
    bug_scatters.append(sc)

ax1.legend(loc='upper right', fontsize=8)
ax1.set_title('Embryo: 10 bugs competing for food')

# Energy plot
ax2.set_xlim(0, 1000)
ax2.set_ylim(0, 1.1)
ax2.set_title('Bug energy over time')
ax2.set_xlabel('Steps')
energy_lines = []
for i in range(10):
    line, = ax2.plot([], [], color=COLORS[i], label=f'Bug {i}')
    energy_lines.append(line)
ax2.legend(loc='upper right', fontsize=8)

energy_history = [[] for _ in range(10)]
step = 0


def update(frame):
    global step
    step += 1

    # Update each bug
    for i in range(10):
        if not world.bug_alive[i]:
            continue
        s = world.sense(i)
        innate = world.innate_bias(i)
        hunger = 1.0 - world.bug_energy[i]
        action = brains[i].decide(s, innate, hunger, world.bug_positions[i])
        world.step(i, action)
        energy_history[i].append(world.bug_energy[i])

    # Update food positions
    fx = [c for r, c in world.food_positions]
    fy = [19-r for r, c in world.food_positions]
    food_scatter.set_offsets(np.c_[fx, fy])

    # Update bug positions
    for i in range(10):
        if world.bug_alive[i]:
            r, c = world.bug_positions[i]
            bug_scatters[i].set_offsets([[c, 19-r]])
        else:
            bug_scatters[i].set_offsets([[-10, -10]])  # hide dead bugs

    # Update energy plot
    for i in range(10):
        if len(energy_history[i]) > 0:
            energy_lines[i].set_data(range(len(energy_history[i])), energy_history[i])
    # Auto-extend x-axis
    ax2.set_xlim(0, max(100, step))

    alive_count = sum(world.bug_alive)
    total_foods = sum(world.bug_foods)
    ax1.set_title(f'Step {step} | Alive: {alive_count}/10 | Total food: {total_foods}')


ani = FuncAnimation(fig, update, frames=1000, interval=50)
plt.show()
