"""Embryo: full ecosystem with sex, breeding, seasons, gene recombination + mutation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']
SEASON_FOOD = [8, 8, 8, 8]  # food count stays constant — eat one, respawn one
SEASON_LENGTH = 300  # steps per season


class Gene:
    """A bug's genome: 5 traits."""
    def __init__(self, smell_weight=0.5, explore_noise=0.5, momentum=0.3, memory_decay=0.999, lr=0.1):
        self.smell_weight = smell_weight
        self.explore_noise = explore_noise
        self.momentum = momentum
        self.memory_decay = memory_decay
        self.lr = lr

    def crossover(self, other):
        """Gene recombination: mix genes from two parents."""
        child = Gene()
        child.smell_weight = np.random.choice([self.smell_weight, other.smell_weight])
        child.explore_noise = np.random.choice([self.explore_noise, other.explore_noise])
        child.momentum = np.random.choice([self.momentum, other.momentum])
        child.memory_decay = np.random.choice([self.memory_decay, other.memory_decay])
        child.lr = np.random.choice([self.lr, other.lr])
        return child

    def mutate(self, rate=0.1):
        """Random mutation."""
        if np.random.random() < rate:
            self.smell_weight *= np.random.uniform(0.8, 1.2)
        if np.random.random() < rate:
            self.explore_noise *= np.random.uniform(0.8, 1.2)
        if np.random.random() < rate:
            self.momentum *= np.random.uniform(0.8, 1.2)


class Bug:
    """A living bug with genes, sex, energy, and learning ability."""
    def __init__(self, pos, gene, sex=None):
        self.pos = np.array(pos)
        self.gene = gene
        self.sex = sex if sex else np.random.choice(['M', 'F'])
        self.energy = 0.8
        self.alive = True
        self.food_eaten = 0
        self.offspring = 0
        self._prev_action = None
        self._prev_sensory = None
        # Learning matrix: sensory (25) -> action (8)
        # This is W — learned through experience, not genetic
        self.W = np.zeros((8, 25))

    def learn(self, sensory, action, reward):
        """Hebbian learning: reinforce or weaken the sensory-action pair.
        reward > 0: good outcome (found food)
        reward < 0: bad outcome (hit wall, starving)
        """
        lr = 0.1
        self.W[action] += lr * reward * sensory
        # Clip to prevent explosion
        self.W = np.clip(self.W, -1.0, 1.0)

    def decide(self, sensory, world):
        """Decision: innate bias + learned associations."""
        # Learned component: what does my experience say?
        learned = self.W @ sensory

        food_in_view = np.argwhere(sensory.reshape(5, 5) == 2)
        if len(food_in_view) > 0:
            # Move towards nearest food (innate reflex)
            dr, dc = food_in_view[0] - 2
            if dr < 0: innate = 0
            elif dr > 0: innate = 4
            else: innate = 2 if dc > 0 else 6
            # Combine: innate + learned
            total = np.zeros(8)
            total[innate] += 2.0  # innate reflex is strong
            total += learned * self.gene.smell_weight  # learned adds on top
            action = np.argmax(total)
        else:
            # No food in view: use learned + genetic exploration
            total = learned * self.gene.smell_weight
            if self._prev_action is not None:
                total[self._prev_action] += self.gene.momentum
            if np.random.random() < self.gene.explore_noise:
                action = np.random.randint(8)
            else:
                action = np.argmax(total)

        self._prev_sensory = sensory.copy()
        self._prev_action = action
        return action


class Ecosystem:
    """The world: grid, food, bugs, seasons, cues, and learned associations."""
    def __init__(self, size=20, seed=42):
        np.random.seed(seed)
        self.size = size
        self.grid = np.zeros((size, size))
        self.food_positions = []
        self.cue_positions = []  # blue cues: predict food nearby
        self.bugs = []
        self.step_count = 0
        self.born_count = 0
        self.died_count = 0

        # Place walls
        for _ in range(10):
            r, c = np.random.randint(2, size-2, 2)
            self.grid[r, c] = -1

        # Place blue cues (they predict food nearby — bugs must learn this association)
        for _ in range(4):
            r, c = np.random.randint(0, size, 2)
            while self.grid[r, c] != 0 or (r, c) in self.cue_positions:
                r, c = np.random.randint(0, size, 2)
            self.cue_positions.append((r, c))

        # Start with 10 bugs
        for _ in range(10):
            r, c = np.random.randint(0, size, 2)
            while self.grid[r, c] != 0:
                r, c = np.random.randint(0, size, 2)
            self.bugs.append(Bug((r, c), Gene()))

        # Place initial food
        self.season = 0
        self.respawn_food()

    @property
    def season_name(self):
        return SEASONS[self.season]

    def respawn_food(self):
        """Respawn food near blue cues — bugs must learn this association."""
        target = SEASON_FOOD[self.season]
        while len(self.food_positions) < target:
            # Pick a random cue, then place food nearby
            if self.cue_positions:
                cue_r, cue_c = self.cue_positions[np.random.randint(len(self.cue_positions))]
                r_offset, c_offset = np.random.randint(-5, 6, 2)
                r, c = cue_r + r_offset, cue_c + c_offset
                r = max(0, min(self.size-1, r))
                c = max(0, min(self.size-1, c))
            else:
                r, c = np.random.randint(0, self.size, 2)
            if self.grid[r, c] != -1 and (r, c) not in self.food_positions:
                self.food_positions.append((r, c))

    def sense(self, bug):
        """What does this bug see? Food, walls, and blue cues."""
        r, c = bug.pos
        view = np.zeros((5, 5))
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    if (rr, cc) in self.food_positions:
                        view[dr+2, dc+2] = 2  # food
                    elif (rr, cc) in self.cue_positions:
                        view[dr+2, dc+2] = 3  # blue cue
                    elif self.grid[rr, cc] == -1:
                        view[dr+2, dc+2] = -1  # wall
        return view.flatten()

    def step(self):
        """One time step in the ecosystem."""
        self.step_count += 1
        bugs_to_add = []

        # Update season
        if self.step_count % SEASON_LENGTH == 0:
            self.season = (self.season + 1) % 4
            self.respawn_food()

        for bug in self.bugs:
            if not bug.alive:
                continue

            # Decide action
            s = self.sense(bug)
            action = bug.decide(s, self)

            # Move
            dr_list = [-1, -1, 0, 1, 1, 1, 0, -1]
            dc_list = [0, 1, 1, 1, 0, -1, -1, -1]
            dr, dc = dr_list[action], dc_list[action]
            nr, nc = int(bug.pos[0] + dr), int(bug.pos[1] + dc)
            bumped = False
            if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != -1:
                bug.pos = np.array([nr, nc])
            else:
                bumped = True  # hit wall or boundary

            # Eat food
            ate = False
            for (fr, fc) in self.food_positions:
                if (nr, nc) == (fr, fc):
                    bug.energy = min(1.0, bug.energy + 0.3)
                    bug.food_eaten += 1
                    ate = True
                    self.food_positions.remove((fr, fc))
                    # Respawn food near a random cue
                    if self.cue_positions:
                        cue_r, cue_c = self.cue_positions[np.random.randint(len(self.cue_positions))]
                        new_r, new_c = cue_r + np.random.randint(-5, 6), cue_c + np.random.randint(-5, 6)
                        new_r = max(0, min(self.size-1, new_r))
                        new_c = max(0, min(self.size-1, new_c))
                    else:
                        new_r, new_c = np.random.randint(0, self.size, 2)
                    while self.grid[new_r, new_c] == -1 or (new_r, new_c) in self.food_positions:
                        new_r, new_c = np.random.randint(0, self.size, 2)
                    self.food_positions.append((new_r, new_c))
                    break

            # Learning: reward for good outcomes, punishment for bad
            if ate:
                bug.learn(s, action, reward=1.0)  # found food!
            elif bumped:
                bug.learn(s, action, reward=-0.5)  # hit wall

            # Energy decay — density dependent: more bugs = more competition = faster decay
            n_bugs = len([b for b in self.bugs if b.alive])
            bug.energy -= 0.002 + (n_bugs - 10) * 0.0005  # baseline + crowding penalty
            if bug.energy <= 0:
                bug.alive = False
                self.died_count += 1
                continue

            # Breeding: check if adjacent to opposite sex with enough energy
            if bug.energy > 0.6:
                n_bugs = len([b for b in self.bugs if b.alive])
                # If population is too low, asexual reproduction (splitting)
                if n_bugs < 5:
                    if np.random.random() < 0.3:  # 30% chance to split
                        n_children = np.random.randint(3, 8)
                        for _ in range(n_children):
                            child_gene = bug.gene  # clone
                            child_gene.mutate()    # but with mutation
                            # Children spawn nearby
                            r_offset, c_offset = np.random.randint(-2, 3, 2)
                            nr, nc = int(bug.pos[0] + r_offset), int(bug.pos[1] + c_offset)
                            nr = max(0, min(self.size-1, nr))
                            nc = max(0, min(self.size-1, nc))
                            child = Bug((nr, nc), child_gene)
                            child.energy = 0.5
                            bugs_to_add.append(child)
                            bug.offspring += 1
                            self.born_count += 1
                        bug.energy -= 0.4  # splitting costs a lot of energy
                else:
                    # Normal sexual reproduction — slower when population is high
                    breed_chance = 0.2 * (10.0 / max(n_bugs, 5))  # more bugs = slower breeding
                    for other in self.bugs:
                        if not other.alive or other is bug:
                            continue
                        if other.sex == bug.sex:
                            continue
                        if np.linalg.norm(bug.pos - other.pos) < 2:
                            if np.random.random() < breed_chance:
                                child_gene = bug.gene.crossover(other.gene)
                                child_gene.mutate()
                                child = Bug(tuple(bug.pos), child_gene)
                                child.energy = 0.5
                                bugs_to_add.append(child)
                                bug.energy -= 0.2
                                other.energy -= 0.2
                                bug.offspring += 1
                                other.offspring += 1
                                self.born_count += 1
                                break

        # Add newborns
        self.bugs.extend(bugs_to_add)

        # Remove dead bugs
        self.bugs = [b for b in self.bugs if b.alive]


# Setup
eco = Ecosystem(size=100, seed=42)

# Plot: left = world, right = population chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.set_aspect('equal')

# Draw walls
for r in range(100):
    for c in range(100):
        if eco.grid[r, c] == -1:
            ax1.plot(c, 99-r, 's', color='darkgreen', markersize=5)

food_scatter = ax1.scatter([], [], color='gold', s=30, marker='s')
cue_scatter = ax1.scatter([], [], color='cyan', s=80, marker='D')  # blue cues — bigger
bug_scatter_m = ax1.scatter([], [], color='blue', s=40, marker='^')  # male = triangle
bug_scatter_f = ax1.scatter([], [], color='pink', s=40, marker='v')  # female = down-triangle

ax1.set_title('Embryo ecosystem')

# Population plot
ax2.set_xlim(0, 1000)
ax2.set_ylim(0, 50)
ax2.set_title('Bug population')
ax2.set_xlabel('Steps')
pop_line, = ax2.plot([], [], color='red', label='Population')
birth_line, = ax2.plot([], [], color='green', label='Born')
death_line, = ax2.plot([], [], color='black', label='Died')
ax2.legend()

pop_history = []
born_history = []
died_history = []


def update(frame):
    eco.step()

    # Update food
    fx = [c for r, c in eco.food_positions]
    fy = [99-r for r, c in eco.food_positions]
    food_scatter.set_offsets(np.c_[fx, fy])

    # Update cues
    cx = [c for r, c in eco.cue_positions]
    cy = [99-r for r, c in eco.cue_positions]
    cue_scatter.set_offsets(np.c_[cx, cy])

    # Update bugs
    m_pos = [[b.pos[1], 99-b.pos[0]] for b in eco.bugs if b.sex == 'M']
    f_pos = [[b.pos[1], 99-b.pos[0]] for b in eco.bugs if b.sex == 'F']
    bug_scatter_m.set_offsets(m_pos if m_pos else [[-10, -10]])
    bug_scatter_f.set_offsets(f_pos if f_pos else [[-10, -10]])

    # Update population
    pop_history.append(len(eco.bugs))
    born_history.append(eco.born_count)
    died_history.append(eco.died_count)
    pop_line.set_data(range(len(pop_history)), pop_history)
    birth_line.set_data(range(len(born_history)), born_history)
    death_line.set_data(range(len(died_history)), died_history)
    ax2.set_xlim(0, max(100, eco.step_count))

    ax1.set_title(f'Step {eco.step_count} | Season: {eco.season_name} | Bugs: {len(eco.bugs)} | Born: {eco.born_count} | Died: {eco.died_count}')


ani = FuncAnimation(fig, update, frames=2000, interval=30)
plt.show()
