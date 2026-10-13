"""L0+L1: Body with homeostasis and innate tropism.
The agent is BORN knowing how to seek food. It doesn't learn that.
Hebbian plasticity only fine-tunes which visual contexts make it better at it.
"""
import numpy as np


class World:
    EMPTY, WALL, FOOD = 0, 1, 2
    # 8 directional actions: 4 cardinal + 4 diagonals
    ACTIONS = [
        (-1, 0),   # 0: up
        (1, 0),    # 1: down
        (0, -1),   # 2: left
        (0, 1),    # 3: right
        (-1, -1),  # 4: up-left
        (-1, 1),   # 5: up-right
        (1, -1),   # 6: down-left
        (1, 1),    # 7: down-right
    ]
    VIEW = 5

    def __init__(self, size=20, n_food=8, n_walls=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.size = size
        self.grid = np.full((size, size), self.EMPTY, dtype=int)
        self._place_walls(n_walls)
        self._place_food(n_food)
        self.pos = np.array([size // 2, size // 2], dtype=int)
        self.energy = 1.0
        self.steps = 0
        self.visit_map = np.zeros((size, size), dtype=np.float32)
        self.last_bumped = False
        # If we spawned on a food cell, clear it (don't start on food)
        if self.grid[self.pos[0], self.pos[1]] == self.FOOD:
            self.grid[self.pos[0], self.pos[1]] = self.EMPTY

    def _place_walls(self, n):
        placed = 0
        while placed < n:
            r, c = np.random.randint(1, self.size - 1, 2)
            if abs(r - self.size // 2) <= 1 and abs(c - self.size // 2) <= 1:
                continue
            if self.grid[r, c] == self.EMPTY:
                self.grid[r, c] = self.WALL
                placed += 1

    def _place_food(self, n):
        placed = 0
        while placed < n:
            r, c = np.random.randint(1, self.size - 1, 2)
            if self.grid[r, c] == self.EMPTY:
                self.grid[r, c] = self.FOOD
                placed += 1

    def _respawn_food(self):
        for _ in range(50):
            r, c = np.random.randint(1, self.size - 1, 2)
            if self.grid[r, c] == self.EMPTY:
                self.grid[r, c] = self.FOOD
                return

    def sense(self):
        """NxN visual field, agent at center. Returns N*N values (0=empty,1=wall,2=food)."""
        center = self.VIEW // 2
        r0, c0 = self.pos[0] - center, self.pos[1] - center
        view = np.zeros((self.VIEW, self.VIEW), dtype=np.float32)
        for i in range(self.VIEW):
            for j in range(self.VIEW):
                nr, nc = r0 + i, c0 + j
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    view[i, j] = self.grid[nr, nc]
                else:
                    view[i, j] = self.WALL
        return view.flatten()

    def innate_tropism(self, view_flat):
        """Hardwired reflexes: seek nearest food, avoid walls, prefer unvisited cells."""
        view = view_flat.reshape(self.VIEW, self.VIEW)
        center = self.VIEW // 2
        food = np.argwhere(view == self.FOOD)
        bias = np.zeros(8)
        if len(food) > 0:
            dists = np.sum((food - center) ** 2, axis=1)
            fy, fx = food[np.argmin(dists)]
            dy = fy - center
            dx = fx - center
            # Direct the bias toward the food (up/down/left/right + diagonals)
            # Vertical component
            if dy < 0:
                bias[0] = -dy  # up
            elif dy > 0:
                bias[1] = dy   # down
            # Horizontal component
            if dx < 0:
                bias[2] = -dx  # left
            elif dx > 0:
                bias[3] = dx   # right
            # Diagonal: if both dx and dy are non-zero, add diagonal bias
            if dy < 0 and dx < 0:
                bias[4] = min(-dy, -dx)  # up-left
            elif dy < 0 and dx > 0:
                bias[5] = min(-dy, dx)   # up-right
            elif dy > 0 and dx < 0:
                bias[6] = min(dy, -dx)   # down-left
            elif dy > 0 and dx > 0:
                bias[7] = min(dy, dx)    # down-right
        # Olfactory sense: smell food from afar, candle-style decay
        foods = np.argwhere(self.grid == self.FOOD)
        smell_bias = np.zeros(8)
        for fy, fx in foods:
            dist = abs(fy - self.pos[0]) + abs(fx - self.pos[1])
            intensity = 1.0 / (dist + 1)
            dy = fy - self.pos[0]
            dx = fx - self.pos[1]
            if dy < 0 and dx < 0:
                smell_bias[4] += intensity
            elif dy < 0 and dx > 0:
                smell_bias[5] += intensity
            elif dy > 0 and dx < 0:
                smell_bias[6] += intensity
            elif dy > 0 and dx > 0:
                smell_bias[7] += intensity
            elif dy < 0:
                smell_bias[0] += intensity
            elif dy > 0:
                smell_bias[1] += intensity
            elif dx < 0:
                smell_bias[2] += intensity
            elif dx > 0:
                smell_bias[3] += intensity
        bias += smell_bias * 0.5  # smell is weaker than vision

        # Avoid walls adjacent
        center_v = view[center-1:center+2, center-1:center+2]
        if center_v[0, 1] == self.WALL:
            bias[0] -= 1.0  # up
        if center_v[2, 1] == self.WALL:
            bias[1] -= 1.0  # down
        if center_v[1, 0] == self.WALL:
            bias[2] -= 1.0  # left
        if center_v[1, 2] == self.WALL:
            bias[3] -= 1.0  # right
        # Diagonal wall avoidance
        if center_v[0, 0] == self.WALL:
            bias[4] -= 1.0  # up-left
        if center_v[0, 2] == self.WALL:
            bias[5] -= 1.0  # up-right
        if center_v[2, 0] == self.WALL:
            bias[6] -= 1.0  # down-left
        if center_v[2, 2] == self.WALL:
            bias[7] -= 1.0  # down-right
        # Exploration bias: prefer directions leading to less-visited cells
        if len(food) == 0:
            r, c = self.pos
            for i, (dr, dc) in enumerate(self.ACTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    bias[i] -= self.visit_map[nr, nc] * 0.08
        return bias

    def step(self, action_idx):
        """Move in absolute direction. Returns True if ate food."""
        self.steps += 1
        self.last_bumped = False
        dr, dc = self.ACTIONS[action_idx]
        nr, nc = int(self.pos[0] + dr), int(self.pos[1] + dc)
        ate = False
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != self.WALL:
            self.pos = np.array([nr, nc], dtype=int)
            self.visit_map[nr, nc] += 1.0
            if self.grid[nr, nc] == self.FOOD:
                self.grid[nr, nc] = self.EMPTY
                self.energy = min(1.0, self.energy + 0.25)
                self._respawn_food()
                ate = True
        else:
            self.last_bumped = True  # hit a wall or boundary
        self.energy = max(0.0, self.energy - 0.001)
        return ate
