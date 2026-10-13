"""Brain: innate reflex + spatial memory + forward model (L5 prediction).

Instead of just reacting, the brain imagines: "if I step this way, what happens?"
It learns from experience: (position, action) -> (next position, did I find food?).
Then at decision time, it simulates a few steps and picks the best path.
"""
import numpy as np


class Brain:
    def __init__(self, n_sensors=25, n_actions=8, lr=0.2, world_size=20, seed=0):
        np.random.seed(seed)
        self.world_size = world_size
        self.user_command = None

        # Genetic params (B - innate biases, set by evolution)
        self.momentum_strength = 0.3     # how much it keeps going straight
        self.explore_noise = 0.5        # how exploratory it is
        self.smell_weight = 0.5        # how much smell guides it

        # Spatial memory: where food was found
        self.memory_map = np.zeros((world_size, world_size), dtype=np.float32)

        # Forward model memory: (pos, action) -> (next_pos, food_found)
        self.experience = {}

        self._prev_pos = None
        self._prev_action = None

    def _imagine_step(self, pos, action):
        """Imagine stepping in a direction. Return predicted next position + food probability."""
        r, c = int(pos[0]), int(pos[1])
        dr, dc = World.ACTIONS[action]
        nr, nc = r + dr, c + dc
        # Check bounds and walls (we'd need grid access, but approximate)
        # Use experience if we've been here before
        key = (r, c, action)
        if key in self.experience:
            return self.experience[key]
        # Default: just predict we move there
        return (nr, nc, 0.0)

    def decide(self, sensory, innate_bias, hunger, pos):
        food_visible = bool((sensory == 2.0).any())
        if food_visible:
            noise_scale = 0.05
        else:
            # Explore noise is genetic — some bugs are more exploratory
            noise_scale = self.explore_noise * (0.3 + hunger * 0.7)

        total = innate_bias.copy()

        # Momentum is genetic — some bugs keep going straight more
        if self._prev_action is not None:
            total[self._prev_action] += self.momentum_strength

        # L5: forward model — imagine each direction, score by predicted food proximity
        if not food_visible and hunger > 0.2:
            r, c = int(pos[0]), int(pos[1])
            best_score = -1
            best_dir = -1
            for action in range(8):
                nr, nc, ate = self._imagine_step(pos, action)
                score = 0
                if 0 <= nr < self.world_size and 0 <= nc < self.world_size:
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            rr, cc = nr + dr, nc + dc
                            if 0 <= rr < self.world_size and 0 <= cc < self.world_size:
                                score += self.memory_map[rr, cc]
                if score > best_score:
                    best_score = score
                    best_dir = action
            if best_dir >= 0 and best_score > 0.05:
                total[best_dir] += best_score * 2.0

        # User command overrides
        if self.user_command is not None:
            total[self.user_command] += 5.0

        noise = np.random.randn(len(total)) * noise_scale
        action = int(np.argmax(total + noise))
        self._prev_pos = pos.copy()
        self._prev_action = action
        return action

    def reinforce(self, pos):
        """Record experience: when I was at prev_pos and did prev_action, I ended up here and ate food."""
        if self._prev_pos is not None and self._prev_action is not None:
            pr, pc = int(self._prev_pos[0]), int(self._prev_pos[1])
            r, c = int(pos[0]), int(pos[1])
            key = (pr, pc, self._prev_action)
            self.experience[key] = (r, c, 1.0)  # ate food
        r, c = int(pos[0]), int(pos[1])
        self.memory_map[r, c] += 1.0

    def decay_memory(self):
        self.memory_map *= 0.9999


# Import World for ACTIONS lookup (circular import workaround)
from embryo.world import World
