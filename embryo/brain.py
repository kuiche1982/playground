"""Brain: Hebbian modulation + spatial memory + random strategy selection (L5)."""
import numpy as np


class Brain:
    def __init__(self, n_sensors=25, n_actions=8, lr=0.2, world_size=20, seed=0):
        np.random.seed(seed)
        self.W = np.zeros((n_actions, n_sensors), dtype=np.float32)
        self.lr = lr
        self._prev_sensory = None
        self._prev_action = None
        self.memory_map = np.zeros((world_size, world_size), dtype=np.float32)
        self.world_size = world_size
        self.user_command = None  # external direction command (0-7 or None)

    def _scan_memory(self, pos, steps_ahead=2):
        """Scan all 8 directions, return (best_value, best_dir) from memory."""
        r, c = int(pos[0]), int(pos[1])
        best_val = 0
        best_dir = -1
        for dr, dc, idx in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3),
                             (-1, -1, 4), (-1, 1, 5), (1, -1, 6), (1, 1, 7)]:
            val = 0.0
            for s in range(1, steps_ahead + 1):
                nr, nc = r + dr * s, c + dc * s
                if 0 <= nr < self.world_size and 0 <= nc < self.world_size:
                    val += self.memory_map[nr, nc] / s
            if val > best_val:
                best_val = val
                best_dir = idx
        return best_val, best_dir

    def decide(self, sensory, innate_bias, hunger, pos):
        food_visible = bool((sensory == 2.0).any())
        if food_visible:
            noise_scale = 0.05
        else:
            noise_scale = 0.3 + hunger * 0.7
        learned = self.W @ sensory
        total = innate_bias + learned

        # L5: random strategy selection — instinct / recall / preview
        # 60% instinct, 20% recall (go to remembered food spot), 20% preview (look ahead)
        strategy = np.random.choice(['instinct', 'recall', 'preview'], p=[0.6, 0.2, 0.2])

        if not food_visible:
            if strategy == 'recall':
                val, dir_idx = self._scan_memory(pos, steps_ahead=3)
                if dir_idx >= 0 and val > 0.1:
                    total[dir_idx] += val * 3.0  # strong recall pull
            elif strategy == 'preview':
                val, dir_idx = self._scan_memory(pos, steps_ahead=2)
                if dir_idx >= 0 and val > 0.05:
                    total[dir_idx] += val * 2.0  # lighter preview

        # User command overrides everything (L6)
        if self.user_command is not None:
            total[self.user_command] += 5.0

        # Momentum
        if self._prev_action is not None:
            total[self._prev_action] += 0.3

        noise = np.random.randn(len(total)) * noise_scale
        action = int(np.argmax(total + noise))
        self._prev_sensory = sensory
        self._prev_action = action
        return action

    def reinforce(self, pos):
        if self._prev_sensory is not None and self._prev_action is not None:
            self.W[self._prev_action] += self.lr * self._prev_sensory
            np.clip(self.W, -1.0, 1.0, out=self.W)
        r, c = int(pos[0]), int(pos[1])
        self.memory_map[r, c] += 1.0

    def decay_memory(self):
        self.memory_map *= 0.9999
