"""
Embryo v3.0: L0-L5 完整系统
  L0 身体+世界：2D grid、4方向动作、能量条
  L1 先天趋性：朝气味浓度高的方向走（硬编码，不学）
  L2 感觉编码：KC稀疏编码（视觉输入）
  L3 习惯层：蘑菇体Hebbian学习，学会"看到cue就往那走"
  L4 空间记忆：海马体，记住食物出现过的位置
  L5 内部模拟：前向模型，在脑子里试动作，不真撞

决策 = 气味 + 视觉习惯 + 空间记忆 + 想象（前向模型）
"""
import numpy as np


class World:
    EMPTY, WALL, FOOD, CUE = 0, 1, 2, 3
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

    def __init__(self, size=15, n_food=6, n_walls=3, n_cues=6, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.size = size
        self.grid = np.full((size, size), self.EMPTY, dtype=int)
        self._place_walls(n_walls)
        self._place_cues_and_foods(n_cues, n_food)
        self.pos = np.array([size // 2, size // 2], dtype=int)
        self.energy = 1.0
        self.steps = 0

    def _place_walls(self, n):
        placed = 0
        while placed < n:
            r, c = np.random.randint(2, self.size - 2, 2)
            if abs(r - self.size // 2) <= 2 and abs(c - self.size // 2) <= 2:
                continue
            if self.grid[r, c] == self.EMPTY:
                self.grid[r, c] = self.WALL
                placed += 1

    def _place_cues_and_foods(self, n_cues, n_food):
        self.cue_positions = []
        placed_cues = 0
        while placed_cues < n_cues:
            r, c = np.random.randint(3, self.size - 3, 2)
            if abs(r - self.size // 2) <= 3 and abs(c - self.size // 2) <= 3:
                continue
            if self.grid[r, c] == self.EMPTY:
                self.grid[r, c] = self.CUE
                self.cue_positions.append((r, c))
                placed_cues += 1
        placed_foods = 0
        for cr, cc in self.cue_positions:
            if placed_foods >= n_food:
                break
            for _ in range(20):
                dr, dc = self.ACTIONS[np.random.randint(4)]
                r, c = cr + dr * 2, cc + dc * 2
                if 0 <= r < self.size and 0 <= c < self.size and self.grid[r, c] == self.EMPTY:
                    self.grid[r, c] = self.FOOD
                    placed_foods += 1
                    break

    def _respawn_food(self):
        for _ in range(100):
            if self.cue_positions:
                cr, cc = self.cue_positions[np.random.randint(len(self.cue_positions))]
                dr, dc = self.ACTIONS[np.random.randint(4)]
                nr, nc = cr + dr * 2, cc + dc * 2
            else:
                nr, nc = np.random.randint(1, self.size - 1, 2)
            if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] == self.EMPTY:
                self.grid[nr, nc] = self.FOOD
                return

    def smell_gradient(self):
        smell = np.zeros(4, dtype=np.float32)
        foods = np.argwhere(self.grid == self.FOOD)
        if len(foods) == 0:
            return smell
        for i, (dr, dc) in enumerate(self.ACTIONS):
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue
            if self.grid[nr, nc] == self.WALL:
                continue
            dists = np.abs(foods[:, 0] - nr) + np.abs(foods[:, 1] - nc)
            min_dist = dists.min()
            if min_dist <= 2:
                smell[i] = 1.0 / (min_dist + 1)
        return smell

    def see(self):
        vision = np.zeros((4, 4), dtype=np.float32)
        for i, (dr, dc) in enumerate(self.ACTIONS):
            for dist in [1, 2]:
                nr, nc = self.pos[0] + dr * dist, self.pos[1] + dc * dist
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    vision[i, 1] = 1.0
                    break
                obj = self.grid[nr, nc]
                if obj == self.WALL:
                    vision[i, 1] = 1.0
                    break
                elif obj == self.CUE:
                    vision[i, 2] = 1.0
                    break
                elif obj == self.FOOD:
                    vision[i, 3] = 1.0
                    break
        return vision.flatten()

    def step(self, action_idx):
        self.steps += 1
        dr, dc = self.ACTIONS[action_idx]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        ate = False
        bumped = False
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != self.WALL:
            self.pos = np.array([nr, nc], dtype=int)
            if self.grid[nr, nc] == self.FOOD:
                self.grid[nr, nc] = self.EMPTY
                self.energy = min(1.0, self.energy + 0.3)
                self._respawn_food()
                ate = True
        else:
            bumped = True
        self.energy = max(0.0, self.energy - 0.004)
        return ate, bumped


class MushroomBody:
    """L2+L3: 蘑菇体（视觉学习）"""
    def __init__(self, n_input=16, n_kc=400, n_actions=4,
                 lr=0.15, trace_decay=0.85, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_actions = n_actions
        self.lr = lr
        self.trace_decay = trace_decay
        self.n_kc = n_kc
        self.W_in = np.random.randn(n_kc, n_input).astype(np.float32) / np.sqrt(n_input)
        self.n_active = max(1, int(n_kc * 0.05))
        self.W_out = np.random.randn(n_actions, n_kc).astype(np.float32) * 0.01
        self.trace = np.zeros((n_actions, n_kc), dtype=np.float32)

    def encode(self, vision):
        kc_acts = self.W_in @ vision
        return np.argpartition(kc_acts, -self.n_active)[-self.n_active:]

    def decide(self, active_kc):
        return self.W_out[:, active_kc].sum(axis=1)

    def update_trace(self, active_kc, chosen_action):
        self.trace *= self.trace_decay
        self.trace[chosen_action, active_kc] = 1.0

    def learn(self, dopamine):
        if dopamine == 0:
            return
        self.W_out += self.lr * dopamine * self.trace
        np.clip(self.W_out, -2.0, 2.0, out=self.W_out)


class Hippocampus:
    """L4: 海马体（空间记忆）"""
    def __init__(self, world_size=15, decay=0.9995):
        self.size = world_size
        self.memory = np.zeros((world_size, world_size), dtype=np.float32)
        self.decay = decay

    def record_food(self, pos):
        r, c = int(pos[0]), int(pos[1])
        self.memory[r, c] += 1.0

    def decay_memory(self):
        self.memory *= self.decay

    def spatial_bias(self, pos):
        bias = np.zeros(4, dtype=np.float32)
        r, c = int(pos[0]), int(pos[1])
        for i, (dr, dc) in enumerate(World.ACTIONS):
            for dist in [1, 2, 3]:
                nr, nc = r + dr * dist, c + dc * dist
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    bias[i] += self.memory[nr, nc] * (1.0 / dist)
        return bias


class ForwardModel:
    """L5: 前向模型（内部模拟）
    学会：在当前状态下，做某个动作 → 下一步会看到什么/得到什么
    决策时在脑子里试每个动作，选预期最好的那个"""

    def __init__(self, n_kc=400, n_actions=4, lr=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_actions = n_actions
        self.lr = lr
        # (当前KC状态, 动作) → 预期价值（会不会吃到食物）
        # 简化：用一个线性模型，KC状态 × 动作 → 预期奖励
        self.W_pred = np.random.randn(n_actions, n_kc).astype(np.float32) * 0.01
        self._prev_kc = None
        self._prev_action = None

    def observe(self, active_kc, action, ate, bumped):
        """每次做完动作，更新前向模型"""
        if self._prev_kc is not None:
            # 实际结果：吃到食物=+1，撞墙=-0.5，otherwise=0
            target = 1.0 if ate else (-0.5 if bumped else 0.0)
            # 预测误差
            pred = self.W_pred[self._prev_action, self._prev_kc].sum()
            error = target - pred
            # 更新权重
            self.W_pred[self._prev_action, self._prev_kc] += self.lr * error
            np.clip(self.W_pred, -1.0, 1.0, out=self.W_pred)
        self._prev_kc = active_kc
        self._prev_action = action

    def imagine(self, active_kc):
        """对4个动作分别模拟预期结果，返回每个动作的预期价值"""
        return self.W_pred[:, active_kc].sum(axis=1)


def run(n_steps=20000, seed=42, verbose=True):
    world = World(size=15, n_food=6, n_walls=3, n_cues=6, seed=seed)
    mb = MushroomBody(n_input=16, n_kc=400, n_actions=4, lr=0.15, trace_decay=0.85, seed=seed)
    hip = Hippocampus(world_size=15, decay=0.9995)
    fm = ForwardModel(n_kc=400, n_actions=4, lr=0.1, seed=seed)

    foods_eaten = 0
    eat_times = []
    last_food_step = 0
    imagination_used = 0  # 靠想象（前向模型）做出的决策次数

    for step in range(n_steps):
        smell = world.smell_gradient()
        vision = world.see()
        active_kc = mb.encode(vision)
        spatial = hip.spatial_bias(world.pos)
        imagined = fm.imagine(active_kc)  # L5：在脑子里试每个动作

        smell_max = smell.max()
        has_cue = (vision.reshape(4, 4)[:, 2] > 0).any()

        # 决策 = 气味 + 视觉习惯 + 空间记忆 + 想象
        habit = mb.decide(active_kc)
        total = smell * 1.0 + habit * 1.0 + spatial * 0.5 + imagined * 0.8
        noise = np.random.randn(4) * 0.2
        action = int(np.argmax(total + noise))

        ate, bumped = world.step(action)
        mb.update_trace(active_kc, action)
        hip.decay_memory()
        fm.observe(active_kc, action, ate, bumped)  # 更新前向模型

        if ate:
            dopamine = 1.0
            foods_eaten += 1
            eat_times.append(step - last_food_step)
            last_food_step = step
            hip.record_food(world.pos)
        elif bumped:
            dopamine = -0.3
        else:
            dopamine = 0.0

        mb.learn(dopamine)

        if world.energy <= 0:
            if verbose:
                recent = eat_times[-5:] if len(eat_times) >= 5 else eat_times
                avg = np.mean(recent) if recent else float('inf')
                print(f"[{step:5d}] 饿死！吃了{foods_eaten}个，平均:{avg:.1f}步")
            world = World(size=15, n_food=6, n_walls=3, n_cues=6, seed=seed + step)
            last_food_step = step

        if verbose and step % 2000 == 0:
            recent = eat_times[-10:] if len(eat_times) >= 10 else eat_times
            avg = np.mean(recent) if recent else float('inf')
            print(f"[{step:5d}] 食物:{foods_eaten:4d}  "
                  f"平均找食:{avg:.1f}步  能量:{world.energy:.2f}")

    print("\n===== 最终结果 =====")
    print(f"总步数: {n_steps}")
    print(f"吃食物总数: {foods_eaten}")
    if len(eat_times) >= 30:
        first_10 = np.mean(eat_times[:10])
        last_10 = np.mean(eat_times[-10:])
        print(f"前10次平均找食: {first_10:.1f} 步")
        print(f"后10次平均找食: {last_10:.1f} 步")
        print(f"提升: {(1 - last_10/first_10)*100:.1f}%")


if __name__ == "__main__":
    run(n_steps=20000, seed=42)
