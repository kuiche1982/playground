"""
Embryo v4.0: L0-L6 完整系统 + 训练过程保存
  L0 身体+世界
  L1 先天趋性（气味）
  L2 感觉编码（KC）
  L3 习惯层（蘑菇体）
  L4 空间记忆（海马体）
  L5 内部模拟（前向模型）
  L6 语言通道

训练过程：
  - 每1000步保存一次权重checkpoint
  - 每步记录训练日志（CSV）
  - 能从checkpoint加载继续训练
"""
import numpy as np
import os
import csv
import time


class World:
    EMPTY, WALL, FOOD, CUE = 0, 1, 2, 3
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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
    def __init__(self, n_input=20, n_kc=500, n_actions=4,
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

        # L7: 记忆巩固
        self.activation_count = np.zeros_like(self.W_out)
        self.plasticity = np.ones_like(self.W_out)  # 1.0=可塑, 0.02=固化
        self.consolidation_threshold = 20
        self.base_decay = 0.99999  # 很弱的衰减，20000步只衰减18%

        # L9: 回放（存成功经历）
        self.success_memories = []  # (active_kc, action) 成功的经历

    def encode(self, sensory, kc_gain=None):
        kc_acts = self.W_in @ sensory
        if kc_gain is not None:
            kc_acts = kc_acts * kc_gain  # L8: 上下文增益调制
        return np.argpartition(kc_acts, -self.n_active)[-self.n_active:]

    def decide(self, active_kc):
        return self.W_out[:, active_kc].sum(axis=1)

    def update_trace(self, active_kc, chosen_action):
        self.trace *= self.trace_decay
        self.trace[chosen_action, active_kc] = 1.0

    def learn(self, dopamine, active_kc=None, chosen_action=None):
        if dopamine == 0:
            return
        # Hebbian更新
        self.W_out += self.lr * dopamine * self.trace
        np.clip(self.W_out, -2.0, 2.0, out=self.W_out)

        # L7: 记忆巩固——统计激活次数，触发固化
        if active_kc is not None and chosen_action is not None and dopamine > 0:
            self.activation_count[chosen_action, active_kc] += 1
            # 达到阈值，可塑性降低（固化）
            mask = self.activation_count[chosen_action, active_kc] >= self.consolidation_threshold
            if mask.any():
                indices = np.where(mask)[0]
                self.plasticity[chosen_action, active_kc[indices]] *= 0.6
                self.plasticity[chosen_action, active_kc[indices]] = np.clip(
                    self.plasticity[chosen_action, active_kc[indices]], 0.02, 1.0)

    def step_decay(self):
        """每步的权重衰减——只对长期不激活的突触衰减（遗忘），经常激活的不衰减"""
        # 只有activation_count=0的突触才衰减（从未被强化过的）
        mask = (self.activation_count == 0)
        self.W_out[mask] *= self.base_decay
        # 激活计数缓慢回落（防止永久锁死）
        self.activation_count *= 0.9995

    def replay(self, n_times=10):
        """L9: 回放——重放成功经历，强化浅层连接"""
        if len(self.success_memories) == 0:
            return
        for _ in range(n_times):
            idx = np.random.randint(len(self.success_memories))
            active_kc, action = self.success_memories[idx]
            # 回放时模拟多巴胺强化（弱强化，不要冲乱权重）
            self.W_out[action, active_kc] += self.lr * 0.1
            np.clip(self.W_out, -2.0, 2.0, out=self.W_out)

    def save(self, path):
        np.savez(path, W_in=self.W_in, W_out=self.W_out,
                 activation_count=self.activation_count, plasticity=self.plasticity)

    def load(self, path):
        data = np.load(path)
        self.W_in = data['W_in']
        self.W_out = data['W_out']
        self.activation_count = data['activation_count']
        self.plasticity = data['plasticity']


class Hippocampus:
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

    def save(self, path):
        np.save(path, self.memory)

    def load(self, path):
        self.memory = np.load(path)


class ForwardModel:
    def __init__(self, n_kc=500, n_actions=4, lr=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_actions = n_actions
        self.lr = lr
        self.W_pred = np.random.randn(n_actions, n_kc).astype(np.float32) * 0.01
        self._prev_kc = None
        self._prev_action = None

    def observe(self, active_kc, action, ate, bumped):
        if self._prev_kc is not None:
            target = 1.0 if ate else (-0.5 if bumped else 0.0)
            pred = self.W_pred[self._prev_action, self._prev_kc].sum()
            error = target - pred
            self.W_pred[self._prev_action, self._prev_kc] += self.lr * error
            np.clip(self.W_pred, -1.0, 1.0, out=self.W_pred)
        self._prev_kc = active_kc
        self._prev_action = action

    def imagine(self, active_kc):
        return self.W_pred[:, active_kc].sum(axis=1)

    def save(self, path):
        np.save(path, self.W_pred)

    def load(self, path):
        self.W_pred = np.load(path)


class LanguageChannel:
    WORDS = ["上", "下", "左", "右"]

    def __init__(self):
        self.command = None

    def set_command(self, word_idx):
        self.command = word_idx

    def get_input(self):
        vec = np.zeros(4, dtype=np.float32)
        if self.command is not None:
            vec[self.command] = 1.0
        return vec


class ContextModulator:
    """L8: 上下文调制——临时调节KC神经元的兴奋性（不改权重）"""
    def __init__(self, n_kc=500):
        self.gain = np.ones(n_kc, dtype=np.float32)  # 临时增益，1.0=正常
        self.target_gain = np.ones(n_kc, dtype=np.float32)

    def set_context(self, context_vec, n_kc):
        """根据上下文（语言指令等）设置目标增益"""
        # 简单：指令方向对应的KC集群增益拉高
        for i in range(n_kc):
            if context_vec.max() > 0:
                # 随机分配一部分KC到对应方向
                if i % 4 == np.argmax(context_vec):
                    self.target_gain[i] = 1.5  # 相关KC增益拉高
                else:
                    self.target_gain[i] = 0.7   # 无关KC增益压低
            else:
                self.target_gain[i] = 1.0

    def step(self):
        """每步缓慢过渡（模拟上下文切换的空白期）"""
        self.gain += (self.target_gain - self.gain) * 0.1

    def apply(self, active_kc):
        """应用增益到激活的KC"""
        return active_kc  # 简化：实际应该缩放KC激活值，这里先占位


class PrefrontalCortex:
    """L10: 前额叶——可学习的小网络，初始权重很小（婴儿状态）"""
    def __init__(self, n_input=6, n_output=4, lr=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        # 初始权重很小——婴儿前额叶没发育，几乎没作用
        self.W = np.random.randn(n_output, n_input).astype(np.float32) * 0.01
        self.trace = np.zeros((n_output, n_input), dtype=np.float32)
        self.trace_decay = 0.9

    def forward(self, lang_cmd, energy, hip_strength):
        """前向：输入6维 → 输出4维任务模式偏好"""
        x = np.concatenate([lang_cmd, [energy], [hip_strength]])
        out = self.W @ x
        # 更新trace
        self.trace *= self.trace_decay
        self.trace[:, :] += x[None, :]  # 所有输出都记录输入trace
        return out

    def learn(self, dopamine):
        """Hebbian + 多巴胺调制"""
        if dopamine == 0:
            return
        self.W += self.lr * dopamine * self.trace
        np.clip(self.W, -0.5, 0.5, out=self.W)

    def get_bias(self, task_bias):
        """输出任务偏置给决策层"""
        return task_bias * 1.0  # 初始偏置小，但学习后会慢慢变大


def run(n_steps=20000, seed=42, ckpt_dir="checkpoints", log_file="training_log.csv",
        resume_mb=None, resume_hip=None, resume_fm=None):
    os.makedirs(ckpt_dir, exist_ok=True)

    world = World(size=15, n_food=6, n_walls=3, n_cues=6, seed=seed)
    mb = MushroomBody(n_input=20, n_kc=500, n_actions=4, lr=0.15, trace_decay=0.85, seed=seed)
    hip = Hippocampus(world_size=15, decay=0.9995)
    fm = ForwardModel(n_kc=500, n_actions=4, lr=0.1, seed=seed)
    lang = LanguageChannel()
    pfc = PrefrontalCortex(n_input=6, n_output=4, lr=0.05, seed=seed)  # L10 前额叶（可学习）
    # ctx = ContextModulator(n_kc=500)   # L8 上下文调制——暂时不用，L10直接输出到决策层

    # 从checkpoint继续训练（调参用）
    if resume_mb:
        mb.load(resume_mb)
        print(f"加载蘑菇体权重: {resume_mb}")
    if resume_hip:
        hip.load(resume_hip)
        print(f"加载海马体权重: {resume_hip}")
    if resume_fm:
        fm.load(resume_fm)
        print(f"加载前向模型权重: {resume_fm}")

    foods_eaten = 0
    eat_times = []
    last_food_step = 0
    command_followed = 0
    command_total = 0
    just_got_command = False
    death_count = 0
    success_memories = []  # L9: 全脑回放记忆

    # 训练日志
    log_rows = []
    start_time = time.time()

    for step in range(n_steps):
        smell = world.smell_gradient()
        vision = world.see()
        lang_input = lang.get_input()

        if step % 200 == 0:
            lang.set_command(np.random.randint(4))
            command_total += 1
            just_got_command = True

        sensory = np.concatenate([vision, lang_input])

        # L10 前额叶：可学习小网络，输出任务偏置（初始几乎没作用）
        hip_strength = hip.memory[world.pos[0], world.pos[1]] if 0 <= world.pos[0] < 15 and 0 <= world.pos[1] < 15 else 0
        task_bias = pfc.forward(lang_input, world.energy, hip_strength)

        active_kc = mb.encode(sensory)
        spatial = hip.spatial_bias(world.pos)
        imagined = fm.imagine(active_kc)

        habit = mb.decide(active_kc)
        language_bias = lang_input * 2.0
        pfc_bias = pfc.get_bias(task_bias)  # L10输出加到决策层
        total = smell * 1.0 + habit * 1.0 + spatial * 0.5 + imagined * 0.8 + language_bias + pfc_bias
        noise = np.random.randn(4) * 0.2
        action = int(np.argmax(total + noise))

        if just_got_command and action == lang.command:
            command_followed += 1
        just_got_command = False

        ate, bumped = world.step(action)
        mb.update_trace(active_kc, action)
        hip.decay_memory()
        fm.observe(active_kc, action, ate, bumped)

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

        mb.learn(dopamine, active_kc, action)
        mb.step_decay()  # L7: 只对长期不激活的突触衰减
        pfc.learn(dopamine)  # L10: 前额叶也用Hebbian学习

        # L9: 存成功经历（全脑回放用）
        if dopamine > 0.5 and len(success_memories) < 1000:
            success_memories.append({
                'active_kc': active_kc.copy(),
                'action': action,
                'lang_input': lang_input.copy(),
                'energy': world.energy,
                'hip_strength': hip_strength,
            })

        # L9: 全脑回放——每2000步重放10次，只强化抽象模块（L10/海马体），不碰蘑菇体
        if step % 2000 == 0 and step > 0 and len(success_memories) > 0:
            for _ in range(10):
                mem = success_memories[np.random.randint(len(success_memories))]
                # 只强化L10前额叶（抽象任务判断），不碰蘑菇体
                pfc.learn(0.3)

        # 饿死重置
        if world.energy <= 0:
            death_count += 1
            world = World(size=15, n_food=6, n_walls=3, n_cues=6, seed=seed + step)
            last_food_step = step

        # 记录训练日志（每100步记一次）
        if step % 100 == 0:
            recent = eat_times[-10:] if len(eat_times) >= 10 else eat_times
            avg_eat = np.mean(recent) if recent else -1
            log_rows.append({
                'step': step,
                'foods_eaten': foods_eaten,
                'avg_eat_steps': round(avg_eat, 2),
                'deaths': death_count,
                'command_follow_rate': round(command_followed / max(1, command_total), 3),
                'energy': round(world.energy, 3),
            })

        # 保存checkpoint（每2000步）
        if step % 2000 == 0 and step > 0:
            mb.save(f"{ckpt_dir}/mb_step{step}.npz")
            hip.save(f"{ckpt_dir}/hip_step{step}.npy")
            fm.save(f"{ckpt_dir}/fm_step{step}.npy")

    # 保存最终模型
    mb.save(f"{ckpt_dir}/mb_final.npz")
    hip.save(f"{ckpt_dir}/hip_final.npy")
    fm.save(f"{ckpt_dir}/fm_final.npy")

    # 写CSV日志
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    elapsed = time.time() - start_time
    print(f"\n===== 训练完成 =====")
    print(f"总步数: {n_steps}")
    print(f"耗时: {elapsed:.1f}秒")
    print(f"吃食物总数: {foods_eaten}")
    print(f"饿死次数: {death_count}")
    print(f"听懂指令: {command_followed}/{command_total} ({command_followed/max(1,command_total)*100:.1f}%)")
    if len(eat_times) >= 30:
        first_10 = np.mean(eat_times[:10])
        last_10 = np.mean(eat_times[-10:])
        print(f"前10次找食: {first_10:.1f}步 → 后10次: {last_10:.1f}步 (提升{(1-last_10/first_10)*100:.1f}%)")
    print(f"checkpoint保存在: {ckpt_dir}/")
    print(f"训练日志保存在: {log_file}")


if __name__ == "__main__":
    run(n_steps=20000, seed=42)
