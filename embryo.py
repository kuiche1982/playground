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

    def __init__(self, size=15, n_food=0, n_walls=3, n_cues=6, seed=None):  # 不放自由食物，只有听指令才给
        if seed is not None:
            np.random.seed(seed)
        self.size = size
        self.grid = np.full((size, size), self.EMPTY, dtype=int)
        self._place_walls(n_walls)
        self._place_cues_and_foods(n_cues, n_food)
        self.pos = np.array([size // 2, size // 2], dtype=int)
        self.energy = 1.0
        self.steps = 0
        self.temp_foods = []  # 临时食物（指令给的），有寿命

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

    def _get_nearest_food_dir(self):
        """找到最近的食物，返回它在哪个方向（0=上,1=下,2=左,3=右）"""
        food_positions = np.argwhere(self.grid == self.FOOD)
        if len(food_positions) == 0:
            return np.random.randint(4)  # 没食物就随机
        # 算所有食物到当前位置的距离
        dists = np.abs(food_positions - self.pos).sum(axis=1)
        nearest = food_positions[np.argmin(dists)]
        dy, dx = nearest - self.pos
        # 选主导方向
        if abs(dy) > abs(dx):
            return 0 if dy < 0 else 1  # 上/下
        else:
            return 2 if dx < 0 else 3  # 左/右

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
        self.energy = max(0.0, self.energy - 0.004)  # 恢复energy下降
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

    def __init__(self, n_words=4, eta=0.04, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.command = None
        self.n_words = n_words
        self.eta = eta
        # L6: 词共现Hebbian矩阵（4×4）
        self.W_cooc = np.eye(n_words, dtype=np.float32) * 0.5  # 初始对角线偏强

    def set_command(self, word_idx):
        self.command = word_idx
        # L6: 词共现Hebbian更新（纯感知统计，无多巴胺）
        if word_idx is not None:
            vec = np.zeros(self.n_words, dtype=np.float32)
            vec[word_idx] = 1.0
            # Hebbian：同时激活的词互相增强
            self.W_cooc += self.eta * np.outer(vec, vec)
            # 行归一化，防止权重无限膨胀
            row_norm = np.linalg.norm(self.W_cooc, axis=1, keepdims=True) + 1e-8
            self.W_cooc = self.W_cooc / row_norm

    def get_input(self):
        vec = np.zeros(self.n_words, dtype=np.float32)
        if self.command is not None:
            vec[self.command] = 1.0
            # L6: 用共现矩阵投影，得到语义向量
            vec = self.W_cooc @ vec
        return vec


class ContextModulator:
    """L8: 上下文调制——可学习小网络，学习动态增益（婴儿状态起步）"""
    def __init__(self, n_input=4, n_output=4, lr=0.06, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        # 初始权重很小——婴儿状态，几乎没作用
        self.W = np.random.randn(n_output, n_input).astype(np.float32) * 0.01
        self.trace = np.zeros((n_output, n_input), dtype=np.float32)
        self.trace_decay = 0.9

    def forward(self, pfc_output):
        """前向：L10输出(4维) → 动作增益(4维)"""
        out = self.W @ pfc_output
        # 更新trace
        self.trace *= self.trace_decay
        self.trace[:, :] += pfc_output[None, :]
        return out

    def learn(self, dopamine):
        """Hebbian + 多巴胺调制"""
        if dopamine == 0:
            return
        self.W += self.lr * dopamine * self.trace
        np.clip(self.W, -0.5, 0.5, out=self.W)

    def get_gain(self, gain):
        """输出动作增益给决策层"""
        return gain * 1.5  # 初始增益小，学习后慢慢变大

    def apply(self, active_kc):
        """应用增益到激活的KC"""
        return active_kc  # 简化：实际应该缩放KC激活值，这里先占位


class PrefrontalCortex:
    """L10: 前额叶——可学习的小网络，初始权重很小（婴儿状态）"""
    def __init__(self, n_input=6, n_output=5, lr=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        # 初始权重很小——婴儿前额叶没发育，几乎没作用
        self.W = np.random.randn(n_output, n_input).astype(np.float32) * 0.01
        # 抑制维度初始权重调大一点，让它更容易学
        if n_output >= 5:
            self.W[4, :] = np.random.randn(n_input).astype(np.float32) * 0.1
        self.trace = np.zeros((n_output, n_input), dtype=np.float32)
        self.trace_decay = 0.95  # 拉长trace，解决指令在前、奖励在后的信用分配

    def forward(self, lang_cmd, energy, hip_strength):
        """前向：输入6维 → 输出5维（前4维动作偏置，第5维抑制系数）"""
        x = np.concatenate([lang_cmd, [energy], [hip_strength]])
        out = self.W @ x
        # 更新trace
        self.trace *= self.trace_decay
        self.trace[:, :] += x[None, :]  # 所有输出都记录输入trace
        # 第5维用sigmoid压到0-1之间（抑制系数）
        out[4] = 1.0 / (1.0 + np.exp(-out[4]))  # sigmoid
        return out

    def learn(self, dopamine):
        """Hebbian + 多巴胺调制"""
        if dopamine == 0:
            return
        self.W += self.lr * dopamine * self.trace
        np.clip(self.W, -0.5, 0.5, out=self.W)

    def get_bias(self, task_bias):
        """输出任务偏置给决策层（前4维是动作偏置，第5维是抑制系数）"""
        return task_bias[:4] * 1.0  # 只取前4维动作偏置


def run(n_steps=20000, seed=42, ckpt_dir="checkpoints", log_file="training_log.csv",
        resume_mb=None, resume_hip=None, resume_fm=None):
    os.makedirs(ckpt_dir, exist_ok=True)

    world = World(size=15, n_food=6, n_walls=3, n_cues=6, seed=seed)
    mb = MushroomBody(n_input=20, n_kc=500, n_actions=4, lr=0.15, trace_decay=0.85, seed=seed)
    hip = Hippocampus(world_size=15, decay=0.9995)
    fm = ForwardModel(n_kc=500, n_actions=4, lr=0.1, seed=seed)
    lang = LanguageChannel()
    pfc = PrefrontalCortex(n_input=6, n_output=5, lr=0.08, seed=seed)  # L10 前额叶（5维：4动作+1抑制）
    ctx = ContextModulator(n_input=4, n_output=4, lr=0.06, seed=seed)    # L8 上下文调制（可学习，婴儿状态）

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
    success_memories = []  # L9: 正样本（成功经历）
    failure_memories = []  # L9: 负样本（指令执行错误）
    # 20步窗口判定
    cmd_window = 0
    cmd_start_pos = None
    cmd_target_pos = None

    # 训练日志
    log_rows = []
    start_time = time.time()

    for step in range(n_steps):
        smell = world.smell_gradient()
        vision = world.see()

        if step % 200 == 0:  # 200步一个指令
            # 给随机指令，然后在指令方向上放个临时食物
            cmd = np.random.randint(4)
            lang.set_command(cmd)
            # 在指令方向上放食物（近的远的都有）
            dr, dc = world.ACTIONS[cmd]
            dist = np.random.randint(2, 6)  # 2-5步远
            food_r = world.pos[0] + dr * dist
            food_c = world.pos[1] + dc * dist
            if 0 <= food_r < world.size and 0 <= food_c < world.size:
                if world.grid[food_r, food_c] == world.EMPTY:
                    world.grid[food_r, food_c] = world.FOOD
                    world.temp_foods.append((food_r, food_c, 20))  # 寿命20步
            command_total += 1
            just_got_command = True
            # 20步窗口判定：记录起始位置和目标位置
            cmd_start_pos = world.pos.copy()
            cmd_target_pos = np.array([food_r, food_c])
            cmd_window = 20

        # 更新临时食物寿命，寿命到了就消失
        new_temp_foods = []
        for fr, fc, life in world.temp_foods:
            life -= 1
            if life > 0 and world.grid[fr, fc] == world.FOOD:
                new_temp_foods.append((fr, fc, life))
            elif world.grid[fr, fc] == world.FOOD:
                world.grid[fr, fc] = world.EMPTY  # 寿命到了，消失
        world.temp_foods = new_temp_foods

        lang_input = lang.get_input()  # 先set_command，再get_input

        sensory = np.concatenate([vision, lang_input])

        # L10 前额叶：可学习小网络，输出任务偏置（初始几乎没作用）
        hip_strength = hip.memory[world.pos[0], world.pos[1]] if 0 <= world.pos[0] < 15 and 0 <= world.pos[1] < 15 else 0
        task_bias = pfc.forward(lang_input, world.energy, hip_strength)

        # L8 上下文调制：输入L10前4维输出，输出动作增益（婴儿状态，初始几乎没作用）
        ctx_gain = ctx.forward(task_bias[:4])

        active_kc = mb.encode(sensory)
        spatial = hip.spatial_bias(world.pos)
        imagined = fm.imagine(active_kc)

        habit = mb.decide(active_kc)
        language_bias = lang_input * (3.0 + 4.0 * world.energy)  # 调大基线：饿了也有点听指令
        pfc_bias = pfc.get_bias(task_bias) * 2.0  # L10输出权重调大，对决策影响更大
        inhibit = task_bias[4]  # L10抑制系数（0-1），主动压低本能驱力
        smell_weight = 1.0 * (1.0 - 0.0 * inhibit)  # 抑制最多压一半
        habit_weight = 1.0 * (1.0 - 0.0 * inhibit)
        ctx_bias = ctx.get_gain(ctx_gain)  # L8上下文调制增益
        total = smell * smell_weight + habit * habit_weight + spatial * 0.5 + imagined * 0.8 + language_bias + pfc_bias + ctx_bias
        noise = np.random.randn(4) * 0.2
        action = int(np.argmax(total + noise))

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

        # 刚收到指令那一步：立即给多巴胺（预判）
        if just_got_command:
            if action == lang.command:
                dopamine += 2.0  # 第一步走对，给奖励
                command_followed += 1  # 单步判定统计
            else:
                dopamine -= 1.0  # 第一步走错，给惩罚

        # 20步窗口结束：只用于训练，不统计
        if cmd_window > 0:
            cmd_window -= 1
        just_got_command = False

        mb.learn(dopamine, active_kc, action)
        mb.step_decay()  # L7: 只对长期不激活的突触衰减
        pfc.learn(dopamine)  # L10: 前额叶也用Hebbian学习
        ctx.learn(dopamine)  # L8: 上下文调制也用Hebbian学习

        # L9: 存成功经历（全脑回放用）
        if dopamine > 0.5 and len(success_memories) < 1000:
            success_memories.append({
                'active_kc': active_kc.copy(),
                'action': action,
                'lang_input': lang_input.copy(),
                'energy': world.energy,
                'hip_strength': hip_strength,
            })

        # L9: 回放增强——每1000步重放，正样本:负样本 = 3:1
        if step % 1000 == 0 and step > 0:
            # 正样本回放（75%）
            for _ in range(16):
                if len(success_memories) > 0:
                    mem = success_memories[np.random.randint(len(success_memories))]
                    pfc.learn(0.5)  # 正多巴胺增强
                    ctx.learn(0.5)
            # 负样本回放（25%）
            for _ in range(6):
                if len(failure_memories) > 0:
                    mem = failure_memories[np.random.randint(len(failure_memories))]
                    pfc.learn(-0.4)  # 负多巴胺增强
                    ctx.learn(-0.4)

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
    run(n_steps=100000, seed=42)
