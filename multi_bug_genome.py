"""Embryo: 基因×经历 个体分化实验（v2 真学习版）

修正 v1 的三个错误：
  1. 基因从"决策公式乘法"移到"学习参数 + 硬件"（lr/trace/多巴胺敏感度/嗅觉灵敏度/代谢）
  2. 打开真学习：mb/pfc/ctx/hip/fm 全部 Hebbian 更新
  3. 共享权重 = 出生起点，各自学习 → 权重漂移 = 个体分化

三组对照（每组 10 虫，共享世界抢食物）：
  A 经历组    ：无基因差异 + 随机出生 + 均匀食物 + 学习      → 纯经历能分化吗
  B 基因+经历 ：基因差异 + 随机出生 + 均匀食物 + 学习         → 基因放大分化
  C 基因+环境 ：基因差异 + 随机出生 + 资源斑块 + 学习         → G×E 交互
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from embryo import World, MushroomBody, Hippocampus, ForwardModel, LanguageChannel, ContextModulator, PrefrontalCortex


def smell_at(world, pos):
    smell = np.zeros(4, dtype=np.float32)
    # 可重叠食物：嗅觉看计数层（food_count>0 即存在）
    if not hasattr(world, 'food_count'):
        foods = np.argwhere(world.grid == world.FOOD)
    else:
        foods = np.argwhere(world.food_count > 0)
    if len(foods) == 0:
        return smell
    for i, (dr, dc) in enumerate(world.ACTIONS):
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < world.size and 0 <= nc < world.size):
            continue
        if world.grid[nr, nc] == world.WALL:
            continue
        dists = np.abs(foods[:, 0] - nr) + np.abs(foods[:, 1] - nc)
        min_dist = dists.min()
        if min_dist <= 2:
            smell[i] = 1.0 / (min_dist + 1)
    return smell


def see_at(world, pos):
    vision = np.zeros((4, 4), dtype=np.float32)
    for i, (dr, dc) in enumerate(world.ACTIONS):
        for dist in [1, 2]:
            nr, nc = pos[0] + dr * dist, pos[1] + dc * dist
            if not (0 <= nr < world.size and 0 <= nc < world.size):
                vision[i, 1] = 1.0
                break
            obj = world.grid[nr, nc]
            if obj == world.WALL:
                vision[i, 1] = 1.0
                break
            elif obj == world.CUE:
                vision[i, 2] = 1.0
                break
            elif obj == world.FOOD:
                vision[i, 3] = 1.0
                break
    return vision.flatten()


QUADRANT_WEIGHTS = [0.45, 0.20, 0.20, 0.15]

def respawn_food_patchy(world):
    weights = getattr(world, 'weights', QUADRANT_WEIGHTS)
    for _ in range(200):
        q = np.random.choice(4, p=weights)
        half = world.size // 2
        if q == 0:
            r, c = np.random.randint(1, half), np.random.randint(1, half)
        elif q == 1:
            r, c = np.random.randint(1, half), np.random.randint(half, world.size - 1)
        elif q == 2:
            r, c = np.random.randint(half, world.size - 1), np.random.randint(1, half)
        else:
            r, c = np.random.randint(half, world.size - 1), np.random.randint(half, world.size - 1)
        if world.grid[r, c] == world.EMPTY:
            world.grid[r, c] = world.FOOD
            world.food_count[r, c] += 1
            return


def respawn_food_uniform(world):
    for _ in range(200):
        r, c = np.random.randint(1, world.size - 1, 2)
        if world.grid[r, c] == world.EMPTY:
            world.grid[r, c] = world.FOOD
            world.food_count[r, c] += 1
            return


class EnvHebbian:
    """环境持续学习网络：无监督Hebbian，不依赖奖惩
    v3 修复三缺陷：
      1. 方向性写入：线索来源方向累加（非脚下）→ density = 空间地图而非路径历史
      2. 全局梯度导航：bias 扫描 10 步朝富集区（远程导航）
      3. 近期访问抑制：走过的格子衰减 → 消除路径锁定"""
    def __init__(self, size=15, decay=0.999, gain=1.0):
        self.size = size
        self.density = np.zeros((size, size), dtype=np.float32)
        self.decay = decay      # 环境跟踪速度（基因可调）
        self.gain = gain        # 环境偏置强度（基因可调）

    def observe(self, pos, smell, vision):
        """方向性写入：smell[i] 表示 i 方向有食物 → density 累加到该方向 2-3 格"""
        r, c = int(pos[0]), int(pos[1])
        for i, (dr, dc) in enumerate(World.ACTIONS):
            strength = float(smell[i])
            if strength > 0:
                for dist in [2, 3]:
                    nr, nc = r + dr * dist, c + dc * dist
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        self.density[nr, nc] = min(5.0, self.density[nr, nc] + strength * 0.5)
        # 脚下只留弱痕迹（避免完全丢失路径信息）
        if 0 <= r < self.size and 0 <= c < self.size:
            self.density[r, c] = min(5.0, self.density[r, c] + 0.1)
        self.density *= self.decay

    def visit_inhibit(self, pos):
        """近期访问抑制：走过的格子短期衰减（模拟海马重复抑制）"""
        r, c = int(pos[0]), int(pos[1])
        if 0 <= r < self.size and 0 <= c < self.size:
            self.density[r, c] *= 0.9

    def bias(self, pos):
        """全局梯度导航：4 方向扫描 10 步，朝 density 累计最大的方向走"""
        bias = np.zeros(4, dtype=np.float32)
        r, c = int(pos[0]), int(pos[1])
        for i, (dr, dc) in enumerate(World.ACTIONS):
            total = 0.0
            for dist in range(1, 11):
                nr, nc = r + dr * dist, c + dc * dist
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                total += self.density[nr, nc] / dist
            bias[i] = total
        return bias * self.gain


def place_stable(world, rng, corner='tl', n_cues=6, n_food=10):
    """在指定角落放 cue + 附近食物（稳定环境结构，产区搬家用）
    稀疏 cue：间距 ≥4，避免食物工厂；角落范围随 world.size 缩放
    食物可重叠堆叠（food_count 计数层+1），消除物理格子上限"""
    world.grid[world.grid == world.FOOD] = world.EMPTY
    world.grid[world.grid == world.CUE] = world.EMPTY
    world.food_count[:] = 0
    world.cue_positions = []
    if corner == 'tl':
        lo, hi = 2, max(7, int(world.size * 0.45))
    else:  # br
        lo, hi = int(world.size * 0.55), world.size - 2
    placed = 0
    attempts = 0
    while placed < n_cues:
        r, c = rng.randint(lo, hi, 2)
        if world.grid[r, c] != world.EMPTY:
            attempts += 1
            if attempts > 800:
                break
            continue
        # 与已有 cue 保持间距
        if any(abs(r - cr) + abs(c - cc) < 4 for cr, cc in world.cue_positions):
            attempts += 1
            if attempts > 800:
                break
            continue
        world.grid[r, c] = world.CUE
        world.cue_positions.append((r, c))
        placed += 1
    if not world.cue_positions:
        return
    nf = 0
    # 食物堆叠：n_food 平均分配到各 cue 附近（可叠在同一格）
    per = max(1, n_food // max(1, len(world.cue_positions)))
    while nf < n_food:
        for cr, cc in world.cue_positions:
            if nf >= n_food:
                break
            dr, dc = world.ACTIONS[rng.randint(4)]
            r, c = cr + dr * 2, cc + dc * 2
            if 0 <= r < world.size and 0 <= c < world.size and world.grid[r, c] != world.WALL:
                world.grid[r, c] = world.FOOD
                world.food_count[r, c] += 1
                nf += 1
                if nf % per == 0:
                    break


def safe_respawn(world):
    """cue 绑定 respawn，食物可堆叠（计数层+1），无需避开已有食物位"""
    bugs = getattr(world, 'bugs', [])
    for _ in range(300):
        if world.cue_positions:
            cr, cc = world.cue_positions[np.random.randint(len(world.cue_positions))]
            dr, dc = world.ACTIONS[np.random.randint(4)]
            nr, nc = cr + dr * 2, cc + dc * 2
        else:
            nr, nc = np.random.randint(1, world.size - 1, 2)
        if 0 <= nr < world.size and 0 <= nc < world.size and world.grid[nr, nc] != world.WALL:
            if all(abs(b.pos[0] - nr) + abs(b.pos[1] - nc) >= 2 for b in bugs):
                world.grid[nr, nc] = world.FOOD
                world.food_count[nr, nc] += 1
                return


class Bug:
    """一只虫子：共享权重出生，基因决定学习规则和硬件，经历改写权重"""

    def __init__(self, pos, genome, ckpt='checkpoints', seed=42, learn_flags=None, shared=None,
                 energy_cost_base=0.002, patchy=False, env_gain=None, env_decay=None,
                 stable=False, world_size=15):
        self.pos = np.array(pos, dtype=int)
        self.world_size = world_size
        self.energy = 0.8
        self.alive = True
        self.foods = 0
        self.deaths = 0
        self.actions = []
        self.energy_cost_base = energy_cost_base
        self.patchy = patchy
        self.stable = stable
        # 学习组件开关（诊断用）
        self.lf = learn_flags or {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True}

        # ---- 基因（出生固定，不随经历改变）----
        g = genome
        self.genes = g
        self.lr_gain = g.get('lr_gain', 1.0)           # 学习速率乘子
        self.trace_gain = g.get('trace_gain', 1.0)     # 记忆痕迹衰减乘子
        self.da_gain = g.get('da_gain', 1.0)           # 多巴胺敏感度
        self.smell_sens = g.get('smell_sens', 1.0)     # 嗅觉硬件灵敏度（感知输入层）
        self.energy_cost = g.get('energy_cost', 1.0)   # 代谢率
        self.explore_noise = g.get('explore_noise', 0.2)
        # 环境持续学习网络（独立Hebbian，不吃奖惩；由 lf['env'] 控制开关）
        self.env = (EnvHebbian(size=world_size,
                               decay=env_decay if env_decay is not None else g.get('env_decay', 0.999),
                               gain=env_gain if env_gain is not None else g.get('env_gain', 1.0))
                    if self.lf.get('env', False) else None)

        if shared is not None:
            # 诊断模式：10 虫共享同一套大脑对象（复现 v1 行为）
            self.mb, self.hip, self.fm, self.pfc, self.ctx = shared
            self.init_W = self.mb.W_out.copy()
            return

        # ---- 大脑：加载同一套训练权重（共享出生点）----
        mb = MushroomBody(n_input=20, n_kc=500, n_actions=4, lr=0.15, trace_decay=0.85, seed=seed)
        mb.load(f'{ckpt}/mb_final.npz')
        mb.lr = 0.03 * self.lr_gain               # 基因 → 增量学习速率（小lr，不冲垮训练权重）
        mb.trace_decay = min(0.99, 0.85 * self.trace_gain)  # 基因 → 痕迹衰减
        self.mb = mb
        self.init_W = mb.W_out.copy()               # 记录出生权重（算漂移）

        hip = Hippocampus(world_size=world_size, decay=0.9995)
        if world_size == 15:
            hip.load(f'{ckpt}/hip_final.npy')
        else:
            # 大世界：训练好的 15×15 本能地图嵌入左上角（对应稳定产区），其余空白
            small = np.load(f'{ckpt}/hip_final.npy')
            hip.memory[:small.shape[0], :small.shape[1]] = small
        self.hip = hip

        fm = ForwardModel(n_kc=500, n_actions=4, lr=0.1, seed=seed)
        fm.load(f'{ckpt}/fm_final.npy')
        self.fm = fm

        self.lang = LanguageChannel()
        # pfc/ctx 无 checkpoint，用同一套 seed 初始化（婴儿状态，同一起点）
        self.pfc = PrefrontalCortex(n_input=6, n_hidden=8, n_output=5, lr=0.25, seed=seed)
        self.ctx = ContextModulator(n_input=4, n_output=4, lr=0.06, seed=seed)

    def decide_and_learn(self, world):
        # ---- 感知（每虫独立）----
        smell = smell_at(world, self.pos) * self.smell_sens   # 基因：嗅觉硬件
        vision = see_at(world, self.pos)
        lang_input = np.zeros(4, dtype=np.float32)
        sensory = np.concatenate([vision, lang_input])

        r, c = int(self.pos[0]), int(self.pos[1])
        hip_strength = self.hip.memory[r, c] if (0 <= r < self.world_size and 0 <= c < self.world_size) else 0.0
        task_bias = self.pfc.forward(lang_input, self.energy, hip_strength)
        ctx_gain = self.ctx.forward(task_bias[:4])
        active_kc = self.mb.encode(sensory)
        spatial = self.hip.spatial_bias(self.pos)
        imagined = self.fm.imagine(active_kc)
        habit = self.mb.decide(active_kc)
        pfc_bias = self.pfc.get_bias(task_bias)
        inhibit = task_bias[4]
        smell_weight = 1.0 * (1.0 - inhibit)      # 基因不再直接乘决策公式
        habit_weight = 1.0 * (1.0 - inhibit)
        ctx_bias = self.ctx.get_gain(ctx_gain)
        env_bias = self.env.bias(self.pos) if self.env else np.zeros(4, dtype=np.float32)
        total = (smell * smell_weight + habit * habit_weight
                 + spatial * 0.5 + imagined * 0.8
                 + pfc_bias * 0.5 + ctx_bias + env_bias)
        noise = np.random.randn(4) * self.explore_noise
        action = int(np.argmax(total + noise))
        self.actions.append(action)

        # ---- 移动 ----
        dr, dc = world.ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        ate = bumped = False
        if 0 <= nr < world.size and 0 <= nc < world.size and world.grid[nr, nc] != world.WALL:
            self.pos = np.array([nr, nc])
            if world.grid[nr, nc] == world.FOOD:
                # 可重叠食物：计数层-1，减到 0 才清格
                world.food_count[nr, nc] -= 1
                if world.food_count[nr, nc] <= 0:
                    world.grid[nr, nc] = world.EMPTY
                self.foods += 1
                self.energy = min(1.0, self.energy + 0.3)
                ate = True
                # 食物补充（v1 有、v2 重构时漏掉的 respawn！）
                if self.stable:
                    safe_respawn(world)      # 绑定 cue + 防原地刷
                elif self.patchy:
                    respawn_food_patchy(world)
                else:
                    respawn_food_uniform(world)
        else:
            bumped = True

        # ---- 学习（经历改写权重）----
        if self.env is not None:
            self.env.observe(self.pos, smell, vision)   # 环境网络每步无监督学习（不吃奖惩）
            self.env.visit_inhibit(self.pos)            # 近期访问抑制：防路径锁定
        if self.lf['mb']:
            self.mb.update_trace(active_kc, action)
        # 多虫竞争下每虫正样本稀疏，训练好的空间记忆不能被 decay 抹光
        if self.lf['hip']:
            # self.hip.decay_memory()   # 关闭：出生共享地图保持为"本能地图"
            if ate:
                self.hip.record_food(self.pos)
        if self.lf['fm']:
            self.fm.lr = 0.02           # 前向模型只微调（避免被"没吃没撞=0"常态冲刷）
            self.fm.observe(active_kc, action, ate, bumped)

        dopamine = 1.0 if ate else (-0.3 if bumped else 0.0)
        dopamine *= self.da_gain                      # 基因：奖励敏感度
        if self.lf['mb']:
            self.mb.learn(dopamine, active_kc, action)
        if self.lf['pfc']:
            self.pfc.learn(dopamine)
        if self.lf['ctx']:
            self.ctx.learn(dopamine)

        # ---- 代谢（基因：能量消耗）----
        self.energy -= self.energy_cost_base * self.energy_cost
        if self.energy <= 0:
            self.alive = False
            self.deaths += 1

    def drift(self):
        """权重漂移 = 当前大脑离出生大脑多远（个体分化的直接证据）"""
        return float(np.linalg.norm(self.mb.W_out - self.init_W))


def make_genome(rng, scale=1.0):
    """基因差异放大：scale 控制变异幅度（1.0=弱, 2.0=强）"""
    return {
        'lr_gain': rng.uniform(1.0 - 0.8 * scale, 1.0 + 0.5 * scale),   # 学得快/慢
        'trace_gain': rng.uniform(1.0 - 0.3 * scale, 1.0 + 0.15 * scale),  # 记得久/忘得快
        'da_gain': rng.uniform(1.0 - 0.5 * scale, 1.0 + 0.5 * scale),  # 奖励敏感/迟钝
        'smell_sens': rng.uniform(1.0 - 0.7 * scale, 1.0 + 1.0 * scale),  # 鼻子灵/钝
        'energy_cost': rng.uniform(1.0 - 0.3 * scale, 1.0 + 0.5 * scale),  # 代谢低/高
        'explore_noise': rng.uniform(0.05 * scale, 0.2 + 0.3 * scale),  # 探索/保守
        'env_decay': rng.uniform(0.997, 0.9999),   # 环境跟踪速度（快=易适应变化/易被噪声骗）
        'env_gain': rng.uniform(0.5, 2.0),         # 环境偏置强度（敏感/迟钝）
    }


def run_group(name, use_genes, patchy, n_bugs=10, steps=6000, seed=42, learn_flags=None,
              share_brain=False, n_food=10, energy_cost_base=0.002, gene_scale=1.0,
              flip_at=None, env_gain=None, env_decay=None, stable=False, n_cues=6,
              world_size=15):
    rng = np.random.RandomState(seed)
    world = World(size=world_size, n_food=n_food, n_walls=3, n_cues=6, seed=seed)
    world.food_count = np.zeros((world_size, world_size), dtype=np.int32)  # 可重叠食物计数层
    # 同步 World.__init__ 放置的初始食物到计数层
    for (fr, fc) in np.argwhere(world.grid == world.FOOD):
        world.food_count[fr, fc] = 1
    if patchy:
        world.weights = QUADRANT_WEIGHTS.copy()
        world.grid = np.full((world_size, world_size), World.EMPTY, dtype=int)
        world._place_walls(3)
        for _ in range(n_food):
            respawn_food_patchy(world)

    bugs = []
    shared = None
    if share_brain:
        # 复现 v1：10 虫共享同一套大脑对象
        mb_s = MushroomBody(n_input=20, n_kc=500, n_actions=4, lr=0.15, trace_decay=0.85, seed=seed)
        mb_s.load('checkpoints/mb_final.npz')
        hip_s = Hippocampus(world_size=world_size, decay=0.9995)
        hip_s.load('checkpoints/hip_final.npy')
        fm_s = ForwardModel(n_kc=500, n_actions=4, lr=0.1, seed=seed)
        fm_s.load('checkpoints/fm_final.npy')
        pfc_s = PrefrontalCortex(n_input=6, n_hidden=8, n_output=5, lr=0.25, seed=seed)
        ctx_s = ContextModulator(n_input=4, n_output=4, lr=0.06, seed=seed)
        shared = (mb_s, hip_s, fm_s, pfc_s, ctx_s)

    # 稳定结构：初始用稀疏 cue 产区（左上）
    if stable:
        place_stable(world, rng, corner='tl', n_cues=n_cues, n_food=n_food)

    birth_lo, birth_hi = 2, world_size - 2
    for i in range(n_bugs):
        r, c = rng.randint(birth_lo, birth_hi, 2)
        while world.grid[r, c] != World.EMPTY:
            r, c = rng.randint(birth_lo, birth_hi, 2)
        genome = make_genome(rng, gene_scale) if use_genes else {}
        bugs.append(Bug([r, c], genome, seed=42, learn_flags=learn_flags, shared=shared,
                        energy_cost_base=energy_cost_base, patchy=patchy,
                        env_gain=env_gain, env_decay=env_decay, stable=stable,
                        world_size=world_size))
    world.bugs = bugs   # 供 safe_respawn 检查距离

    pair_dists = []
    total_deaths = 0
    flip_mark = None
    cue_arr = np.array(world.cue_positions) if stable else None
    near_cue_steps = 0   # 大世界指标：产区停留步数

    for step in range(steps):
        if flip_at is not None and step == flip_at:
            # 环境突变：食物产区从左上迁移到右下
            if stable:
                place_stable(world, rng, corner='br', n_cues=n_cues, n_food=n_food)
                cue_arr = np.array(world.cue_positions)
            else:
                world.weights = [0.15, 0.20, 0.20, 0.45]
                world.grid[world.grid == world.FOOD] = world.EMPTY
                for _ in range(n_food):
                    respawn_food_patchy(world)
            flip_mark = {id(b): b.foods for b in bugs}
        for bug in bugs:
            if bug.alive:
                bug.decide_and_learn(world)
                if cue_arr is not None and len(cue_arr):
                    d = np.abs(cue_arr - bug.pos).sum(axis=1).min()
                    if d <= 3:
                        near_cue_steps += 1
            else:
                # 饿死复活：保留大脑和基因，换个出生点继续发育
                bug.energy = 0.8
                bug.pos = np.array([rng.randint(birth_lo, birth_hi), rng.randint(birth_lo, birth_hi)], dtype=int)
                bug.alive = True
                total_deaths += 1
        if step % 100 == 0:
            dists = [np.abs(bugs[i].pos - bugs[j].pos).sum()
                     for i in range(len(bugs)) for j in range(i + 1, len(bugs))]
            pair_dists.append(np.mean(dists))

    foods = np.array([b.foods for b in bugs])
    gini = np.abs(foods[:, None] - foods[None, :]).sum() / (2 * len(foods) ** 2 * foods.mean() + 1e-9)
    ents = []
    for b in bugs:
        if b.actions:
            hist = np.bincount(np.array(b.actions), minlength=4).astype(float)
            p = hist / hist.sum()
            ents.append(-(p * np.log(p + 1e-9)).sum())
    drifts = np.array([b.drift() for b in bugs])
    post_flip = None
    if flip_mark is not None:
        post_flip = np.array([b.foods - flip_mark[id(b)] for b in bugs])
    result = {
        'name': name, 'foods': foods, 'foods_mean': foods.mean(), 'foods_std': foods.std(),
        'gini': gini, 'pair_dist_mean': np.mean(pair_dists[-20:]),
        'entropy_mean': np.mean(ents), 'deaths': total_deaths,
        'drift_mean': drifts.mean(), 'drift_std': drifts.std(), 'drifts': drifts,
        'pair_dists': pair_dists,
        'post_flip': post_flip,
        'near_cue_ratio': near_cue_steps / max(1, steps * n_bugs),
    }
    return result


def evolve(n_gens=10, steps_per_gen=3000, n_bugs=20, seed=42, n_cues=2,
           test_steps=2000, env_pool=None, top_k=8, world_size=15, n_food=10,
           cross=False, learn_env=True, migrate_at=None):
    """Baldwin效应验证：固化指标 = 出生测试时到产区的平均距离
    新生儿不学习 → 若环境适应已固化进基因，出生就有"去产区"的空间偏好
    env_pool: None=跨代稳定；'shuffle'=每代换布局（对照组）
    learn_env: 个体是否带 EnvHebbian 环境网络（耦合验证）
    migrate_at: 第 N 代起产区从左上迁到右下（固化惯性验证）"""
    rng = np.random.RandomState(seed)
    pool = [make_genome(rng, 1.0) for _ in range(n_bugs)]
    birth_dist = []   # 出生测试：平均距产区距离（越小=固化越深）
    birth_food = []   # 出生测试：食物数（固化对信号 → 出生就能吃到；固化错信号 → 饿死）
    birth_deaths = [] # 出生测试：饿死次数
    explore_seq, smell_seq = [], []
    learn = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True, 'env': learn_env}
    nolearn = {'mb': False, 'hip': False, 'fm': False, 'pfc': False, 'ctx': False, 'env': False}

    def birth_test(pool, wseed, corner='tl'):
        """新生儿纯基因测试：不学习
        指标1 = 平均距产区距离（空间偏好固化程度）
        指标2 = 测试期食物数均值（固化对信号 → 出生就能吃到；固化错信号 → 饿死）"""
        tw = World(size=world_size, n_food=n_food, n_walls=3, n_cues=n_cues, seed=wseed)
        tw.food_count = np.zeros((world_size, world_size), dtype=np.int32)
        place_stable(tw, rng, corner=corner, n_cues=n_cues, n_food=n_food)
        test_bugs = []
        for i in range(n_bugs):
            r, c = rng.randint(2, world_size - 2, 2)
            while tw.grid[r, c] != World.EMPTY:
                r, c = rng.randint(2, world_size - 2, 2)
            test_bugs.append(Bug([r, c], pool[i], seed=42, learn_flags=nolearn, stable=True,
                                 patchy=False, world_size=world_size))
        tw.bugs = test_bugs
        cue_arr = np.array(tw.cue_positions)
        total_d = 0.0
        n = 0
        deaths = 0
        for step in range(test_steps):
            for bug in test_bugs:
                if bug.alive:
                    bug.decide_and_learn(tw)
                    d = np.abs(cue_arr - bug.pos).sum(axis=1).min()
                    total_d += d
                    n += 1
                else:
                    bug.energy = 0.8
                    bug.pos = np.array([rng.randint(2, world_size - 2), rng.randint(2, world_size - 2)], dtype=int)
                    bug.alive = True
                    deaths += 1
        foods_arr = np.array([b.foods for b in test_bugs])
        return total_d / max(1, n), foods_arr.mean(), deaths

    for gen in range(n_gens):
        wseed = seed if env_pool is None else seed + gen * 7
        corner = 'br' if (migrate_at is not None and gen >= migrate_at) else 'tl'
        world = World(size=world_size, n_food=n_food, n_walls=3, n_cues=n_cues, seed=wseed)
        world.food_count = np.zeros((world_size, world_size), dtype=np.int32)
        place_stable(world, rng, corner=corner, n_cues=n_cues, n_food=n_food)

        bugs = []
        for i in range(n_bugs):
            r, c = rng.randint(2, world_size - 2, 2)
            while world.grid[r, c] != World.EMPTY:
                r, c = rng.randint(2, world_size - 2, 2)
            g = pool[i]
            bugs.append(Bug([r, c], g, seed=42, learn_flags=learn, stable=True,
                            env_gain=g.get('env_gain', 1.0), env_decay=g.get('env_decay', 0.999),
                            patchy=False, world_size=world_size))
        world.bugs = bugs

        for step in range(steps_per_gen):
            for bug in bugs:
                if bug.alive:
                    bug.decide_and_learn(world)
                else:
                    bug.energy = 0.8
                    bug.pos = np.array([rng.randint(2, world_size - 2), rng.randint(2, world_size - 2)], dtype=int)
                    bug.alive = True

        # 繁殖：Top-k 轮转复制 + 高斯突变，填满种群
        # cross=True：双亲各取一半基因位点（交叉重组）再突变
        ranked = sorted(bugs, key=lambda b: b.foods, reverse=True)
        new_pool = []
        while len(new_pool) < n_bugs:
            idx = len(new_pool) % top_k
            p1 = ranked[idx]
            child = {}
            for k in p1.genes:
                if cross and rng.rand() < 0.5:
                    p2 = ranked[(idx + 1) % top_k]   # 相邻父本交叉
                    child[k] = p2.genes[k]
                else:
                    child[k] = p1.genes[k]
                child[k] *= float(rng.uniform(0.9, 1.1))
            new_pool.append(child)
        pool = new_pool

        d, bf, bd = birth_test(pool, wseed, corner)
        birth_dist.append(d)
        birth_food.append(bf)
        birth_deaths.append(bd)
        explore_seq.append(np.mean([p['explore_noise'] for p in pool]))
        smell_seq.append(np.mean([p['smell_sens'] for p in pool]))
        gen_foods = np.array([b.foods for b in bugs])
        print(f"G{gen}: 学习期食物 {gen_foods.mean():.1f}±{gen_foods.std():.1f} | "
              f"出生距产区 {d:.2f} 出生食物 {bf:.1f} 出生饿死 {bd} | "
              f"smell {smell_seq[-1]:.2f} explore {explore_seq[-1]:.2f}")

    return {'birth': birth_dist, 'explore': explore_seq, 'smell': smell_seq,
            'birth_food': birth_food, 'birth_deaths': birth_deaths}


if __name__ == '__main__':
    import sys
    if '--cross' in sys.argv:
        # 繁殖策略对比：纯突变 vs 突变+基因交叉（Baldwin 框架内）
        # 交叉 = 双亲各取一半基因位点重组 → 有利组合更快扩散？
        print("===== 繁殖策略：纯突变 vs 突变+交叉（15×15, 12代×20虫, 稳定环境）=====\n")
        print("纯突变:")
        mut = evolve(n_gens=12, cross=False)
        print("\n突变+交叉:")
        cr = evolve(n_gens=12, cross=True)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        plt.plot(range(12), mut['birth'], 'o-', label='纯突变', color='#adb5bd', linewidth=2)
        plt.plot(range(12), cr['birth'], 's--', label='突变+交叉', color='#f72585', linewidth=2)
        plt.xlabel('世代'); plt.ylabel('出生距产区')
        plt.title('交叉重组对固化速度/终点的影响')
        plt.legend()
        plt.tight_layout()
        plt.savefig('multi_bug_cross_result.png', dpi=110)
        print("\n图表已保存: multi_bug_cross_result.png")
        sys.exit(0)
    if '--multiseed' in sys.argv:
        # 多 seed 坐实 Baldwin + EnvHebbian 耦合
        # 3 seeds × env on/off → 平均曲线+误差棒；env on 是否加速/加固化
        print("===== 多seed坐实Baldwin + EnvHebbian耦合（15×15, 12代×20虫, 3 seeds）=====\n")
        seeds = [42, 43, 44]
        env_runs, noenv_runs = [], []
        for s in seeds:
            print(f"\n--- seed {s} env ON ---")
            env_runs.append(evolve(n_gens=12, seed=s, learn_env=True))
            print(f"--- seed {s} env OFF ---")
            noenv_runs.append(evolve(n_gens=12, seed=s, learn_env=False))

        import matplotlib.pyplot as plt
        env_b = np.array([r['birth'] for r in env_runs])
        noenv_b = np.array([r['birth'] for r in noenv_runs])
        env_e = np.array([r['explore'] for r in env_runs])
        noenv_e = np.array([r['explore'] for r in noenv_runs])
        gens = range(12)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].errorbar(gens, env_b.mean(0), yerr=env_b.std(0), fmt='o-', label='env ON (3 seeds)', color='#f72585', capsize=3)
        axes[0].errorbar(gens, noenv_b.mean(0), yerr=noenv_b.std(0), fmt='s--', label='env OFF (3 seeds)', color='#adb5bd', capsize=3)
        axes[0].set_xlabel('世代'); axes[0].set_ylabel('出生距产区')
        axes[0].set_title('固化趋势（多seed平均±std）'); axes[0].legend()
        axes[1].errorbar(gens, env_e.mean(0), yerr=env_e.std(0), fmt='o-', label='env ON', color='#f72585', capsize=3)
        axes[1].errorbar(gens, noenv_e.mean(0), yerr=noenv_e.std(0), fmt='s--', label='env OFF', color='#adb5bd', capsize=3)
        axes[1].set_xlabel('世代'); axes[1].set_ylabel('explore 基因均值')
        axes[1].set_title('探索基因演化（固化载体）'); axes[1].legend()
        plt.tight_layout()
        plt.savefig('multi_bug_multiseed_result.png', dpi=110)
        print(f"\n终点出生距产区: env ON {env_b[:,-1].mean():.2f}±{env_b[:,-1].std():.2f} | "
              f"env OFF {noenv_b[:,-1].mean():.2f}±{noenv_b[:,-1].std():.2f}")
        print(f"终点 explore: env ON {env_e[:,-1].mean():.3f} | env OFF {noenv_e[:,-1].mean():.3f}")
        print("图表已保存: multi_bug_multiseed_result.png")
        sys.exit(0)
    if '--bigpop' in sys.argv:
        # 升级规模 + 固化信号正确性诊断
        # 32虫×15代×2 seeds，稳定 vs 打乱；n_food=16 降出生测试噪声
        # 新增指标：出生测试食物数/饿死 = 固化对信号（出生就能吃）还是固化错信号（饿死淘汰）
        print("===== 升级规模：32虫×15代×2seeds + 出生测试食物数（n_food=16）=====\n")
        seeds = [42, 43]
        st_runs, sh_runs = [], []
        for s in seeds:
            print(f"\n--- seed {s} 稳定环境 ---")
            st_runs.append(evolve(n_gens=15, n_bugs=32, seed=s, steps_per_gen=1500,
                                  test_steps=1200, env_pool=None, n_food=16))
            print(f"\n--- seed {s} 打乱环境 ---")
            sh_runs.append(evolve(n_gens=15, n_bugs=32, seed=s, steps_per_gen=1500,
                                  test_steps=1200, env_pool='shuffle', n_food=16))

        import matplotlib.pyplot as plt
        st_b = np.array([r['birth'] for r in st_runs])
        sh_b = np.array([r['birth'] for r in sh_runs])
        st_f = np.array([r['birth_food'] for r in st_runs])
        sh_f = np.array([r['birth_food'] for r in sh_runs])
        st_e = np.array([r['explore'] for r in st_runs])
        sh_e = np.array([r['explore'] for r in sh_runs])
        x = range(15)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        axes[0].errorbar(x, st_b.mean(0), yerr=st_b.std(0), fmt='o-', label='稳定', color='#f72585', capsize=3)
        axes[0].errorbar(x, sh_b.mean(0), yerr=sh_b.std(0), fmt='s--', label='打乱', color='#adb5bd', capsize=3)
        axes[0].set_xlabel('世代'); axes[0].set_ylabel('出生距产区')
        axes[0].set_title('固化趋势（距产区）'); axes[0].legend()
        axes[1].errorbar(x, st_f.mean(0), yerr=st_f.std(0), fmt='o-', label='稳定', color='#f72585', capsize=3)
        axes[1].errorbar(x, sh_f.mean(0), yerr=sh_f.std(0), fmt='s--', label='打乱', color='#adb5bd', capsize=3)
        axes[1].set_xlabel('世代'); axes[1].set_ylabel('出生测试食物数')
        axes[1].set_title('固化信号正确性（出生就能吃=固化对）'); axes[1].legend()
        axes[2].errorbar(x, st_e.mean(0), yerr=st_e.std(0), fmt='o-', label='稳定', color='#f72585', capsize=3)
        axes[2].errorbar(x, sh_e.mean(0), yerr=sh_e.std(0), fmt='s--', label='打乱', color='#adb5bd', capsize=3)
        axes[2].set_xlabel('世代'); axes[2].set_ylabel('explore 基因')
        axes[2].set_title('探索基因演化'); axes[2].legend()
        plt.tight_layout()
        plt.savefig('multi_bug_bigpop_result.png', dpi=110)
        print(f"\n终点（G14）稳定 vs 打乱: 距产区 {st_b[:,-1].mean():.2f} vs {sh_b[:,-1].mean():.2f} | "
              f"出生食物 {st_f[:,-1].mean():.1f} vs {sh_f[:,-1].mean():.1f} | "
              f"explore {st_e[:,-1].mean():.3f} vs {sh_e[:,-1].mean():.3f}")
        print("图表已保存: multi_bug_bigpop_result.png")
        sys.exit(0)
    if '--migrate' in sys.argv:
        # 固化惯性：稳定 8 代（固化到左上）→ 第 8 代产区迁到右下，继续 4 代
        # 固化组 vs 打乱组：迁移后出生距右下产区下降速度 = 惯性/反转能力
        print("===== 环境迁移固化惯性（15×15, 12代×20虫：稳定8代→迁移右下4代）=====\n")
        print("固化组（稳定8代→迁移）:")
        fix = evolve(n_gens=12, seed=42, migrate_at=8, env_pool=None)
        print("\n打乱组（每代换产区，无固化）:")
        shuf = evolve(n_gens=12, seed=42, migrate_at=8, env_pool='shuffle')

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(12)
        axes[0].plot(x, fix['birth'], 'o-', label='固化组（8代固化→迁移）', color='#f72585', linewidth=2)
        axes[0].plot(x, shuf['birth'], 's--', label='打乱组（无固化）', color='#adb5bd', linewidth=2)
        axes[0].axvline(8, color='#333', linestyle=':', label='产区迁移')
        axes[0].set_xlabel('世代'); axes[0].set_ylabel('出生距当前产区')
        axes[0].set_title('迁移后惯性（固化组反转慢=惯性大）'); axes[0].legend()
        axes[1].plot(x, fix['explore'], 'o-', label='固化组', color='#f72585', linewidth=2)
        axes[1].plot(x, shuf['explore'], 's--', label='打乱组', color='#adb5bd', linewidth=2)
        axes[1].axvline(8, color='#333', linestyle=':')
        axes[1].set_xlabel('世代'); axes[1].set_ylabel('explore 基因')
        axes[1].set_title('迁移后探索基因反转'); axes[1].legend()
        plt.tight_layout()
        plt.savefig('multi_bug_migrate_result.png', dpi=110)
        print(f"\n迁移后4代（G8-11）距产区均值: 固化组 {np.mean(fix['birth'][8:]):.2f} | 打乱组 {np.mean(shuf['birth'][8:]):.2f}")
        print("图表已保存: multi_bug_migrate_result.png")
        sys.exit(0)
    if '--evolve' in sys.argv:
        print("===== Baldwin效应：产区偏好跨代固化（15×15 12代×20虫）=====\n")
        print("稳定环境（产区固定左上）:")
        stable = evolve(n_gens=12, env_pool=None)
        print("\n环境每代打乱（对照组）:")
        shuffle = evolve(n_gens=12, env_pool='shuffle')

        print("\n===== 大世界 Baldwin：30×30（12代×16虫，n_food=16, n_cues=2）=====\n")
        print("稳定环境:")
        big_stable = evolve(n_gens=12, n_bugs=16, steps_per_gen=1500, test_steps=1500,
                            env_pool=None, world_size=30, n_food=16)
        print("\n环境每代打乱:")
        big_shuffle = evolve(n_gens=12, n_bugs=16, steps_per_gen=1500, test_steps=1500,
                             env_pool='shuffle', world_size=30, n_food=16)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].plot(range(12), stable['birth'], 'o-', label='15×15 稳定', color='#f72585', linewidth=2)
        axes[0].plot(range(12), shuffle['birth'], 's--', label='15×15 打乱', color='#adb5bd', linewidth=2)
        axes[0].set_xlabel('世代'); axes[0].set_ylabel('出生距产区（15×15 尺度）')
        axes[0].set_title('15×15 固化趋势'); axes[0].legend()
        axes[1].plot(range(12), big_stable['birth'], 'o-', label='30×30 稳定', color='#f72585', linewidth=2)
        axes[1].plot(range(12), big_shuffle['birth'], 's--', label='30×30 打乱', color='#adb5bd', linewidth=2)
        axes[1].set_xlabel('世代'); axes[1].set_ylabel('出生距产区（30×30 尺度）')
        axes[1].set_title('30×30 大世界固化（距离尺度更大=更敏感）'); axes[1].legend()
        plt.tight_layout()
        plt.savefig('multi_bug_evolve_result.png', dpi=110)
        print("\n图表已保存: multi_bug_evolve_result.png")
        sys.exit(0)
    if '--stableenv' in sys.argv:
        # 稳定结构环境（食物固定 cue 附近）验证：
        # 环境有可学结构 → env 网络才有预测价值；产区迁移 → 验证跟随能力
        print("===== 稳定结构环境：env网络价值 + 产区迁移跟随（5000步，cue绑定食物）=====\n")
        base_learn = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True}
        env_learn = dict(base_learn); env_learn['env'] = True

        r1 = run_group('无env 稳定环境', use_genes=False, patchy=False,
                       learn_flags=base_learn, steps=5000, stable=True, n_cues=2)
        r2 = run_group('env(g0.3) 稳定环境', use_genes=False, patchy=False,
                       learn_flags=env_learn, steps=5000, stable=True,
                       env_gain=0.3, env_decay=0.995, n_cues=2)
        r3 = run_group('无env 稳定+产区迁移', use_genes=False, patchy=False,
                       learn_flags=base_learn, steps=5000, stable=True, flip_at=2500, n_cues=2)
        r4 = run_group('env(g0.3) 稳定+产区迁移', use_genes=False, patchy=False,
                       learn_flags=env_learn, steps=5000, stable=True, flip_at=2500,
                       env_gain=0.3, env_decay=0.995, n_cues=2)

        print(f"{'组':<22} {'食物均值':>7} {'食物std':>7} {'漂移std':>7} {'熵':>5} {'迁移后食物':>8}")
        for r in [r1, r2, r3, r4]:
            pf = f"{r['post_flip'].mean():>8.1f}" if r['post_flip'] is not None else f"{'-':>8}"
            print(f"{r['name']:<22} {r['foods_mean']:>7.1f} {r['foods_std']:>7.1f} {r['drift_std']:>7.2f} "
                  f"{r['entropy_mean']:>5.2f} {pf}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        rr = [r1, r2, r3, r4]
        names = ['无env', 'env0.3', '无env+迁移', 'env0.3+迁移']
        colors = ['#adb5bd', '#4cc9f0', '#adb5bd', '#f72585']
        axes[0].bar(range(4), [r['foods_mean'] for r in rr], color=colors)
        axes[0].set_xticks(range(4)); axes[0].set_xticklabels(names, fontsize=9)
        axes[0].set_title('食物均值'); axes[0].set_ylabel('foods')
        pf_vals = [r['post_flip'].mean() if r['post_flip'] is not None else 0 for r in rr]
        axes[1].bar(range(4), pf_vals, color=colors)
        axes[1].set_xticks(range(4)); axes[1].set_xticklabels(names, fontsize=9)
        axes[1].set_title('产区迁移后2500步食物获取（越大=越能跟上）')
        axes[1].set_ylabel('post-flip foods')
        plt.tight_layout()
        plt.savefig('multi_bug_stableenv_result.png', dpi=110)
        print("\n图表已保存: multi_bug_stableenv_result.png")
        sys.exit(0)

    if '--envsens' in sys.argv:
        # env 网络 gain×decay 敏感性：找 env 不拖累主网络的工作点
        # 环境场景改为"稳定结构"：食物固定在 cue 附近（结构稳定，环境模型才有预测价值）
        print("===== env网络 gain×decay 敏感性（3000步，斑块环境）=====\n")
        base_learn = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True}
        import multi_bug_genome as M

        # 用固定基因（只有 env 参数变化）跑
        def run_with_env(gain, decay):
            lf = dict(base_learn); lf['env'] = True
            return run_group(f'env g{gain} d{decay}', use_genes=False, patchy=True,
                             learn_flags=lf, steps=3000, env_gain=gain, env_decay=decay)

        rows = []
        for gain in [0.2, 0.5, 1.0]:
            for decay in [0.99, 0.999]:
                r = run_with_env(gain, decay)
                rows.append((gain, decay, r))
        print(f"{'gain':>5} {'decay':>6} {'食物均值':>7} {'食物std':>7} {'熵':>5}")
        for gain, decay, r in rows:
            print(f"{gain:>5.1f} {decay:>6.3f} {r['foods_mean']:>7.1f} {r['foods_std']:>7.1f} {r['entropy_mean']:>5.2f}")
        sys.exit(0)

    if '--env' in sys.argv:
        # 环境持续学习网络验证：
        # 组1: 无env网络（对照，环境突变后靠老记忆/碰运气）
        # 组2: +env网络，主网络不学（环境网络独立持续学习，验证用户假设）
        # 组3: +env网络 + 主网络学习（双通道）
        # 组4: 组3 + 环境中途突变（2500步食物从左上搬到右下）→ 跟随能力
        print("===== 环境持续学习网络验证（5000步，10虫/组，斑块环境）=====\n")
        base_learn = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True}
        env_only = {'mb': False, 'hip': False, 'fm': False, 'pfc': False, 'ctx': False, 'env': True}
        env_learn = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True, 'env': True}

        r1 = run_group('无env网络 主网络学', use_genes=False, patchy=True,
                       learn_flags=base_learn, steps=5000)
        r2 = run_group('env网络 主网络不学', use_genes=False, patchy=True,
                       learn_flags=env_only, steps=5000)
        r3 = run_group('env网络 主网络学', use_genes=False, patchy=True,
                       learn_flags=env_learn, steps=5000)
        r5 = run_group('无env网络 主网络学+突变', use_genes=False, patchy=True,
                       learn_flags=base_learn, steps=5000, flip_at=2500)
        r4 = run_group('env网络 主网络学+突变', use_genes=False, patchy=True,
                       learn_flags=env_learn, steps=5000, flip_at=2500)

        print(f"{'组':<24} {'食物均值':>7} {'食物std':>7} {'漂移std':>7} {'熵':>5} {'突变后食物':>8}")
        for r in [r1, r2, r3, r5, r4]:
            pf = f"{r['post_flip'].mean():>8.1f}" if r['post_flip'] is not None else f"{'-':>8}"
            print(f"{r['name']:<24} {r['foods_mean']:>7.1f} {r['foods_std']:>7.1f} {r['drift_std']:>7.2f} "
                  f"{r['entropy_mean']:>5.2f} {pf}")

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        rr = [r1, r2, r3, r5, r4]
        names = [r['name'].split(' ')[0] + ('+突' if '突变' in r['name'] else '')
                 + ('+env' if 'env网络' in r['name'] else '') for r in rr]
        axes[0].bar(range(5), [r['foods_mean'] for r in rr],
                    color=['#adb5bd', '#4cc9f0', '#4895ef', '#adb5bd', '#f72585'])
        axes[0].set_xticks(range(5)); axes[0].set_xticklabels(names, fontsize=8)
        axes[0].set_title('食物均值（吃得多=环境感知好）'); axes[0].set_ylabel('foods')
        pf_vals = [r['post_flip'].mean() if r['post_flip'] is not None else 0 for r in rr]
        axes[1].bar(range(5), pf_vals, color=['#adb5bd', '#4cc9f0', '#4895ef', '#adb5bd', '#f72585'])
        axes[1].set_xticks(range(5)); axes[1].set_xticklabels(names, fontsize=8)
        axes[1].set_title('环境中途突变后2500步食物获取（越大=越能跟上环境变化）')
        axes[1].set_ylabel('post-flip foods')
        plt.tight_layout()
        plt.savefig('multi_bug_env_result.png', dpi=110)
        print("\n图表已保存: multi_bug_env_result.png")
        sys.exit(0)

    if '--scan' in sys.argv:
        # 三因素扫描：基因强度 × 环境异质性 × 学习（经历写入）
        # 学习关 = 纯环境直接作用；学习开 = 经历累积（环境通过行为进入大脑）
        print("===== 基因×环境×经历 三因素扫描（4000步，10虫/组）=====\n")
        learn_on = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True}
        learn_off = {'mb': False, 'hip': False, 'fm': False, 'pfc': False, 'ctx': False}
        combos = [
            ('基因0 环境均匀 不学', 0.0, False, learn_off),
            ('基因0 环境斑块 不学', 0.0, True,  learn_off),
            ('基因0 环境均匀 学习', 0.0, False, learn_on),
            ('基因0 环境斑块 学习', 0.0, True,  learn_on),
            ('基因弱 环境均匀 学习', 1.0, False, learn_on),
            ('基因弱 环境斑块 学习', 1.0, True,  learn_on),
            ('基因强 环境均匀 学习', 2.0, False, learn_on),
            ('基因强 环境斑块 学习', 2.0, True,  learn_on),
        ]
        print(f"{'组合':<22} {'食物均值':>6} {'食物std':>7} {'漂移std':>7} {'漂移均值':>7} {'熵':>5} {'距离':>5}")
        rows = []
        for name, gs, patchy, flags in combos:
            r = run_group(name, use_genes=(gs > 0), patchy=patchy, learn_flags=flags,
                          steps=4000, gene_scale=gs)
            rows.append(r)
            print(f"{name:<22} {r['foods_mean']:>6.1f} {r['foods_std']:>7.1f} {r['drift_std']:>7.2f} "
                  f"{r['drift_mean']:>7.2f} {r['entropy_mean']:>5.2f} {r['pair_dist_mean']:>5.1f}")

        # 主效应计算（漂移std = 个体发育分叉的直接证据）
        drift_std = {r['name']: r['drift_std'] for r in rows}
        learn_effect = (drift_std['基因0 环境均匀 学习'] + drift_std['基因0 环境斑块 学习']
                        - drift_std['基因0 环境均匀 不学'] - drift_std['基因0 环境斑块 不学']) / 2
        env_effect_learn = (drift_std['基因0 环境斑块 学习'] + drift_std['基因弱 环境斑块 学习']
                            - drift_std['基因0 环境均匀 学习'] - drift_std['基因弱 环境均匀 学习']) / 2
        gene_effect_strong = (drift_std['基因强 环境均匀 学习'] + drift_std['基因强 环境斑块 学习']
                              - drift_std['基因弱 环境均匀 学习'] - drift_std['基因弱 环境斑块 学习']) / 2
        print(f"\n主效应（漂移std贡献，越大=该因素越让个体分叉）:")
        print(f"  经历(学习写入): {learn_effect:+.2f}")
        print(f"  环境(斑块化)  : {env_effect_learn:+.2f}")
        print(f"  基因(弱→强)   : {gene_effect_strong:+.2f}")

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        labels = [r['name'] for r in rows]
        vals = [r['drift_std'] for r in rows]
        colors = ['#adb5bd', '#4cc9f0', '#f72585']
        bar_colors = []
        for n in labels:
            if '不学' in n:
                bar_colors.append('#adb5bd')
            elif '基因强' in n:
                bar_colors.append('#f72585')
            elif '基因弱' in n or ('基因0' in n and '学习' in n):
                bar_colors.append('#4cc9f0')
            else:
                bar_colors.append('#adb5bd')
        axes[0].barh(range(len(labels))[::-1], vals, color=bar_colors)
        axes[0].set_yticks(range(len(labels))[::-1]); axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].set_xlabel('权重漂移std（个体大脑分叉程度）')
        axes[0].set_title('三因素对个体分化贡献')
        # 主效应
        axes[1].bar(['基因\n(弱→强)', '经历\n(学习写入)', '环境\n(斑块化)'],
                    [gene_effect_strong, learn_effect, env_effect_learn],
                    color=['#f72585', '#4cc9f0', '#adb5bd'])
        axes[1].set_title('主效应：漂移std贡献'); axes[1].set_ylabel('贡献量')
        for i, v in enumerate([gene_effect_strong, learn_effect, env_effect_learn]):
            axes[1].text(i, v + 0.05, f'{v:+.2f}', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig('multi_bug_scan_result.png', dpi=110)
        print("\n图表已保存: multi_bug_scan_result.png")
        sys.exit(0)

    if '--diag' in sys.argv:
        # 组件级诊断：逐个开关学习组件，定位谁毁掉找食
        print("===== 学习组件诊断（A组配置，无基因，均匀食物）=====\n")
        all_on = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True}
        variants = [
            ('v2 n_food8 ecb0.003 (=v1参数)', dict(mb=False, hip=False, fm=False, pfc=False, ctx=False), 8, 0.003),
            ('v2 n_food8 ecb0.002',            dict(mb=False, hip=False, fm=False, pfc=False, ctx=False), 8, 0.002),
            ('v2 n_food10 ecb0.003',           dict(mb=False, hip=False, fm=False, pfc=False, ctx=False), 10, 0.003),
            ('v2 n_food10 ecb0.002 (=当前)',   dict(mb=False, hip=False, fm=False, pfc=False, ctx=False), 10, 0.002),
        ]
        print(f"{'变体':<30} {'食物均值':>7} {'食物std':>7} {'饿死':>4}")
        for name, flags, nf, ecb in variants:
            r = run_group(name, use_genes=False, patchy=False, learn_flags=flags,
                          steps=2000, n_food=nf, energy_cost_base=ecb)
            print(f"{name:<30} {r['foods_mean']:>7.1f} {r['foods_std']:>7.1f} {r['deaths']:>4}")
        sys.exit(0)

    if '--bigworld' in sys.argv:
        # 大世界验证 v2：30×30 + 稀疏稳定产区，扫描 n_food 验证"产区内部边际递减"
        # 假设：env 远程导航已兑现（产区停留↑），但 24 食物 + 10 虫抢 → 觅食收益被稀缺掩盖
        # 若 n_food↑ 时 env 增益↑ → 边际递减假设成立（导航价值被食物稀缺限制）
        print("===== 大世界 30×30 + 稳定产区：n_food 扫描（8000步，10虫/组）=====\n")
        print("发现：食物越富集 env 增益越负（路径锁定抑制探索）→ 补稀缺档验证价值反转\n")
        base_learn = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True}
        env_learn = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True, 'env': True}

        all_no, all_env = [], []
        print(f"{'n_food':>6} {'组':<16} {'食物均值':>7} {'食物std':>7} {'产区停留%':>8} {'漂移std':>7}")
        for nf in [8, 24, 48, 72]:
            r_no = run_group('无env', use_genes=False, patchy=False, stable=True,
                             learn_flags=base_learn, steps=8000, n_food=nf, n_cues=4,
                             world_size=30)
            r_env = run_group('env(g1.0)', use_genes=False, patchy=False, stable=True,
                              learn_flags=env_learn, steps=8000, n_food=nf, n_cues=4,
                              env_gain=1.0, world_size=30)
            all_no.append(r_no); all_env.append(r_env)
            gain = (r_env['foods_mean'] - r_no['foods_mean']) / max(0.1, r_no['foods_mean']) * 100
            print(f"{nf:>6} {r_no['name']:<16} {r_no['foods_mean']:>7.1f} {r_no['foods_std']:>7.1f} "
                  f"{r_no['near_cue_ratio']*100:>7.1f}% {r_no['drift_std']:>7.2f}")
            print(f"{'':>6} {r_env['name']:<16} {r_env['foods_mean']:>7.1f} {r_env['foods_std']:>7.1f} "
                  f"{r_env['near_cue_ratio']*100:>7.1f}% {r_env['drift_std']:>7.2f}  "
                  f"(env增益 {gain:+.1f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        nfs = [8, 24, 48, 72]
        axes[0].bar(np.arange(4) - 0.15, [r['foods_mean'] for r in all_no], 0.3, label='无env', color='#adb5bd')
        axes[0].bar(np.arange(4) + 0.15, [r['foods_mean'] for r in all_env], 0.3, label='env(g1.0)', color='#f72585')
        axes[0].set_xticks(range(4)); axes[0].set_xticklabels([f'{n}' for n in nfs])
        axes[0].set_title('食物均值 vs n_food（稀缺时env增益?富集时拖累?）')
        axes[0].set_xlabel('n_food'); axes[0].set_ylabel('foods'); axes[0].legend()
        axes[1].bar(np.arange(4) - 0.15, [r['near_cue_ratio']*100 for r in all_no], 0.3, label='无env', color='#adb5bd')
        axes[1].bar(np.arange(4) + 0.15, [r['near_cue_ratio']*100 for r in all_env], 0.3, label='env(g1.0)', color='#f72585')
        axes[1].set_xticks(range(4)); axes[1].set_xticklabels([f'{n}' for n in nfs])
        axes[1].set_title('产区停留% vs n_food（env锁在产区=路径锁定）')
        axes[1].set_xlabel('n_food'); axes[1].set_ylabel('%'); axes[1].legend()
        plt.tight_layout()
        plt.savefig('multi_bug_bigworld_result.png', dpi=110)
        print("\n图表已保存: multi_bug_bigworld_result.png")
        sys.exit(0)

    print("===== 基因×经历 个体分化实验（v2 真学习版）=====\n")
    results = [
        run_group('A 经历（无基因,均匀食物）', use_genes=False, patchy=False),
        run_group('B 基因+经历（均匀食物）', use_genes=True, patchy=False),
        run_group('C 基因+环境（资源斑块）', use_genes=True, patchy=True),
    ]

    print(f"{'组':<22} {'食物均值':>6} {'食物std':>7} {'基尼':>6} {'两两距离':>8} {'决策熵':>6} {'饿死':>4} {'权重漂移':>8}")
    for r in results:
        print(f"{r['name']:<22} {r['foods_mean']:>6.1f} {r['foods_std']:>7.1f} {r['gini']:>6.3f} "
              f"{r['pair_dist_mean']:>8.1f} {r['entropy_mean']:>6.3f} {r['deaths']:>4} "
              f"{r['drift_mean']:>8.3f}±{r['drift_std']:.3f}")

    print("\n每虫 (id, 食物, 权重漂移):")
    for r in results:
        detail = "  ".join(f"#{i}:{f}(Δw{d:.2f})" for i, (f, d) in enumerate(zip(r['foods'], r['drifts'])))
        print(f"  [{r['name']}]  {detail}")

    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    names = [r['name'].split('（')[0] for r in results]
    x = np.arange(3)
    colors = ['#adb5bd', '#4cc9f0', '#f72585']

    axes[0].bar(x, [r['foods_std'] for r in results], color=colors)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15, fontsize=9)
    axes[0].set_title('食物数标准差（能力分化）')

    axes[1].bar(x, [r['drift_mean'] for r in results], color=colors, yerr=[r['drift_std'] for r in results])
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15, fontsize=9)
    axes[1].set_title('权重漂移均值（大脑长得多不同）')

    axes[2].bar(x, [r['entropy_mean'] for r in results], color=colors)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=15, fontsize=9)
    axes[2].set_title('决策熵（行为多样性）')

    for r in results:
        axes[3].plot(np.arange(len(r['pair_dists'])) * 100, r['pair_dists'],
                     label=r['name'].split('（')[0], linewidth=2)
    axes[3].set_title('两两距离（空间分叉）'); axes[3].set_xlabel('step')
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('multi_bug_genome_result.png', dpi=110)
    print("\n图表已保存: multi_bug_genome_result.png")
