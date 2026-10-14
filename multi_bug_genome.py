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
    foods = np.argwhere(world.grid == world.FOOD)
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
    for _ in range(200):
        q = np.random.choice(4, p=QUADRANT_WEIGHTS)
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
            return


def respawn_food_uniform(world):
    for _ in range(200):
        r, c = np.random.randint(1, world.size - 1, 2)
        if world.grid[r, c] == world.EMPTY:
            world.grid[r, c] = world.FOOD
            return


class Bug:
    """一只虫子：共享权重出生，基因决定学习规则和硬件，经历改写权重"""

    def __init__(self, pos, genome, ckpt='checkpoints', seed=42, learn_flags=None, shared=None,
                 energy_cost_base=0.002, patchy=False):
        self.pos = np.array(pos, dtype=int)
        self.energy = 0.8
        self.alive = True
        self.foods = 0
        self.deaths = 0
        self.actions = []
        self.energy_cost_base = energy_cost_base
        self.patchy = patchy
        # 学习组件开关（诊断用）
        self.lf = learn_flags or {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True}

        # ---- 基因（出生固定，不随经历改变）----
        g = genome
        self.lr_gain = g.get('lr_gain', 1.0)           # 学习速率乘子
        self.trace_gain = g.get('trace_gain', 1.0)     # 记忆痕迹衰减乘子
        self.da_gain = g.get('da_gain', 1.0)           # 多巴胺敏感度
        self.smell_sens = g.get('smell_sens', 1.0)     # 嗅觉硬件灵敏度（感知输入层）
        self.energy_cost = g.get('energy_cost', 1.0)   # 代谢率
        self.explore_noise = g.get('explore_noise', 0.2)

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

        hip = Hippocampus(world_size=15, decay=0.9995)
        hip.load(f'{ckpt}/hip_final.npy')
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
        hip_strength = self.hip.memory[r, c] if (0 <= r < 15 and 0 <= c < 15) else 0.0
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
        total = (smell * smell_weight + habit * habit_weight
                 + spatial * 0.5 + imagined * 0.8
                 + pfc_bias * 0.5 + ctx_bias)
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
                world.grid[nr, nc] = world.EMPTY
                self.foods += 1
                self.energy = min(1.0, self.energy + 0.3)
                ate = True
                # 食物补充（v1 有、v2 重构时漏掉的 respawn！）
                (respawn_food_patchy if self.patchy else respawn_food_uniform)(world)
        else:
            bumped = True

        # ---- 学习（经历改写权重）----
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


def make_genome(rng):
    return {
        'lr_gain': rng.uniform(0.2, 1.5),        # 学得快/慢（增量学习）
        'trace_gain': rng.uniform(0.7, 1.15),    # 记得久/忘得快
        'da_gain': rng.uniform(0.5, 1.5),        # 奖励敏感/迟钝
        'smell_sens': rng.uniform(0.3, 2.0),     # 鼻子灵/钝
        'energy_cost': rng.uniform(0.7, 1.5),    # 代谢低/高
        'explore_noise': rng.uniform(0.02, 0.5), # 探索/保守
    }


def run_group(name, use_genes, patchy, n_bugs=10, steps=6000, seed=42, learn_flags=None,
              share_brain=False, n_food=10, energy_cost_base=0.002):
    rng = np.random.RandomState(seed)
    world = World(size=15, n_food=n_food, n_walls=3, n_cues=6, seed=seed)
    if patchy:
        world.grid = np.full((15, 15), World.EMPTY, dtype=int)
        world._place_walls(3)
        for _ in range(n_food):
            respawn_food_patchy(world)

    bugs = []
    shared = None
    if share_brain:
        # 复现 v1：10 虫共享同一套大脑对象
        mb_s = MushroomBody(n_input=20, n_kc=500, n_actions=4, lr=0.15, trace_decay=0.85, seed=seed)
        mb_s.load('checkpoints/mb_final.npz')
        hip_s = Hippocampus(world_size=15, decay=0.9995)
        hip_s.load('checkpoints/hip_final.npy')
        fm_s = ForwardModel(n_kc=500, n_actions=4, lr=0.1, seed=seed)
        fm_s.load('checkpoints/fm_final.npy')
        pfc_s = PrefrontalCortex(n_input=6, n_hidden=8, n_output=5, lr=0.25, seed=seed)
        ctx_s = ContextModulator(n_input=4, n_output=4, lr=0.06, seed=seed)
        shared = (mb_s, hip_s, fm_s, pfc_s, ctx_s)

    for i in range(n_bugs):
        r, c = rng.randint(2, 13, 2)
        while world.grid[r, c] != World.EMPTY:
            r, c = rng.randint(2, 13, 2)
        genome = make_genome(rng) if use_genes else {}
        bugs.append(Bug([r, c], genome, seed=42, learn_flags=learn_flags, shared=shared,
                        energy_cost_base=energy_cost_base, patchy=patchy))

    pair_dists = []
    total_deaths = 0

    for step in range(steps):
        for bug in bugs:
            if bug.alive:
                bug.decide_and_learn(world)
            else:
                # 饿死复活：保留大脑和基因，换个出生点继续发育
                bug.energy = 0.8
                bug.pos = np.array([rng.randint(2, 13), rng.randint(2, 13)], dtype=int)
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
    return {
        'name': name, 'foods': foods, 'foods_mean': foods.mean(), 'foods_std': foods.std(),
        'gini': gini, 'pair_dist_mean': np.mean(pair_dists[-20:]),
        'entropy_mean': np.mean(ents), 'deaths': total_deaths,
        'drift_mean': drifts.mean(), 'drift_std': drifts.std(), 'drifts': drifts,
        'pair_dists': pair_dists,
    }


if __name__ == '__main__':
    import sys
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
