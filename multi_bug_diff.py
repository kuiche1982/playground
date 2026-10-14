"""Embryo: 本地个体差异化对照实验
三组对照，每组 10 只虫子，全部加载同一套训练权重（mb/fm/hip 共享、不学习）：
  A 基线组      ：无基因差异（增益=1.0, noise=0.2）+ 中心出生 + 均匀食物
  B 基因放大组  ：基因 0.3-2.0 调制决策通道 + 随机出生 + 均匀食物
  C 基因+环境组 ：B + 资源斑块化（4 象限食物密度不同）

量化指标：
  - 每虫食物数（标准差/基尼系数）——吃得多不多
  - 两两轨迹距离（越大=散得越开）
  - 决策熵（越大=行为越多样）
  - 饿死次数
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from embryo import World, MushroomBody, Hippocampus, ForwardModel, LanguageChannel, ContextModulator, PrefrontalCortex


def smell_at(world, pos):
    """复刻 World.smell_gradient，但基于任意位置（修复多虫感知 bug）"""
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
    """复刻 World.see，但基于任意位置"""
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


# ---- 资源斑块化：4 个象限，各自食物密度权重 ----
QUADRANT_WEIGHTS = [0.45, 0.20, 0.20, 0.15]  # 左上最密，右下最稀

def quadrant_of(r, c, size=15):
    half = size // 2
    if r < half and c < half:
        return 0
    if r < half:
        return 1
    if c < half:
        return 2
    return 3


def respawn_food_patchy(world, weights=QUADRANT_WEIGHTS):
    """按象限权重放食物（斑块化）"""
    for _ in range(200):
        q = np.random.choice(4, p=weights)
        half = world.size // 2
        if q == 0:
            r = np.random.randint(1, half)
            c = np.random.randint(1, half)
        elif q == 1:
            r = np.random.randint(1, half)
            c = np.random.randint(half, world.size - 1)
        elif q == 2:
            r = np.random.randint(half, world.size - 1)
            c = np.random.randint(1, half)
        else:
            r = np.random.randint(half, world.size - 1)
            c = np.random.randint(half, world.size - 1)
        if world.grid[r, c] == world.EMPTY:
            world.grid[r, c] = world.FOOD
            return


def respawn_food_uniform(world):
    """均匀随机放食物"""
    for _ in range(200):
        r, c = np.random.randint(1, world.size - 1, 2)
        if world.grid[r, c] == world.EMPTY:
            world.grid[r, c] = world.FOOD
            return


class Bug:
    def __init__(self, pos, genes=None):
        self.pos = np.array(pos, dtype=int)
        self.energy = 0.8
        self.alive = True
        self.foods = 0
        self.deaths = 0
        # 基因（None = 基线默认值）
        self.smell_gain = genes['smell_gain'] if genes else 1.0
        self.habit_gain = genes['habit_gain'] if genes else 1.0
        self.spatial_gain = genes['spatial_gain'] if genes else 1.0
        self.imagine_gain = genes['imagine_gain'] if genes else 1.0
        self.explore_noise = genes['explore_noise'] if genes else 0.2
        self.actions = []  # 决策记录（算熵用）


def make_genes(rng):
    """基因差异放大：0.3-2.0，不是 0.8-1.2"""
    return {
        'smell_gain': rng.uniform(0.3, 2.0),
        'habit_gain': rng.uniform(0.3, 2.0),
        'spatial_gain': rng.uniform(0.3, 1.8),
        'imagine_gain': rng.uniform(0.3, 1.8),
        'explore_noise': rng.uniform(0.02, 0.6),
    }


def run_group(name, use_genes, patchy, spawn_center, n_bugs=10,
              steps=8000, seed=42, ckpt='checkpoints'):
    rng = np.random.RandomState(seed)
    world = World(size=15, n_food=8, n_walls=3, n_cues=6, seed=seed)
    # 初始食物也按斑块权重放
    if patchy:
        world.grid = np.full((15, 15), World.EMPTY, dtype=int)
        world._place_walls(3)
        for _ in range(8):
            respawn_food_patchy(world)

    # 共享大脑（mb/fm/hip 加载训练权重，推理不学习）
    mb = MushroomBody(n_input=20, n_kc=500, n_actions=4, lr=0.15, trace_decay=0.85, seed=seed)
    fm = ForwardModel(n_kc=500, n_actions=4, lr=0.1, seed=seed)
    hip = Hippocampus(world_size=15, decay=0.9995)
    mb.load(f'{ckpt}/mb_final.npz')
    fm.load(f'{ckpt}/fm_final.npy')
    hip.load(f'{ckpt}/hip_final.npy')
    pfc = PrefrontalCortex(n_input=6, n_hidden=8, n_output=5, lr=0.25, seed=seed)
    ctx = ContextModulator(n_input=4, n_output=4, lr=0.06, seed=seed)

    # 出生位置
    bugs = []
    for i in range(n_bugs):
        if spawn_center:
            r, c = 7 + rng.randint(-2, 3), 7 + rng.randint(-2, 3)
        else:
            r, c = rng.randint(2, 13, 2)
            while world.grid[r, c] != World.EMPTY:
                r, c = rng.randint(2, 13, 2)
        bugs.append(Bug([r, c], make_genes(rng) if use_genes else None))

    pair_dists = []   # 每100步的两两距离均值
    foods_history = []  # 每500步的食物累计（画图用）
    total_deaths = 0

    for step in range(steps):
        for bug in bugs:
            if not bug.alive:
                continue
            # 每虫独立感知（修复：不再用 world.pos）
            smell = smell_at(world, bug.pos)
            vision = see_at(world, bug.pos)
            lang_input = np.zeros(4, dtype=np.float32)  # 无指令
            sensory = np.concatenate([vision, lang_input])

            hip_strength = hip.memory[bug.pos[0], bug.pos[1]]
            task_bias = pfc.forward(lang_input, bug.energy, hip_strength)
            ctx_gain = ctx.forward(task_bias[:4])
            active_kc = mb.encode(sensory)
            spatial = hip.spatial_bias(bug.pos)
            imagined = fm.imagine(active_kc)
            habit = mb.decide(active_kc)
            pfc_bias = pfc.get_bias(task_bias)
            inhibit = task_bias[4]
            smell_weight = bug.smell_gain * (1.0 - inhibit)
            habit_weight = bug.habit_gain * (1.0 - inhibit)
            ctx_bias = ctx.get_gain(ctx_gain)
            total = (smell * smell_weight + habit * habit_weight
                     + spatial * 0.5 * bug.spatial_gain
                     + imagined * 0.8 * bug.imagine_gain
                     + pfc_bias * 0.5 + ctx_bias)
            noise = np.random.randn(4) * bug.explore_noise
            action = int(np.argmax(total + noise))
            bug.actions.append(action)

            # 移动
            dr, dc = world.ACTIONS[action]
            nr, nc = bug.pos[0] + dr, bug.pos[1] + dc
            if 0 <= nr < world.size and 0 <= nc < world.size and world.grid[nr, nc] != world.WALL:
                bug.pos = np.array([nr, nc])
                if world.grid[nr, nc] == world.FOOD:
                    bug.foods += 1
                    bug.energy = min(1.0, bug.energy + 0.3)
                    world.grid[nr, nc] = world.EMPTY
                    (respawn_food_patchy if patchy else respawn_food_uniform)(world)

            bug.energy -= 0.003
            if bug.energy <= 0:
                bug.alive = False
                bug.deaths += 1
                total_deaths += 1
                # 复活：新出生位置 + 满能量（保持种群规模）
                bug.energy = 0.8
                bug.pos = np.array([rng.randint(2, 13), rng.randint(2, 13)], dtype=int)
                bug.alive = True

        # 指标采样
        if step % 100 == 0:
            dists = []
            for i in range(len(bugs)):
                for j in range(i + 1, len(bugs)):
                    dists.append(np.abs(bugs[i].pos - bugs[j].pos).sum())
            pair_dists.append(np.mean(dists))
        if step % 500 == 0:
            foods_history.append([b.foods for b in bugs])

    # 汇总指标
    foods = np.array([b.foods for b in bugs])
    gini = np.abs(foods[:, None] - foods[None, :]).sum() / (2 * len(foods) ** 2 * foods.mean() + 1e-9)
    ents = []
    for b in bugs:
        if not b.actions:
            continue
        hist = np.bincount(np.array(b.actions), minlength=4).astype(float)
        p = hist / hist.sum()
        ents.append(-(p * np.log(p + 1e-9)).sum())
    # 基因-表现相关性（只对 use_genes 组有意义）
    gene_corr = {}
    if use_genes:
        for k in ['smell_gain', 'habit_gain', 'spatial_gain', 'imagine_gain', 'explore_noise']:
            vals = np.array([getattr(b, k) for b in bugs])
            gene_corr[k] = float(np.corrcoef(vals, foods)[0, 1])
    per_bug = [(i, b.foods, round(b.smell_gain, 2), round(b.explore_noise, 2)) for i, b in enumerate(bugs)]
    return {
        'name': name,
        'foods': foods,
        'foods_mean': foods.mean(),
        'foods_std': foods.std(),
        'gini': gini,
        'pair_dist_mean': np.mean(pair_dists[-20:]),   # 最近2000步的平均距离
        'pair_dist_last': pair_dists[-1],
        'entropy_mean': np.mean(ents),
        'deaths': total_deaths,
        'pair_dists': pair_dists,
        'foods_history': foods_history,
        'gene_corr': gene_corr,
        'per_bug': per_bug,
    }


if __name__ == '__main__':
    print("===== 本地个体差异化对照实验 =====\n")
    results = [
        run_group('A 基线（无基因差异,中心出生,均匀食物）', use_genes=False, patchy=False, spawn_center=True),
        run_group('B 基因放大（0.3-2.0,随机出生,均匀食物）', use_genes=True, patchy=False, spawn_center=False),
        run_group('C 基因+环境（基因放大,随机出生,资源斑块）', use_genes=True, patchy=True, spawn_center=False),
    ]

    print(f"{'组':<38} {'食物均值':>7} {'食物标准差':>8} {'基尼':>6} {'两两距离':>8} {'决策熵':>6} {'饿死':>4}")
    for r in results:
        print(f"{r['name']:<38} {r['foods_mean']:>7.1f} {r['foods_std']:>8.1f} {r['gini']:>6.3f} "
              f"{r['pair_dist_mean']:>8.1f} {r['entropy_mean']:>6.3f} {r['deaths']:>4}")

    print("\n每虫明细 (id, 食物数, 嗅觉增益, 探索噪声):")
    for r in results:
        print(f"  [{r['name'].split('（')[0]}]  " + "  ".join(
            f"#{i}:{f}(smell{g:.1f}/noise{n:.2f})" for i, f, g, n in r['per_bug']))

    print("\n基因与觅食表现相关性 (B/C 组):")
    for r in results:
        if r['gene_corr']:
            corr = "  ".join(f"{k}={v:+.2f}" for k, v in r['gene_corr'].items())
            print(f"  [{r['name'].split('（')[0]}]  {corr}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = [r['name'].split('（')[0] for r in results]

    ax = axes[0]
    x = np.arange(3)
    ax.bar(x, [r['foods_std'] for r in results], color=['#adb5bd', '#4cc9f0', '#f72585'])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=9)
    ax.set_title('每虫食物数标准差（越大=能力分化越明显）'); ax.set_ylabel('std')

    ax = axes[1]
    for r in results:
        ax.plot(np.arange(len(r['pair_dists'])) * 100, r['pair_dists'],
                label=r['name'].split('（')[0], linewidth=2)
    ax.set_title('两两虫子平均距离（越大=散得越开）'); ax.set_xlabel('step'); ax.set_ylabel('曼哈顿距离')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(x, [r['entropy_mean'] for r in results], color=['#adb5bd', '#4cc9f0', '#f72585'])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=9)
    ax.set_title('决策熵（越大=行为越多样）'); ax.set_ylabel('entropy')

    plt.tight_layout()
    plt.savefig('multi_bug_diff_result.png', dpi=110)
    print("\n图表已保存: multi_bug_diff_result.png")
