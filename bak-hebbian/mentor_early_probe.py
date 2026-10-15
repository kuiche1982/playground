"""探针：6步教学窗到底有没有帮子代在 EnvHebbian 里积累环境经验？
对照：
  A: mentor_gene_mode=True, imitation_prob=0.8（正常C组，父代带子走）
  B: mentor_gene_mode=True, imitation_prob=0.0（父代在但子代不模仿动作——纯靠自己位置感知）
跑 seed=44 各 4000 步，对比终点种群/幼死/饿死/早期子代首次进食步数。
另外直接追踪若干新生子代的 env.density 非零格数随步数的变化。
"""
import numpy as np
import multi_bug_genome as M

# 先跑宏观对照
print('=== 宏观对照：imitation_prob=0.8 vs 0.0（seed=44, 4000步）===')
for name, ip in [('A_模仿0.8', 0.8), ('B_模仿0.0', 0.0)]:
    r = M.natural_evolve(seed=44, steps=4000, n_bugs=24, max_pop=80, n_food=48,
                          mentor_gene_mode=True, imitation_prob=ip,
                          record_full=True, verbose_every=100000)
    pop = r['pop'][-1] if r['pop'] else 0
    early = int((r['death_ages'] < 100).sum())
    starve = r['death_causes'].count('starve')
    mt = r['mentor_stats'][-1]['mean'] if r['mentor_stats'] else -1
    print(f"{name}: 种群 {pop:>3} | 幼死<100 {early:>3} | 饿死 {starve:>3} | 收敛教学窗 {mt:.0f}")

# 微观探针：追踪新生子代 env.density 积累
print('\n=== 微观探针：新生子代前20步 env.density 非零格数 ===')

# 跑一个短 run，在 step 1000 后追踪新生虫子
import types

def run_with_probe(seed, imitation_prob, steps=1500, track_n=20):
    rng = np.random.RandomState(seed)
    world = M.World(size=15, n_food=48, n_walls=3, n_cues=6, seed=seed)
    world.food_count = np.zeros((15, 15), dtype=np.int32)
    for (fr, fc) in np.argwhere(world.grid == world.FOOD):
        world.food_count[fr, fc] = 1
    M.place_stable(world, rng, corner='tl', n_cues=6, n_food=48)
    learn = {'mb': True, 'hip': True, 'fm': True, 'pfc': True, 'ctx': True, 'env': True}
    bugs = []
    for i in range(24):
        if world.cue_positions:
            cr, cc = world.cue_positions[i % len(world.cue_positions)]
        else:
            cr, cc = rng.randint(2, 13, 2)
        for _ in range(30):
            dr, dc = world.ACTIONS[rng.randint(4)]
            r, c = cr + dr * 2, cc + dc * 2
            if 0 <= r < 15 and 0 <= c < 15 and world.grid[r, c] != world.WALL:
                if all(abs(b.pos[0]-r)+abs(b.pos[1]-c) >= 2 for b in bugs):
                    break
        else:
            r, c = cr, cc
        bugs.append(M.Bug([r, c], M.make_genome(rng, 1.0), seed=42, learn_flags=learn,
                         stable=True, patchy=False, natural=True,
                         reproduce_threshold=0.4, birth_energy=1.0, litter_size=3,
                         smell_range=5, max_age=500, world_size=15,
                         mentor_ticks=0, imitation_prob=imitation_prob, energy_transfer=0.5))
        bugs[-1].mentor_fixed = False  # 基因自由
    world.bugs = bugs

    tracked = []  # (bug_ref, birth_step, first_food_step, density_trace)
    for step in range(steps):
        alive = [b for b in bugs if b.alive]
        for b in alive:
            b.decide_and_learn(world)
        for b in [x for x in alive if x.alive]:
            n_alive = len([x for x in bugs if x.alive])
            if rng.rand() < max(0.0, (n_alive - 40) / 40):
                continue
            ns = b.reproduce(world, rng)
            if ns:
                bugs.extend(ns)
                if len(tracked) < track_n:
                    for child in ns:
                        tracked.append({'bug': child, 'birth': step, 'first_food': None,
                                       'dens_trace': []})
        world.bugs = bugs
        # 记录 tracked 虫子的 density
        for t in tracked:
            b = t['bug']
            if not b.alive:
                continue
            age = step - t['birth']
            if age <= 20:
                nz = int((b.env.density > 0.01).sum())
                t['dens_trace'].append((age, nz, b.foods))
                if t['first_food'] is None and b.foods > 0:
                    t['first_food'] = age

    return tracked

for name, ip in [('A_模仿0.8', 0.8), ('B_模仿0.0', 0.0)]:
    tracked = run_with_probe(44, ip, steps=1500, track_n=30)
    # 汇总：前20步 density 非零格数均值
    age_grid = {}
    for t in tracked:
        for age, nz, foods in t['dens_trace']:
            age_grid.setdefault(age, []).append(nz)
    first_foods = [t['first_food'] for t in tracked if t['first_food'] is not None]
    print(f"\n{name}:")
    print(f"  追踪 {len(tracked)} 只新生子代, 其中 {len(first_foods)} 只在20步内吃到食物")
    if first_foods:
        print(f"  首次进食步数: mean={np.mean(first_foods):.1f}, median={np.median(first_foods):.0f}")
    for age in [0, 1, 3, 6, 10, 15, 20]:
        if age in age_grid:
            print(f"  第{age:>2}步 density 非零格数: mean={np.mean(age_grid[age]):.1f}")
