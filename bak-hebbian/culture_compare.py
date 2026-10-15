"""文化传递（亲代教学）ON/OFF 快速对比：seed 42/43/44 × 4000 步
指标：终点种群 / 最大世代 / 幼年期(<100步)死亡 / 终点平均食物 / 死因构成
用法: python culture_compare.py
"""
import multi_bug_genome as M

print('=== 文化传递 mentor ON(80) vs OFF(0)，3 seeds × 4000 步 ===')
for seed in (42, 43, 44):
    for name, mt in (('OFF', 0), ('ON', 80)):
        r = M.natural_evolve(seed=seed, steps=4000, n_bugs=24, max_pop=80, n_food=48,
                             mentor_ticks=mt, verbose_every=100000)
        pop_end = r['pop'][-1] if r['pop'] else 0
        food_end = r['food'][-1] if r['food'] else 0.0
        early = int((r['death_ages'] < 100).sum())
        causes = {c: r['death_causes'].count(c) for c in set(r['death_causes'])}
        print(f"seed {seed} {name}: 终点种群 {pop_end} | 最大世代 {r['gen_max']:>2} | "
              f"幼年死亡<100 {early:>3} | 终点均食 {food_end:>4.1f} | 死因 {causes}")
