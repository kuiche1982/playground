"""方案 C：mentor_ticks 基因化 vs 固定 0 / 固定 80
3 seeds × 4000 步
指标：终点种群 / 最大世代 / 幼年死亡(<100步) / 终点均食 / 死因构成 / C组终点教学窗基因收敛值
用法: python mentor_gene_compare.py
"""
import numpy as np
import multi_bug_genome as M

print('=== mentor_ticks 基因化(C) vs 固定0(A) vs 固定80(B)，3 seeds × 4000 步 ===\n')

rows = []
for seed in (42, 43, 44):
    # A: 固定 0（繁殖即死，不教）
    rA = M.natural_evolve(seed=seed, steps=4000, n_bugs=24, max_pop=80, n_food=48,
                           mentor_ticks=0, mentor_gene_mode=False,
                           record_full=True, verbose_every=100000)
    # B: 固定 80（当前硬编码教学窗）
    rB = M.natural_evolve(seed=seed, steps=4000, n_bugs=24, max_pop=80, n_food=48,
                           mentor_ticks=80, mentor_gene_mode=False,
                           record_full=True, verbose_every=100000)
    # C: 基因化（初始 0~120 均匀，自然选择收敛）
    rC = M.natural_evolve(seed=seed, steps=4000, n_bugs=24, max_pop=80, n_food=48,
                           mentor_ticks=0, mentor_gene_mode=True,
                           record_full=True, verbose_every=100000)
    for name, r in (('A_固定0', rA), ('B_固定80', rB), ('C_基因化', rC)):
        pop_end = r['pop'][-1] if r['pop'] else 0
        food_end = r['food'][-1] if r['food'] else 0.0
        early = int((r['death_ages'] < 100).sum())
        causes = {c: r['death_causes'].count(c) for c in set(r['death_causes'])}
        if r['mentor_stats']:
            last = r['mentor_stats'][-1]
            mt_final = f"mean={last['mean']:.0f} (p10={last['p10']:.0f}/p50={last['p50']:.0f}/p90={last['p90']:.0f})"
        else:
            mt_final = '-'
        rows.append((seed, name, pop_end, r['gen_max'], early, food_end, causes, mt_final))
        print(f"seed {seed} {name:8s}: 种群 {pop_end:>3} | 世代 {r['gen_max']:>2} | "
              f"幼死<100 {early:>3} | 均食 {food_end:>4.1f} | 死因 {causes} | 教学窗 {mt_final}")

print('\n=== 3 seeds 汇总 ===')
for name in ('A_固定0', 'B_固定80', 'C_基因化'):
    sub = [x for x in rows if x[1] == name]
    print(f"{name:8s}: 平均种群 {np.mean([x[2] for x in sub]):.0f} | "
          f"平均世代 {np.mean([x[3] for x in sub]):.1f} | "
          f"平均幼死 {np.mean([x[4] for x in sub]):.0f} | "
          f"平均均食 {np.mean([x[5] for x in sub]):.1f}")
