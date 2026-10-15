"""成熟年龄基因收敛分析（自然生命周期进化）

大种群（max_pop=150）下单 seed × 15000 步，每 1000 步即时落盘 CSV，
记录成熟年龄基因分布演化（mean/std/min/p10/p50/p90/max）。

用法:
  python natural_convergence.py --seed 42     # 单 seed，CSV 落到 natural_mature_s42.csv
  python natural_convergence.py --merge       # 合并已有 CSV -> natural_mature_convergence.csv + 图
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multi_bug_genome as M

STEPS = 15000
FIELDS = ['seed', 'step', 'mean', 'std', 'min', 'p10', 'p50', 'p90', 'max', 'n']


def run_seed(seed):
    csv_path = f'natural_mature_s{seed}.csv'
    # 续跑：已存在的行跳过
    done_steps = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done_steps.add(int(row['step']))
    print(f"\n--- seed {seed} (已记录 {len(done_steps)} 个采样点) ---")
    r = M.natural_evolve(seed=seed, steps=STEPS, n_bugs=50, max_pop=150,
                         n_food=72, litter_size=3, record_full=True)
    fresh = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not fresh:
            writer.writeheader()
        for st in r['mature_stats']:
            if st['step'] in done_steps:
                continue
            writer.writerow({'seed': seed, **st})
    print(f"终点: 种群 {r['pop'][-1]} | 最大世代 {r['gen_max']} | "
          f"饿死 {r['death_causes'].count('starve')} | CSV 已保存: {csv_path}")
    # 即时打印收敛结论
    stats = r['mature_stats']
    tail = stats[len(stats) // 4 * 3:]
    m = np.mean([t['mean'] for t in tail])
    p10 = np.mean([t['p10'] for t in tail])
    p90 = np.mean([t['p90'] for t in tail])
    print(f"seed {seed} 收敛: 起点 {stats[0]['mean']:.1f} → 终点均值 {m:.1f} "
          f"(10%~90% 分位 {p10:.0f}~{p90:.0f})")


def merge_and_plot():
    seeds = []
    for fn in sorted(os.listdir('.')):
        if fn.startswith('natural_mature_s') and fn.endswith('.csv'):
            seeds.append(fn)
    if not seeds:
        print("没有找到 natural_mature_s*.csv，先跑 --seed")
        return
    all_rows = []
    for fn in seeds:
        with open(fn) as f:
            all_rows.extend(csv.DictReader(f))
    all_rows.sort(key=lambda r: (r['seed'], int(r['step'])))
    with open('natural_mature_convergence.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"合并 CSV: {len(all_rows)} 行 -> natural_mature_convergence.csv")

    by_seed = {}
    for r in all_rows:
        by_seed.setdefault(r['seed'], []).append(r)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {str(s): c for s, c in zip(sorted(by_seed), ['#f72585', '#4361ee', '#2a9d8f'])}
    for s, rows in by_seed.items():
        x = [int(r['step']) for r in rows]
        mean = [float(r['mean']) for r in rows]
        p10 = [float(r['p10']) for r in rows]
        p90 = [float(r['p90']) for r in rows]
        ax.plot(x, mean, 'o-', label=f'seed {s}', color=colors[s], linewidth=2)
        ax.fill_between(x, p10, p90, alpha=0.15, color=colors[s])
    ax.axhline(50, color='#999', linestyle=':', label='初始均值≈50')
    ax.set_xlabel('step'); ax.set_ylabel('成熟年龄基因均值')
    ax.set_title('成熟年龄基因收敛（大种群 150，带 10%~90% 分位带）')
    ax.legend()
    plt.tight_layout()
    plt.savefig('multi_bug_mature_convergence.png', dpi=110)
    print("图表已保存: multi_bug_mature_convergence.png")


if __name__ == '__main__':
    if '--merge' in sys.argv:
        merge_and_plot()
    else:
        seed = 42
        for i, a in enumerate(sys.argv):
            if a == '--seed' and i + 1 < len(sys.argv):
                seed = int(sys.argv[i + 1])
        run_seed(seed)
