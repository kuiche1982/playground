"""极简验证：Hebbian 层挂在固定底座上，能不能实时学会找食物？
对照：
  A 纯底座：感官编码固定，动作权重也固定（不学习）
  B 混合架构：感官编码固定（模拟大模型底座），动作权重 Hebbian 实时更新
指标：吃到食物数、平均找食步数
"""
import numpy as np

SIZE = 10
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右

class MiniWorld:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.food = (rng.randint(0, SIZE), rng.randint(0, SIZE))
        self.pos = (rng.randint(0, SIZE), rng.randint(0, SIZE))
    def step(self, action):
        dr, dc = ACTIONS[action]
        nr = max(0, min(SIZE-1, self.pos[0] + dr))
        nc = max(0, min(SIZE-1, self.pos[1] + dc))
        self.pos = (nr, nc)
        ate = (self.pos == self.food)
        if ate:
            self.food = (np.random.randint(0, SIZE), np.random.randint(0, SIZE))
        return ate

def encode(pos):
    """模拟大模型 embedding：把位置 one-hot 投影到 8 维（固定不学习）"""
    vec = np.zeros(SIZE * SIZE)
    vec[pos[0] * SIZE + pos[1]] = 1.0
    return W_fixed @ vec  # W_fixed 是随机固定矩阵

# 固定底座：随机投影矩阵（模拟预训练 embedding）
rng_fixed = np.random.RandomState(42)
W_fixed = rng_fixed.randn(8, SIZE * SIZE) * 0.5

def run(with_hebbian, steps=2000, seed=0):
    rng = np.random.RandomState(seed)
    W_action = rng.randn(4, 8) * 0.1  # 动作层权重
    world = MiniWorld(seed=seed)
    foods_eaten = 0
    step_history = []
    for t in range(steps):
        x = encode(world.pos)  # 底座编码（固定）
        logits = W_action @ x
        action = int(np.argmax(logits + rng.randn(4) * 0.3))  # 探索噪声
        ate = world.step(action)
        if ate:
            foods_eaten += 1
            step_history.append(t)
        # Hebbian 更新：吃到食物 DA=+1，撞墙/没吃到 DA=0
        if with_hebbian:
            da = 1.0 if ate else 0.0
            # Δw = lr * 突触后激活 * 突触前激活 * DA
            post = np.zeros(4); post[action] = 1.0
            pre = x
            W_action += 0.1 * np.outer(post, pre) * da
    return foods_eaten, step_history

# 跑对照
print("=== 极简验证：Hebbian 层挂在固定底座上 ===")
for seed in (42, 43, 44):
    a_foods, a_hist = run(with_hebbian=False, steps=2000, seed=seed)
    b_foods, b_hist = run(with_hebbian=True, steps=2000, seed=seed)
    print(f"seed {seed}: 纯底座吃 {a_foods:>3} | 混合架构吃 {b_foods:>3} | "
          f"提升 {b_foods - a_foods:+d}")

print("\n结论：如果混合架构组吃到食物显著更多，说明 Hebbian 层在固定底座上能实时学习。")
