"""
参数搜索脚本：自动跑多组参数，找听懂指令和找食物的平衡点
"""
import subprocess
import re
import os

# 要测试的参数组合
configs = [
    {"name": "基线(无抑制)", "language": "3.0 + 4.0 * world.energy", "inhibit": "0.0"},
    {"name": "抑制0.25", "language": "3.0 + 4.0 * world.energy", "inhibit": "0.25"},
    {"name": "抑制0.5", "language": "3.0 + 4.0 * world.energy", "inhibit": "0.5"},
    {"name": "抑制0.75", "language": "3.0 + 4.0 * world.energy", "inhibit": "0.75"},
    {"name": "抑制1.0(全压)", "language": "3.0 + 4.0 * world.energy", "inhibit": "1.0"},
]

results = []

for cfg in configs:
    print(f"\n=== 跑: {cfg['name']} ===")
    
    # 临时改embryo.py
    with open("embryo.py", "r") as f:
        code = f.read()
    
    # 改抑制强度
    old_inhibit = "smell_weight = 1.0 * (1.0 - 0.5 * inhibit)"
    new_inhibit = f"smell_weight = 1.0 * (1.0 - {cfg['inhibit']} * inhibit)"
    code = code.replace(old_inhibit, new_inhibit)
    
    old_inhibit2 = "habit_weight = 1.0 * (1.0 - 0.5 * inhibit)"
    new_inhibit2 = f"habit_weight = 1.0 * (1.0 - {cfg['inhibit']} * inhibit)"
    code = code.replace(old_inhibit2, new_inhibit2)
    
    with open("embryo.py", "w") as f:
        f.write(code)
    
    # 跑训练
    result = subprocess.run(["python", "embryo.py"], capture_output=True, text=True)
    output = result.stdout
    
    # 提取结果
    food = int(re.search(r"吃食物总数: (\d+)", output).group(1))
    starve = int(re.search(r"饿死次数: (\d+)", output).group(1))
    listen = float(re.search(r"听懂指令: (\d+)/100", output).group(1))
    improve = float(re.search(r"提升(-?[\d.]+)%", output).group(1))
    
    results.append({
        "name": cfg["name"],
        "food": food,
        "starve": starve,
        "listen": listen,
        "improve": improve,
    })
    
    print(f"吃食物: {food}, 饿死: {starve}, 听懂指令: {listen}%, 找食提升: {improve}%")

# 恢复原始代码
with open("embryo.py", "r") as f:
    code = f.read()
code = code.replace("smell_weight = 1.0 * (1.0 - 0.25 * inhibit)", "smell_weight = 1.0 * (1.0 - 0.5 * inhibit)")
code = code.replace("habit_weight = 1.0 * (1.0 - 0.25 * inhibit)", "habit_weight = 1.0 * (1.0 - 0.5 * inhibit)")
code = code.replace("smell_weight = 1.0 * (1.0 - 0.75 * inhibit)", "smell_weight = 1.0 * (1.0 - 0.5 * inhibit)")
code = code.replace("habit_weight = 1.0 * (1.0 - 0.75 * inhibit)", "habit_weight = 1.0 * (1.0 - 0.5 * inhibit)")
code = code.replace("smell_weight = 1.0 * (1.0 - 1.0 * inhibit)", "smell_weight = 1.0 * (1.0 - 0.5 * inhibit)")
code = code.replace("habit_weight = 1.0 * (1.0 - 1.0 * inhibit)", "habit_weight = 1.0 * (1.0 - 0.5 * inhibit)")
with open("embryo.py", "w") as f:
    f.write(code)

# 打印汇总表
print("\n" + "="*60)
print(f"{'配置':<15} {'吃食物':<10} {'饿死':<10} {'听懂指令':<12} {'找食提升':<10}")
print("="*60)
for r in results:
    print(f"{r['name']:<15} {r['food']:<10} {r['starve']:<10} {r['listen']}%{'':<8} {r['improve']}%")
print("="*60)
