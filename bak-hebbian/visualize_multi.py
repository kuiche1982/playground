"""Embryo: Multi-bug version with trained brain, 10 bugs competing for food"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from embryo import World, MushroomBody, Hippocampus, ForwardModel, LanguageChannel, ContextModulator, PrefrontalCortex


class TrainedBug:
    """A bug with trained brain + genetic variation."""
    def __init__(self, seed=0, color_idx=0):
        np.random.seed(seed)
        self.color_idx = color_idx  # for visualization
        self.pos = np.array([np.random.randint(3, 11), np.random.randint(3, 11)])
        self.energy = 0.8
        self.alive = True
        self.foods_eaten = 0
        
        # Genetic variation - each bug has slightly different parameters
        self.smell_gain = np.random.uniform(0.8, 1.2)  # smell sensitivity varies
        self.explore_noise = np.random.uniform(0.1, 0.4)  # exploration tendency varies
        self.habit_gain = np.random.uniform(0.8, 1.2)  # habit strength varies
        
        # Initialize brain
        self.mb = MushroomBody(n_input=20, n_kc=500, n_actions=4, lr=0.15, trace_decay=0.85, seed=seed)
        self.hip = Hippocampus(world_size=20, decay=0.9995)
        self.fm = ForwardModel(n_kc=500, n_actions=4, lr=0.1, seed=seed)
        self.lang = LanguageChannel()
        self.pfc = PrefrontalCortex(n_input=6, n_hidden=8, n_output=5, lr=0.25, seed=seed)
        self.ctx = ContextModulator(n_input=4, n_output=4, lr=0.06, seed=seed)
        
        # Load trained weights
        try:
            self.mb.load('checkpoints/mb_final.npz')
            self.hip.load('checkpoints/hip_final.npy')
            self.fm.load('checkpoints/fm_final.npy')
        except:
            pass
    
    def decide(self, world):
        smell = world.smell_gradient()
        vision = world.see()
        lang_input = self.lang.get_input()
        
        sensory = np.concatenate([vision, lang_input])
        r, c = int(self.pos[0]), int(self.pos[1])
        # Boundary check for hippocampus memory
        if 0 <= r < 15 and 0 <= c < 15:
            hip_strength = self.hip.memory[r, c]
        else:
            hip_strength = 0.0
        task_bias = self.pfc.forward(lang_input, self.energy, hip_strength)
        ctx_gain = self.ctx.forward(task_bias[:4])
        active_kc = self.mb.encode(sensory)
        spatial = self.hip.spatial_bias(self.pos)
        imagined = self.fm.imagine(active_kc)
        habit = self.mb.decide(active_kc)
        language_bias = lang_input * (3.0 + 4.0 * self.energy)
        pfc_bias = self.pfc.get_bias(task_bias)
        inhibit = task_bias[4]
        smell_weight = self.smell_gain * (1.0 - inhibit)
        habit_weight = self.habit_gain * (1.0 - inhibit)
        ctx_bias = self.ctx.get_gain(ctx_gain)
        total = smell * smell_weight + habit * habit_weight + spatial * 0.5 + imagined * 0.8 + language_bias + pfc_bias * 0.5 + ctx_bias
        noise = np.random.randn(4) * self.explore_noise
        return int(np.argmax(total + noise))


def main():
    np.random.seed(42)
    world_size = 20
    
    world = World(size=15, n_food=8, n_walls=3, n_cues=6, seed=42)
    
    # Start with 10 bugs, different colors
    bug_colors = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # different colors for each bug
    bugs = [TrainedBug(seed=i, color_idx=bug_colors[i]) for i in range(10)]
    
    foods_eaten_by_bug = [0] * 10
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Update each bug
        for i, bug in enumerate(bugs):
            if not bug.alive:
                continue
            
            action = bug.decide(world)
            
            # Move bug
            dr, dc = world.ACTIONS[action]
            nr, nc = bug.pos[0] + dr, bug.pos[1] + dc
            
            # Check wall
            if 0 <= nr < world.size and 0 <= nc < world.size and world.grid[nr, nc] != world.WALL:
                bug.pos = np.array([nr, nc])
                # Clamp to safe bounds (leave room for spatial_bias - 3 steps)
                bug.pos[0] = np.clip(bug.pos[0], 3, 11)
                bug.pos[1] = np.clip(bug.pos[1], 3, 11)
                
                # Check food
                if world.grid[nr, nc] == world.FOOD:
                    bug.foods_eaten += 1
                    foods_eaten_by_bug[i] += 1
                    bug.energy = min(1.0, bug.energy + 0.3)
                    world.grid[nr, nc] = world.EMPTY
                    # Respawn food
                    world._respawn_food()
            
            bug.energy -= 0.003
            if bug.energy <= 0:
                bug.alive = False
        
        # Draw world
        grid = world.grid.copy()
        for bug in bugs:
            if bug.alive:
                grid[bug.pos[0], bug.pos[1]] = bug.color_idx
        
        from matplotlib.colors import ListedColormap
        colors = ['#3a0ca3', '#43aa8b', '#ffd60a', '#4895ef', '#ffffff',
                  '#f72585', '#4cc9f0', '#7209b7', '#f8961e', '#f94144',
                  '#90be6d', '#577590', '#e63946', '#a8dadc', '#1d3557']
        cmap = ListedColormap(colors)
        ax1.imshow(grid, cmap=cmap, vmin=0, vmax=14)
        ax1.set_title(f'Step {frame} | 10 Bugs competing for food')
        ax1.set_xlabel('Each color = one bug, Yellow = Food, Green = Wall')
        
        # Foods eaten by each bug
        ax2.bar(range(10), foods_eaten_by_bug)
        ax2.set_title('Foods eaten by each bug')
        ax2.set_xlabel('Bug ID')
        ax2.set_ylabel('Foods eaten')
        
        return ax1, ax2
    
    ani = animation.FuncAnimation(fig, update, frames=5000, interval=50)
    plt.show()


if __name__ == "__main__":
    main()
