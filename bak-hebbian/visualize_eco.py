"""Embryo: Ecosystem version with trained brain, sex, breeding, seasons"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from embryo import World, MushroomBody, Hippocampus, ForwardModel, LanguageChannel, ContextModulator, PrefrontalCortex


class TrainedBug:
    """A bug with trained brain."""
    def __init__(self, seed=0, gender=None):
        np.random.seed(seed)
        self.gender = gender if gender is not None else np.random.choice(['M', 'F'])
        self.pos = np.array([7, 7])
        self.energy = 0.8
        self.alive = True
        self.foods_eaten = 0
        
        # Initialize brain
        self.mb = MushroomBody(n_input=20, n_kc=500, n_actions=4, lr=0.15, trace_decay=0.85, seed=seed)
        self.hip = Hippocampus(world_size=15, decay=0.9995)
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
        hip_strength = self.hip.memory[self.pos[0], self.pos[1]]
        task_bias = self.pfc.forward(lang_input, self.energy, hip_strength)
        ctx_gain = self.ctx.forward(task_bias[:4])
        active_kc = self.mb.encode(sensory)
        spatial = self.hip.spatial_bias(self.pos)
        imagined = self.fm.imagine(active_kc)
        habit = self.mb.decide(active_kc)
        language_bias = lang_input * (3.0 + 4.0 * self.energy)
        pfc_bias = self.pfc.get_bias(task_bias)
        inhibit = task_bias[4]
        smell_weight = 1.0 * (1.0 - inhibit)
        habit_weight = 1.0 * (1.0 - inhibit)
        ctx_bias = self.ctx.get_gain(ctx_gain)
        total = smell * smell_weight + habit * habit_weight + spatial * 0.5 + imagined * 0.8 + language_bias + pfc_bias * 0.5 + ctx_bias
        noise = np.random.randn(4) * 0.2
        return int(np.argmax(total + noise))


def main():
    np.random.seed(42)
    
    world = World(size=15, n_food=8, n_walls=3, n_cues=6, seed=42)
    
    # Start with 10 bugs
    bugs = [TrainedBug(seed=i) for i in range(10)]
    
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    season_step = 0
    foods_eaten_total = 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def update(frame):
        nonlocal season_step, foods_eaten_total
        
        ax1.clear()
        ax2.clear()
        
        # Season cycle
        season_step = (frame // 500) % 4
        current_season = seasons[season_step]
        
        # Season effect on food
        if current_season == 'Winter':
            food_multiplier = 0.3
        elif current_season == 'Autumn':
            food_multiplier = 0.7
        else:
            food_multiplier = 1.0
        
        # Update each bug
        alive_bugs = []
        for bug in bugs:
            if not bug.alive:
                continue
            
            action = bug.decide(world)
            ate, bumped = world.step(action)
            
            # Update bug position
            dr, dc = world.ACTIONS[action]
            nr, nc = bug.pos[0] + dr, bug.pos[1] + dc
            if 0 <= nr < world.size and 0 <= nc < world.size and world.grid[nr, nc] != world.WALL:
                bug.pos = np.array([nr, nc])
            
            if ate:
                bug.foods_eaten += 1
                bug.energy = min(1.0, bug.energy + 0.3)
                foods_eaten_total += 1
            
            bug.energy -= 0.003
            if bug.energy <= 0:
                bug.alive = False
            
            alive_bugs.append(bug)
        
        # Breeding: if two bugs meet and both have high energy
        for i in range(len(alive_bugs)):
            for j in range(i+1, len(alive_bugs)):
                if alive_bugs[i].gender != alive_bugs[j].gender:
                    dist = np.abs(alive_bugs[i].pos - alive_bugs[j].pos).sum()
                    if dist < 2 and alive_bugs[i].energy > 0.7 and alive_bugs[j].energy > 0.7:
                        # Breed!
                        child = TrainedBug(seed=np.random.randint(1000))
                        child.pos = (alive_bugs[i].pos + alive_bugs[j].pos) // 2
                        child.energy = 0.5
                        bugs.append(child)
                        alive_bugs.append(child)
                        alive_bugs[i].energy -= 0.2
                        alive_bugs[j].energy -= 0.2
        
        # Auto-split if too few bugs
        if len(alive_bugs) < 5:
            for bug in alive_bugs[:2]:
                for _ in range(3):
                    child = TrainedBug(seed=np.random.randint(1000))
                    child.pos = bug.pos.copy()
                    child.energy = 0.5
                    bugs.append(child)
                    alive_bugs.append(child)
        
        # Draw world
        grid = world.grid.copy()
        for bug in bugs:
            if bug.alive:
                color = 5 if bug.gender == 'M' else 6  # 5=male, 6=female
                grid[bug.pos[0], bug.pos[1]] = color
        
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#3a0ca3', '#43aa8b', '#ffd60a', '#4895ef', '#ffffff', '#4cc9f0', '#f72585'])
        ax1.imshow(grid, cmap=cmap, vmin=0, vmax=6)
        ax1.set_title(f'Step {frame} | Season: {current_season} | Bugs: {len(alive_bugs)} | Food: {foods_eaten_total}')
        ax1.set_xlabel('Blue=Male, Pink=Female, Yellow=Food, Green=Wall')
        
        # Population curve
        if not hasattr(update, 'pop_history'):
            update.pop_history = []
        update.pop_history.append(len(alive_bugs))
        ax2.plot(update.pop_history[-100:])
        ax2.set_title('Population over time')
        ax2.set_ylim(0, 30)
        ax2.set_ylabel('Bugs alive')
        
        return ax1, ax2
    
    ani = animation.FuncAnimation(fig, update, frames=5000, interval=50)
    plt.show()


if __name__ == "__main__":
    main()
