"""Embryo: Visualization with keyboard control"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from embryo import World, MushroomBody, Hippocampus, ForwardModel, LanguageChannel, ContextModulator, PrefrontalCortex


def main():
    np.random.seed(42)
    
    # Initialize all modules
    world = World(size=15, n_food=10, n_walls=3, n_cues=6, seed=42)
    mb = MushroomBody(n_input=20, n_kc=500, n_actions=4, lr=0.15, trace_decay=0.85, seed=42)
    hip = Hippocampus(world_size=15, decay=0.9995)
    fm = ForwardModel(n_kc=500, n_actions=4, lr=0.1, seed=42)
    lang = LanguageChannel()
    pfc = PrefrontalCortex(n_input=6, n_hidden=8, n_output=5, lr=0.25, seed=42)
    ctx = ContextModulator(n_input=4, n_output=4, lr=0.06, seed=42)
    
    # Load trained weights
    try:
        mb.load('checkpoints/mb_final.npz')
        hip.load('checkpoints/hip_final.npy')
        fm.load('checkpoints/fm_final.npy')
        print("Loaded trained weights from checkpoints")
    except:
        print("No checkpoint found, using fresh weights")
    
    foods_eaten = 0
    energy_history = []
    current_command = None  # Keyboard command
    command_steps = 0  # How long since command was given
    COMMAND_TIMEOUT = 20  # Command expires after 20 steps
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Keyboard control
    def on_key(event):
        nonlocal current_command, command_steps
        if event.key == 'up':
            current_command = 0
            command_steps = 0
            lang.set_command(0)
            print("Command: UP")
        elif event.key == 'down':
            current_command = 1
            command_steps = 0
            lang.set_command(1)
            print("Command: DOWN")
        elif event.key == 'left':
            current_command = 2
            command_steps = 0
            lang.set_command(2)
            print("Command: LEFT")
        elif event.key == 'right':
            current_command = 3
            command_steps = 0
            lang.set_command(3)
            print("Command: RIGHT")
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    def update(frame):
        nonlocal foods_eaten, command_steps, current_command
        
        ax1.clear()
        ax2.clear()
        
        # Command timeout - forget after N steps
        command_steps += 1
        if command_steps > COMMAND_TIMEOUT:
            current_command = None
            lang.set_command(None)
        
        # Update temp food lifespan
        new_temp_foods = []
        for fr, fc, life in world.temp_foods:
            life -= 1
            if life > 0 and world.grid[fr, fc] == world.FOOD:
                new_temp_foods.append((fr, fc, life))
            elif world.grid[fr, fc] == world.FOOD:
                world.grid[fr, fc] = world.EMPTY
        world.temp_foods = new_temp_foods
        
        # Sense
        smell = world.smell_gradient()
        vision = world.see()
        lang_input = lang.get_input()
        
        # Decide
        sensory = np.concatenate([vision, lang_input])
        hip_strength = hip.memory[world.pos[0], world.pos[1]]
        task_bias = pfc.forward(lang_input, world.energy, hip_strength)
        ctx_gain = ctx.forward(task_bias[:4])
        active_kc = mb.encode(sensory)
        spatial = hip.spatial_bias(world.pos)
        imagined = fm.imagine(active_kc)
        habit = mb.decide(active_kc)
        language_bias = lang_input * (3.0 + 4.0 * world.energy)
        pfc_bias = pfc.get_bias(task_bias)
        inhibit = task_bias[4]
        smell_weight = 1.0 * (1.0 - inhibit)
        habit_weight = 1.0 * (1.0 - inhibit)
        ctx_bias = ctx.get_gain(ctx_gain)
        total = smell * smell_weight + habit * habit_weight + spatial * 0.5 + imagined * 0.8 + language_bias + pfc_bias * 0.5 + ctx_bias
        noise = np.random.randn(4) * 0.2
        action = int(np.argmax(total + noise))
        
        # Execute
        ate, bumped = world.step(action)
        mb.update_trace(active_kc, action)
        hip.decay_memory()
        fm.observe(active_kc, action, ate, bumped)
        
        if ate:
            foods_eaten += 1
            hip.record_food(world.pos)
        
        # Draw world
        grid = world.grid.copy()
        grid[world.pos[0], world.pos[1]] = 4
        
        cmd_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        cmd_str = cmd_names[current_command] if current_command is not None else 'NONE'
        
        ax1.imshow(grid, cmap='tab10', vmin=0, vmax=4)
        ax1.set_title(f'Step {frame} | Energy: {world.energy:.2f} | Food: {foods_eaten} | Cmd: {cmd_str}')
        ax1.set_xlabel('0=Empty, 1=Wall, 2=Food, 3=Cue, 4=Bug')
        ax1.text(0.5, -0.15, 'Use arrow keys to give commands', transform=ax1.transAxes, ha='center')
        
        # Energy curve
        energy_history.append(world.energy)
        ax2.plot(energy_history[-100:])
        ax2.set_title('Energy over time')
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Energy')
        
        return ax1, ax2
    
    ani = animation.FuncAnimation(fig, update, frames=10000, interval=50)
    plt.show()


if __name__ == "__main__":
    main()
