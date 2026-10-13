"""Embryo: a creature born knowing how to seek food, learning as it lives."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from embryo.world import World
from embryo.brain import Brain


def main():
    np.random.seed(0)
    world = World(size=20, n_food=8, n_walls=10, seed=1)
    brain = Brain(n_sensors=25, n_actions=8, lr=0.2, world_size=20, seed=2)

    food_count = 0
    energy_history = []
    _bumped_last = False

    fig, (ax_world, ax_log) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Embryo — innate tropism + Hebbian learning", fontsize=13)

    def update(frame):
        nonlocal food_count, _bumped_last
        sensory = world.sense()
        innate = world.innate_tropism(sensory)
        hunger = 1.0 - world.energy
        action = brain.decide(sensory, innate, hunger, world.pos)
        if _bumped_last:
            action = (action + np.random.randint(1, 4)) % 4
        ate = world.step(action)
        _bumped_last = world.last_bumped
        brain.decay_memory()
        if ate:
            brain.reinforce(world.pos)
            food_count += 1
        energy_history.append(world.energy)

        ax_world.clear()
        ax_world.imshow(world.grid, cmap='viridis', vmin=0, vmax=2)
        ax_world.plot(world.pos[1], world.pos[0], 'r*', markersize=18)
        ax_world.set_title(f"Step {world.steps} | Energy {world.energy:.2f} | Food {food_count}")
        ax_world.set_xticks([]); ax_world.set_yticks([])

        ax_log.clear()
        ax_log.plot(energy_history, color='steelblue', linewidth=0.8)
        ax_log.set_title("Homeostasis: energy over time")
        ax_log.set_xlabel("steps")
        ax_log.set_ylim(0, 1.1)

    ani = animation.FuncAnimation(fig, update, interval=30, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
