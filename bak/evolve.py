"""Evolution: 10 bugs, random params, natural selection.
Each round: run all bugs, keep the best, breed next generation."""
import numpy as np
from embryo.world import World
from embryo.brain import Brain


class Bug:
    """A bug with its own genetic params."""
    def __init__(self, params=None):
        if params is None:
            # Random genome
            self.params = {
                'smell_weight': np.random.uniform(0.1, 1.0),      # how strong smell is
                'explore_noise': np.random.uniform(0.2, 1.0),      # how exploratory
                'momentum': np.random.uniform(0.0, 0.8),           # how much it keeps going straight
                'memory_decay': np.random.uniform(0.999, 0.9999),  # how fast memory fades
                'lr': np.random.uniform(0.05, 0.3),                # hebbian learning rate
            }
        else:
            self.params = params.copy()

    def mutate(self, rate=0.2):
        """Small random mutation for breeding."""
        p = self.params.copy()
        for k in p:
            if np.random.random() < rate:
                p[k] *= np.random.uniform(0.8, 1.2)
        return Bug(p)


def run_bug(params, steps=2000, seed=0):
    """Run one bug in its own world, return score."""
    np.random.seed(seed)
    # Very hard world: 1 food, big map, many walls — real survival pressure
    w = World(size=30, n_food=1, n_walls=20, n_cue=6, seed=seed)
    brain = Brain(n_sensors=25, n_actions=8, lr=params['lr'], world_size=30, seed=seed)
    brain.momentum_strength = params['momentum']
    brain.explore_noise = params['explore_noise']
    brain.smell_weight = params['smell_weight']

    foods = 0
    bumped_last = False
    for i in range(steps):
        s = w.sense()
        innate = w.innate_tropism(s)
        hunger = 1.0 - w.energy
        action = brain.decide(s, innate, hunger, w.pos)
        if bumped_last:
            action = (action + np.random.randint(1, 8)) % 8
        ate = w.step(action)
        bumped_last = w.last_bumped
        brain.decay_memory()
        if ate:
            brain.reinforce(w.pos)
            foods += 1
        if w.energy <= 0:
            break  # starved

    # Score: food eaten + steps survived
    score = foods * 10 + i / steps * 2  # food is much more valuable now
    return score, foods, i


def main():
    np.random.seed(42)
    population = [Bug() for _ in range(10)]
    generations = 10

    for gen in range(generations):
        print(f"\n=== Generation {gen+1} ===")
        scores = []
        for i, bug in enumerate(population):
            score, foods, steps = run_bug(bug.params, steps=2000, seed=i)
            scores.append((score, i, foods, steps))
            print(f"  Bug {i}: score={score:.2f}, foods={foods}, survived={steps}")

        # Sort by score, keep top 3
        scores.sort(reverse=True)
        best_params = population[scores[0][1]].params
        print(f"\n  Best bug: #{scores[0][1]}, score={scores[0][0]:.2f}")
        print(f"  Best params: {best_params}")

        # Breed: top 3, each makes ~3 children with mutation
        next_gen = []
        top_bugs = [population[s[1]] for s in scores[:3]]
        for bug in top_bugs:
            next_gen.append(bug)  # keep original
            for _ in range(3):
                next_gen.append(bug.mutate())
        population = next_gen[:10]


if __name__ == "__main__":
    main()
