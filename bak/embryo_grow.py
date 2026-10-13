"""Embryo: growing neural network.
Not designed — grown from a single cell with simple rules.
Like embryonic development: one cell -> division -> differentiation -> complex brain.
"""
import numpy as np
import matplotlib.pyplot as plt


class GrowingNeuron:
    """A single neuron that can divide and connect."""
    def __init__(self, pos, neuron_type='intermediate'):
        self.pos = np.array(pos, dtype=float)
        self.neuron_type = neuron_type  # sensory, intermediate, motor
        self.inputs = []  # list of (source_neuron, weight)
        self.output = 0.0

    def forward(self, input_signal):
        """Process input."""
        self.output = np.tanh(input_signal)
        return self.output


class GrowingBrain:
    """A brain that grows from a single cell.
    Simple rules:
    1. Start with one neuron in the center
    2. Neurons can divide (create a child nearby)
    3. Neurons connect to nearby neurons
    4. Edge neurons become sensory or motor, center stays intermediate
    """
    def __init__(self, size=10, seed=0):
        np.random.seed(seed)
        self.size = size
        self.neurons = []
        self.connections = []  # list of (from_idx, to_idx, weight)
        
        # Start with one neuron in the center
        center = size // 2
        self.neurons.append(GrowingNeuron((center, center), 'intermediate'))
        
        # Grow for N steps
        for step in range(50):
            self.grow_step()

    def grow_step(self):
        """One step of growth."""
        # 1. Pick a random neuron to divide
        if len(self.neurons) >= 50:
            return
        
        parent_idx = np.random.randint(len(self.neurons))
        parent = self.neurons[parent_idx]
        
        # 2. Create child — random direction, not always outward
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(1.0, 2.5)
        child_pos = parent.pos + np.array([np.cos(angle), np.sin(angle)]) * distance
        child_pos = np.clip(child_pos, 0, self.size)
        
        # 3. Determine neuron type based on position
        # Left edge = sensory, right edge = motor, center = intermediate
        x, y = child_pos
        if x < self.size * 0.2:
            neuron_type = 'sensory'
        elif x > self.size * 0.8:
            neuron_type = 'motor'
        else:
            neuron_type = 'intermediate'
        
        child = GrowingNeuron(child_pos, neuron_type)
        self.neurons.append(child)
        
        # 4. Connect parent to child
        self.connections.append((parent_idx, len(self.neurons)-1, 1.0))
        
        # 5. Connect child to nearby neurons
        for i, other in enumerate(self.neurons[:-1]):
            dist = np.linalg.norm(child.pos - other.pos)
            if dist < 2.0 and np.random.random() < 0.3:
                weight = np.random.randn() * 0.3
                self.connections.append((i, len(self.neurons)-1, weight))

    def visualize(self):
        """Visualize the growing brain."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw connections
        for from_idx, to_idx, weight in self.connections:
            from_pos = self.neurons[from_idx].pos
            to_pos = self.neurons[to_idx].pos
            color = 'blue' if weight > 0 else 'red'
            ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                    color=color, alpha=0.3, linewidth=abs(weight))
        
        # Draw neurons
        sensory_x = [n.pos[0] for n in self.neurons if n.neuron_type == 'sensory']
        sensory_y = [n.pos[1] for n in self.neurons if n.neuron_type == 'sensory']
        intermediate_x = [n.pos[0] for n in self.neurons if n.neuron_type == 'intermediate']
        intermediate_y = [n.pos[1] for n in self.neurons if n.neuron_type == 'intermediate']
        motor_x = [n.pos[0] for n in self.neurons if n.neuron_type == 'motor']
        motor_y = [n.pos[1] for n in self.neurons if n.neuron_type == 'motor']
        
        ax.scatter(sensory_x, sensory_y, c='green', s=100, label='Sensory')
        ax.scatter(intermediate_x, intermediate_y, c='gray', s=100, label='Intermediate')
        ax.scatter(motor_x, motor_y, c='red', s=100, label='Motor')
        
        ax.set_xlim(-1, self.size+1)
        ax.set_ylim(-1, self.size+1)
        ax.legend()
        ax.set_title(f'Growing Brain: {len(self.neurons)} neurons, {len(self.connections)} connections')
        plt.show()


if __name__ == "__main__":
    brain = GrowingBrain(size=10, seed=42)
    brain.visualize()
    print(f"Neurons: {len(brain.neurons)}")
    print(f"Connections: {len(brain.connections)}")
    sensory = sum(1 for n in brain.neurons if n.neuron_type == 'sensory')
    intermediate = sum(1 for n in brain.neurons if n.neuron_type == 'intermediate')
    motor = sum(1 for n in brain.neurons if n.neuron_type == 'motor')
    print(f"Sensory: {sensory}, Intermediate: {intermediate}, Motor: {motor}")
