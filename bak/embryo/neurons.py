"""L2: Mushroom body (Kenyon cells, sparse encoding).
L3: Actor-critic with eligibility traces (dopamine-modulated, biologically plausible).

Learning rule (TD(λ)):
  δ = r + γ V(s') − V(s)              # dopamine = reward prediction error
  e[action, kc] *= γλ; e[chosen, active_kc] += 1   # short-term credit trace
  ΔW[action, kc] = lr · δ · e[action, kc]
  ΔV[kc]         = lr_v · δ · e_v[kc]

This is how a fly's mushroom body assigns credit across the few seconds
leading up to a reward — no backprop, only local traces + global DA.
"""
import numpy as np


class KenyonLayer:
    """~2000 Kenyon cells. Random fixed connections. Winner-take-all sparse firing."""

    def __init__(self, n_input=26, n_kc=2000, sparse_ratio=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_kc = n_kc
        self.n_active = max(1, int(n_kc * sparse_ratio))
        self.W = np.random.randn(n_kc, n_input).astype(np.float32) / np.sqrt(n_input)

    def encode(self, sensory):
        acts = self.W @ sensory
        return np.argpartition(acts, -self.n_active)[-self.n_active:]


class ActorCritic:
    """KC -> 4 actions (actor) + KC -> value (critic).
    Plastic weights modulated by TD error (dopamine)."""

    def __init__(self, n_kc=2000, n_actions=4,
                 lr_actor=0.05, lr_critic=0.05,
                 gamma=0.95, lam=0.7, exploration=0.3, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.exploration = exploration

        # Actor: action preference (n_actions, n_kc)
        self.W = (np.random.randn(n_actions, n_kc).astype(np.float32) * 0.01)
        # Critic: value weights (n_kc,)
        self.V = np.zeros(n_kc, dtype=np.float32)
        # Eligibility traces (replacing, not accumulating)
        self.e = np.zeros((n_actions, n_kc), dtype=np.float32)
        self.e_v = np.zeros(n_kc, dtype=np.float32)

    def value(self, active_kc):
        """V(s) = sum of critic weights for active KC cells."""
        return self.V[active_kc].sum()

    def act(self, active_kc):
        logits = self.W[:, active_kc].sum(axis=1)
        if np.random.random() < self.exploration:
            return np.random.randint(self.n_actions)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return np.random.choice(self.n_actions, p=p)

    def learn(self, active_kc, chosen_action, reward, next_value):
        """One TD(λ) update with replacing traces."""
        v = self.value(active_kc)
        delta = reward + self.gamma * next_value - v
        delta = np.clip(delta, -1.0, 1.0)  # dopamine bounded

        # Decay traces
        self.e *= self.gamma * self.lam
        self.e_v *= self.gamma * self.lam
        # Replacing traces: set to 1 (not += 1) for active cells
        self.e[chosen_action, active_kc] = 1.0
        self.e_v[active_kc] = 1.0

        # Update weights by TD error
        self.W += self.lr_actor * delta * self.e
        self.V += self.lr_critic * delta * self.e_v

        # Keep bounded
        np.clip(self.W, -1.0, 1.0, out=self.W)
        np.clip(self.V, -1.0, 1.0, out=self.V)
        return delta
