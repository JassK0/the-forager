"""Plant biomass on every walkable cell.

Each cell grows logistically towards a carrying capacity set by soil fertility
(Verhulst 1838): dB/dt = g(t) r B (1 - B/K), where g(t) is the seasonal growth
multiplier. Bare cells are recolonised by seed rain from their four
neighbours, limited by the same free space, so grazed-out ground recovers
from its edges inward.
"""

import base64

import numpy as np


class Vegetation:
    def __init__(self, terrain, cfg, rng: np.random.Generator):
        self.capacity = cfg.plant_capacity
        self.r = cfg.plant_growth
        self.seeding = cfg.plant_seeding
        self.B = np.zeros((terrain.height, terrain.width))
        self.refresh_capacity(terrain)
        self.B[:] = self.K * rng.uniform(0.4, 1.0, self.K.shape)
        self._nb = np.empty_like(self.B)

    def refresh_capacity(self, terrain) -> None:
        """Recompute carrying capacity after the terrain changes."""
        self.K = self.capacity * terrain.fertility
        self._inv_k = np.divide(1.0, self.K, out=np.zeros_like(self.K), where=self.K > 0)
        np.minimum(self.B, self.K, out=self.B)

    def step(self, growth: float = 1.0) -> None:
        B, nb = self.B, self._nb
        nb.fill(0.0)
        nb[1:] += B[:-1]
        nb[:-1] += B[1:]
        nb[:, 1:] += B[:, :-1]
        nb[:, :-1] += B[:, 1:]
        B += growth * (self.r * B + 0.25 * self.seeding * nb) * (1.0 - B * self._inv_k)
        np.minimum(B, self.K, out=B)  # also keeps non-growing cells (K = 0) bare
        np.maximum(B, 0.0, out=B)

    def fertilise(self, cx: float, cy: float, radius: float) -> None:
        """Sandbox tool: bring the plants around (cx, cy) up to carrying capacity."""
        h, w = self.B.shape
        yy, xx = np.ogrid[:h, :w]
        inside = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= radius * radius
        self.B[inside] = self.K[inside]

    def total(self) -> float:
        return float(self.B.sum())

    def encode(self) -> str:
        """Biomass as base64 bytes, 0..255 of the maximum carrying capacity."""
        q = np.clip(self.B * (255.0 / self.capacity), 0, 255).astype(np.uint8)
        return base64.b64encode(q.tobytes()).decode("ascii")
