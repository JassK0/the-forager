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
    def __init__(self, terrain, cfg, rng: np.random.Generator, productivity=1.0):
        self.capacity = cfg.plant_capacity
        self.productivity = productivity  # climate: drier and colder land carries less
        self.r = cfg.plant_growth
        self.seeding = cfg.plant_seeding
        self.B = np.zeros((terrain.height, terrain.width))
        self.refresh_capacity(terrain)
        self.B[:] = self.K * rng.uniform(0.4, 1.0, self.K.shape)
        self._nb = np.empty_like(self.B)

    def refresh_capacity(self, terrain) -> None:
        """Recompute carrying capacity after the terrain changes."""
        self.K = self.capacity * terrain.fertility * self.productivity
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


# Wild fruit: brambles, wild apples and plums along woodland edges. Red foxes
# eat a lot of fruit in late summer and autumn (e.g. about a quarter of their
# autumn diet in some studies), which carries them through lean spells.
BUSH_SHARE = 0.12  # share of woodland-edge cells with a fruiting bush
FRUIT_CAP = 40.0  # ripe fruit per bush at its best, in energy units
FRUIT_RATE = 0.002  # ripening per tick at the height of the season, as a share of the unripe crop
FRUIT_ROT = 0.995  # per tick out of season, unpicked fruit falls and rots
FRUIT_PEAK, FRUIT_WIDTH = 0.55, 0.18  # season of ripe fruit, as a fraction of the year (0 = start of spring)


class Fruit:
    """Fruit on bushes at woodland edges, ripening from late summer into autumn."""

    def __init__(self, terrain):
        self.F = np.zeros(terrain.types.shape)
        self.refresh(terrain)

    def refresh(self, terrain) -> None:
        """Bushes grow on open ground next to trees (recomputed when the terrain is painted)."""
        from .terrain import FOREST, GRASS, TREE

        t = terrain.types
        tree = t == TREE
        near = np.zeros_like(tree)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    near |= np.roll(np.roll(tree, dy, axis=0), dx, axis=1)
        # A fixed hash of the cell decides which edges have a bush, so painting
        # never consumes random numbers and runs stay reproducible.
        idx = np.arange(t.size, dtype=np.uint64).reshape(t.shape)
        h = ((idx * np.uint64(2654435761)) % np.uint64(4294967296)).astype(np.float64) / 4294967296.0
        self.bush = near & ((t == GRASS) | (t == FOREST)) & (h < BUSH_SHARE)
        self.F[~self.bush] = 0.0

    @staticmethod
    def ripeness(year_fraction: float, seasons: bool) -> float:
        if not seasons:
            return 0.5
        return max(0.0, 1.0 - ((year_fraction - FRUIT_PEAK) / FRUIT_WIDTH) ** 2)

    def step(self, ripeness: float) -> None:
        if ripeness > 0.0:
            self.F += (FRUIT_RATE * ripeness) * (FRUIT_CAP - self.F) * self.bush
        else:
            self.F *= FRUIT_ROT

    def total(self) -> float:
        return float(self.F.sum())

    def encode(self) -> str:
        """Ripe fruit per cell as base64 bytes, 0..255 of a full bush."""
        q = np.clip(self.F * (255.0 / FRUIT_CAP), 0, 255).astype(np.uint8)
        return base64.b64encode(q.tobytes()).decode("ascii")

    def encode_bushes(self) -> str:
        return base64.b64encode(self.bush.astype(np.uint8).tobytes()).decode("ascii")
