"""Procedural terrain: elevation, biomes, soil fertility and a water flow field.

Cells are one of six types. Water, trees and rock block movement; sand, grass
and forest floor can be walked on. The terrain can be repainted at runtime
(sandbox tools), after which everything derived from the cell types is
recomputed.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

WATER, SAND, GRASS, FOREST, TREE, ROCK = range(6)
TYPE_NAMES = ("water", "sand", "grass", "forest", "tree", "rock")
WALKABLE = np.array([False, True, True, True, False, False])
FERTILITY_FACTOR = np.array([0.0, 0.35, 1.0, 0.6, 0.0, 0.0])
N8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def fractal_noise(rng: np.random.Generator, w: int, h: int, scale: float, octaves: int = 5) -> np.ndarray:
    """Sum of value-noise octaves, each half the size and amplitude of the last."""
    total = np.zeros((h, w))
    amp, norm, cell = 1.0, 0.0, float(scale)
    for _ in range(octaves):
        if cell < 1.5:
            break
        lattice = rng.random((int(h / cell) + 3, int(w / cell) + 3))
        ys = np.arange(h) / cell + rng.random()
        xs = np.arange(w) / cell + rng.random()
        y0, x0 = ys.astype(int), xs.astype(int)
        ty, tx = _fade(ys - y0)[:, None], _fade(xs - x0)[None, :]
        a = lattice[np.ix_(y0, x0)]
        b = lattice[np.ix_(y0, x0 + 1)]
        c = lattice[np.ix_(y0 + 1, x0)]
        d = lattice[np.ix_(y0 + 1, x0 + 1)]
        top = a + (b - a) * tx
        bottom = c + (d - c) * tx
        total += amp * (top + (bottom - top) * ty)
        norm += amp
        amp *= 0.5
        cell /= 2
    out = total / norm
    return (out - out.min()) / (np.ptp(out) + 1e-12)


def _components(walk: np.ndarray) -> np.ndarray:
    """Label 4-connected components of walkable cells (0 = not walkable)."""
    h, w = walk.shape
    open_ = walk.ravel().tolist()
    label = [0] * (w * h)
    n = 0
    for start in range(w * h):
        if not open_[start] or label[start]:
            continue
        n += 1
        label[start] = n
        q = deque([start])
        while q:
            i = q.popleft()
            y, x = divmod(i, w)
            for j in (i - w if y else -1, i + w if y < h - 1 else -1, i - 1 if x else -1, i + 1 if x < w - 1 else -1):
                if j >= 0 and open_[j] and not label[j]:
                    label[j] = n
                    q.append(j)
    return np.array(label).reshape(h, w)


def _bfs(walk: np.ndarray, sources: np.ndarray) -> list[int]:
    """Grid distance over walkable cells from the sources, without cutting corners."""
    h, w = walk.shape
    open_ = walk.ravel().tolist()
    dist = [-1] * (w * h)
    q = deque()
    for i in np.flatnonzero(sources.ravel()).tolist():
        dist[i] = 0
        q.append(i)
    while q:
        i = q.popleft()
        y, x = divmod(i, w)
        d = dist[i] + 1
        for dx, dy in N8:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            j = ny * w + nx
            if not open_[j] or dist[j] >= 0:
                continue
            if dx and dy and not (open_[y * w + nx] and open_[ny * w + x]):
                continue
            dist[j] = d
            q.append(j)
    return dist


@dataclass
class Terrain:
    width: int
    height: int
    elevation: np.ndarray  # 0..1
    types: np.ndarray  # uint8 cell types
    moisture: np.ndarray  # 0..1 broad soil moisture
    level: float  # elevation of the original waterline
    rock_level: float
    version: int = 0

    def __post_init__(self):
        self.recompute()

    # ------------------------------------------------------------------ derived layers

    def recompute(self) -> None:
        t = self.types
        self.water = t == WATER
        walk = WALKABLE[t]
        wet = np.zeros_like(self.water)
        wet[1:] |= self.water[:-1]
        wet[:-1] |= self.water[1:]
        wet[:, 1:] |= self.water[:, :-1]
        wet[:, :-1] |= self.water[:, 1:]
        shore = walk & wet
        self.shore_dist = _bfs(walk, shore)
        self.walk = bytearray(walk.ravel().astype(np.uint8).tobytes())
        self.shore = bytearray(shore.ravel().astype(np.uint8).tobytes())

        near = np.exp(-np.maximum(np.array(self.shore_dist).reshape(t.shape), 0) / 10.0)
        fert = np.clip(0.15 + 0.55 * self.moisture + 0.4 * near, 0.0, 1.0)
        self.fertility = fert * FERTILITY_FACTOR[t]

        # Render heights: terraced land, mountains on rock, a shallow bed under water.
        rel = np.clip((self.elevation - self.level) / max(1e-6, 1.0 - self.level), 0.0, 1.0)
        land = 1.0 + 0.2 * np.floor(rel * 12)
        rock_rel = np.clip((self.elevation - self.rock_level) / max(1e-6, 1.0 - self.rock_level), 0.0, 1.0)
        rock = land + 0.8 + 0.4 * np.floor(rock_rel * 8)
        self.heights = np.where(t == WATER, 0.45, np.where(t == ROCK, rock, land))

    def is_shore(self, cx: int, cy: int) -> bool:
        return bool(self.shore[cy * self.width + cx])

    def water_heading(self, x: float, y: float) -> float | None:
        """Heading down the distance-to-water gradient from (x, y).

        Animals are assumed to know where water lies in their landscape (a
        cognitive map), so a thirsty animal follows the shortest land path to
        the nearest shore instead of searching blindly.
        """
        w, h = self.width, self.height
        cx, cy = int(x), int(y)
        dist, walk = self.shore_dist, self.walk
        best = dist[cy * w + cx]
        if best < 0:
            return None
        target = None
        for dx, dy in N8:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if dx and dy and not (walk[cy * w + nx] and walk[ny * w + cx]):
                continue
            d = dist[ny * w + nx]
            if 0 <= d < best:
                best, target = d, (nx, ny)
        if target is None:
            return None
        return math.atan2(target[1] + 0.5 - y, target[0] + 0.5 - x)

    # ------------------------------------------------------------------ editing

    def paint(self, cx: float, cy: float, radius: float, kind: int, rng: np.random.Generator) -> int:
        """Set the cells within ``radius`` of (cx, cy) to ``kind``. Returns cells changed."""
        h, w = self.types.shape
        x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
        y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
        if x0 >= x1 or y0 >= y1:
            return 0
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= radius * radius
        if kind == TREE:
            inside &= rng.random(inside.shape) < 0.35  # scatter trees instead of a solid block
        region = self.types[y0:y1, x0:x1]
        changed = int(np.count_nonzero(inside & (region != kind)))
        region[inside] = kind
        if kind in (GRASS, SAND, FOREST, TREE):
            # Raised ground: lift painted land above the waterline if needed.
            e = self.elevation[y0:y1, x0:x1]
            e[inside] = np.maximum(e[inside], self.level + 0.02)
        if changed:
            self.version += 1
            self.recompute()
        return changed


def generate_terrain(rng: np.random.Generator, cfg) -> Terrain:
    w, h = cfg.width, cfg.height
    elevation = fractal_noise(rng, w, h, cfg.terrain_scale)
    moisture = fractal_noise(rng, w, h, cfg.terrain_scale * 0.7)
    woods = fractal_noise(rng, w, h, cfg.terrain_scale * 0.6)

    level = float(np.quantile(elevation, min(0.6, max(0.03, cfg.water_fraction))))
    rock_level = float(np.quantile(elevation, 1.0 - min(0.4, max(0.0, cfg.rock_fraction)))) if cfg.rock_fraction > 0 else 2.0
    types = np.full((h, w), GRASS, dtype=np.uint8)
    types[elevation < level] = WATER
    types[elevation >= rock_level] = ROCK
    if cfg.forest_fraction > 0:
        forest = (woods > np.quantile(woods, 1.0 - cfg.forest_fraction)) & (types == GRASS)
        types[forest] = FOREST
        types[forest & (rng.random((h, w)) < 0.25)] = TREE
    types[(types == GRASS) & (rng.random((h, w)) < 0.006)] = TREE

    water = types == WATER
    wet = np.zeros_like(water)
    wet[1:] |= water[:-1]
    wet[:-1] |= water[1:]
    wet[:, 1:] |= water[:, :-1]
    wet[:, :-1] |= water[:, 1:]
    types[wet & (types == GRASS) & (elevation < level + 0.035)] = SAND

    # Every walkable cell must be reachable from every other: pockets and
    # islands cut off from the main area become thickets.
    labels = _components(WALKABLE[types])
    if labels.max() > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        types[(labels > 0) & (labels != sizes.argmax())] = TREE

    return Terrain(width=w, height=h, elevation=elevation, types=types, moisture=moisture,
                   level=level, rock_level=rock_level)
