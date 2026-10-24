"""Climate: what each part of the map is like to live in.

Every cell has a local temperature offset (added to the seasonal temperature)
and a dryness. Together they set, cell by cell:

* snow cover: where it is cold enough, and not too dry for snow to fall;
* how much plants can grow (capacity) and when: frost stops growth in cold
  places, summer heat makes desert plants go dormant;
* how much water animals get from what they eat;
* the cold and heat each animal has to pay for (thermal_cost in animal.py);
* the ground's colour, and so how well a coat is camouflaged against it.

A climate preset sets the whole map (Arctic tundra, desert, mountains...), or
"mixed" lays a north-south gradient across it, from a cold, snowy north to a
hot, dry south, so several regions sit side by side and animals can adapt to
each. Mountains cool with altitude (a lapse rate), giving tree lines and snowy
tops. Biome names follow Whittaker's classification by temperature and
moisture (Whittaker 1975).

The temperature scale is the simulation's own: the temperate seasons run from
-1 (midwinter) to +1 (midsummer), and snow lies below about -0.35.
"""

import base64

import numpy as np

from .terrain import FOREST, GRASS, ROCK, SAND, TREE, WALKABLE, _components, fractal_noise

SNOW_FROM = 0.35  # snow lies where the local temperature is below -SNOW_FROM...
SNOW_RANGE = 0.4  # ...and is complete this much colder
TEMP_RANGE = (-2.0, 1.8)

# label, icon; base temperature offset; north-south temperature gradient (top
# of the map minus bottom is -gradient); cooling with altitude; dryness and its
# gradient; terrain the preset suggests; starting numbers relative to the
# temperate meadow.
CLIMATES = {
    "temperate": dict(label="Temperate meadow", icon="🌳", temp=0.0, gradient=0.0, lapse=0.0, dry=0.0, dry_gradient=0.0,
                      terrain=dict(water=12, forest=12, rock=5), animals=1.0,
                      text="mild seasons: snowy winters, green summers"),
    "arctic": dict(label="Arctic tundra", icon="❄️", temp=-0.5, gradient=0.0, lapse=0.3, dry=0.1, dry_gradient=0.0,
                   terrain=dict(water=14, forest=4, rock=8), animals=0.6,
                   text="long snowy winters, a short summer for plants to grow, few trees"),
    "desert": dict(label="Desert", icon="🏜️", temp=0.45, gradient=0.0, lapse=0.3, dry=0.85, dry_gradient=0.0,
                   terrain=dict(water=4, forest=2, rock=6), animals=0.45,
                   text="scorching summers, cool winters with little snow, sparse plants and few waterholes"),
    "mountains": dict(label="Mountains", icon="⛰️", temp=0.0, gradient=0.0, lapse=1.4, dry=0.1, dry_gradient=0.0,
                      terrain=dict(water=8, forest=14, rock=24), animals=0.75,
                      text="green valleys under cold, snowy heights; it gets colder the higher you go"),
    "grassland": dict(label="Grassland plains", icon="🌾", temp=0.15, gradient=0.0, lapse=0.0, dry=0.38, dry_gradient=0.0,
                      terrain=dict(water=6, forest=2, rock=1), animals=1.1,
                      text="open plains with warm summers and little cover"),
    "mixed": dict(label="Mixed: cold north, hot south", icon="🌍", temp=0.0, gradient=1.7, lapse=0.6, dry=0.35,
                  dry_gradient=1.0, terrain=dict(water=10, forest=10, rock=7), animals=0.85,
                  text="tundra in the north, temperate land in the middle, desert in the south, cold mountain tops"),
}
BIOMES = ("tundra", "boreal", "temperate", "grassland", "desert", "alpine")


def _smooth(a: float, b: float, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Climate:
    def __init__(self, name: str, terrain, seed: int):
        self.name = name if name in CLIMATES else "temperate"
        spec = CLIMATES[self.name]
        self.spec = spec
        h, w = terrain.types.shape
        self.width, self.height = w, h
        # Its own random stream, so a temperate map is exactly what it always was.
        rng = np.random.default_rng(seed + 7919)
        north = np.linspace(-0.5, 0.5, h)[:, None] * np.ones((1, w))  # -0.5 at the top (north)
        wobble = fractal_noise(rng, w, h, 40.0) - 0.5
        rel = np.clip((terrain.elevation - terrain.level) / max(1e-6, 1.0 - terrain.level), 0.0, 1.0)
        offset = spec["temp"] + spec["gradient"] * north - spec["lapse"] * np.clip((rel - 0.35) / 0.65, 0.0, 1.0)
        dry = spec["dry"] + spec["dry_gradient"] * north
        if spec["gradient"] or spec["dry_gradient"]:
            offset = offset + 0.25 * wobble * (spec["gradient"] > 0)
            dry = dry + 0.3 * (fractal_noise(rng, w, h, 32.0) - 0.5) * (spec["dry_gradient"] > 0)
        self.uniform = self.name == "temperate"
        self.offset = np.clip(offset, -1.2, 0.9) if not self.uniform else np.zeros((h, w))
        self.dry = np.clip(dry, 0.0, 1.0) if not self.uniform else np.zeros((h, w))
        cold = np.clip(1.0 + 0.5 * np.minimum(self.offset, 0.0), 0.4, 1.0)
        # Plant capacity: less in the cold and in dry country (where sand also
        # leaves only patches of scrub to graze).
        self.productivity = (1.0 - 0.45 * self.dry) * cold
        self.moisture = 1.0 - 0.45 * self.dry  # water in forage, relative to temperate (desert plants hold some)
        self.snow_fall = 1.0 - 0.8 * self.dry  # deserts are too dry for much snow
        # Flat lists for fast lookups from per-animal code.
        self._off = self.offset.ravel().tolist()
        self._moist = self.moisture.ravel().tolist()
        self._dry = self.dry.ravel().tolist()
        self._snowfall = self.snow_fall.ravel().tolist()
        self._growth = None
        self._growth_tick = -10 ** 9

    # ------------------------------------------------------------------ shaping the land

    def shape_terrain(self, terrain, rng: np.random.Generator) -> None:
        """Deserts turn dry grassland to sand and scrub; trees stop at the tree line."""
        if self.uniform:
            return
        t = terrain.types
        noise = rng.random(t.shape)
        # Desert plants grow in patches: in moist hollows and dry washes, and as
        # oases round water. The rest of dry country is bare sand.
        shore = np.array(terrain.shore_dist).reshape(t.shape)
        oasis = (shore >= 0) & (shore <= 3)
        aridity = self.dry + 0.6 * (0.5 - terrain.moisture) + 0.2 * (noise - 0.5)
        t[(t == GRASS) & (aridity > 0.8) & ~oasis] = SAND
        t[((t == FOREST) | (t == TREE)) & (self.dry > 0.55)] = SAND
        cold = (t == FOREST) | (t == TREE)
        t[cold & (self.offset < -0.55) & (noise > 0.08)] = GRASS  # above the tree line: a few stunted trees only
        t[(t == FOREST) & (self.offset < -0.4) & (noise < 0.15)] = TREE  # conifer taiga thickens
        labels = _components(WALKABLE[t])
        if labels.max() > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            t[(labels > 0) & (labels != sizes.argmax())] = TREE
        terrain.version += 1
        terrain.recompute()

    # ------------------------------------------------------------------ local conditions

    def cell(self, x: float, y: float) -> int:
        cx = min(self.width - 1, max(0, int(x)))
        cy = min(self.height - 1, max(0, int(y)))
        return cy * self.width + cx

    def temperature(self, season_temp: float, x: float, y: float) -> float:
        t = season_temp + self._off[self.cell(x, y)]
        return min(TEMP_RANGE[1], max(TEMP_RANGE[0], t))

    def snow_at(self, season_temp: float, x: float, y: float) -> float:
        k = self.cell(x, y)
        t = season_temp + self._off[k]
        return min(1.0, max(0.0, (-t - SNOW_FROM) / SNOW_RANGE)) * self._snowfall[k]

    def dry_at(self, x: float, y: float) -> float:
        return self._dry[self.cell(x, y)]

    def moisture_at(self, x: float, y: float) -> float:
        return self._moist[self.cell(x, y)]

    def growth(self, season_growth: float, season_temp: float, tick: int):
        """Plant growth rate per cell: the season's, stopped by frost in cold places
        and slowed by summer heat in hot ones. A scalar on a temperate map."""
        if self.uniform:
            return season_growth
        if tick - self._growth_tick >= 20 or self._growth is None:
            local = np.clip(season_temp + self.offset, *TEMP_RANGE)
            frost = np.clip((local + 1.5) / 0.5, 0.0, 1.0)
            dormancy = np.clip((1.8 - local) / 0.7, 0.2, 1.0)
            self._growth = season_growth * frost * dormancy
            self._growth_tick = tick
        return self._growth

    def snow_share(self, season_temp: float, land: np.ndarray) -> float:
        """Share of the land under snow now."""
        snow = np.clip((-(season_temp + self.offset) - SNOW_FROM) / SNOW_RANGE, 0.0, 1.0) * self.snow_fall
        return float(snow[land].mean()) if land.any() else 0.0

    # ------------------------------------------------------------------ biomes

    def biomes(self, terrain) -> np.ndarray:
        """Biome index per cell (see BIOMES), from local temperature, dryness and height."""
        rel = np.clip((terrain.elevation - terrain.level) / max(1e-6, 1.0 - terrain.level), 0.0, 1.0)
        out = np.full(self.offset.shape, BIOMES.index("temperate"), dtype=np.int8)
        out[self.dry >= 0.35] = BIOMES.index("grassland")
        out[self.dry >= 0.6] = BIOMES.index("desert")
        out[self.offset <= -0.3] = BIOMES.index("boreal")
        out[self.offset <= -0.55] = BIOMES.index("tundra")
        out[(self.offset <= -0.45) & (rel > 0.6)] = BIOMES.index("alpine")
        return out

    def payload(self) -> dict:
        """For the browser: offset and dryness as bytes, to draw snow and biomes."""
        off = np.clip((self.offset + 1.2) / 2.1 * 255, 0, 255).astype(np.uint8)
        dry = np.clip(self.dry * 255, 0, 255).astype(np.uint8)
        return {"name": self.name, "label": self.spec["label"], "icon": self.spec["icon"],
                "offset": base64.b64encode(off.tobytes()).decode("ascii"),
                "dry": base64.b64encode(dry.tobytes()).decode("ascii")}
