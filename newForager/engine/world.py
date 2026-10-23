"""The world: terrain, vegetation, animals, the clock and the simulation loop."""

import base64
import bisect
import math
import random
from collections import Counter

import numpy as np

from . import brain
from .animal import ALERT_TICKS, BRAIN_LOG_EVERY, DANGER_MEMORY_DAYS, FIGHT_LOST, FIGHT_WON, FOOD_MEMORY_DAYS, \
    PAIN_CHASED, REGION, REWARD_MATING, THUMP_RADIUS, Act, Animal, Cache, Carrion
from .coats import PARTS, PRESETS
from .config import SimConfig, settings_of
from .genetics import GenomeSpec
from .narrator import OBSERVE, Narrator
from .navigation import find_path
from .terrain import GRASS, ROCK, SAND, TREE, TYPE_NAMES, WATER, generate_terrain
from .vegetation import Vegetation

TAU = 2 * math.pi
BUCKET_SHIFT = 3  # spatial hash buckets are 8 x 8 cells
HISTORY_LIMIT = 20000  # samples kept before the history is thinned
GRAVEYARD_SIZE = 500  # recently dead animals remembered for the inspector
EVENT_LIMIT = 400
PREY_SPOT = 0.8  # chance per look that a predator notices a visible, nearby prey animal
FROZEN_SPOT = 0.6  # motionless prey are harder to notice
CACHE_DAYS = 4  # buried food lasts this long
CACHE_DECAY = 0.998  # per tick: cached meat slowly spoils
CAUSES = ("predation", "starvation", "dehydration", "old age", "disease", "fighting", "emigrated", "removed")
CONTEST_CHANCE = 0.5  # chance a floater stands its ground against a resident instead of fleeing
OWNER_ADVANTAGE = 1.3  # residents win more contests than their size alone predicts
FIGHT_COST = 0.05  # reserves each contestant burns in a fight
FIGHT_DEATH = (0.005, 0.02)  # chance the loser is fatally injured, outside and during the mating season
TERRITORY_SCALE = (0.6, 1.6)  # how far food can shrink or stretch the spacing between territories
SEASONS = ("spring", "summer", "autumn", "winter")
PAINTABLE = {"water": WATER, "grass": GRASS, "sand": SAND, "tree": TREE, "rock": ROCK}
EVENTS = {
    "drought": "Plant growth almost stops for two days",
    "bloom": "Every plant regrows to full size",
    "cold": "A cold snap: freezing weather for a day and a half",
    "plague": "A disease outbreak in one species for a day",
}

# Which decision option each action belongs to, so that animals stick with the
# option they are already pursuing (hysteresis) unless another clearly wins.
ACT_OPTION = np.full(len(Act), brain.EXPLORE, dtype=np.int64)
for _act, _option in ((Act.FORAGE, brain.EAT), (Act.EAT, brain.EAT), (Act.HUNT, brain.EAT),
                      (Act.CACHE, brain.EAT), (Act.SEEK_WATER, brain.DRINK), (Act.DRINK, brain.DRINK),
                      (Act.SEEK_MATE, brain.MATE), (Act.HOME, brain.REST), (Act.REST, brain.REST),
                      (Act.FLEE, brain.FLEE), (Act.FREEZE, brain.FREEZE), (Act.ALERT, brain.HIDE),
                      (Act.HIDE, brain.HIDE)):
    ACT_OPTION[_act] = _option

_NAME_START = ("Ash", "Bram", "Clo", "Dew", "Elm", "Fen", "Fig", "Hazel", "Juni", "Kit", "Lum", "Moss",
               "Nib", "Oak", "Pip", "Quill", "Rue", "Sorr", "Tansy", "Umber", "Vale", "Wren", "Yarr", "Zinn",
               "Bil", "Cas", "Dot", "Ember", "Gus", "Holl")
_NAME_END = ("", "a", "ie", "o", "y", "et", "kin", "ly", "wick", "bell", "en", "ton", "ber", "sy", "row")


def animal_name(uid: int) -> str:
    return _NAME_START[uid % len(_NAME_START)] + _NAME_END[(uid // len(_NAME_START)) % len(_NAME_END)]


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode("ascii")


def _num(v, digits: int = 3):
    """A float for JSON, or None for NaN."""
    v = float(v)
    return None if v != v else round(v, digits)


def _smooth(a: float, b: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


class World:
    def __init__(self, cfg: SimConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        self.terrain = generate_terrain(self.nrng, cfg)
        self.veg = Vegetation(self.terrain, cfg, self.nrng)
        self.width, self.height = cfg.width, cfg.height
        self.plant_energy = cfg.plant_energy

        self.species = cfg.species
        self.genomes = [GenomeSpec(s) for s in self.species]
        self.narrator = Narrator(self)
        index = {s.name: i for i, s in enumerate(self.species)}
        self.prey_of = [tuple(index[p] for p in s.prey) for s in self.species]
        self.predators_of = [
            tuple(j for j, other in enumerate(self.species) if s.name in other.prey)
            for s in self.species
        ]

        n = len(self.species)
        self._nbx = (self.width >> BUCKET_SHIFT) + 1
        self._buckets = [{} for _ in range(n)]
        self.animals: list[Animal] = []
        self.by_id: dict[int, Animal] = {}
        self.region_cols = (self.width + REGION - 1) // REGION
        self.brains = self._brain_tables()
        self._born: list[Animal] = []
        self._deaths_pending = False
        self._next_id = 1
        self.tick = 0
        self._holders = [{} for _ in range(n)]
        self._graveyard: dict[int, dict] = {}
        self._events: list = []
        self.carrion: list[Carrion] = []
        self.caches: list[Cache] = []
        self.huddle: Counter = Counter()
        self.effects: list[dict] = []
        self.event_log: list[dict] = []
        self.end_on_extinction = cfg.end_on_extinction
        self.finished: dict | None = None
        self._reset_counters()

        self.in_season = [True] * n
        self.resting = [False] * n
        self.rest_remaining = [0] * n
        self.plague = [0.0] * n
        self._environment()

        self._index_land()
        self._reset_history()
        for si, s in enumerate(self.species):
            for _ in range(s.initial):
                self._add(self._founder(si, self._land, mature=False))
        self._record()

    def _reset_counters(self) -> None:
        n = len(self.species)
        self.births = [0] * n
        self.stillborn = [0] * n
        self.rejections = [0] * n
        self.deaths = [Counter() for _ in range(n)]
        self.hunts = [Counter() for _ in range(n)]
        self.ever = [False] * n
        self.max_generation = [0] * n
        # Hall of fame: (value, id, name) for the oldest, most prolific and deadliest.
        self.records = [{"oldest": None, "parent": None, "hunter": None} for _ in range(n)]
        self.social = [Counter() for _ in range(n)]  # contests, takeovers, inheritances, helpers
        # How often evolved brains still choose what the founders' instincts would,
        # counting only clear-cut decisions (close calls flip with any small variation).
        self.agreement = [[0, 0] for _ in range(n)]
        self.choice_pairs = [Counter() for _ in range(n)]  # (instinct, brain) disagreements since last read
        self.decisions = [0] * n  # decisions made since last read
        self.start_tick = self.tick

    def _brain_tables(self):
        return [brain.BrainTable() for _ in self.species] if self.cfg.brains == "neural" else None

    def _index_land(self) -> None:
        walk = np.frombuffer(bytes(self.terrain.walk), dtype=np.uint8).astype(bool)
        land = np.flatnonzero(walk)
        self._land = land.tolist()
        lx, ly = land % self.width, land // self.width
        edge = land[(lx < 3) | (ly < 3) | (lx >= self.width - 3) | (ly >= self.height - 3)]
        self._edge = edge.tolist() or self._land

    # ------------------------------------------------------------------ clock and weather

    def _environment(self) -> None:
        cfg = self.cfg
        tpd = cfg.ticks_per_day
        t = self.tick + tpd // 4  # the world starts at 06:00 on the first day of spring
        day, tod = divmod(t, tpd)
        hours = 24.0 * tod / tpd
        year_len = tpd * cfg.days_per_year
        f = (t % year_len) / year_len
        active = {e["kind"] for e in self.effects}
        if cfg.seasons:
            temp = math.cos(TAU * (f - 0.35))
            growth = 1.0 + 0.6 * math.cos(TAU * (f - 0.3))
        else:
            temp, growth = 0.2, 1.0
        if "cold" in active:
            temp -= 0.9
        if "drought" in active:
            growth *= 0.1
        day_length = min(16.0, max(8.0, 12.0 + 3.0 * temp))
        sunrise, sunset = 12.0 - day_length / 2, 12.0 + day_length / 2
        self.light = _smooth(sunrise - 0.75, sunrise + 0.75, hours) * (1.0 - _smooth(sunset - 0.75, sunset + 0.75, hours))
        self.cold = max(0.0, -temp)
        self.heat = max(0.0, temp)
        self.snow = min(1.0, max(0.0, (-temp - 0.35) / 0.4))
        self.growth = growth
        # Water content of forage: lush in the growing season, dry in winter and drought.
        self.forage_moisture = min(1.0, max(0.15, 0.2 + 0.55 * growth))
        self.bg_lum = 0.32 + 0.55 * self.snow
        for si, s in enumerate(self.species):
            a, b = s.breeding
            self.in_season[si] = (not cfg.seasons) or (a <= f < b if a < b else (f >= a or f < b))
            rest_from, rest_to = sunrise + s.rest_after_sunrise, sunset - s.rest_before_sunset
            self.resting[si] = rest_from <= hours < rest_to
            self.rest_remaining[si] = int((rest_to - hours) / 24.0 * tpd) + 1 if self.resting[si] else 0
        self.env = {
            "day": int(day) + 1,
            "hours": round(hours, 3),
            "year": int(t // year_len) + 1,
            "year_fraction": round(f, 4),
            "season": SEASONS[int(f * 4) % 4],
            "light": round(self.light, 3),
            "temperature": round(temp, 3),
            "snow": round(self.snow, 3),
            "growth": round(growth, 3),
            "sunrise": round(sunrise, 2),
            "sunset": round(sunset, 2),
        }

    def visibility(self, a: Animal) -> float:
        """How conspicuous an animal is: its coat's contrast with the ground, scaled by daylight."""
        contrast = min(2.0, max(0.6, 0.8 + 1.4 * abs(a.coat_lum - self.bg_lum)))
        return 1.0 + (contrast - 1.0) * self.light

    # ------------------------------------------------------------------ geometry

    def walkable(self, x: float, y: float) -> bool:
        return 0.0 <= x < self.width and 0.0 <= y < self.height and self.terrain.walk[int(y) * self.width + int(x)] == 1

    def clear_path(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True if a straight walk from (x0, y0) to (x1, y1) stays on walkable ground."""
        dx, dy = x1 - x0, y1 - y0
        n = int(max(abs(dx), abs(dy)) * 2) + 1
        walk, w = self.terrain.walk, self.width
        for i in range(1, n + 1):
            t = i / n
            if not walk[int(y0 + dy * t) * w + int(x0 + dx * t)]:
                return False
        return True

    def _random_cell(self, cells) -> tuple[float, float]:
        i = self.rng.choice(cells)
        y, x = divmod(i, self.width)
        return x + self.rng.random(), y + self.rng.random()

    def _near(self, x: float, y: float, r: float) -> tuple[float, float]:
        for _ in range(8):
            nx, ny = x + self.rng.uniform(-r, r), y + self.rng.uniform(-r, r)
            if self.walkable(nx, ny):
                return nx, ny
        return x, y

    def nearest_walkable(self, x: float, y: float, limit: int = 30) -> tuple[float, float] | None:
        cx, cy = int(x), int(y)
        for r in range(limit + 1):
            for dy in range(-r, r + 1):
                for dx in (-r, r) if abs(dy) != r else range(-r, r + 1):
                    nx, ny = cx + dx, cy + dy
                    if self.walkable(nx + 0.5, ny + 0.5):
                        return nx + 0.5, ny + 0.5
        return None

    # ------------------------------------------------------------------ spatial queries

    def _rebuild_buckets(self) -> None:
        nbx = self._nbx
        for b in self._buckets:
            b.clear()
        huddle = self.huddle
        huddle.clear()
        for a in self.animals:
            if a.concealed:
                huddle[(int(a.x), int(a.y))] += 1
            key = (int(a.x) >> BUCKET_SHIFT) + (int(a.y) >> BUCKET_SHIFT) * nbx
            bucket = self._buckets[a.si].get(key)
            if bucket is None:
                self._buckets[a.si][key] = [a]
            else:
                bucket.append(a)

    def query(self, species_idx, x: float, y: float, r: float, exclude=None) -> list:
        """Living animals of the given species within ``r`` of (x, y), as (dist^2, animal)."""
        out = []
        r2 = r * r
        nbx = self._nbx
        bx0, bx1 = max(0, int(x - r)) >> BUCKET_SHIFT, int(min(self.width - 1, x + r)) >> BUCKET_SHIFT
        by0, by1 = max(0, int(y - r)) >> BUCKET_SHIFT, int(min(self.height - 1, y + r)) >> BUCKET_SHIFT
        for si in species_idx:
            buckets = self._buckets[si]
            for by in range(by0, by1 + 1):
                row = by * nbx
                for bx in range(bx0, bx1 + 1):
                    members = buckets.get(row + bx)
                    if not members:
                        continue
                    for a in members:
                        if a.alive and a is not exclude:
                            dx, dy = a.x - x, a.y - y
                            d2 = dx * dx + dy * dy
                            if d2 <= r2:
                                out.append((d2, a))
        return out

    def find_path(self, x0: float, y0: float, x1: float, y1: float, max_cost: float = 80.0):
        """A* route over walkable ground: (waypoints, length), or None."""
        return find_path(self.terrain.walk, self.width, self.height, x0, y0, x1, y1, max_cost=max_cost)

    def vulnerability(self, a: Animal) -> float:
        """How hard a prey animal is to catch, as a multiplier on its distance.

        Predators favour the young (FitzGibbon 1990; Annear et al. 2023),
        animals on their own rather than in a group (the selfish herd), and
        prey that are not watching them.
        """
        f = 1.0
        if not a.adult:
            f *= 0.7
        if not self.query((a.si,), a.x, a.y, 3.0, exclude=a):
            f *= 0.85
        if a.scanning or a.act in (Act.FREEZE, Act.ALERT):
            f *= 1.3
        return f

    def find_prey(self, me: Animal, radius: float | None = None, skip: Animal | None = None):
        """The best prey the predator notices this look, as (prey, cost), or None.

        Noticing is probabilistic: likelier for close, conspicuous prey, much
        less likely if the prey is frozen still, and a little less likely at
        night, when foxes hunt by smell and hearing. Of the prey noticed, the
        one with the lowest cost (route length x vulnerability) is chosen;
        prey only reachable by a long detour round water is passed over.
        """
        ignore = me._ignored(self)
        senses = 0.75 + 0.25 * self.light
        reach = me.vision if radius is None else radius
        noticed = []
        for d2, a in self.query(me.prey_idx, me.x, me.y, reach, exclude=me):
            if a.concealed or a is ignore or a is skip:
                continue
            p = PREY_SPOT * senses * self.visibility(a) * (1.0 - math.sqrt(d2) / me.vision)
            if a.act == Act.FREEZE:
                p *= FROZEN_SPOT
            if self.rng.random() < p:
                noticed.append((math.sqrt(d2) * (0.7 if not a.adult else 1.0), a.id, a))
        noticed.sort()
        best = None
        for _, _, a in noticed[:3]:
            d = math.hypot(a.x - me.x, a.y - me.y)
            if self.clear_path(me.x, me.y, a.x, a.y):
                route = d
            else:
                found = self.find_path(me.x, me.y, a.x, a.y, max_cost=1.5 * d + 6.0)
                if found is None:
                    continue
                route = found[1]
            cost = route * self.vulnerability(a)
            if best is None or cost < best[1]:
                best = (a, cost)
        return best

    def find_carrion(self, me: Animal):
        """The nearest fresh carcass within the scavenger's vision (smell carries far)."""
        best, best_d2 = None, (1.5 * me.vision) ** 2
        ignore = me._ignored(self)
        for c in self.carrion:
            if c.energy <= 0.0 or c is ignore or c.si not in me.prey_idx:
                continue
            d2 = (c.x - me.x) ** 2 + (c.y - me.y) ** 2
            if d2 < best_d2:
                best, best_d2 = c, d2
        return best

    @staticmethod
    def related(a: Animal, b: Animal) -> bool:
        """Kin recognition: parents, offspring and (half-)siblings avoid mating."""
        if a.id in (b.mother_id, b.father_id) or b.id in (a.mother_id, a.father_id):
            return True
        return (a.mother_id is not None and a.mother_id == b.mother_id) or \
            (a.father_id is not None and a.father_id == b.father_id)

    def find_cache(self, me: Animal):
        """Food the animal can retrieve: its own caches (remembered anywhere), its
        den's larder, its mother's larder when young, or any cache it smells close by."""
        best, best_d2 = None, math.inf
        for c in self.caches:
            if c.energy <= 0.0:
                continue
            d2 = (c.x - me.x) ** 2 + (c.y - me.y) ** 2
            mine = c.owner == me.id or c.den == me.id or (c.den is not None and c.den == me.mother_id and d2 < 400)
            if (mine and d2 < 3600) or d2 < 9:
                if d2 < best_d2:
                    best, best_d2 = c, d2
        return best

    def add_cache(self, owner: Animal, energy: float, den: int | None) -> None:
        self.caches.append(Cache(owner.x, owner.y, energy, self.tick + CACHE_DAYS * self.cfg.ticks_per_day,
                                 owner.si, owner.id, den))
        self._event("cache", owner)

    def thump(self, a: Animal) -> None:
        """Alarm drumming: rabbits nearby stop what they are doing and look."""
        self._event("thump", a)
        until = self.tick + ALERT_TICKS
        for _, other in self.query((a.si,), a.x, a.y, THUMP_RADIUS, exclude=a):
            if not other.concealed and other.mother is None:
                other.alert_until = until

    def pounce(self, a: Animal) -> None:
        self._event("pounce", a)

    def evict(self, intruder: Animal, resident: Animal, d2: float) -> None:
        floater = intruder.adult and not intruder.territorial and not intruder.natal and intruder.helper_of is None
        if floater and intruder.energy > 0.4 * intruder.max_energy and self.rng.random() < CONTEST_CHANCE:
            self.contest(intruder, resident, d2)
            return
        intruder._switch(Act.FLEE)
        intruder.reward_events -= PAIN_CHASED
        intruder.threats = [(d2, resident)]
        intruder.flee_until = self.tick + 20
        self._event("chase", resident)

    def territory_push(self, a: Animal) -> float | None:
        """Heading away from nearby territory holders of the same sex (their scent marks)."""
        core = a.sp.territory / 2
        fx = fy = 0.0
        for h in self._holders[a.si].values():
            if h is a or h.female != a.female:
                continue
            dx, dy = a.x - h.home_x, a.y - h.home_y
            d = math.hypot(dx, dy)
            if 1e-6 < d < core:
                w = (core - d) / core / d
                fx += dx * w
                fy += dy * w
        return math.atan2(fy, fx) if (fx or fy) else None

    def find_mate(self, me: Animal):
        ignore = me._ignored(self)
        candidates = sorted(
            (d2, a.id, a)
            for d2, a in self.query((me.si,), me.x, me.y, me.sp.mate_range, exclude=me)
            if a.female != me.female and a is not ignore and a.ready_to_mate(self) and not self.related(me, a)
        )
        for d2, _, a in candidates[:3]:
            d = math.sqrt(d2)
            if self.clear_path(me.x, me.y, a.x, a.y) or self.find_path(me.x, me.y, a.x, a.y, 2.0 * d + 10.0):
                return a
        return None

    def find_plant(self, me: Animal):
        """Best grazing cell in view by rate of energy gain (travel + handling time).

        A simple optimal-foraging rule (Charnov 1976): prefer the patch that
        gives the most food per unit time, not merely the nearest one.
        """
        sp, r, x, y = me.sp, me.vision, me.x, me.y
        x0, x1 = max(0, int(x - r)), min(self.width, int(x + r) + 1)
        y0, y1 = max(0, int(y - r)), min(self.height, int(y + r) + 1)
        window = self.veg.B[y0:y1, x0:x1]
        dx = np.arange(x0, x1) + 0.5 - x
        dy = np.arange(y0, y1) + 0.5 - y
        dist = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
        ok = (window >= sp.graze_min) & (dist <= r)
        if not ok.any():
            return None
        speed = max(me.speed * sp.cruise, 1e-6)
        intake = max(me.intake / self.plant_energy, 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(ok, window / (dist / speed + window / intake), -1.0)
        if me.danger_mem:
            # Patches where a predator was met recently look less attractive.
            keys = (np.arange(x0, x1) // REGION)[None, :] + (np.arange(y0, y1) // REGION)[:, None] * self.region_cols
            for key in np.unique(keys).tolist():
                danger = me.recall(self, me.danger_mem, key, DANGER_MEMORY_DAYS)
                if danger > 0.05:
                    rate = np.where(keys == key, rate * (1.0 - 0.7 * danger), rate)
        ignore = me._ignored(self)
        cols = x1 - x0
        for k in np.argsort(rate, axis=None)[::-1][:6].tolist():
            if rate.flat[k] <= 0:
                break
            cell = (x0 + k % cols, y0 + k // cols)
            if cell != ignore and self.clear_path(x, y, cell[0] + 0.5, cell[1] + 0.5):
                return cell
        return None

    # ------------------------------------------------------------------ life and death

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def _add(self, a: Animal) -> None:
        self.animals.append(a)
        self.by_id[a.id] = a

    def _make(self, si: int, genome, x: float, y: float, **kw) -> Animal | None:
        expr = self.genomes[si].express(genome, self.rng)
        return Animal(self, self._new_id(), si, genome, expr, x, y, **kw)

    def _founder(self, si: int, cells, mature: bool, preset: str | None = None, at=None) -> Animal:
        s, gs, rng = self.species[si], self.genomes[si], self.rng
        while True:
            genome = gs.founder(rng, male=rng.random() < 0.5, preset=preset)
            expr = gs.express(genome, rng)
            if not expr.lethal:
                break
        x, y = at if at is not None else self._random_cell(cells)
        q = (expr.traits["mass"] / s.ref_mass) ** 0.25
        age = s.maturity * q + 1 if mature else rng.uniform(0.0, 0.5) * s.lifespan
        a = Animal(self, self._new_id(), si, genome, expr, x, y, age=int(age))
        a.energy = a.max_energy * rng.uniform(0.6, 0.9)
        a.water = a.max_water * rng.uniform(0.6, 0.9)
        if a.adult and not a.territorial:
            self.settle(a, a.sp.dispersal)
        return a

    def dispersal_time(self, si: int) -> bool:
        season = self.species[si].dispersal_season
        if season is None:
            return True
        f = self.env["year_fraction"]
        a, b = season
        return a <= f < b if a <= b else (f >= a or f < b)

    def prey_richness(self, si: int, x: float, y: float) -> float:
        """Prey within a territory's width of (x, y), relative to the reference density."""
        sp = self.species[si]
        if not sp.territory_prey or not any(self._buckets[p] for p in self.prey_of[si]):
            return 1.0
        n = sum(len(self.query((p,), x, y, sp.territory)) for p in self.prey_of[si])
        expected = sp.territory_prey / 1000.0 * math.pi * sp.territory ** 2
        return n / expected

    def territory_spacing(self, si: int, x: float, y: float) -> float:
        """Territories are smaller where food is rich and larger where it is poor."""
        sp = self.species[si]
        f = 1.0 / math.sqrt(max(self.prey_richness(si, x, y), 0.05))
        return sp.territory * min(TERRITORY_SCALE[1], max(TERRITORY_SCALE[0], f))

    def helpers_of(self, mother: Animal) -> int:
        return sum(1 for a in self.animals if a.alive and a.helper_of is mother)

    def nearest_rival(self, a: Animal, reach: float):
        best, best_d = None, reach
        for h in self._holders[a.si].values():
            if h.female == a.female and h is not a:
                d = math.hypot(h.home_x - a.x, h.home_y - a.y)
                if d < best_d:
                    best, best_d = h, d
        return best

    def contest(self, intruder: Animal, resident: Animal, d2: float) -> None:
        """A ritualised fight over a territory (Maynard Smith & Parker 1976).

        Bigger animals in better condition usually win, and residents have an
        owner's advantage. A winning intruder takes over the territory and the
        old resident becomes a floater. Fights cost both sides energy and are
        rarely fatal, more often in the mating season.
        """
        si = intruder.si
        rhp_i = intruder.mass * (0.5 + intruder.energy / intruder.max_energy)
        rhp_r = OWNER_ADVANTAGE * resident.mass * (0.5 + resident.energy / resident.max_energy)
        won = self.rng.random() < rhp_i ** 2 / (rhp_i ** 2 + rhp_r ** 2)
        self._event("fight", resident)
        self.social[si]["contest"] += 1
        for a in (intruder, resident):
            a.energy -= FIGHT_COST * a.max_energy
            a.stamina = 0.0
        winner, loser = (intruder, resident) if won else (resident, intruder)
        winner.reward_events += FIGHT_WON
        loser.reward_events -= FIGHT_LOST
        if won:
            self.social[si]["takeover"] += 1
            self._holders[si].pop(resident.id, None)
            resident.territorial = False
            resident.failed_claims = 0
            intruder.home_x, intruder.home_y = resident.home_x, resident.home_y
            intruder.territorial = True
            intruder.failed_claims = 0
            self._holders[si][intruder.id] = intruder
            for a in self.animals:
                if a.helper_of is resident:
                    a.helper_of = None  # the old family breaks up
        loser._switch(Act.FLEE)
        loser.threats = [(d2, winner)]
        loser.flee_until = self.tick + 20
        if winner.act == Act.FLEE:
            winner._switch(Act.WANDER)
        if self.rng.random() < FIGHT_DEATH[1 if self.in_season[si] else 0]:
            self.kill(loser, "fighting")

    def settle(self, a: Animal, reach: float, here: bool = True) -> bool:
        """Look for a territory within `reach`: first join a holder of the opposite
        sex that has no mate (red foxes hold territories as pairs), then try where
        the animal is (if `here`), then prospect a few spots round about."""
        rng, half = self.rng, a.sp.territory / 2
        holders = list(self._holders[a.si].values())
        lone = []
        for h in holders:
            if h is a or h.female == a.female:
                continue
            d = math.hypot(h.home_x - a.x, h.home_y - a.y)
            if d <= reach and not any(o is not a and o.female == a.female and
                                      math.hypot(o.home_x - h.home_x, o.home_y - h.home_y) < half for o in holders):
                lone.append((d, h.id, h))
        for _, _, h in sorted(lone):
            if self.claim_territory(a, h.home_x, h.home_y):
                return True
        if here and self.claim_territory(a, a.x, a.y):
            return True
        for _ in range(8):
            d, angle = rng.uniform(0.3, 1.0) * reach, rng.uniform(-math.pi, math.pi)
            x, y = a.x + d * math.cos(angle), a.y + d * math.sin(angle)
            if self.walkable(x, y) and self.claim_territory(a, x, y):
                return True
        return False

    def claim_territory(self, a: Animal, x: float, y: float) -> bool:
        """Settle a territory centred on (x, y) unless a same-sex holder is too close."""
        r2 = self.territory_spacing(a.si, x, y) ** 2
        for other in self._holders[a.si].values():
            if other is not a and other.female == a.female:
                dx, dy = other.home_x - x, other.home_y - y
                if dx * dx + dy * dy < r2:
                    return False
        a.home_x, a.home_y = x, y
        a.territorial = True
        self._holders[a.si][a.id] = a
        return True

    def _event(self, kind: str, a: Animal) -> None:
        if len(self._events) < EVENT_LIMIT:
            self._events.append([kind, round(a.x, 2), round(a.y, 2), a.si, a.id])

    def _note_record(self, a: Animal) -> None:
        rec = self.records[a.si]
        name = animal_name(a.id)
        for key, value in (("oldest", a.age), ("parent", a.children), ("hunter", a.kills)):
            if value and (rec[key] is None or value > rec[key][0]):
                rec[key] = (value, a.id, name)

    def kill(self, a: Animal, cause: str) -> None:
        self._note_record(a)
        a.alive = False
        a.cause = cause
        if self.brains is not None and a.brain_row >= 0:
            self.brains[a.si].release(a.brain_row)
            a.brain_row = -1
        self.deaths[a.si][cause] += 1
        self.by_id.pop(a.id, None)
        if a.territorial and self._holders[a.si].pop(a.id, None) is not None and a.sp.max_helpers:
            # A subordinate inherits the dead breeder's territory (Macdonald 1979).
            heirs = [h for h in self.animals if h.alive and h.helper_of is a]
            for h in heirs:
                h.helper_of = None
            if heirs:
                heir = max(heirs, key=lambda h: h.age)
                if self.claim_territory(heir, a.home_x, a.home_y):
                    self.social[a.si]["inherited"] += 1
                    for h in heirs:
                        if h is not heir:
                            h.helper_of = heir
        self._deaths_pending = True
        self._event(cause, a)
        self.narrator.windows[a.si].death(a, cause)
        if cause in ("starvation", "dehydration", "old age", "disease") and self.predators_of[a.si]:
            # The body lies where it fell for two days, available to scavengers.
            self.carrion.append(Carrion(a.x, a.y, 0.6 * a.mass * a.sp.carcass_energy,
                                        self.tick + 2 * self.cfg.ticks_per_day, a.si))
        self._graveyard[a.id] = {"id": a.id, "name": animal_name(a.id), "species": a.sp.name,
                                 "dead": True, "cause": cause, "age": a.age}
        if len(self._graveyard) > GRAVEYARD_SIZE:
            del self._graveyard[next(iter(self._graveyard))]

    def mate(self, a: Animal, b: Animal) -> None:
        female, male = (a, b) if a.female else (b, a)
        female.pregnant = True
        female.gestation_left = female.gestation
        female.mate_genome = male.genome
        female.mate_traits = male.traits
        female.mate_genotype = male.genotype
        female.mate_id = male.id
        female.mate_generation = male.generation
        male.partner = female
        for x in (female, male):
            x.reward_events += REWARD_MATING * x.traits.get("libido", 1.0)
        female.refractory = max(female.sp.refractory, female.gestation)
        male.refractory = 0.1 * male.sp.refractory
        for x in (female, male):
            x._switch(Act.WANDER)

    def give_birth(self, mother: Animal) -> None:
        sp, rng, gs = mother.sp, self.rng, self.genomes[mother.si]
        # Mothers in poor condition reabsorb part of the litter, so fecundity
        # falls as food per head runs short (density-dependent reproduction).
        condition = mother.energy / mother.max_energy
        mean = mother.traits["litter"] * min(1.0, max(0.15, (condition - 0.15) / 0.55))
        n = int(mean) + (rng.random() < mean - int(mean))
        if n == 0:
            mother.pregnant = False
            mother.mate_genome = None
            return
        budget = mother.energy * sp.investment
        share = budget / n
        generation = max(mother.generation, mother.mate_generation) + 1
        born = 0
        for _ in range(n):
            genome = gs.child(mother.genome, mother.mate_genome, rng, self.cfg.mutation)
            expr = gs.express(genome, rng)
            # Viability selection: lethal genotypes and low-vigour embryos die before birth.
            if expr.lethal or rng.random() > expr.vigor:
                self.stillborn[mother.si] += 1
                window = self.narrator.windows[mother.si]
                if expr.lethal:
                    window.lethal += 1
                else:
                    window.weak += 1
                continue
            x, y = self._near(mother.x, mother.y, 0.7)
            child = Animal(self, self._new_id(), mother.si, genome, expr, x, y,
                           generation=generation, mother_id=mother.id, father_id=mother.mate_id)
            child.home_x, child.home_y = mother.home_x, mother.home_y
            child.mother = mother
            # The young stay hidden in the nest where they were born.
            child.nest_x, child.nest_y = mother.x, mother.y
            child.x, child.y = mother.x, mother.y
            child.act = Act.NEST
            child.concealed = True
            mother.litter.append(child)
            # Provisioning the newborn can't use more than it can hold.
            child.energy = min(share, child.max_energy)
            mother.energy -= child.energy
            child.water = 0.9 * child.max_water
            self._born.append(child)
            mother.children_ids.append(child.id)
            self._event("birth", child)
            self.narrator.windows[mother.si].birth(mother, mother.mate_traits, mother.mate_genotype)
            born += 1
        mother.children += born
        mother.nest_x, mother.nest_y = mother.x, mother.y
        # The first feed comes straight after birth; then once per nursing interval.
        for child in mother.litter:
            if child.alive and child.mother is mother:
                child._suckle(mother)
        mother.next_nurse = self.tick + mother.sp.nurse_interval
        if born:
            self.max_generation[mother.si] = max(self.max_generation[mother.si], generation)
        mother.pregnant = False
        mother.mate_genome = None
        self.births[mother.si] += born

    # ------------------------------------------------------------------ sandbox tools

    def spawn(self, si: int, x: float, y: float, count: int = 1, preset: str | None = None) -> list[int]:
        ids = []
        for _ in range(max(1, min(50, count))):
            spot = self._near(x, y, 1.5) if self.walkable(x, y) else self.nearest_walkable(x, y)
            if spot is None:
                break
            a = self._founder(si, self._land, mature=True, preset=preset, at=spot)
            self.narrator.windows[si].arrivals += 1
            self._born.append(a)
            self._event("birth", a)
            ids.append(a.id)
        self._flush()
        return ids

    def remove_area(self, x: float, y: float, radius: float) -> int:
        n = 0
        for a in self.animals:
            if a.alive and (a.x - x) ** 2 + (a.y - y) ** 2 <= radius * radius:
                self.kill(a, "removed")
                n += 1
        self._flush()
        return n

    def fertilise(self, x: float, y: float, radius: float) -> None:
        self.veg.fertilise(x, y, radius)

    def paint(self, x: float, y: float, radius: float, kind: str) -> int:
        changed = self.terrain.paint(x, y, radius, PAINTABLE[kind], self.nrng)
        if not changed:
            return 0
        self.veg.refresh_capacity(self.terrain)
        if kind in ("grass", "sand"):
            self.veg.fertilise(x, y, radius)
        self._index_land()
        for a in self.animals:
            if not self.walkable(a.x, a.y):
                spot = self.nearest_walkable(a.x, a.y)
                if spot is None:
                    self.kill(a, "removed")
                    continue
                a.x, a.y = spot
            if not self.walkable(a.home_x, a.home_y):
                spot = self.nearest_walkable(a.home_x, a.home_y)
                if spot is not None:
                    a.home_x, a.home_y = spot
        self._flush()
        return changed

    def repopulate(self) -> None:
        """Replace every animal with a fresh founding population on the current landscape."""
        for a in self.animals:
            a.alive = False
        self.animals = []
        self.by_id = {}
        self.brains = self._brain_tables()
        self._born = []
        self.carrion = []
        self.caches = []
        self.finished = None
        self.event_log = []
        self._holders = [{} for _ in self.species]
        self._reset_counters()
        self._reset_history()
        for si, s in enumerate(self.species):
            for _ in range(s.initial):
                self._add(self._founder(si, self._land, mature=False))
        self._record()
        counts = ", ".join(f"{s.initial} {s.name}s" if not s.name.endswith("x") else f"{s.initial} {s.name}es"
                           for s in self.species)
        self.narrator.restart(f"{counts} were released with fresh genomes drawn from the founding allele frequencies. "
                              "Earlier trends no longer apply.")

    def breed(self, a: Animal, b: Animal) -> str | None:
        """Force a mating between two animals. Returns an error message if impossible."""
        if a.si != b.si:
            return "They are different species."
        if a.female == b.female:
            return "They are the same sex."
        female = a if a.female else b
        if not (a.adult and b.adult):
            return "Both must be adults."
        if female.pregnant:
            return "She is already pregnant."
        self.mate(a, b)
        return None

    def clone(self, a: Animal) -> int | None:
        spot = self._near(a.x, a.y, 1.5)
        child = self._make(a.si, a.genome.copy(), *spot, generation=a.generation)
        child.age = int(child.maturity) + 1
        child.born = self.tick - child.age
        child.adult = True
        child._allometry()
        child.energy, child.water = 0.85 * child.max_energy, 0.85 * child.max_water
        if not child.territorial:
            self.claim_territory(child, *spot)
        self._born.append(child)
        self._event("birth", child)
        self._flush()
        return child.id

    def edit_gene(self, a: Animal, symbol: str, alleles: tuple[str, str]) -> None:
        gs = self.genomes[a.si]
        gs.set_gene(a.genome, symbol, alleles)
        expr = gs.express(a.genome, self.rng)
        a.genotype = expr.genotype
        a.coat = expr.coat
        a.coat_lum = gs.coats[expr.coat].lum
        a.vigor = expr.vigor
        a._allometry()

    def trigger(self, kind: str, si: int | None = None) -> None:
        tpd = self.cfg.ticks_per_day
        self.event_log.append({"tick": self.tick, "day": self.env["day"], "kind": kind,
                               "species": None if si is None else self.species[si].name})
        if kind == "bloom":
            self.veg.B[:] = self.veg.K
            return
        duration = {"drought": 2 * tpd, "cold": int(1.5 * tpd), "plague": tpd}[kind]
        effect = {"kind": kind, "until": self.tick + duration, "species": si}
        self.narrator.effect_started(effect)
        self.effects.append(effect)
        self._apply_effects()

    def _apply_effects(self) -> None:
        for e in self.effects:
            if e["until"] <= self.tick:
                self.narrator.effect_ended(e)
        self.effects = [e for e in self.effects if e["until"] > self.tick]
        self.plague = [0.0] * len(self.species)
        for e in self.effects:
            if e["kind"] == "plague" and e["species"] is not None:
                self.plague[e["species"]] = 0.002

    # ------------------------------------------------------------------ main loop

    def _flush(self) -> None:
        if self._deaths_pending:
            self.animals = [a for a in self.animals if a.alive]
            self._deaths_pending = False
        if self._born:
            for a in self._born:
                if a.alive:
                    self._add(a)
            self._born.clear()

    def finish(self, si: int) -> None:
        """End the run: a species has died out."""
        env = self.env
        self.finished = {"tick": self.tick, "day": env["day"], "season": env["season"], "year": env["year"],
                         "species": self.species[si].name, "reason": "extinction"}
        self.narrator.observe()

    def resume(self) -> None:
        """Carry on after an ending, without ending again."""
        self.end_on_extinction = False
        self.finished = None

    def summary(self) -> dict:
        return self.narrator.summary()

    def step(self) -> None:
        if self.finished:
            return
        self.tick += 1
        if self.effects:
            self._apply_effects()
        self._environment()
        self.veg.step(self.growth)
        self._rebuild_buckets()
        self._think()
        order = self.animals[:]
        self.rng.shuffle(order)  # random update order avoids systematic first-mover advantages
        for a in order:
            if a.alive:
                a.update(self)
        self._flush()
        for si in range(len(self.species)):
            if self.rng.random() < self.cfg.immigration:
                self._add(self._founder(si, self._edge, mature=True))
                self.narrator.windows[si].arrivals += 1
        if self.carrion and self.tick % 20 == 0:
            self.carrion = [c for c in self.carrion if c.energy > 0.0 and c.until > self.tick]
        if self.caches and self.tick % 10 == 0:
            spoil = CACHE_DECAY ** 10
            for c in self.caches:
                c.energy *= spoil
            self.caches = [c for c in self.caches if c.energy > 5.0 and c.until > self.tick]
        for si, n in enumerate(self.counts()):
            if n:
                self.ever[si] = True
            elif self.ever[si] and self.end_on_extinction and self.finished is None:
                self.finish(si)
        if self.tick % OBSERVE == 0:
            self.narrator.observe()
        if self.tick % self._interval == 0:
            self._record()

    # ------------------------------------------------------------------ statistics

    def _think(self) -> None:
        """Every free animal senses its situation; then each species chooses at once.

        In neural mode the whole species goes through its networks in one
        batched matrix product per layer, and (with learning on) every animal
        then learns from the reward it got since its last decision. The
        instinct choice is computed too, as a reference for how far brains have
        evolved and learned away from their ancestors' instincts.
        """
        learning = self.brains is not None and self.cfg.learning
        groups = [[] for _ in self.species]
        for a in self.animals:
            if a.alive and a.mother is None:
                a.sense(self)
                groups[a.si].append(a)
        for si, members in enumerate(groups):
            if not members:
                continue
            x = np.ones((len(members), brain.N_IN), dtype=np.float32)
            x[:, :-1] = [a.inputs for a in members]
            mask = np.array([a.mask for a in members], dtype=bool)
            current = ACT_OPTION[[int(a.act) for a in members]]
            instinct = brain.squash(brain.rule_scores(x))
            rule_choice = brain.choose(instinct, mask, current)
            if self.brains is None:
                scores, hidden, choice = instinct, None, rule_choice
            else:
                table = self.brains[si]
                rows = np.array([a.brain_row for a in members])
                scores, hidden = brain.forward(x, table.w1[rows], table.w2[rows])
                greedy = brain.choose(scores, mask, current)
                if learning:
                    choice, probs = brain.policy(scores, mask, current, self.nrng)
                    value = table.value(rows, hidden)
                    reward = np.array([a.reward for a in members], dtype=np.float32)
                    plasticity = np.array([a.plasticity() for a in members], dtype=np.float32)
                    delta = table.learn(rows, x, hidden, value, reward, plasticity, choice, probs, mask)
                    for k, a in enumerate(members):
                        a.surprise = float(delta[k])
                        a.value_now = float(value[k])
                        a.wellbeing += 0.02 * (a.reward - a.wellbeing)
                        a.surprise_avg += 0.02 * (abs(a.surprise) - a.surprise_avg)
                        if (self.tick + a.id) % BRAIN_LOG_EVERY == 0:
                            self._log_brain(a, table)
                else:
                    choice = greedy
                clear = brain.margin(instinct, mask) >= brain.CLEAR
                n_clear = int(clear.sum())
                differ = np.nonzero((greedy != rule_choice) & clear)[0]
                self.agreement[si][0] += n_clear - len(differ)
                self.agreement[si][1] += n_clear
                self.decisions[si] += n_clear
                pairs = self.choice_pairs[si]
                for k in differ.tolist():
                    pairs[(int(rule_choice[k]), int(greedy[k]))] += 1
            choice = choice.tolist()
            rule_choice = rule_choice.tolist()
            for k, a in enumerate(members):
                a.planned = choice[k]
                a.rule_choice = rule_choice[k]
                a.scores = scores[k]
                a.hidden = None if hidden is None else hidden[k]

    def _learned(self, a: Animal, table) -> tuple[float, float]:
        """How far an animal's wiring has moved from what it was born with (RMS),
        and in what share of its recent situations it now decides differently."""
        row = a.brain_row
        d1 = table.w1[row] - table.i1[row]
        d2 = table.w2[row] - table.i2[row]
        change = float(np.sqrt((np.sum(d1 * d1) + np.sum(d2 * d2)) / brain.N_WEIGHTS))
        if not a.recent:
            return change, 0.0
        x = np.array([s for s, _ in a.recent], dtype=np.float32)
        mask = np.array([m for _, m in a.recent], dtype=bool)
        none = np.full(len(x), -1)
        n = len(x)
        now, _ = brain.forward(x, np.broadcast_to(table.w1[row], (n,) + table.w1.shape[1:]),
                               np.broadcast_to(table.w2[row], (n,) + table.w2.shape[1:]))
        born, _ = brain.forward(x, np.broadcast_to(table.i1[row], (n,) + table.i1.shape[1:]),
                                np.broadcast_to(table.i2[row], (n,) + table.i2.shape[1:]))
        # Only situations where its inborn wiring had a clear preference count:
        # close calls flip with the slightest change.
        clear = brain.margin(born, mask) >= brain.CLEAR
        if not clear.any():
            return change, 0.0
        differs = float(np.mean((brain.choose(now, mask, none) != brain.choose(born, mask, none))[clear]))
        return change, differs

    def _log_brain(self, a: Animal, table) -> None:
        change, differs = self._learned(a, table)
        a.brain_log.append([round(a.age / self.cfg.ticks_per_day, 2), round(a.wellbeing * 1000, 3),
                            round(a.surprise_avg * 1000, 3), round(change, 4), round(differs, 3)])
        if len(a.brain_log) > 160:
            del a.brain_log[::2]  # keep the whole life, at lower resolution

    def counts(self) -> list[int]:
        out = [0] * len(self.species)
        for a in self.animals:
            out[a.si] += 1
        return out

    def _reset_history(self) -> None:
        self._interval = self.cfg.stats_interval
        self.history = {"tick": [], "plants": [], "species": {}}
        for s in self.species:
            self.history["species"][s.name] = {
                "count": [],
                "juveniles": [],
                "traits": {t: {"mean": [], "sd": [], "genetic": []} for t in s.traits},
                "genes": {g.symbol: {"freq": [], "ho": [], "he": []} for g in s.genes},
                "diversity": {"he": [], "ho": []},
                "vigor": [],
                "load": [],
                "instinct": [],
                # Lifetime learning: how much brains have changed since birth, and
                # habits (thresholds) they were born with versus have now.
                "brain": {k: [] for k in ("learned", "differs", "eat_born", "eat_now", "drink_born", "drink_now",
                                          "flee_born", "flee_now")},
            }

    def _record(self) -> None:
        h = self.history
        h["tick"].append(self.tick)
        h["plants"].append(round(self.veg.total(), 1))
        groups = [[] for _ in self.species]
        for a in self.animals:
            groups[a.si].append(a)
        for si, (s, members) in enumerate(zip(self.species, groups)):
            gs = self.genomes[si]
            row = h["species"][s.name]
            n = len(members)
            row["count"].append(n)
            row["juveniles"].append(sum(1 for a in members if not a.adult))
            same, total = self.agreement[si]
            row["instinct"].append(round(same / total, 4) if total else None)
            self.agreement[si] = [0, 0]
            self._record_learning(si, members, row["brain"])
            for t, series in row["traits"].items():
                if n:
                    values = [a.traits[t] for a in members]
                    mean = sum(values) / n
                    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
                    genetic = s.traits[t].mean * sum(a.breeding[t] for a in members) / n
                    series["mean"].append(round(mean, 4))
                    series["sd"].append(round(sd, 4))
                    series["genetic"].append(round(genetic, 4))
                else:
                    for k in ("mean", "sd", "genetic"):
                        series[k].append(None)
            for g in s.genes:
                series = row["genes"][g.symbol]
                i = gs.gene_at[g.symbol]
                counts = [0] * len(g.alleles)
                het = 0
                for a in members:
                    x, y = a.genome.mat[i], a.genome.pat[i]
                    counts[x] += 1
                    counts[y] += 1
                    het += x != y
                if n:
                    freq = [round(c / (2 * n), 4) for c in counts]
                    series["freq"].append(freq)
                    series["ho"].append(round(het / n, 4))
                    series["he"].append(round(1.0 - sum(p * p for p in freq), 4))
                else:
                    series["freq"].append(None)
                    series["ho"].append(None)
                    series["he"].append(None)
            if n:
                he = 0.0
                for i in gs.marker_at:
                    counts = Counter()
                    for a in members:
                        counts[a.genome.mat[i]] += 1
                        counts[a.genome.pat[i]] += 1
                    he += 1.0 - sum((c / (2 * n)) ** 2 for c in counts.values())
                he /= max(1, len(gs.marker_at))
                ho = sum(a.het for a in members) / (n * max(1, len(gs.marker_at)))
                row["diversity"]["he"].append(round(he, 4))
                row["diversity"]["ho"].append(round(ho, 4))
                row["vigor"].append(round(sum(a.vigor for a in members) / n, 4))
                row["load"].append(round(sum(a.load for a in members) / n, 3))
            else:
                row["diversity"]["he"].append(None)
                row["diversity"]["ho"].append(None)
                row["vigor"].append(None)
                row["load"].append(None)
        if len(h["tick"]) > HISTORY_LIMIT:
            self._thin(h)
            self._interval *= 2

    def _record_learning(self, si: int, members: list, out: dict) -> None:
        """Population averages of lifetime learning, from a sample of animals."""
        free = [a for a in members if a.mother is None and a.brain_row >= 0][:40] if self.brains is not None else []
        if not free:
            for series in out.values():
                series.append(None)
            return
        table = self.brains[si]
        rows = [a.brain_row for a in free]
        learned = [self._learned(a, table) for a in free]
        out["learned"].append(round(sum(c for c, _ in learned) / len(learned), 4))
        out["differs"].append(round(sum(d for _, d in learned) / len(learned), 4))
        prey = bool(self.predators_of[si])
        born = brain.thresholds(table.i1[rows], table.i2[rows], prey)
        now = brain.thresholds(table.w1[rows], table.w2[rows], prey)
        for k in ("eat", "drink", "flee"):
            for tag, src in (("born", born), ("now", now)):
                vals = src.get(k)
                ok = vals[~np.isnan(vals)] if vals is not None else []
                out[f"{k}_{tag}"].append(round(float(np.mean(ok)), 4) if len(ok) else None)

    def _thin(self, node) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                self._thin(value)
            else:
                node[key] = value[::2]

    def history_since(self, since: int) -> dict:
        h = self.history
        start = bisect.bisect_right(h["tick"], since)

        def cut(node):
            return {k: cut(v) if isinstance(v, dict) else v[start:] for k, v in node.items()}

        return cut(h)

    # ------------------------------------------------------------------ serialisation

    def terrain_payload(self) -> dict:
        t = self.terrain
        return {
            "version": t.version,
            "types": _b64(t.types),
            "heights": _b64(np.clip(t.heights * 20, 0, 255)),
            "fertility": _b64(t.fertility * 255),
        }

    def describe(self) -> dict:
        """Static information the client needs once per world."""
        species = []
        for si, s in enumerate(self.species):
            gs = self.genomes[si]
            species.append({
                "name": s.name,
                "icon": s.icon,
                "color": s.color,
                "diet": s.diet,
                "prey": list(s.prey),
                "territorial": s.territory > 0,
                "burrows": s.burrow_reach > 0,
                "ref_mass": s.ref_mass,
                "traits": {t: {"label": spec.label, "unit": spec.unit, "mean": spec.mean,
                               "heritability": spec.heritability, "loci": spec.loci}
                           for t, spec in s.traits.items()},
                "genes": [{"symbol": g.symbol, "name": g.name, "alleles": list(g.alleles),
                           "labels": list(g.labels or g.alleles), "note": g.note} for g in s.genes],
                "coats": [{"name": c.name, "colors": {p: c.colors.get(p, c.colors["body"]) for p in PARTS},
                           "stripe": c.stripe} for c in gs.coats],
                "presets": list(PRESETS[s.coat]),
                "chromosomes": s.chromosomes,
            })
        return {
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "terrain": self.terrain_payload(),
            "terrain_types": list(TYPE_NAMES),
            "species": species,
            "acts": [a.name.lower().replace("_", " ") for a in Act],
            "causes": list(CAUSES),
            "events": EVENTS,
            "plant_capacity": self.veg.capacity,
            "plant_capacity_total": round(float(self.veg.K.sum()), 1),
            "clock": {"ticks_per_day": self.cfg.ticks_per_day, "days_per_year": self.cfg.days_per_year,
                      "seasons": self.cfg.seasons},
            "settings": settings_of(self.cfg),
            "brain": {"inputs": list(brain.INPUTS) + ["bias"], "options": list(brain.OPTIONS), "hidden": brain.N_HID,
                      "mode": self.cfg.brains, "learning": self.cfg.brains == "neural" and self.cfg.learning,
                      "training": brain.training_report()},
        }

    def frame(self, plants: bool = True, insights_since: int | None = None) -> dict:
        """Dynamic state for one rendered frame."""
        out = {
            "tick": self.tick,
            "env": self.env,
            "animals": [
                [a.id, a.si, round(a.x, 2), round(a.y, 2), round(a.heading, 2), int(a.act),
                 int(a.female) | (a.pregnant << 1) | ((not a.adult) << 2) | (a.scanning << 3)
                 | (a.stalking << 4) | (a.concealed << 5) | ((a.act == Act.CACHE and a.carried > 0) << 6), a.coat,
                 round((a.mass / a.sp.ref_mass) ** (1 / 3), 2), round(a.vision, 1)]
                for a in self.animals
            ],
            "counts": self.counts(),
            "births": self.births,
            "stillborn": self.stillborn,
            "deaths": [{c: d[c] for c in CAUSES} for d in self.deaths],
            "hunts": [dict(h) for h in self.hunts],
            "events": self._events,
            "effects": [{"kind": e["kind"], "left": e["until"] - self.tick, "species": e["species"]}
                        for e in self.effects],
            "terrain_version": self.terrain.version,
            "plant_total": round(self.veg.total(), 1),
            "carrion": [[round(c.x, 2), round(c.y, 2)] for c in self.carrion if c.energy > 0.0][:300],
            "caches": [[round(c.x, 2), round(c.y, 2)] for c in self.caches][:300],
            "finished": self.finished,
            "end_on_extinction": self.end_on_extinction,
            "brains": self.narrator.brain_state if self.brains is not None else None,
        }
        self._events = []
        if plants:
            out["plants"] = self.veg.encode()
        if insights_since is not None:
            out["insights"] = self.narrator.since(insights_since)
        return out

    def _ref(self, uid: int | None) -> dict | None:
        if uid is None:
            return None
        a = self.by_id.get(uid)
        if a is not None:
            return {"id": uid, "name": animal_name(uid), "alive": True}
        dead = self._graveyard.get(uid)
        return {"id": uid, "name": animal_name(uid), "alive": False, "cause": dead["cause"] if dead else None}

    def inspect(self, uid: int) -> dict | None:
        a = self.by_id.get(uid)
        if a is None:
            return self._graveyard.get(uid)
        gs = self.genomes[a.si]
        target = a.target
        if isinstance(target, (Animal, Carrion)):
            target = [round(target.x, 2), round(target.y, 2)]
        elif target is not None:
            target = [target[0] + 0.5, target[1] + 0.5]
        genes = []
        for g in a.sp.genes:
            pair = a.genotype[g.symbol]
            labels = g.labels or g.alleles
            genes.append({"symbol": g.symbol, "name": g.name,
                          "alleles": [labels[g.alleles.index(x)] for x in pair],
                          "raw": list(pair), "note": g.note})
        living = [self._ref(c) for c in a.children_ids if c in self.by_id]
        return {
            "id": a.id,
            "name": animal_name(a.id),
            "species": a.sp.name,
            "sex": "female" if a.female else "male",
            "act": a.act.name.lower().replace("_", " "),
            "x": round(a.x, 2),
            "y": round(a.y, 2),
            "vision": round(a.vision, 2),
            "home": [round(a.home_x, 2), round(a.home_y, 2)],
            "home_radius": a.sp.home_radius,
            "target": target,
            "age": a.age,
            "maturity": round(a.maturity),
            "lifespan": round(a.lifespan),
            "adult": a.adult,
            "dependent": a.mother is not None,
            "in_nest": a.mother is not None and a.concealed,
            "litter": sum(1 for k in a.litter if k.alive and k.mother is a),
            "next_nurse": max(0, a.next_nurse - self.tick) if a.litter else None,
            "fear": round(a.fear, 2),
            "partner": self._ref(a.partner.id) if a.partner is not None else None,
            "caches": sum(1 for c in self.caches if c.owner == a.id),
            "route": [[round(x, 2), round(y, 2)] for x, y in a.path[:40]] if a.path else None,
            "territory": None if a.sp.territory <= 0 or not a.adult else a.territorial,
            "role": self._role(a),
            "helpers": self.helpers_of(a) if a.territorial and a.female else 0,
            "generation": a.generation,
            "mother": self._ref(a.mother_id),
            "father": self._ref(a.father_id),
            "children": a.children,
            "living_children": living[:24],
            "kills": a.kills,
            "pregnant": a.pregnant,
            "gestation_left": round(a.gestation_left) if a.pregnant else None,
            "mate": self._ref(a.mate_id) if a.pregnant else None,
            "mass": round(a.mass, 3),
            "energy": round(a.energy / a.max_energy, 3),
            "water": round(a.water / a.max_water, 3),
            "stamina": round(a.stamina / a.max_stamina, 3),
            "upkeep": round(a.upkeep * a.thermal_cost(self), 4),
            "traits": {t: {"value": round(v, 3), "genetic": round(a.sp.traits[t].mean * a.breeding[t], 3)}
                       for t, v in a.traits.items()},
            "coat": gs.coats[a.coat].name,
            "visibility": round(self.visibility(a), 2),
            "genes": genes,
            "vigor": round(a.vigor, 3),
            "load": a.load,
            "load_hom": a.load_hom,
            "het": a.het,
            "markers": len(gs.marker_at),
            "chromosomes": gs.describe_genome(a.genome),
            "brain": self._brain_view(a),
            "memory": self._memory_view(a),
        }

    def _role(self, a: Animal) -> str | None:
        if a.sp.territory <= 0 or not a.adult:
            return None
        if a.territorial:
            return "breeder"
        if a.helper_of is not None:
            return "helper"
        return "natal" if a.natal else "floater"

    def _brain_view(self, a: Animal) -> dict | None:
        if a.inputs is None:
            return None
        out = {
            "mode": self.cfg.brains,
            "inputs": [round(float(v), 3) for v in a.inputs] + [1.0],
            "mask": list(a.mask),
            "scores": None if a.scores is None else [round(float(v), 3) for v in a.scores],
            "choice": a.planned,
            "instinct": a.rule_choice,
            "hidden": None if a.hidden is None else [round(float(v), 3) for v in a.hidden],
        }
        if self.brains is not None and a.brain_row >= 0:
            table = self.brains[a.si]
            row = a.brain_row
            w1, w2, i1, i2 = table.w1[row], table.w2[row], table.i1[row], table.i2[row]
            out["w1"] = np.round(w1, 3).tolist()
            out["w2"] = np.round(w2, 3).tolist()
            out["born_w1"] = np.round(i1, 3).tolist()
            out["born_w2"] = np.round(i2, 3).tolist()
            # How far its inborn wiring (its genes) differs from the founders'.
            ref1, ref2 = brain.unpack(self.genomes[a.si].brain_init)
            diff = np.concatenate([(i1 - ref1).ravel(), (i2 - ref2).ravel()])
            out["drift"] = round(float(np.sqrt(np.mean(diff ** 2))), 3)
            out["learning"] = self.cfg.learning
            change, differs = self._learned(a, table)
            out["change"] = round(change, 4)
            out["differs"] = round(differs, 3)
            out["value"] = round(a.value_now, 3)
            out["surprise"] = round(a.surprise, 4)
            out["reward"] = round(a.reward, 4)
            out["plasticity"] = round(a.plasticity(), 2)
            # Why it chose what it did: how much each sense pushed that option up or down.
            x = np.array(a.inputs + [1.0], dtype=np.float32)
            out["why"] = [round(float(v), 3) for v in brain.attributions(x, w1, w2, a.planned)]
            # Habits it was born with and has now.
            prey = bool(self.predators_of[a.si])
            born = brain.thresholds(i1[None], i2[None], prey)
            now = brain.thresholds(w1[None], w2[None], prey)
            out["habits"] = {k: [_num(born[k][0]), _num(now[k][0])] for k in born}
            # Which options it has come to like more or less, over its recent situations.
            if a.recent:
                xs = np.array([s for s, _ in a.recent], dtype=np.float32)
                ms = np.array([m for _, m in a.recent], dtype=bool)
                n = len(xs)
                s_now, _ = brain.forward(xs, np.broadcast_to(w1, (n,) + w1.shape), np.broadcast_to(w2, (n,) + w2.shape))
                s_born, _ = brain.forward(xs, np.broadcast_to(i1, (n,) + i1.shape), np.broadcast_to(i2, (n,) + i2.shape))
                count = ms.sum(axis=0)
                total = np.where(ms, s_now - s_born, 0.0).sum(axis=0)
                out["shift"] = [_num(t / c) if c else None for t, c in zip(total, count)]
            out["log"] = a.brain_log
        return out

    def _memory_view(self, a: Animal) -> dict:
        def regions(memory, half_life):
            out = []
            for key in memory:
                value = a.recall(self, memory, key, half_life)
                if value >= 0.05:
                    out.append([(key % self.region_cols) * REGION, (key // self.region_cols) * REGION, round(value, 2)])
            return out
        return {
            "region": REGION,
            "food": regions(a.food_mem, FOOD_MEMORY_DAYS),
            "danger": regions(a.danger_mem, DANGER_MEMORY_DAYS),
            "experience": round(a.experience, 3),
            "skill": round(a.skill, 3) if self.prey_of[a.si] else None,
            "goal": [round(a.goal[0], 2), round(a.goal[1], 2)] if a.goal else None,
        }
