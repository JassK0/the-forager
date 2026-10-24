"""The run's diary: one row per simulated day, a log of the last deaths of each
species, every extinction, and what the user did with the sandbox tools.

It also keeps the evidence for natural selection: the summed traits of every
animal that died (by cause) and of every parent (weighted by young born), so
"the rabbits foxes caught were 8% slower than average" can be read straight
off the records.

Rows store running totals (births, deaths by cause, meals...) alongside the
state of each species at the end of the day, so any span of days can be summed
by subtracting two rows, even after old rows have been thinned out. This is
the evidence the AI analyst (and the Deaths & births charts) work from.
"""

from collections import Counter, deque

import numpy as np

LEDGER_LIMIT = 6000  # rows kept; beyond this, older rows are thinned (totals stay exact)
LAST_DEATHS = 60  # deaths of each species remembered in detail
REGIONS = (4, 4)  # columns x rows of the map grid used for "where" questions
SEASONS = ("spring", "summer", "autumn", "winter")


def region_of(world, x: float, y: float) -> int:
    cols, rows = REGIONS
    c = min(cols - 1, max(0, int(x * cols / world.width)))
    r = min(rows - 1, max(0, int(y * rows / world.height)))
    return r * cols + c


def region_name(k: int) -> str:
    cols, _ = REGIONS
    return f"{'ABCD'[k % cols]}{k // cols + 1}"


class Ledger:
    def __init__(self, world, causes: tuple[str, ...]):
        self.w = world
        self.causes = causes
        self.cause_index = {c: i for i, c in enumerate(causes)}
        n = len(world.species)
        self.rows: list[dict] = []
        self.last_deaths = [deque(maxlen=LAST_DEATHS) for _ in range(n)]
        self.extinctions: list[dict] = []
        self.actions: list[dict] = []
        self.region_deaths = np.zeros((n, len(causes), REGIONS[0] * REGIONS[1]), dtype=np.int32)
        self.trait_names = [list(sp.traits) for sp in world.species]
        # Running sums of traits (last column: how many), for the dead by cause and for parents.
        self.dead_traits = [np.zeros((len(causes), len(t) + 1)) for t in self.trait_names]
        self.parent_traits = [np.zeros(len(t) + 1) for t in self.trait_names]
        self.dead_gap = [np.zeros(len(causes)) for _ in world.species]  # summed camouflage gaps of the dead, by cause
        self._alive = [False] * n
        self._temp = self._snow = 0.0
        self._samples = 0

    # ------------------------------------------------------------------ recording

    def tick(self) -> None:
        """Called every tick: averages the weather over the day and notices extinctions."""
        env = self.w.env
        self._temp += env["temperature"]
        self._snow += env["snow"]
        self._samples += 1

    def counts_changed(self, counts: list[int]) -> None:
        w = self.w
        for si, n in enumerate(counts):
            if n and not self._alive[si]:
                self._alive[si] = True
                for e in reversed(self.extinctions):
                    if e["si"] == si and e["recolonised_day"] is None:
                        e["recolonised_day"] = w.env["day"]
                        break
            elif not n and self._alive[si]:
                self._alive[si] = False
                env = w.env
                self.extinctions.append({
                    "si": si, "species": w.species[si].name, "tick": w.tick, "day": env["day"], "year": env["year"],
                    "season": env["season"], "last_deaths": list(self.last_deaths[si]), "recolonised_day": None,
                })

    def death(self, a, cause: str) -> None:
        w = self.w
        tpd = w.cfg.ticks_per_day
        dependents = sum(1 for c in a.litter if c.alive and not c.adult) if a.litter else 0
        self.last_deaths[a.si].append({
            "tick": w.tick, "day": w.env["day"], "hour": int(w.env["hours"]), "id": a.id,
            "sex": "F" if a.female else "M", "age_days": round(a.age / tpd, 1), "adult": a.adult, "cause": cause,
            "x": round(a.x), "y": round(a.y), "region": region_name(region_of(w, a.x, a.y)),
            "energy": round(a.energy / a.max_energy, 2), "water": round(a.water / a.max_water, 2),
            "injury": round(a.injury, 2), "role": w._role(a), "pregnant": a.pregnant, "dependents": dependents,
            "generation": a.generation,
        })
        k = self.cause_index.get(cause)
        if k is not None:
            self.region_deaths[a.si, k, region_of(w, a.x, a.y)] += 1
            self.dead_gap[a.si][k] += w.camouflage_gap(a)
            row = self.dead_traits[a.si][k]
            for j, t in enumerate(self.trait_names[a.si]):
                row[j] += a.traits[t]
            row[-1] += 1

    def parents(self, mother, father_traits: dict | None, young: int) -> None:
        """A litter was born: its parents' average traits count once per young."""
        acc = self.parent_traits[mother.si]
        for j, t in enumerate(self.trait_names[mother.si]):
            v = mother.traits[t] if father_traits is None else 0.5 * (mother.traits[t] + father_traits[t])
            acc[j] += young * v
        acc[-1] += young

    def action(self, kind: str, detail: str = "", amount: int = 1, species: str | None = None) -> None:
        """A sandbox action. Repeats on the same day (brush strokes) are merged into one entry."""
        env = self.w.env
        last = self.actions[-1] if self.actions else None
        if last and last["kind"] == kind and last["detail"] == detail and last["species"] == species \
                and last["day"] == env["day"]:
            last["amount"] += amount
            return
        self.actions.append({"tick": self.w.tick, "day": env["day"], "kind": kind, "detail": detail,
                             "amount": amount, "species": species})
        del self.actions[:-300]

    def record(self) -> None:
        """One row: the state at the end of a day, plus running totals."""
        w = self.w
        cfg = w.cfg
        dpy = cfg.days_per_year
        day = w.env["day"] - 1 if w.tick else 0
        groups = [[] for _ in w.species]
        for a in w.animals:
            groups[a.si].append(a)
        species = []
        for si, (sp, members) in enumerate(zip(w.species, groups)):
            adults = [a for a in members if a.adult]
            names = self.trait_names[si]
            row = {
                "n": len(members), "adults": len(adults),
                "af": sum(1 for a in adults if a.female), "am": sum(1 for a in adults if not a.female),
                "pregnant": sum(1 for a in members if a.pregnant),
                "condition": round(sum(a.energy / a.max_energy for a in members) / len(members), 3) if members else None,
                "hydration": round(sum(a.water / a.max_water for a in members) / len(members), 3) if members else None,
                "injured": sum(1 for a in members if a.injury > 0.2),
                "births": w.births[si], "stillborn": w.stillborn[si], "arrivals": w.arrivals[si],
                "matings": w.matings[si], "rejections": w.rejections[si],
                "deaths": dict(w.deaths[si]), "hunts": dict(w.hunts[si]), "social": dict(w.social[si]),
                # Average traits of the living (as they are, and as genes predict), and the selection sums.
                "pheno": [round(sum(a.traits[t] for a in members) / len(members), 4) for t in names] if members else None,
                "genetic": [round(sp.traits[t].mean * sum(a.breeding[t] for a in members) / len(members), 4)
                            for t in names] if members else None,
                "dead_traits": self.dead_traits[si].copy(), "parent_traits": self.parent_traits[si].copy(),
                "gap": round(sum(w.camouflage_gap(a) for a in members) / len(members), 4) if members else None,
                "dead_gap": self.dead_gap[si].copy(),
            }
            if w.cfg.super_animals and adults:
                powers = sorted((w.relative_power(a) for a in adults), reverse=True)
                row["power"] = [round(sum(powers) / len(powers), 2), round(powers[0], 2)]
            if sp.territory > 0:
                holders = list(w._holders[si].values())
                homes = {}
                for h in holders:
                    homes.setdefault((h.home_x, h.home_y), set()).add(h.female)
                row["territories"] = len(homes)
                row["paired"] = sum(1 for sexes in homes.values() if len(sexes) == 2)
                row["floaters"] = sum(1 for a in adults if w._role(a) == "floater")
                row["helpers"] = sum(1 for a in adults if w._role(a) == "helper")
            species.append(row)
        k = float(w.veg.K.sum()) or 1.0
        samples = max(1, self._samples)
        self.rows.append({
            "tick": w.tick, "day": day, "year": (day - 1) // dpy + 1 if day else 1,
            "season": SEASONS[((day - 1) % dpy) * 4 // dpy] if (day and cfg.seasons) else ("spring" if cfg.seasons else None),
            "temp": round(self._temp / samples, 2) if self._samples else w.env["temperature"],
            "snow": round(self._snow / samples, 2) if self._samples else w.env["snow"],
            "cover": round(w.veg.total() / k, 3), "voles": round(w.voles.total()), "fruit": round(w.fruit.total()),
            "effects": sorted({e["kind"] for e in w.effects}), "species": species,
            "region_deaths": self.region_deaths.copy(),
        })
        self._temp = self._snow = 0.0
        self._samples = 0
        if len(self.rows) > LEDGER_LIMIT:
            half = len(self.rows) // 2
            self.rows = self.rows[:half:2] + self.rows[half:]

    # ------------------------------------------------------------------ reading

    @staticmethod
    def delta(new: dict, old: dict, si: int) -> dict:
        """What happened to species si between two rows."""
        a, b = new["species"][si], old["species"][si]
        out = {k: a[k] - b[k] for k in ("births", "stillborn", "arrivals", "matings", "rejections")}
        out["deaths"] = Counter(a["deaths"]) - Counter(b["deaths"])
        out["hunts"] = Counter(a["hunts"]) - Counter(b["hunts"])
        out["social"] = Counter(a["social"]) - Counter(b["social"])
        return out
