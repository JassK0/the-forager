"""The narrator: explains what is happening in the ecosystem, and why.

No guesswork is needed: the simulation knows the true cause of every death,
which animals bred, what genes they and the dead carried, and what the weather
and food supply were. Every few hours of simulated time the narrator takes a
snapshot, compares it with earlier ones, and when something notable happens
writes an explanation from that evidence:

* Population swings are attributed to the recorded causes of death and to
  births, with the context that drove them (food, season, predators, heat).
* Allele-frequency changes are tested against genetic drift: the expected
  change from drift alone has standard deviation sqrt(p(1-p) t / 2Ne) over t
  generations (Wright 1931). Changes much larger than that, or alleles that
  are over-represented among the dead or under-represented among parents,
  point to selection.
* Trait evolution is explained with the breeder's equation, R = h^2 S
  (Lush 1937): the selection differential S is the difference between the
  parents of new young and the average adult, and the dead are compared with
  the living to show which pressures were at work.
* Evolving brains are compared, decision by decision, with the instincts the
  founders were trained on, and the commonest changes of mind are reported.
"""

import math
from collections import Counter, deque

OBSERVE = 60  # ticks between snapshots (a quarter of a day)
KEEP = 48  # snapshots kept (12 days)
LOG_LIMIT = 300
NE_RATIO = 0.5  # effective population size as a fraction of the census size
SEASONS = ("spring", "summer", "autumn", "winter")
BRAIN_CHECK_DAYS = 2  # days of decisions pooled before brains are compared with instincts
BRAIN_MILESTONES = (0.85, 0.75, 0.65, 0.5)  # instinct agreement levels worth reporting

EVOLVE_WORDS = {
    "mass": ("heavier bodies", "lighter bodies"),
    "speed": ("faster legs", "slower legs"),
    "vision": ("sharper eyesight", "poorer eyesight"),
    "litter": ("larger litters", "smaller litters"),
    "vigilance": ("more watchfulness", "less watchfulness"),
    "insulation": ("thicker fur", "thinner fur"),
    "stamina": ("more endurance", "less endurance"),
    "boldness": ("bolder temperaments", "shyer temperaments"),
    "freeze": ("a stronger urge to freeze", "less freezing"),
    "appetite": ("bigger appetites", "smaller appetites"),
    "libido": ("a stronger mating drive", "a weaker mating drive"),
    "persistence": ("more persistent chasing", "shorter chases"),
    "learning": ("faster learning", "slower learning"),
}
COMPARE_WORDS = {
    "mass": ("heavier", "lighter"),
    "speed": ("faster", "slower"),
    "vision": ("sharper-sighted", "shorter-sighted"),
    "litter": ("more fecund", "less fecund"),
    "vigilance": ("more watchful", "less watchful"),
    "insulation": ("thicker-furred", "thinner-furred"),
    "stamina": ("tougher", "less enduring"),
    "boldness": ("bolder", "shyer"),
    "freeze": ("quicker to freeze", "slower to freeze"),
    "appetite": ("hungrier", "less hungry"),
    "libido": ("keener to mate", "less keen to mate"),
    "persistence": ("more persistent", "quicker to give up"),
    "learning": ("quicker learners", "slower learners"),
}
# Why a trait difference between victims and survivors matters: (trait, cause, victims lower?)
MECHANISMS = {
    ("mass", "starvation", True): "small bodies hold less fat and run out of reserves first",
    ("mass", "starvation", False): "big bodies burn more energy when food is short",
    ("mass", "predation", True): "small animals are easier to overpower",
    ("speed", "predation", True): "slower animals are easier to run down",
    ("speed", "starvation", False): "fast muscles are expensive to maintain",
    ("vision", "predation", True): "short-sighted prey spot predators later",
    ("vision", "starvation", False): "big eyes and brains are costly to run",
    ("vigilance", "predation", True): "less watchful prey get taken by surprise",
    ("vigilance", "starvation", False): "watchful animals spend less time eating",
    ("insulation", "starvation", True): "thin fur costs extra energy to keep warm",
    ("insulation", "starvation", False): "thick fur overheats and wastes energy in warm weather",
    ("insulation", "dehydration", False): "thick fur overheats in the heat",
    ("stamina", "starvation", False): "endurance muscle is expensive to maintain",
    ("litter", "starvation", False): "mothers of big litters wear themselves out",
    ("boldness", "predation", False): "bold animals let predators get too close before running",
    ("boldness", "starvation", True): "shy animals flee at every shadow and lose feeding time",
    ("freeze", "predation", False): "freezers that sit tight too long get caught",
    ("freeze", "predation", True): "animals that bolt instead of freezing catch a predator's eye",
    ("appetite", "starvation", True): "animals that wait too long to eat run out of reserves",
    ("appetite", "predation", False): "hungry animals spend more time out feeding, exposed",
    ("libido", "starvation", False): "animals that put mating before eating run down their reserves",
    ("persistence", "starvation", True): "hunters that give up chases too early catch too little",
    ("persistence", "starvation", False): "long chases burn more energy than they win",
    ("learning", "starvation", False): "a fast-learning brain is expensive to run",
    ("learning", "predation", True): "slow learners keep going back to dangerous places",
}
# What each decision option means, for explanations ("where instincts said X, they now Y").
OPTION_VERBS = {"eat": "eat", "drink": "drink", "mate": "look for a mate", "rest": "rest",
                "explore": "explore", "flee": "flee", "freeze": "freeze", "hide": "take cover"}
CAUSE_PHRASES = {
    "starvation": "starvation", "predation": "predation", "dehydration": "thirst",
    "old age": "old age", "disease": "disease", "fighting": "territorial fights", "emigrated": "emigration",
    "removed": "removal by you",
}


def _pct(x: float) -> str:
    if 0 < x < 0.01:
        return f"{x * 100:.1f}%"
    if 0.99 < x < 1:
        return f"{x * 100:.1f}%"
    return f"{round(x * 100)}%"


def _signed(x: float) -> str:
    return f"{'+' if x >= 0 else '−'}{abs(round(x * 100))}%"


def _plural(name: str) -> str:
    return name + ("es" if name.endswith(("s", "x", "z", "ch", "sh")) else "s")


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _when(f: float) -> str:
    season = SEASONS[int(f * 4) % 4]
    part = (f * 4) % 1
    return ("early " if part < 0.34 else "mid " if part < 0.67 else "late ") + season


class Window:
    """Everything that happened to one species between two snapshots."""

    def __init__(self, sp):
        self.deaths = Counter()
        self.juvenile_deaths = Counter()
        self.trait_sums = {}  # cause -> {trait: sum}
        self.lum_sums = Counter()  # cause -> sum of coat luminance
        self.allele_deaths = {}  # cause -> {symbol: [copies per allele]}
        self.parents = 0
        self.parent_traits = {t: 0.0 for t in sp.traits}
        self.parent_alleles = {g.symbol: [0] * len(g.alleles) for g in sp.genes}
        self.births = 0
        self.arrivals = 0  # immigrants and animals added with the sandbox
        self.lethal = 0
        self.weak = 0
        self._sp = sp

    def death(self, a, cause: str) -> None:
        self.deaths[cause] += 1
        if not a.adult:
            self.juvenile_deaths[cause] += 1
        sums = self.trait_sums.setdefault(cause, {t: 0.0 for t in self._sp.traits})
        for t, v in a.traits.items():
            sums[t] += v
        self.lum_sums[cause] += a.coat_lum
        alleles = self.allele_deaths.setdefault(cause, {g.symbol: [0] * len(g.alleles) for g in self._sp.genes})
        for g in self._sp.genes:
            for x in a.genotype[g.symbol]:
                alleles[g.symbol][g.alleles.index(x)] += 1

    def birth(self, mother, father_traits, father_genotype) -> None:
        self.births += 1
        self.parents += 1
        for t in self.parent_traits:
            self.parent_traits[t] += 0.5 * (mother.traits[t] + father_traits[t])
        for g in self._sp.genes:
            counts = self.parent_alleles[g.symbol]
            for x in mother.genotype[g.symbol] + father_genotype[g.symbol]:
                counts[g.alleles.index(x)] += 0.5

    @staticmethod
    def merge(windows: list["Window"], sp) -> "Window":
        out = Window(sp)
        for w in windows:
            out.deaths.update(w.deaths)
            out.juvenile_deaths.update(w.juvenile_deaths)
            out.lum_sums.update(w.lum_sums)
            for cause, sums in w.trait_sums.items():
                dst = out.trait_sums.setdefault(cause, {t: 0.0 for t in sp.traits})
                for t, v in sums.items():
                    dst[t] += v
            for cause, alleles in w.allele_deaths.items():
                dst = out.allele_deaths.setdefault(cause, {g.symbol: [0] * len(g.alleles) for g in sp.genes})
                for sym, counts in alleles.items():
                    dst[sym] = [a + b for a, b in zip(dst[sym], counts)]
            out.parents += w.parents
            for t, v in w.parent_traits.items():
                out.parent_traits[t] += v
            for sym, counts in w.parent_alleles.items():
                out.parent_alleles[sym] = [a + b for a, b in zip(out.parent_alleles[sym], counts)]
            out.births += w.births
            out.arrivals += w.arrivals
            out.lethal += w.lethal
            out.weak += w.weak
        return out


class Narrator:
    def __init__(self, world):
        self.w = world
        self.snaps: deque = deque(maxlen=KEEP)
        self.log: list[dict] = []
        self._next = 1
        self._cooldown: dict[str, int] = {}
        self.reset_windows()
        self._reset_brains()

    def _reset_brains(self) -> None:
        n = len(self.w.species)
        since = getattr(self.w, "tick", 0)
        self._brain_acc = [[Counter(), 0, since] for _ in range(n)]  # pairs, decisions, since
        self._brain_next = [0] * n  # index of the next milestone to report
        self._brain_base: list[float | None] = [None] * n  # the founders' agreement
        self.brain_state: list[dict | None] = [None] * n

    def reset_windows(self) -> None:
        self.windows = [Window(s) for s in self.w.species]

    # ------------------------------------------------------------------ bookkeeping

    def snapshot(self) -> dict:
        w = self.w
        groups = [[] for _ in w.species]
        for a in w.animals:
            groups[a.si].append(a)
        species = []
        for si, (sp, members) in enumerate(zip(w.species, groups)):
            n = len(members)
            freqs = {}
            for g in sp.genes:
                counts = [0] * len(g.alleles)
                for a in members:
                    for x in a.genotype[g.symbol]:
                        counts[g.alleles.index(x)] += 1
                freqs[g.symbol] = [c / (2 * n) for c in counts] if n else None
            adults = [a for a in members if a.adult]
            species.append({
                "n": n,
                "adults": len(adults),
                "freqs": freqs,
                "traits": {t: sum(a.traits[t] for a in adults) / len(adults) for t in sp.traits} if adults else None,
                "breeding": {t: sp.traits[t].mean * sum(a.breeding[t] for a in members) / n for t in sp.traits} if n else None,
                "lum": sum(a.coat_lum for a in members) / n if n else None,
                "energy": sum(a.energy / a.max_energy for a in members) / n if n else None,
                "vigor": sum(a.vigor for a in members) / n if n else None,
                "het": sum(a.het for a in members) / (n * max(1, sp.markers)) if n else None,
                "window": self.windows[si],
            })
        total_k = float(w.veg.K.sum()) or 1.0
        return {
            "tick": w.tick,
            "env": dict(w.env),
            "cover": w.veg.total() / total_k,
            "effects": {e["kind"] for e in w.effects},
            "species": species,
        }

    def observe(self) -> None:
        snap = self.snapshot()
        self.reset_windows()
        prev = self.snaps[-1] if self.snaps else None
        self.snaps.append(snap)
        if prev is None:
            return
        if snap["env"]["season"] != prev["env"]["season"] and self.w.cfg.seasons:
            self._season(snap)
        tpd = self.w.cfg.ticks_per_day
        for si in range(len(self.w.species)):
            if self.w.brains is not None:
                self._brains(si)
                if self.w.cfg.learning:
                    self._learning(si)
            self._extinction(si, snap, prev)
            day_ago = self._ago(tpd)
            if day_ago is not None:
                self._population(si, day_ago, snap)
            days3 = self._ago(3 * tpd)
            if days3 is not None:
                self._alleles(si, days3, snap)
                self._traits(si, days3, snap)
                self._diversity(si, days3, snap)

    def _verb(self, si: int, option: int) -> str:
        from .brain import OPTIONS  # local import keeps this module free of numpy
        name = OPTIONS[option]
        if name == "eat" and self.w.species[si].diet != "plants":
            return "hunt"
        return OPTION_VERBS[name]

    def _brains(self, si: int) -> dict | None:
        """Compare evolved brains with the founders' instincts, a couple of days at a time."""
        w = self.w
        acc = self._brain_acc[si]
        acc[0].update(w.choice_pairs[si])
        acc[1] += w.decisions[si]
        w.choice_pairs[si].clear()
        w.decisions[si] = 0
        if w.tick - acc[2] < BRAIN_CHECK_DAYS * w.cfg.ticks_per_day:
            return None
        pairs, n = acc[0], acc[1]
        self._brain_acc[si] = [Counter(), 0, w.tick]
        if n < 300:
            return None
        changed = sum(pairs.values())
        agree = 1.0 - changed / n
        top = [{"instinct": self._verb(si, a), "now": self._verb(si, b), "share": k / changed}
               for (a, b), k in pairs.most_common(3)] if changed else []
        first = self._brain_base[si] is None
        if first:
            self._brain_base[si] = agree  # the founding generation sets the baseline
        self.brain_state[si] = {"agreement": agree, "changes": top, "decisions": n, "founders": self._brain_base[si]}
        if first:
            return None
        k = self._brain_next[si]
        if k >= len(BRAIN_MILESTONES) or agree >= BRAIN_MILESTONES[k]:
            return None
        while k < len(BRAIN_MILESTONES) and agree < BRAIN_MILESTONES[k]:
            k += 1
        self._brain_next[si] = k
        sp = w.species[si]
        names = _plural(sp.name)
        n_now = sum(1 for a in w.animals if a.si == si)
        text = (f"In clear-cut situations, {names} now make the choice their founders' instincts would only "
                f"{_pct(agree)} of the time (over the last {BRAIN_CHECK_DAYS} days), down from "
                f"{_pct(self._brain_base[si])} in the first generation. ")
        if top:
            t = top[0]
            text += (f"The commonest change: where instinct would have them {t['instinct']}, they now "
                     f"{t['now']} instead ({_pct(t['share'])} of changed choices). ")
        text += ("Every connection in the brain is set by a pair of genes, shuffled at mating and now and then "
                 "mutated, so behaviour evolves like any other trait. ")
        if n_now < 30:
            text += (f"With only {n_now} {names}, much of this is likely chance (genetic drift) rather than "
                     "adaptation; watch whether survival improves.")
        else:
            text += ("Changes of mind that help animals survive and breed spread by natural selection; "
                     "harmful ones die out with their owners.")
        title = (f"{names.capitalize()} are starting to think differently" if k == 1
                 else f"{names.capitalize()}' brains keep evolving")
        return self._emit(icon="🧠", title=title, text=text, species=sp.name, level="minor" if k == 1 else "major",
                          chart={"tab": "brains", "species": sp.name})

    def _habits(self, si: int) -> dict | None:
        """Latest population averages of lifetime learning, from the history."""
        row = self.w.history["species"][self.w.species[si].name].get("brain")
        if not row or not row["learned"]:
            return None
        last = {k: next((v for v in reversed(series[-5:]) if v is not None), None) for k, series in row.items()}
        return last if last["learned"] is not None else None

    def _habit_phrases(self, si: int, h: dict) -> list[str]:
        sp = self.w.species[si]
        eat = "eating" if sp.diet == "plants" else "hunting"
        out = []
        if h["eat_born"] is not None and h["eat_now"] is not None and abs(h["eat_now"] - h["eat_born"]) >= 0.04:
            out.append(f"to start {eat} {'sooner' if h['eat_now'] < h['eat_born'] else 'later'} (at {_pct(h['eat_now'])} "
                       f"hunger instead of the {_pct(h['eat_born'])} they were born with)")
        if h["drink_born"] is not None and h["drink_now"] is not None and abs(h["drink_now"] - h["drink_born"]) >= 0.04:
            out.append(f"to drink {'sooner' if h['drink_now'] < h['drink_born'] else 'later'} (at {_pct(h['drink_now'])} "
                       f"thirst instead of {_pct(h['drink_born'])})")
        if h["flee_born"] is not None and h["flee_now"] is not None and abs(h["flee_now"] - h["flee_born"]) >= 0.06:
            out.append(f"to flee {'earlier' if h['flee_now'] > h['flee_born'] else 'later'} (when a predator is within "
                       f"{h['flee_now']:.2f}× their inborn flight distance, not {h['flee_born']:.2f}×)")
        return out

    def _learning(self, si: int) -> dict | None:
        """Report what a species has been learning during life, now and then."""
        h = self._habits(si)
        if h is None or h["differs"] is None:
            return None
        phrases = self._habit_phrases(si, h)
        if (not phrases and h["differs"] < 0.15) or not self._ready(f"learning{si}", 4 * self.w.cfg.ticks_per_day):
            return None
        sp = self.w.species[si]
        names = _plural(sp.name)
        text = (f"On average {names} now decide differently from how they were born in {_pct(h['differs'])} of "
                "clear-cut situations. ")
        if phrases:
            text += f"They have learned {_join(phrases)}. "
        text += ("None of this is hard-coded: each animal rewires its own network from the rewards and punishments it "
                 "experiences (a reward prediction error, like dopamine). The lessons die with them, but faster "
                 "learners leave more young, so evolution can favour the ability to learn.")
        return self._emit(icon="🎓", title=f"{names.capitalize()} are learning from experience", text=text,
                          species=sp.name, level="minor", chart={"tab": "brains", "species": sp.name})

    def _ago(self, ticks: int):
        """The newest snapshot at least ``ticks`` old, or None."""
        now = self.snaps[-1]["tick"]
        for s in reversed(self.snaps):
            if now - s["tick"] >= ticks:
                return s
        return None

    def _between(self, old: dict, new: dict) -> list[dict]:
        return [s for s in self.snaps if old["tick"] < s["tick"] <= new["tick"]]

    def _merged(self, si: int, old: dict, new: dict) -> Window:
        return Window.merge([s["species"][si]["window"] for s in self._between(old, new)], self.w.species[si])

    def _ready(self, key: str, ticks: int) -> bool:
        if self._cooldown.get(key, -1) > self.w.tick:
            return False
        self._cooldown[key] = self.w.tick + ticks
        return True

    def _emit(self, *, icon, title, text, species=None, level="major", chart=None) -> dict:
        env = self.w.env
        item = {
            "id": self._next, "tick": self.w.tick, "day": env["day"], "hours": env["hours"],
            "season": env["season"], "year": env["year"], "level": level, "icon": icon,
            "species": species, "title": title, "text": text, "chart": chart,
        }
        self._next += 1
        self.log.append(item)
        if len(self.log) > LOG_LIMIT:
            del self.log[0]
        return item

    def since(self, last_id: int) -> list[dict]:
        return [i for i in self.log if i["id"] > last_id]

    # ------------------------------------------------------------------ helpers for explanations

    def _span(self, old: dict, new: dict) -> str:
        hours = (new["tick"] - old["tick"]) / self.w.cfg.ticks_per_day * 24
        if hours < 20:
            return f"{max(1, round(hours))} hours"
        days = hours / 24
        return "day" if days < 1.5 else f"{days:.0f} days"

    def _generation(self, si: int) -> float:
        sp = self.w.species[si]
        cfg = self.w.cfg
        gen = sp.maturity + sp.gestation
        a, b = sp.breeding
        window = (b - a) % 1.0 or 1.0
        if cfg.seasons and window < 0.35:
            gen = max(gen, cfg.ticks_per_day * cfg.days_per_year)
        return gen

    def _harmonic_n(self, si: int, old: dict, new: dict) -> float:
        ns = [max(1, s["species"][si]["n"]) for s in self._between(old, new)] or [max(1, new["species"][si]["n"])]
        return len(ns) / sum(1 / n for n in ns)

    def _drift_sd(self, p: float, si: int, old: dict, new: dict) -> float:
        t = max(0.05, (new["tick"] - old["tick"]) / self._generation(si))
        ne = max(1.0, NE_RATIO * self._harmonic_n(si, old, new))
        return math.sqrt(max(p * (1 - p), 1e-6) * min(1.0, t / (2 * ne)))

    def _breeding_season(self, si: int) -> str:
        a, b = self.w.species[si].breeding
        return f"{_when(a)} to {_when((b - 0.01) % 1.0)}"

    def _allele_label(self, si: int, symbol: str, k: int) -> tuple[str, object]:
        gene = next(g for g in self.w.species[si].genes if g.symbol == symbol)
        labels = gene.labels or gene.alleles
        return labels[k], gene

    def _predator_counts(self, si: int, snap: dict) -> int:
        return sum(snap["species"][j]["n"] for j in self.w.predators_of[si])

    def _prey_counts(self, si: int, snap: dict) -> int:
        return sum(snap["species"][j]["n"] for j in self.w.prey_of[si])

    # ------------------------------------------------------------------ detectors

    def _season(self, snap: dict) -> None:
        season = snap["env"]["season"]
        text = {
            "spring": "Plants regrow quickly and rabbits begin breeding. Fox cubs born in late winter are being weaned.",
            "summer": "Plant growth peaks. Young rabbits disperse to find homes, and the heat raises water needs and makes thick fur a burden.",
            "autumn": "Growth slows and rabbits stop breeding, so their numbers usually peak about now before falling through winter.",
            "winter": "Plants barely grow and the cold raises every animal's energy bill, so expect starvation. "
                      "Snow makes pale coats better camouflage than dark ones. Foxes mate in late winter.",
        }[season]
        icon = {"spring": "🌱", "summer": "☀️", "autumn": "🍂", "winter": "❄️"}[season]
        self._emit(icon=icon, title=f"{season.capitalize()} has arrived", text=text, level="info")

    def _extinction(self, si: int, snap: dict, prev: dict) -> None:
        sp = self.w.species[si]
        n, before = snap["species"][si]["n"], prev["species"][si]["n"]
        if n == 0 and before > 0:
            old = self._ago(2 * self.w.cfg.ticks_per_day) or self.snaps[0]
            win = self._merged(si, old, snap)
            causes = self._cause_list(win)
            text = f"The last {sp.name} is gone. Over the final days they were lost to {causes}."
            if self.w.prey_of[si]:
                text += f" Only {self._prey_counts(si, snap)} prey animals remained to hunt."
            if self.w.cfg.immigration > 0:
                days = 1 / self.w.cfg.immigration / self.w.cfg.ticks_per_day
                text += f" Immigrants arrive about every {days:.0f} days, so they may recolonise, but a lone newcomer needs a mate."
            self._emit(icon="💀", title=f"{_plural(sp.name).capitalize()} have died out", text=text, species=sp.name,
                       chart={"tab": "populations"})
        elif 0 < n <= 4 and before > 4 and self._ready(f"rare:{si}", 3 * self.w.cfg.ticks_per_day):
            self._emit(icon="⚠️", title=f"Only {n} {_plural(sp.name)} left",
                       text=f"{_plural(sp.name).capitalize()} are close to local extinction. With so few animals, "
                            "one unlucky death or a failed season could finish them off, and inbreeding becomes likely.",
                       species=sp.name, chart={"tab": "populations"})

    def _cause_list(self, win: Window) -> str:
        total = sum(win.deaths.values())
        if not total:
            return "nothing in particular"
        parts = [f"{CAUSE_PHRASES.get(c, c)} ({_pct(k / total)})" for c, k in win.deaths.most_common(3)]
        return _join(parts)

    def _population(self, si: int, old: dict, new: dict, force: bool = False) -> dict | None:
        sp = self.w.species[si]
        n0, n1 = old["species"][si]["n"], new["species"][si]["n"]
        if n0 == 0 and n1 == 0:
            return None
        change = (n1 - n0) / max(1, n0)
        falling = change <= -0.2 and n0 >= 15
        rising = change >= 0.4 and n0 >= 10
        if not force and not (falling or rising):
            return None
        if not force and not self._ready(f"pop:{si}:{'down' if falling else 'up'}", self.w.cfg.ticks_per_day):
            return None
        win = self._merged(si, old, new)
        deaths = sum(win.deaths.values())
        name, names = sp.name, _plural(sp.name)
        lead = f"From {n0} to {n1} over the last {self._span(old, new)}: {deaths} died and {win.births} were born"
        lead += f", and {win.arrivals} arrived from elsewhere." if win.arrivals else "."
        if deaths:
            lead += f" Deaths were from {self._cause_list(win)}."
        reasons = self._reasons(si, old, new, win, deaths)
        if change < -0.02:
            title = f"{names.capitalize()} fell {abs(round(change * 100))}%"
            icon = "📉"
        elif change > 0.02:
            title = f"{names.capitalize()} grew {round(change * 100)}%"
            icon = "📈"
            reasons = self._growth_reasons(si, old, new, win)
        else:
            title = f"{names.capitalize()} are holding steady"
            icon = "⚖️"
        text = " ".join([lead] + reasons)
        return self._emit(icon=icon, title=title, text=text, species=name, chart={"tab": "populations"},
                          level="major" if not force else "report")

    def _reasons(self, si: int, old: dict, new: dict, win: Window, deaths: int) -> list[str]:
        sp = self.w.species[si]
        env, s0, s1 = new["env"], old["species"][si], new["species"][si]
        out = []
        if not deaths:
            return out
        share = {c: k / deaths for c, k in win.deaths.items()}
        if share.get("starvation", 0) >= 0.3:
            starved = win.deaths["starvation"]
            if sp.diet == "plants":
                cover0, cover1 = old["cover"], new["cover"]
                if cover1 >= 0.5:
                    text = (f"There is still grass overall (plant cover {_pct(cover1)}), so these were local shortages "
                            "around crowded burrows and animals too weak to travel to better pasture.")
                elif cover1 < cover0 - 0.03:
                    text = f"Food ran short: plant cover fell from {_pct(cover0)} to {_pct(cover1)} of capacity."
                else:
                    text = f"Food is still short: plant cover is only {_pct(cover1)} of capacity."
                if "drought" in new["effects"]:
                    text += " The drought has almost stopped regrowth."
                elif env["growth"] < 0.7 and self.w.cfg.seasons:
                    text += f" {env['season'].capitalize()} has slowed plant growth to {env['growth']:.1f}× its yearly average."
                recent = self._ago(4 * self.w.cfg.ticks_per_day)
                peak = max(s["species"][si]["n"] for s in self.snaps if recent is None or s["tick"] >= recent["tick"])
                if cover1 < 0.5 and peak > 1.3 * max(1, s0["n"]) and peak > 50:
                    text += f" Numbers peaked at {peak} a few days ago, more than the pasture could feed (overgrazing)."
            else:
                prey0 = self._prey_counts(si, old) / max(1, s0["n"])
                prey1 = self._prey_counts(si, new) / max(1, s1["n"])
                text = f"Prey is scarce: there are {prey1:.0f} prey animals per {sp.name}, down from {prey0:.0f}." \
                    if prey1 < prey0 else f"Hunting is hard: there are {prey1:.0f} prey animals per {sp.name}."
            if win.juvenile_deaths["starvation"] >= 0.5 * starved:
                text += " Most of the starved were young animals with small reserves."
            if env["season"] == "winter" and self.w.cfg.seasons:
                text += " The cold raises everyone's energy needs."
            out.append(text)
        if share.get("predation", 0) >= 0.25:
            preds = self._predator_counts(si, new)
            preds0 = self._predator_counts(si, old)
            text = f"Predators took {win.deaths['predation']}: there are {preds} predators now (from {preds0})."
            lum_dead = win.lum_sums["predation"] / max(1, win.deaths["predation"])
            if s1["lum"] is not None and new["env"]["snow"] > 0.5 and lum_dead < s1["lum"] - 0.04:
                text += " On the snow, darker coats stand out, and the victims were darker than average."
            elif s1["lum"] is not None and new["env"]["snow"] < 0.2 and lum_dead > s1["lum"] + 0.04:
                text += " Pale coats stand out against the grass, and the victims were paler than average."
            out.append(text)
        if share.get("dehydration", 0) >= 0.2 and win.deaths["dehydration"] >= 3:
            text = f"{win.deaths['dehydration']} died of thirst."
            if env["temperature"] > 0.5:
                text += " Summer heat raises water loss."
            out.append(text)
        if share.get("old age", 0) >= 0.3 and win.deaths["old age"] >= 3:
            out.append(f"{win.deaths['old age']} simply died of old age: a generation reaching the end of its lifespan.")
        if share.get("disease", 0) >= 0.2:
            out.append(f"Disease killed {win.deaths['disease']}." + (" A plague is raging." if "plague" in new["effects"] else ""))
        if share.get("emigrated", 0) >= 0.2 and win.deaths["emigrated"] >= 2:
            out.append(f"{win.deaths['emigrated']} young {_plural(sp.name)} left the area because every territory was taken.")
        if share.get("removed", 0) >= 0.3:
            out.append(f"You removed {win.deaths['removed']} with the sandbox tool.")
        if win.births < 0.3 * deaths:
            if self.w.cfg.seasons and not self.w.in_season[si]:
                out.append(f"Losses aren't being replaced: {_plural(sp.name)} only breed from {self._breeding_season(si)}.")
            elif s1["energy"] is not None and s1["energy"] < 0.55:
                out.append("Few litters are born: mothers in poor condition reabsorb their embryos.")
        if win.lethal + win.weak >= 3:
            out.append(f"{win.lethal + win.weak} embryos also died before birth from genetic defects.")
        return out

    def _growth_reasons(self, si: int, old: dict, new: dict, win: Window) -> list[str]:
        sp = self.w.species[si]
        out = []
        if win.births and (self.w.in_season[si] or not self.w.cfg.seasons):
            out.append(f"It's the breeding season, and {win.births} young were born.")
        if sp.diet == "plants":
            cover = new["cover"]
            out.append(f"Food is plentiful: plant cover is {_pct(cover)} of capacity." if cover >= 0.5
                       else f"Plant cover is {_pct(cover)} of capacity.")
        else:
            ratio = self._prey_counts(si, new) / max(1, new["species"][si]["n"])
            out.append(f"There are {ratio:.0f} prey animals per {sp.name}" + (", plenty to hunt." if ratio >= 20 else "."))
        if self.w.predators_of[si] and win.deaths["predation"] < 0.15 * max(1, sum(win.deaths.values())):
            out.append("Predators are taking few of them.")
        return out

    def _alleles(self, si: int, old: dict, new: dict, force: bool = False) -> list[dict]:
        sp = self.w.species[si]
        s0, s1 = old["species"][si], new["species"][si]
        if not s1["n"] or not s0["n"]:
            return []
        out = []
        win = None
        candidates = []
        for g in sp.genes:
            f0, f1 = s0["freqs"][g.symbol], s1["freqs"][g.symbol]
            # One story per gene: the allele that was lost, else the one that moved most.
            best = None
            for k in range(len(g.alleles)):
                p0, p1 = f0[k], f1[k]
                lost = p0 > 0 and p1 == 0
                fixed = p0 < 1 and p1 == 1
                score = abs(p1 - p0) + (2 if lost else 1.5 if fixed else 0)
                if best is None or score > best[0]:
                    best = (score, g, k, p0, p1, lost, fixed)
            score, g, k, p0, p1, lost, fixed = best
            if lost or fixed or abs(p1 - p0) >= 0.12 or force:
                candidates.append(best)
        candidates.sort(key=lambda c: -c[0])
        for _, g, k, p0, p1, lost, fixed in candidates[: (1 if force else 3)]:
            if force and abs(p1 - p0) < 0.02 and not (lost or fixed):
                continue
            key = f"allele:{si}:{g.symbol}:{k}:{'lost' if lost else 'fixed' if fixed else 'shift'}"
            if not force and not self._ready(key, (12 if lost or fixed else 3) * self.w.cfg.ticks_per_day):
                continue
            win = win or self._merged(si, old, new)
            out.append(self._explain_allele(si, g, k, p0, p1, lost, fixed, win, old, new, force))
        return out

    def _explain_allele(self, si, g, k, p0, p1, lost, fixed, win, old, new, force) -> dict:
        sp = self.w.species[si]
        label, _ = self._allele_label(si, g.symbol, k)
        names = _plural(sp.name)
        what = f"the {label} allele of the {g.name} gene"
        if lost:
            title = f"The {label} allele ({g.name}) has vanished from {names}"
            icon = "🧬"
        elif fixed:
            title = f"Every {sp.name} now carries only {label} ({g.name})"
            icon = "🧬"
        else:
            verb = "spreading" if p1 > p0 else "declining"
            title = f"{label} ({g.name}) is {verb} in {names}"
            icon = "🧬"
        text = [f"The frequency of {what} went from {_pct(p0)} to {_pct(p1)} over the last {self._span(old, new)}."]
        evidence = []
        if g.lethal == g.alleles[k] and win.lethal:
            evidence.append(f"It is lethal in a double dose: {win.lethal} embryos inherited two copies and died, so carriers "
                            "who mate with other carriers lose a quarter of their litter. Selection purges it.")
        removed = win.allele_deaths.get("removed", {}).get(g.symbol)
        if removed and removed[k] and p1 < p0:
            lost_copies = (p0 - p1) * 2 * max(1, old["species"][si]["n"])
            if removed[k] >= 0.5 * lost_copies:
                evidence.append(f"Most of the lost copies ({removed[k]}) were carried by animals you removed with the sandbox tool.")
        # Over- or under-representation among the dead and among parents.
        pmean = max(0.005, (p0 + p1) / 2)
        best = None
        for cause, alleles in win.allele_deaths.items():
            n = win.deaths[cause]
            # Only compare when enough copies are expected for the difference to mean something.
            if n < 8 or 2 * n * pmean < 5 or cause in ("removed", "emigrated"):
                continue
            q = alleles[g.symbol][k] / (2 * n)
            ratio = q / pmean
            weight = abs(math.log(max(ratio, 1e-3))) * math.sqrt(n)
            if best is None or weight > best[0]:
                best = (weight, cause, ratio, q, n)
        if best and (best[2] >= 1.35 or best[2] <= 0.7):
            _, cause, ratio, q, n = best
            more = ratio > 1
            evidence.append(
                f"Among the {n} that died of {CAUSE_PHRASES[cause]}, {_pct(q)} of gene copies were {label}, against "
                f"{_pct(pmean)} in the population, so carriers were {'over' if more else 'under'}-represented "
                f"({ratio:.1f}×). That points to selection {'against' if more else 'for'} it.")
            if cause == "predation" and g.symbol in [x.symbol for x in sp.genes]:
                snow = new["env"]["snow"]
                ground = "snow" if snow > 0.5 else "green grass"
                evidence.append(f"Coat colour matters for camouflage, and against {ground} some coats are easier for predators to see.")
        if win.parents >= 8 and 2 * win.parents * pmean >= 5:
            q = win.parent_alleles[g.symbol][k] / (2 * win.parents)
            ratio = q / pmean
            if ratio >= 1.35 or ratio <= 0.7:
                evidence.append(f"Carriers also {'produced more' if ratio > 1 else 'produced fewer'} young than expected: "
                                f"{_pct(q)} of the copies among parents were {label}.")
        sd = self._drift_sd(max(min(p0, 0.99), 0.01), si, old, new)
        z = abs(p1 - p0) / sd if sd > 0 else 0
        ne = NE_RATIO * self._harmonic_n(si, old, new)
        expected = (f"with an effective population of about {ne:.0f}, random genetic drift is expected to move an "
                    f"allele like this by about ±{_pct(sd)} over this period")
        if z < 1.5:
            drift = ("Drift may also have played a part: " if evidence else "Chance alone could easily do this: ") + expected + "."
            if (lost or fixed) and min(p0, 1 - p0) < 0.1:
                drift += " It was rare to begin with, so a few carriers dying without offspring was enough."
            evidence.append(drift)
        elif z < 2.5:
            evidence.append(f"That is more than drift usually produces ({expected}), but not by enough to rule chance out.")
        elif not evidence:
            n_max = max(s["species"][si]["n"] for s in self._between(old, new) + [old])
            crashed = new["species"][si]["n"] < 0.6 * n_max
            if crashed:
                evidence.append(
                    f"The change is about {z:.0f}× larger than drift would typically produce in a population of about "
                    f"{ne:.0f}. No cause of death or breeding pattern singles it out, though, and the population crashed "
                    f"from {n_max} to {new['species'][si]['n']}. A crash leaves the future to a few survivors, which "
                    "speeds up drift, so chance may still be the explanation.")
            else:
                evidence.append(f"The change is about {z:.0f}× larger than drift would typically produce in a population "
                                f"of about {ne:.0f}, which points to selection, though no single cause of death or "
                                "breeding pattern stands out.")
        if lost:
            evidence.append("Mutation can occasionally recreate it, or an immigrant may bring it back.")
        return self._emit(icon=icon, title=title, text=" ".join(text + evidence), species=sp.name,
                          chart={"tab": "genes", "species": sp.name, "gene": g.symbol},
                          level="report" if force else "major")

    def _traits(self, si: int, old: dict, new: dict, force: bool = False) -> list[dict]:
        sp = self.w.species[si]
        s0, s1 = old["species"][si], new["species"][si]
        if not s0["breeding"] or not s1["breeding"] or s1["n"] < (5 if force else 15):
            return []
        changes = sorted(((s1["breeding"][t] / s0["breeding"][t] - 1, t) for t in sp.traits), key=lambda c: -abs(c[0]))
        out = []
        for change, t in changes[:1]:
            if abs(change) < (0.01 if force else 0.04):
                continue
            if not force and not self._ready(f"trait:{si}:{t}", 3 * self.w.cfg.ticks_per_day):
                continue
            win = self._merged(si, old, new)
            out.append(self._explain_trait(si, t, change, win, old, new, force))
        return out

    def _explain_trait(self, si, t, change, win, old, new, force) -> dict:
        sp = self.w.species[si]
        spec = sp.traits[t]
        up = change > 0
        names = _plural(sp.name)
        title = f"{names.capitalize()} are evolving {EVOLVE_WORDS.get(t, (f'higher {t}', f'lower {t}'))[0 if up else 1]}"
        text = [f"The genetic average for {spec.label.lower()} (the breeding value, which offspring inherit) "
                f"moved {_signed(change)} over the last {self._span(old, new)}. This is evolution, not just individuals changing."]
        pop = new["species"][si]["traits"] or old["species"][si]["traits"]
        if pop and win.parents >= 8:
            s = win.parent_traits[t] / win.parents / pop[t] - 1
            if abs(s) >= 0.01:
                more, less = COMPARE_WORDS.get(t, ("higher", "lower"))
                text.append(f"Parents of the {win.parents} young born in that time were {_signed(s)} {more if s > 0 else less} "
                            f"than the average adult (the selection differential). With heritability {spec.heritability}, "
                            f"the breeder's equation predicts a {_signed(spec.heritability * s)} shift per generation.")
        if pop:
            best = None
            for cause, sums in win.trait_sums.items():
                n = win.deaths[cause]
                if n < 8 or cause in ("removed", "emigrated"):
                    continue
                rel = sums[t] / n / pop[t] - 1
                if best is None or abs(rel) * math.sqrt(n) > abs(best[1]) * math.sqrt(best[2]):
                    best = (cause, rel, n)
            if best and abs(best[1]) >= 0.03:
                cause, rel, n = best
                more, less = COMPARE_WORDS.get(t, ("higher", "lower"))
                why = MECHANISMS.get((t, cause, rel < 0))
                text.append(f"The {n} that died of {CAUSE_PHRASES[cause]} were {_signed(rel)} "
                            f"{more if rel > 0 else less} than average" + (f": {why}." if why else "."))
        if new["species"][si]["n"] < 40:
            text.append(f"With only {new['species'][si]['n']} animals, chance (drift) may account for part of this.")
        return self._emit(icon="📏", title=title, text=" ".join(text), species=sp.name,
                          chart={"tab": "traits", "species": sp.name, "trait": t},
                          level="report" if force else "major")

    def _diversity(self, si: int, old: dict, new: dict, force: bool = False) -> dict | None:
        sp = self.w.species[si]
        s0, s1 = old["species"][si], new["species"][si]
        if s0["het"] is None or s1["het"] is None or not s0["het"]:
            return None
        drop = 1 - s1["het"] / s0["het"]
        vigor_drop = (s0["vigor"] or 0) - (s1["vigor"] or 0)
        if not force and not (drop >= 0.15 or vigor_drop >= 0.04):
            return None
        if not force and not self._ready(f"div:{si}", 4 * self.w.cfg.ticks_per_day):
            return None
        win = self._merged(si, old, new)
        n = self._harmonic_n(si, old, new)
        text = (f"Marker heterozygosity went from {s0['het']:.2f} to {s1['het']:.2f} while the population averaged about {n:.0f}. "
                f"Small populations lose genetic variation to drift at roughly 1/(2Ne) per generation, and relatives end up "
                f"mating. Mean vigour is {_pct(s1['vigor'])}")
        text += f", and {win.weak} embryos died of inherited defects." if win.weak else "."
        if force and drop < 0.05:
            text = (f"Diversity is stable: marker heterozygosity is {s1['het']:.2f} (was {s0['het']:.2f}) "
                    f"and mean vigour is {_pct(s1['vigor'])}.")
        return self._emit(icon="🧪" if drop >= 0.05 else "🌈", title=f"{_plural(sp.name).capitalize()}: genetic diversity "
                          + ("is falling" if drop >= 0.05 else "check"), text=text, species=sp.name,
                          chart={"tab": "diversity", "species": sp.name}, level="report" if force else "major")

    # ------------------------------------------------------------------ events and reports

    def effect_started(self, effect: dict) -> None:
        effect["start"] = {
            "cover": self.w.veg.total() / (float(self.w.veg.K.sum()) or 1.0),
            "counts": self.w.counts(),
            "deaths": [Counter(d) for d in self.w.deaths],
        }

    def effect_ended(self, effect: dict) -> None:
        start = effect.get("start")
        if not start:
            return
        w = self.w
        cover = w.veg.total() / (float(w.veg.K.sum()) or 1.0)
        counts = w.counts()
        name = {"drought": "The drought", "cold": "The cold snap", "plague": "The plague"}.get(effect["kind"], "The event")
        parts = []
        for si, sp in enumerate(w.species):
            if effect["kind"] == "plague" and effect["species"] != si:
                continue
            delta = w.deaths[si] - start["deaths"][si]
            top = ", ".join(f"{k} {CAUSE_PHRASES.get(c, c)}" for c, k in delta.most_common(2))
            parts.append(f"{_plural(sp.name)} went from {start['counts'][si]} to {counts[si]}" + (f" ({top})" if top else ""))
        text = f"{name} is over. "
        if effect["kind"] == "drought":
            text += f"Plant cover went from {_pct(start['cover'])} to {_pct(cover)}. "
        text += "During it, " + _join(parts) + "."
        self._emit(icon="📋", title=f"{name} has ended", text=text, level="major", chart={"tab": "populations"})

    def report(self) -> list[dict]:
        """An on-demand explanation of the last day (populations) and last three days (genetics)."""
        if not self.snaps:
            self.observe_now()
        # A provisional snapshot whose window is the one still accumulating.
        now = self.snapshot()
        self.snaps.append(now)
        tpd = self.w.cfg.ticks_per_day
        day_ago = self._ago(tpd) or self.snaps[0]
        days3 = self._ago(3 * tpd) or self.snaps[0]
        items = []
        for si in range(len(self.w.species)):
            for item in (self._population(si, day_ago, now, force=True),
                         *self._alleles(si, days3, now, force=True),
                         *self._traits(si, days3, now, force=True),
                         self._diversity(si, days3, now, force=True)):
                if item:
                    items.append(item)
        self.snaps.pop()
        # Reports are answers, not news: keep them out of the event log.
        ids = {i["id"] for i in items}
        self.log = [i for i in self.log if i["id"] not in ids]
        return items

    def observe_now(self) -> None:
        self.snaps.append(self.snapshot())
        self.reset_windows()

    # ------------------------------------------------------------------ end-of-run summary

    @staticmethod
    def _cycles(counts: list) -> int:
        """Boom-and-bust cycles: rises to a peak followed by a fall of at least 30%."""
        peaks, rising = 0, True
        hi = lo = counts[0] if counts else 0
        for v in counts:
            if v is None:
                continue
            if rising:
                if v > hi:
                    hi = v
                elif hi >= 20 and v < 0.7 * hi:
                    peaks += 1
                    rising, lo = False, v
            else:
                if v < lo:
                    lo = v
                elif v > 1.4 * max(lo, 1) and v - lo >= 10:
                    rising, hi = True, v
        return peaks

    def summary(self) -> dict:
        """A breakdown of the whole run: what happened to each species, and why."""
        w = self.w
        cfg = w.cfg
        tpd = cfg.ticks_per_day
        year = tpd * cfg.days_per_year
        h = w.history
        ticks = h["tick"]
        elapsed = w.tick - w.start_tick
        day_of = lambda t: int((t + tpd // 4) // tpd) + 1  # noqa: E731
        finished = w.finished
        story, species_out = [], []

        if finished:
            headline = f"{_plural(finished['species']).capitalize()} died out on day {finished['day']}"
            outcome = (f" It ended on day {finished['day']} ({finished['season']} of year {finished['year']}), "
                       f"when the last {finished['species']} died.")
        else:
            headline = f"The story so far: day {w.env['day']}"
            outcome = " It is still running."
        story.append(f"The simulation ran for {elapsed / tpd:.0f} days, about {elapsed / year:.1f} simulated years.{outcome}")

        groups = [[] for _ in w.species]
        for a in w.animals:
            groups[a.si].append(a)
        for si, sp in enumerate(w.species):
            row = h["species"][sp.name]
            counts = row["count"]
            if not counts:
                continue
            names = _plural(sp.name)
            peak = max(counts)
            peak_i = counts.index(peak)
            low = min(counts[peak_i:]) if peak_i < len(counts) else counts[-1]
            low_i = peak_i + counts[peak_i:].index(low)
            after_peak = low < 0.9 * peak
            if not after_peak:
                # The peak came at (or near) the end: report the run's low point instead,
                # skipping the founding days.
                start = len(counts) // 20
                low = min(counts[start:])
                low_i = start + counts[start:].index(low)
            deaths = w.deaths[si]
            total_deaths = sum(deaths.values())
            cycles = self._cycles(counts)

            traits = []
            for t, spec in sp.traits.items():
                series = [v for v in row["traits"][t]["genetic"] if v is not None]
                if len(series) >= 2:
                    traits.append({"trait": t, "label": spec.label, "unit": spec.unit, "start": round(series[0], 3),
                                   "end": round(series[-1], 3), "change": round(series[-1] / series[0] - 1, 4)})
            alleles = []
            for g in sp.genes:
                freqs = [f for f in row["genes"][g.symbol]["freq"] if f is not None]
                if len(freqs) < 2:
                    continue
                labels = g.labels or g.alleles
                for k, label in enumerate(labels):
                    a0, a1 = freqs[0][k], freqs[-1][k]
                    status = "lost" if a0 > 0 and a1 == 0 else "fixed" if a0 < 1 and a1 == 1 else "changed"
                    if status != "changed" or abs(a1 - a0) >= 0.05:
                        alleles.append({"gene": g.name, "symbol": g.symbol, "allele": label,
                                        "start": round(a0, 3), "end": round(a1, 3), "status": status})
            he = [v for v in row["diversity"]["he"] if v is not None]
            vigor = [v for v in row["vigor"] if v is not None]

            rec = {k: v for k, v in w.records[si].items()}
            for a in groups[si]:
                for key, value in (("oldest", a.age), ("parent", a.children), ("hunter", a.kills)):
                    if value and (rec[key] is None or value > rec[key][0]):
                        rec[key] = (value, a.id, None)
            from .world import animal_name  # local import: world imports this module
            records = {}
            if rec["oldest"]:
                records["oldest"] = {"name": rec["oldest"][2] or animal_name(rec["oldest"][1]),
                                     "days": round(rec["oldest"][0] / tpd, 1)}
            if rec["parent"]:
                records["parent"] = {"name": rec["parent"][2] or animal_name(rec["parent"][1]), "young": rec["parent"][0]}
            if rec["hunter"]:
                records["hunter"] = {"name": rec["hunter"][2] or animal_name(rec["hunter"][1]), "kills": rec["hunter"][0]}

            species_out.append({
                "name": sp.name, "icon": sp.icon, "final": counts[-1], "peak": peak, "peak_day": day_of(ticks[peak_i]),
                "low": low, "low_day": day_of(ticks[low_i]), "low_after_peak": after_peak,
                "births": w.births[si], "stillborn": w.stillborn[si],
                "deaths": dict(deaths), "generations": w.max_generation[si], "cycles": cycles,
                "traits": traits, "alleles": alleles,
                "diversity": [he[0], he[-1]] if he else None, "vigor": [vigor[0], vigor[-1]] if vigor else None,
                "records": records, "hunts": dict(w.hunts[si]), "social": dict(w.social[si]),
            })

            # The story of this species.
            if after_peak:
                text = (f"{names.capitalize()} peaked at {peak} on day {day_of(ticks[peak_i])} and fell to {low} "
                        f"by day {day_of(ticks[low_i])}")
            else:
                text = (f"{names.capitalize()} were at their lowest, {low}, on day {day_of(ticks[low_i])} and "
                        f"peaked at {peak} on day {day_of(ticks[peak_i])}")
            text += f", going through {cycles} boom-and-bust cycle{'s' if cycles != 1 else ''}." if cycles else "."
            if total_deaths:
                main = deaths.most_common(1)[0][0]
                causes = _join([f"{CAUSE_PHRASES.get(c, c)} ({_pct(k / total_deaths)})" for c, k in deaths.most_common(3)])
                text += f" In all {w.births[si]:,} were born and {total_deaths:,} died, mostly of {causes}."
                if main == "starvation":
                    text += " Food, more than predators, kept their numbers in check." if w.predators_of[si] \
                        else " Food supply was the main limit on their numbers."
                elif main == "predation":
                    text += " Predators were the main check on their numbers."
                elif main == "emigrated":
                    text += " Territories were the bottleneck: most young had to leave to find a vacancy."
            evolved = sorted((t for t in traits if abs(t["change"]) >= 0.02), key=lambda t: -abs(t["change"]))[:2]
            if evolved and not w.max_generation[si]:
                text += (" No young were born, so the small shifts in their average traits came only from "
                         "which individuals happened to die, not from evolution.")
            elif evolved:
                parts = []
                for t in evolved:
                    up, down = EVOLVE_WORDS.get(t["trait"], (f"higher {t['trait']}", f"lower {t['trait']}"))
                    parts.append(f"{abs(round(t['change'] * 100))}% {up if t['change'] > 0 else down}")
                text += f" Over {w.max_generation[si]} generations they evolved {_join(parts)}."
            elif w.max_generation[si]:
                text += f" Over {w.max_generation[si]} generations their traits barely changed."
            lost = [f"{a['allele']} ({a['gene']})" for a in alleles if a["status"] == "lost"]
            if lost:
                text += f" They lost the {_join(lost)} allele{'s' if len(lost) > 1 else ''}."
            shifts = sorted((a for a in alleles if a["status"] == "changed"), key=lambda a: -abs(a["end"] - a["start"]))[:1]
            for a in shifts:
                text += f" The {a['allele']} allele of {a['gene']} went from {_pct(a['start'])} to {_pct(a['end'])}."
            if he and he[0] and he[-1] < 0.85 * he[0]:
                text += (f" Genetic diversity fell from {he[0]:.2f} to {he[-1]:.2f}, the mark of small population "
                         "sizes and inbreeding.")
            social = w.social[si]
            if social:
                bits = []
                if social["contest"]:
                    bits.append(f"{social['contest']} territorial fight{'s' if social['contest'] != 1 else ''} "
                                f"({social['takeover']} won by the challenger)")
                if social["stayed as helper"]:
                    bits.append(f"{social['stayed as helper']} daughter{'s' if social['stayed as helper'] != 1 else ''} "
                                "stayed home as helpers")
                if social["inherited"]:
                    bits.append(f"{social['inherited']} helper{'s' if social['inherited'] != 1 else ''} inherited "
                                "a territory")
                if bits:
                    text += f" Family life: {_join(bits)}."
            state = self.brain_state[si]
            if state is not None:
                text += f" By the end their brains matched their founders' instincts in {_pct(state['agreement'])} of clear-cut decisions"
                if state["changes"] and state["agreement"] < 0.97:
                    c = state["changes"][0]
                    text += f"; most often, where instinct would have them {c['instinct']}, they would {c['now']}"
                text += "."
            species_out[-1]["brain"] = state
            habits = self._habits(si) if w.brains is not None and w.cfg.learning else None
            if habits:
                phrases = self._habit_phrases(si, habits)
                if phrases:
                    text += f" During their lives they learned {_join(phrases)}."
            story.append(text)

        if finished:
            ext = [i for i in self.log if i["species"] == finished["species"] and "died out" in i["title"]]
            if ext:
                story.append(ext[-1]["text"])

        step = max(1, len(ticks) // 300)
        series = {
            "tick": ticks[::step],
            "plants": h["plants"][::step],
            "species": {sp.name: h["species"][sp.name]["count"][::step] for sp in w.species},
        }
        moments = [{"day": i["day"], "icon": i["icon"], "title": i["title"], "species": i["species"]}
                   for i in self.log if i["level"] == "major"][-24:]
        return {
            "final": bool(finished), "finished": finished, "headline": headline, "story": story,
            "days": round(elapsed / tpd, 1), "years": round(elapsed / year, 2), "seed": w.seed,
            "species": species_out, "moments": moments, "events": w.event_log, "series": series,
            "end_on_extinction": w.end_on_extinction,
        }

    def restart(self, message: str | None = None) -> None:
        """Forget past snapshots (e.g. after a new founding population), keeping the log."""
        self.snaps.clear()
        self._cooldown.clear()
        self.reset_windows()
        self._reset_brains()
        if message:
            self._emit(icon="🐣", title="A new founding population", text=message, level="info",
                       chart={"tab": "populations"})
