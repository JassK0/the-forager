"""Evidence for the AI analyst: the run's records turned into a compact brief
(text tables) plus a map image, and the daily series behind the Deaths &
births charts.

Everything here is read from what the simulation recorded (engine/ledger.py,
the death counters, the narrator's log); nothing is estimated or guessed, so
the analyst can be asked to back every claim with a day and a number.
"""

import bisect
import struct
import zlib
from collections import Counter

import numpy as np

from .climate import BIOMES
from .ledger import REGIONS, region_name
from .narrator import _plural
from .rodents import VOLE_ENERGY
from .terrain import FOREST, GRASS, ROCK, SAND, TREE, WATER, WALKABLE

CAUSE_SHORT = {"predation": "pred", "starvation": "starv", "dehydration": "thirst", "old age": "old",
               "disease": "disease", "fighting": "fight", "emigrated": "emig", "removed": "removed"}
BIN_CHOICES = (1, 3, 6, 12, 24, 36, 60, 120, 240)  # days per timeline row (3 = a season, 12 = a year)
MAX_TIMELINE = 60  # rows in the whole-run timeline
DETAIL_DAYS = 24  # days shown one by one before the end (or before an extinction)
NOTES = 40  # narrator notes passed on
SCALE = 4  # map pixels per cell


# ---------------------------------------------------------------- small helpers

def _pct(x) -> str:
    return "–" if x is None else f"{round(100 * x)}%"


def _num(x) -> str:
    if x is None:
        return "–"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def _day_of_year(frac: float, dpy: int) -> float:
    return frac * dpy + 1


def _row_at(rows: list, day: float) -> dict:
    """The last row recorded on or before `day`."""
    days = [r["day"] for r in rows]
    i = bisect.bisect_right(days, day) - 1
    return rows[max(0, i)]


def _rows_between(rows: list, d0: float, d1: float) -> list:
    return [r for r in rows if d0 < r["day"] <= d1]


def vole_value(world) -> float:
    """Voles as rabbit-equivalents of food (the same conversion territories use)."""
    for si, sp in enumerate(world.species):
        if sp.eats_voles and world.prey_of[si]:
            prey = world.species[world.prey_of[si][0]]
            return VOLE_ENERGY / (prey.traits["mass"].mean * prey.carcass_energy)
    return 0.0


# ---------------------------------------------------------------- the map image

def encode_png(img: np.ndarray) -> bytes:
    h, w, _ = img.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _box(img, cx, cy, half, color):
    h, w, _ = img.shape
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = color


def _ring(img, cx, cy, half, color, width=2):
    for k in range(width):
        r = half - k
        h, w, _ = img.shape
        x0, x1 = max(0, cx - r), min(w - 1, cx + r)
        y0, y1 = max(0, cy - r), min(h - 1, cy + r)
        if x0 > x1 or y0 > y1:
            continue
        img[y0, x0:x1 + 1] = color
        img[y1, x0:x1 + 1] = color
        img[y0:y1 + 1, x0] = color
        img[y0:y1 + 1, x1] = color


def _cross(img, cx, cy, half, color):
    h, w, _ = img.shape
    for d in range(-half, half + 1):
        for x, y in ((cx + d, cy + d), (cx + d, cy - d), (cx + d + 1, cy + d), (cx + d + 1, cy - d)):
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = color


MAP_KEY = (
    "Map key (top-down, {w}x{h} cells, {s} pixels per cell). Ground: blue = water, pale yellow = sand, "
    "green = grass (bright green = lush, yellow-brown = grazed bare), dark green = forest floor, "
    "very dark green = trees (impassable), grey = rock (impassable), near-white ground = snow. Small white "
    "squares = rabbits; "
    "orange squares with a black border = foxes; hollow orange squares = fox territory centres "
    "(thick = held by a pair, thin = held by one fox). Crosses = deaths in the last 2 days: red = predation, "
    "black = starvation, yellow = thirst, magenta = other causes.{focus} Thin white lines divide the map into "
    "the regions of the region table: columns A-D from left to right, rows 1-4 from top to bottom."
)


def render_map(world, focus: dict | None = None) -> bytes:
    """A top-down picture of the landscape, the animals, territories and recent deaths."""
    t, veg = world.terrain, world.veg
    cover = np.divide(veg.B, veg.K, out=np.zeros_like(veg.B), where=veg.K > 0).clip(0, 1)[..., None]
    types = t.types
    img = np.zeros(types.shape + (3,), dtype=np.float32)
    lush, bare = np.array([79, 154, 58]), np.array([196, 180, 110])
    img[:] = bare + (lush - bare) * cover
    forest = types == FOREST
    img[forest] = (np.array([120, 118, 72]) + (np.array([47, 107, 53]) - np.array([120, 118, 72])) * cover)[forest]
    img[types == WATER] = (59, 125, 216)
    img[types == SAND] = (230, 213, 154)
    img[types == TREE] = (31, 77, 37)
    img[types == ROCK] = (138, 138, 138)
    cl = world.climate
    snow = np.clip((-(world.season_temp + cl.offset) - 0.35) / 0.4, 0.0, 1.0) * cl.snow_fall
    snow[types == WATER] = 0.0
    img += (np.array([242, 245, 250]) - img) * (0.85 * snow)[..., None]
    img = np.repeat(np.repeat(img.astype(np.uint8), SCALE, axis=0), SCALE, axis=1)
    hp, wp, _ = img.shape
    cols, rows = REGIONS
    for c in range(1, cols):
        x = int(c * world.width / cols) * SCALE
        img[:, x] = (img[:, x] * 0.4 + 255 * 0.6).astype(np.uint8)
    for r in range(1, rows):
        y = int(r * world.height / rows) * SCALE
        img[y, :] = (img[y, :] * 0.4 + 255 * 0.6).astype(np.uint8)

    px = lambda v: int(v * SCALE + SCALE / 2)  # noqa: E731
    for si, sp in enumerate(world.species):
        if sp.territory > 0:
            homes = Counter()
            for h in world._holders[si].values():
                homes[(h.home_x, h.home_y)] += 1
            for (hx, hy), k in homes.items():
                _ring(img, px(hx), px(hy), 7, (255, 170, 0), width=3 if k >= 2 else 1)
    for a in world.animals:
        if world.species[a.si].diet == "plants":
            _box(img, px(a.x), px(a.y), 1 if a.adult else 0, (255, 255, 255))
    for a in world.animals:
        if world.species[a.si].diet != "plants":
            _box(img, px(a.x), px(a.y), 3, (0, 0, 0))
            _box(img, px(a.x), px(a.y), 2, (255, 110, 0))
    recent = world.tick - 2 * world.cfg.ticks_per_day
    colours = {"predation": (220, 20, 20), "starvation": (0, 0, 0), "dehydration": (255, 220, 0)}
    for log in world.ledger.last_deaths:
        for d in log:
            if d["tick"] >= recent and d["cause"] not in ("emigrated", "removed"):
                _cross(img, px(d["x"]), px(d["y"]), 2, colours.get(d["cause"], (200, 0, 200)))
    if focus:
        for d in focus["last_deaths"]:
            _cross(img, px(d["x"]), px(d["y"]), 4, (0, 230, 255))
    return encode_png(img)


# ---------------------------------------------------------------- tables

def _species_facts(world) -> list[str]:
    cfg = world.cfg
    tpd, dpy = cfg.ticks_per_day, cfg.days_per_year
    out = []
    for si, sp in enumerate(world.species):
        a, b = sp.breeding
        season = (f"breeds only from day {_day_of_year(a, dpy):.1f} to day {_day_of_year(b, dpy):.1f} of each year"
                  if cfg.seasons else "breeds all year (seasons are off)")
        food = "eats grass and plants" if sp.diet == "plants" else \
            f"hunts {', '.join(p + 's' for p in sp.prey)}"
        if sp.eats_voles:
            food += ", catches voles by 'mousing' (its staple when rabbits are scarce)"
        if sp.eats_fruit:
            food += ", eats wild fruit in late summer and autumn, scavenges carcasses and digs up food it cached"
        line = (f"- {sp.name.capitalize()} {sp.icon}: {food}; needs to drink. {season.capitalize()}; pregnancy lasts "
                f"{sp.gestation / tpd:.2f} days, young mature at about {sp.maturity / tpd:.1f} days and live about "
                f"{sp.lifespan / tpd:.0f} days at most; litters of about {sp.traits['litter'].mean:.0f}. "
                f"Started with {sp.initial}.")
        if sp.territory > 0:
            line += (f" Lives in pair territories (one male + one female, centres about {sp.territory:.0f} cells apart, "
                     "closer where food is rich). Only territory holders breed. Young adults leave home in the "
                     "dispersal season and look for a lone mate's territory or room for a new one; until they find "
                     "one they are 'floaters'. ")
            line += ("Floaters that keep failing eventually emigrate (leave the map: cause 'emig')."
                     if cfg.emigration else "Emigration is switched OFF in this world: floaters never leave.")
            if sp.max_helpers:
                line += " Some daughters stay home as non-breeding helpers and can inherit the territory."
        if world.predators_of[si]:
            line += " Escapes predators by fleeing, freezing or diving into burrows; thumps alarms."
        out.append(line)
    return out


def _settings(world) -> list[str]:
    cfg = world.cfg
    tpd, dpy = cfg.ticks_per_day, cfg.days_per_year
    out = [f"- Map {world.width}x{world.height} cells; seed {world.seed}.",
           f"- Calendar: {tpd} ticks per day, {dpy} days per year. " + (
               "Seasons are 3 days each: days 1-3 of each year are spring, 4-6 summer, 7-9 autumn, 10-12 winter "
               "(cold raises energy needs, plants barely grow, snow)." if cfg.seasons else "Seasons are OFF.")]
    if cfg.immigration > 0:
        out.append(f"- Immigration: on average one new adult of each species arrives at the map edge every "
                   f"{1 / cfg.immigration / tpd:.1f} days ('arrivals' also count animals the user added).")
    else:
        out.append("- Immigration is OFF: a species that dies out cannot come back on its own.")
    cl = world.climate
    out.append(f"- Climate: {cl.spec['label']} ({cl.spec['text']}). Each place has its own temperature and dryness: "
               "they set snow cover, how much and when plants grow (frost stops growth, summer heat makes desert "
               "plants dormant), the water animals get from their food, and the cost of cold (less with thick fur, "
               "in a burrow, or huddled) and of heat (more with thick fur; heat also raises water loss)."
               + (" Temperatures run from -1 (temperate midwinter) to +1 (temperate midsummer); snow lies below about "
                  "-0.35." if not cl.uniform else ""))
    out.append("- Camouflage: how easily an animal is spotted (by a predator, or by prey watching for one) depends on "
               "how different its coat looks from the ground under it, to a fox's or rabbit's eye. Both are dichromats "
               "(red-green colour-blind): they judge lightness, and yellow against blue, so an orange fox matches green "
               "grass. Growing grass is mid-dark green, dead grass pale straw, grazed earth brown, leaf litter dark, "
               "sand pale, snow white. Camouflage works while an animal keeps still; moving gives it away, and at night "
               "colour hardly matters. Animals with the winter-white gene (Mo) moult to a white "
               "coat from mid-autumn to early spring, timed by day length rather than by snow, so they can be "
               "mismatched when snow comes late or not at all.")
    out.append(f"- Brains: {'evolving neural networks' if cfg.brains == 'neural' else 'fixed instincts'}"
               + (", with learning during life." if cfg.brains == "neural" and cfg.learning else "."))
    if cfg.super_animals:
        out.append(f"- SUPER ANIMALS MODE (not realistic): mutation is {cfg.evolution_speed:g}x as frequent, inherited "
                   "variation is larger and more heritable, and traits have no natural ceiling (up to 3x the normal "
                   f"maximum). Brains learn {cfg.learning_speed:g}x as fast. Young start life with "
                   f"{cfg.inherit_learning:.0%} of their parents' know-how (learned brain wiring, the mother's hunting "
                   "skill and wariness), so know-how accumulates across generations as well as genes. Every few hours "
                   "the 3 strongest adults of each species are crowned alphas.")
    if cfg.width * cfg.height < 160 * 100:
        out.append(f"- A small map ({cfg.width}x{cfg.height} cells): few fox territories fit, so chance events matter more.")
    if cfg.mutation != 1.0:
        out.append(f"- Mutation rate x{cfg.mutation:g}.")
    out.append(f"- The run {'stops' if world.end_on_extinction else 'keeps going'} when a species dies out.")
    return out


def _timeline(world, rows: list, si: int, d_from: float, d_to: float, bin_days: float) -> list[str]:
    sp = world.species[si]
    territorial = sp.territory > 0
    meat = sp.diet != "plants"
    head = ["days", "count end (low-high)", "adults F/M", "pregnant", "births", "deaths (by cause)", "arrived",
            "condition"]
    if territorial:
        head += ["territories (paired)", "floaters"]
    if meat:
        head += ["food: rabbits/voles/carrion/caches/fruit"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    start = d_from
    while start < d_to - 1e-9:
        end = min(d_to, start + bin_days)
        r0, r1 = _row_at(rows, start), _row_at(rows, end)
        inside = _rows_between(rows, start, end) or [r1]
        counts = [r["species"][si]["n"] for r in inside]
        s1 = r1["species"][si]
        dl = world.ledger.delta(r1, r0, si)
        deaths = ", ".join(f"{CAUSE_SHORT.get(c, c)} {k}" for c, k in dl["deaths"].most_common()) or "0"
        span = f"{start + 1:g}" if end - start <= 1 else f"{start + 1:g}-{end:g}"
        n_text = f"{s1['n']} ({min(counts)}-{max(counts)})" if len(inside) > 1 else str(s1["n"])
        cells = [span, n_text, f"{s1['af']}/{s1['am']}", str(s1["pregnant"]),
                 str(dl["births"]), deaths, str(dl["arrivals"]), _pct(s1["condition"])]
        if territorial:
            cells += [f"{s1.get('territories', 0)} ({s1.get('paired', 0)})", str(s1.get("floaters", 0))]
        if meat:
            h = dl["hunts"]
            cells += [f"{h['kill'] + h['surplus kill']}/{h['vole']}/{h['scavenged']}/{h['cache']}/{h['fruit']}"]
        out.append("| " + " | ".join(cells) + " |")
        start = end
    return out


def _environment(world, rows: list, d_from: float, d_to: float, bin_days: float) -> list[str]:
    out = ["| days | season | mean temperature (-1 cold, +1 hot) | snow | plant cover (of capacity) | voles | "
           "events |", "|---|---|---|---|---|---|---|"]
    start = d_from
    while start < d_to - 1e-9:
        end = min(d_to, start + bin_days)
        inside = _rows_between(world.ledger.rows, start, end) or [_row_at(rows, end)]
        seasons = []
        for r in inside:
            if r["season"] and r["season"] not in seasons:
                seasons.append(r["season"])
        effects = sorted({e for r in inside for e in r["effects"]})
        span = f"{start + 1:g}" if end - start <= 1 else f"{start + 1:g}-{end:g}"
        mean = lambda k: sum(r[k] for r in inside) / len(inside)  # noqa: E731
        out.append(f"| {span} | {'/'.join(seasons) or '-'} | {mean('temp'):+.2f} | {mean('snow'):.2f} | "
                   f"{_pct(mean('cover'))} | {mean('voles'):,.0f} | {', '.join(effects) or '-'} |")
        start = end
    return out


def _period_mean(rows: list, si: int, d0: float, d1: float, key: str = "pheno"):
    """Average traits of the living over a span of days, weighted by numbers."""
    total, acc = 0, None
    for r in _rows_between(rows, d0, d1):
        sr = r["species"][si]
        if sr.get(key) is None:
            continue
        v = np.array(sr[key]) * sr["n"]
        acc = v if acc is None else acc + v
        total += sr["n"]
    return acc / total if total else None


def _selection(world, rows: list, si: int, d0: float, d1: float) -> list[str]:
    """Traits of the parents and of each cause's victims, against the average animal alive."""
    r0, r1 = _row_at(rows, d0), _row_at(rows, d1)
    s0, s1 = r0["species"][si], r1["species"][si]
    living = _period_mean(rows, si, d0, d1)
    if living is None or "parent_traits" not in s1:
        return []
    names = world.ledger.trait_names[si]
    sp = world.species[si]
    causes = world.ledger.causes
    groups = []
    par = s1["parent_traits"] - s0["parent_traits"]
    if par[-1] >= 5:
        groups.append((f"parents (per young, {int(par[-1])} young)", par[:-1] / par[-1]))
    dead = s1["dead_traits"] - s0["dead_traits"]
    for k in np.argsort(-dead[:, -1]):
        if dead[k, -1] >= 8 and causes[k] != "removed" and len(groups) < 5:
            groups.append((f"died of {causes[k]} ({int(dead[k, -1])})", dead[k, :-1] / dead[k, -1]))
    if not groups:
        return []
    head = ["trait", "living average"] + [g[0] for g in groups]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for j, t in enumerate(names):
        spec = sp.traits[t]
        cells = [spec.label, f"{living[j]:.3g} {spec.unit}".strip()]
        cells += [f"{100 * (g[1][j] / living[j] - 1):+.0f}%" if living[j] else "–" for g in groups]
        out.append("| " + " | ".join(cells) + " |")
    # Camouflage: how much the dead stood out against the ground, against the living.
    gaps = [r["species"][si].get("gap") for r in _rows_between(rows, d0, d1)]
    gaps = [g for g in gaps if g is not None]
    if gaps and "dead_gap" in s1:
        living_gap = sum(gaps) / len(gaps)
        dgap = s1["dead_gap"] - s0["dead_gap"]
        cells = ["Camouflage gap (coat against the ground; lower = better hidden)", f"{living_gap:.2f}"]
        for label, _ in groups:
            if label.startswith("died of "):
                k = causes.index(label[len("died of "):].rsplit(" (", 1)[0])
                n = dead[k, -1]
                cells.append(f"{100 * (dgap[k] / n / living_gap - 1):+.0f}%" if n and living_gap else "–")
            else:
                cells.append("–")
        out.append("| " + " | ".join(cells) + " |")
    return out


def _evolution(world, rows: list, last_day: float) -> list[str]:
    """How each species changed (genes and learned behaviour) and the selection behind it."""
    cfg = world.cfg
    step = next((b for b in BIN_CHOICES if last_day / b <= 10), last_day / 10)
    edges = [0.0]
    while edges[-1] < last_day - 1e-9:
        edges.append(min(last_day, edges[-1] + step))
    h = world.history
    out = ["", "## How they evolved, and the evidence for why",
           "Breeding values are the genetic averages offspring inherit (shown as change from the founders); "
           "selection tables compare the parents of the young born, and the animals that died of each cause, "
           "with the average animal alive at the time (+ = higher). Traits that make parents above average and "
           "victims below average are being favoured by natural selection; with few animals, chance (drift) also "
           f"moves averages. Mutation rate x{cfg.mutation * (cfg.evolution_speed if cfg.super_animals else 1):g}."]
    for si, sp in enumerate(world.species):
        names = world.ledger.trait_names[si]
        base = next((r["species"][si]["genetic"] for r in rows if r["species"][si].get("genetic")), None)
        if base is None:
            continue
        labels = [sp.traits[t].label for t in names]
        gens = world.max_generation[si]
        out += ["", f"### {_plural(sp.name).capitalize()}: {gens} generation{'s' if gens != 1 else ''} born"
                + ("" if gens else " (no young yet, so any change in averages comes only from which animals died or "
                                   "arrived, not from evolution)"),
                "Founders' averages: " + ", ".join(f"{sp.traits[t].label.lower()} {base[j]:.3g} {sp.traits[t].unit}".strip()
                                                    for j, t in enumerate(names)) + ".",
                "", "Breeding values over time (change from the founders):",
                "| day | " + " | ".join(labels) + " |", "|" + "---|" * (len(labels) + 1)]
        for d in edges[1:]:
            g = _row_at(rows, d)["species"][si].get("genetic")
            cells = [f"{100 * (g[j] / base[j] - 1):+.0f}%" if g and base[j] else "–" for j in range(len(names))]
            out.append(f"| {d:g} | " + " | ".join(cells) + " |")
        sel = _selection(world, rows, si, 0, last_day)
        if sel:
            out += ["", "Selection over the whole run (difference from the average animal alive):", *sel]
        if last_day > 2 * cfg.days_per_year:
            recent = _selection(world, rows, si, last_day - cfg.days_per_year, last_day)
            if recent:
                out += ["", f"Selection over the last year (days {last_day - cfg.days_per_year + 1:g}-{last_day:g}):", *recent]
        series = h["species"][sp.name]
        if world.brains is not None and h["tick"]:
            out += ["", "Behaviour over time (averages within each span; 'instinct' = share of clear-cut decisions that "
                        "match the founders' instincts; 'changed in life' = share decided differently from how they "
                        "were born" + (", where 'born with' includes know-how taught by parents" if world.teaching else "")
                    + "; thresholds are the hunger or thirst at which they start eating or drinking, or how close a "
                      "predator gets, in units of their flight distance, before they flee):",
                    "| days | instinct | changed in life | eat at hunger: born/now | drink at thirst: born/now | "
                    "flee distance: born/now |", "|---|---|---|---|---|---|"]
            for d0, d1 in zip(edges, edges[1:]):
                t0, t1 = _row_at(rows, d0)["tick"], _row_at(rows, d1)["tick"]
                lo, hi = bisect.bisect_right(h["tick"], t0), bisect.bisect_right(h["tick"], t1)

                def avg(values):
                    vals = [v for v in values[lo:hi] if v is not None]
                    return sum(vals) / len(vals) if vals else None

                b = series["brain"]
                pair = lambda k, f: f"{f(avg(b[k + '_born']))}/{f(avg(b[k + '_now']))}"  # noqa: E731
                out.append(f"| {d0 + 1:g}-{d1:g} | {_pct(avg(series['instinct']))} | {_pct(avg(b['differs']))} | "
                           f"{pair('eat', _pct)} | {pair('drink', _pct)} | "
                           f"{pair('flee', lambda v: '–' if v is None else f'{v:.2f}x')} |")
        if cfg.super_animals:
            power = [(d, _row_at(rows, d)["species"][si].get("power")) for d in edges[1:]]
            power = [f"day {d:g}: average {p[0]:g}x, top {p[1]:g}x" for d, p in power if p]
            if power:
                out += ["", "Prowess of adults relative to the average founder (speed, senses, size, health, "
                            "condition, skill, learning): " + "; ".join(power) + "."]
    return out


def _bin_days(days: float, cfg) -> float:
    for b in BIN_CHOICES:
        if days / b <= MAX_TIMELINE:
            return b
    return days / MAX_TIMELINE


def _last_deaths_table(deaths: list) -> list[str]:
    out = ["| day hh:00 | animal | sex | age (days) | adult | cause | region | energy | water | injury | role | "
           "pregnant | dependent young |", "|" + "---|" * 13]
    for d in deaths:
        out.append(f"| {d['day']} {d['hour']:02d}:00 | #{d['id']} | {d['sex']} | {d['age_days']} | "
                   f"{'yes' if d['adult'] else 'no'} | {d['cause']} | {d['region']} | {_pct(d['energy'])} | "
                   f"{_pct(d['water'])} | {_pct(d['injury'])} | {d['role'] or '-'} | {'yes' if d['pregnant'] else '-'} | "
                   f"{d['dependents'] or '-'} |")
    return out


def _regions(world, r0: dict, r1: dict, label: str) -> list[str]:
    cols, rows = REGIONS
    t, veg = world.terrain, world.veg
    h, w = t.types.shape
    vv = vole_value(world)
    animals = [Counter() for _ in range(cols * rows)]
    from .ledger import region_of
    for a in world.animals:
        animals[region_of(world, a.x, a.y)][a.si] += 1
    homes = Counter()
    for si, sp in enumerate(world.species):
        if sp.territory > 0:
            for k in {(hh.home_x, hh.home_y) for hh in world._holders[si].values()}:
                homes[region_of(world, *k)] += 1
    died = r1["region_deaths"] - r0["region_deaths"]
    causes = world.ledger.causes
    names = [s.name for s in world.species]
    head = ["region", "water", "trees/rock", "grass+forest", "plant cover", "voles"] + \
           [f"{_plural(n)} now" for n in names] + ["territories", f"deaths {label}"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k in range(cols * rows):
        c, r = k % cols, k // cols
        x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
        y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
        types = t.types[y0:y1, x0:x1]
        n = types.size
        kk = veg.K[y0:y1, x0:x1].sum()
        cover = veg.B[y0:y1, x0:x1].sum() / kk if kk > 0 else None
        voles = world.voles.V[y0:y1, x0:x1].sum()
        bits = []
        for si, name in enumerate(names):
            parts = [f"{CAUSE_SHORT.get(causes[ci], causes[ci])} {died[si, ci, k]}"
                     for ci in range(len(causes)) if died[si, ci, k]]
            if parts:
                bits.append(f"{_plural(name)}: " + ", ".join(parts))
        cells = [region_name(k), _pct((types == WATER).sum() / n), _pct(((types == TREE) | (types == ROCK)).sum() / n),
                 _pct(((types == GRASS) | (types == FOREST)).sum() / n), _pct(cover),
                 f"{voles:,.0f} (≈{voles * vv:.0f} rabbits of food)" if vv else f"{voles:,.0f}"]
        cells += [str(animals[k][si]) for si in range(len(names))]
        cells += [str(homes[k]), "; ".join(bits) or "none"]
        out.append("| " + " | ".join(cells) + " |")
    walk = WALKABLE[t.types].mean()
    out.append(f"\nWalkable land: {_pct(walk)} of the map.")
    out += ["", *_adaptation(world)]
    return out


def _adaptation(world) -> list[str]:
    """Living conditions now, and how the animals living there are adapted, region by region."""
    cols, rows = REGIONS
    cl = world.climate
    biomes = cl.biomes(world.terrain)
    t = world.terrain
    h, w = t.types.shape
    groups = [[[] for _ in world.species] for _ in range(cols * rows)]
    from .ledger import region_of
    for a in world.animals:
        groups[region_of(world, a.x, a.y)][a.si].append(a)
    head = ["region", "biome", "temperature now", "snow now"]
    for sp in world.species:
        head += [f"{_plural(sp.name)}: winter-white", "camouflage gap", "insulation"]
    out = ["Living conditions and adaptation by region (camouflage gap = how different the coat looks from the "
           "ground under the animal, to a fox's or rabbit's eye: lightness, and yellow-blue at half weight; 0 is a "
           "perfect match, above about 0.3 stands out; insulation is the fur trait, 1 = founders' average):", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    walk = WALKABLE[t.types]
    for k in range(cols * rows):
        c, r = k % cols, k // cols
        x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
        y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
        land = walk[y0:y1, x0:x1]
        if not land.any():
            continue
        kinds = biomes[y0:y1, x0:x1][land]
        biome = BIOMES[int(np.bincount(kinds, minlength=len(BIOMES)).argmax())]
        temp = world.season_temp + float(cl.offset[y0:y1, x0:x1][land].mean())
        snow = float((np.clip((-(world.season_temp + cl.offset[y0:y1, x0:x1]) - 0.35) / 0.4, 0, 1)
                      * cl.snow_fall[y0:y1, x0:x1])[land].mean())
        cells = [region_name(k), biome, f"{temp:+.2f}", _pct(snow)]
        for si in range(len(world.species)):
            members = groups[k][si]
            if members:
                gap = sum(world.camouflage_gap(a) for a in members) / len(members)
                ins = sum(a.traits["insulation"] for a in members) / len(members)
                cells += [_pct(sum(a.moults for a in members) / len(members)), f"{gap:.2f}", f"{ins:.2f}"]
            else:
                cells += ["–", "–", "–"]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _totals(world, summary: dict) -> list[str]:
    out = []
    for sp in summary["species"]:
        si = next(i for i, s in enumerate(world.species) if s.name == sp["name"])
        deaths = Counter(sp["deaths"])
        total = sum(deaths.values())
        line = (f"- {_plural(sp['name']).capitalize()}: now {sp['final']}, peak {sp['peak']} (day {sp['peak_day']}), "
                f"lowest {sp['low']} (day {sp['low_day']}); {sp['births']:,} born, {sp['stillborn']:,} died before "
                f"birth, {world.arrivals[si]:,} arrived from outside; {world.matings[si]:,} matings, "
                f"{world.rejections[si]:,} courtships turned down; {total:,} deaths: "
                + (", ".join(f"{c} {k:,} ({_pct(k / total)})" for c, k in deaths.most_common()) if total else "none")
                + f"; {sp['generations']} generations; {sp['cycles']} boom-bust cycles.")
        hunts = sp.get("hunts") or {}
        if world.species[si].diet != "plants" and hunts:
            line += (f" Meals: {hunts.get('kill', 0) + hunts.get('surplus kill', 0):,} rabbits caught "
                     f"({hunts.get('surplus kill', 0):,} surplus kills cached), {hunts.get('vole', 0):,} voles, "
                     f"{hunts.get('scavenged', 0):,} carcasses, {hunts.get('cache', 0):,} cached meals, "
                     f"{hunts.get('fruit', 0):,} fruit; {hunts.get('missed lunge', 0):,} missed lunges.")
        social = {k: v for k, v in (sp.get("social") or {}).items() if v}
        if social:
            line += " Social: " + ", ".join(f"{k} {v}" for k, v in social.items()) + "."
        moved = [t for t in sp["traits"] if abs(t["change"]) >= 0.02]
        if moved:
            line += " Evolved (breeding values, start→end): " + ", ".join(
                f"{t['label'].lower()} {t['start']:g}→{t['end']:g} {t['unit']} ({t['change']:+.0%})" for t in moved) + "."
        alleles = [a for a in sp["alleles"] if a["status"] != "changed" or abs(a["end"] - a["start"]) >= 0.1]
        if alleles:
            line += " Gene pool: " + ", ".join(
                f"{a['gene']} {a['allele']} {a['start']:.0%}→{a['end']:.0%}" + (f" ({a['status']})" if a["status"] != "changed" else "")
                for a in alleles[:6]) + "."
        if sp.get("diversity"):
            line += f" Genetic diversity (He) {sp['diversity'][0]:.2f}→{sp['diversity'][1]:.2f}."
        if sp.get("brain"):
            line += f" Brains now match the founders' instincts in {_pct(sp['brain']['agreement'])} of clear-cut decisions."
        out.append(line)
    return out


def _now(world) -> list[str]:
    row = world.ledger.rows[-1] if world.ledger.rows else None
    groups = [[] for _ in world.species]
    for a in world.animals:
        groups[a.si].append(a)
    out = []
    for si, (sp, members) in enumerate(zip(world.species, groups)):
        adults = [a for a in members if a.adult]
        line = (f"- {_plural(sp.name).capitalize()}: {len(members)} ({sum(1 for a in adults if a.female)} adult females, "
                f"{sum(1 for a in adults if not a.female)} adult males, {len(members) - len(adults)} young; "
                f"{sum(1 for a in members if a.pregnant)} pregnant)")
        if members:
            line += (f", mean energy {_pct(sum(a.energy / a.max_energy for a in members) / len(members))}, "
                     f"water {_pct(sum(a.water / a.max_water for a in members) / len(members))}")
        if sp.territory > 0:
            roles = Counter(world._role(a) for a in adults)
            line += (f"; {roles['breeder']} territory holders, {roles['floater']} floaters, {roles['helper']} helpers, "
                     f"{roles['natal']} grown young still at home")
        out.append(line + ".")
    if row:
        out.append(f"- Plant cover {_pct(row['cover'])} of capacity; about {row['voles']:,} voles.")
    return out


# ---------------------------------------------------------------- the brief

def evidence(world, kind: str = "summary", species: str | None = None) -> dict:
    """The analyst's brief: text sections plus a PNG map. `kind` is "summary",
    "extinction" (needs `species`) or "question"."""
    ledger = world.ledger
    rows = ledger.rows
    cfg = world.cfg
    tpd = cfg.ticks_per_day
    summary = world.narrator.summary()
    env = world.env
    days_now = (world.tick - world.start_tick) / tpd
    last_day = rows[-1]["day"] if rows else 0
    focus = None
    if kind == "extinction":
        matches = [e for e in ledger.extinctions if e["species"] == species]
        focus = matches[-1] if matches else None

    out = ["# Run records", ""]
    status = "still running"
    if world.finished:
        f = world.finished
        status = f"FINISHED on day {f['day']} ({f['season']}, year {f['year']}): the last {f['species']} died"
    out.append(f"Now: day {env['day']} ({env['season'] if cfg.seasons else 'no seasons'}, year {env['year']}), "
               f"{days_now:.1f} days since the population was founded; {status}.")
    out += ["", "## How this world works", *_settings(world), *_species_facts(world),
            "- Causes of death: pred = killed by a fox; starv = energy ran out (hunger); thirst = dehydration; "
            "old = old age; disease = background illness and accidents (e.g. mange); fight = killed in a "
            "territorial fight; emig = emigrated (left the area for lack of a territory: lost to this population); "
            "removed = removed by the user.",
            "- Condition = average energy reserves as a share of capacity (below about 50% females don't breed and "
            "mothers reabsorb litters)."]

    out += ["", "## Totals for the whole run", *_totals(world, summary)]
    out += ["", "## Now", *_now(world)]

    if ledger.extinctions:
        out += ["", "## Extinctions (the species' count reached 0)"]
        for e in ledger.extinctions:
            back = f"recolonised by immigrants on day {e['recolonised_day']}" if e["recolonised_day"] else "never recolonised"
            out.append(f"- {_plural(e['species']).capitalize()} died out on day {e['day']} ({e['season']}, year {e['year']}); {back}.")

    # Whole-run timeline.
    if rows and last_day > 0:
        bin_days = _bin_days(last_day, cfg)
        out += ["", f"## Timeline of the whole run ({bin_days:g} day{'s' if bin_days != 1 else ''} per row; counts are at "
                    "the end of each row; births, deaths, arrivals and food are totals within the row)"]
        for si, sp in enumerate(world.species):
            out += ["", f"### {_plural(sp.name).capitalize()}", *_timeline(world, rows, si, 0, last_day, bin_days)]
        out += ["", "### Environment", *_environment(world, rows, 0, last_day, bin_days)]
        out += _evolution(world, rows, last_day)

        # Day-by-day detail at the end, or before the extinction being asked about.
        end = focus["day"] if focus else last_day
        end = min(end, last_day)
        start = max(0, end - DETAIL_DAYS)
        if bin_days > 1 and end > start:
            what = f"the {end - start:g} days up to the extinction of the {_plural(focus['species'])}" if focus \
                else f"the last {end - start:g} days"
            out += ["", f"## Day by day: {what}"]
            for si, sp in enumerate(world.species):
                out += ["", f"### {_plural(sp.name).capitalize()}", *_timeline(world, rows, si, start, end, 1)]
            out += ["", "### Environment", *_environment(world, rows, start, end, 1)]

    # The last deaths.
    if focus:
        out += ["", f"## The last {len(focus['last_deaths'])} {_plural(focus['species'])} to die before the extinction "
                    f"(oldest first). 'role' for foxes: breeder = territory holder, floater = adult without a territory, "
                    "helper = daughter helping at home, natal = grown young not yet dispersed.",
                *_last_deaths_table(focus["last_deaths"])]
    else:
        for si, sp in enumerate(world.species):
            log = list(ledger.last_deaths[si])[-25:]
            if log:
                out += ["", f"## The last {len(log)} {_plural(sp.name)} to die (oldest first)", *_last_deaths_table(log)]

    # The map.
    if rows:
        if focus:
            r1 = _row_at(rows, focus["day"])
            r0 = _row_at(rows, focus["day"] - 6)
            label = f"in the 6 days before the {_plural(focus['species'])} died out (days {r0['day'] + 1}-{r1['day']})"
        else:
            r1 = rows[-1]
            r0 = _row_at(rows, r1["day"] - 3)
            label = f"in the last 3 days (days {r0['day'] + 1}-{r1['day']})"
        out += ["", "## The map, by region (animals now; deaths " + label + ")", *_regions(world, r0, r1, label)]
    key = MAP_KEY.format(w=world.width, h=world.height, s=SCALE, focus=(
        f" Big cyan crosses = where the last {_plural(focus['species'])} died before the extinction." if focus else ""))
    out += ["", "## The map image (attached)", key, "The map shows the landscape and animals NOW" + (
        ", which may be long after the extinction." if focus and focus["day"] < env["day"] - 1 else ".")]

    # The user's sandbox actions and events.
    if ledger.actions:
        out += ["", "## What the user did with the sandbox tools (day: action)"]
        for a in ledger.actions[-60:]:
            what = a["kind"] + (f" {a['detail']}" if a["detail"] else "") + (f" ({a['species']})" if a["species"] else "")
            amount = f" x{a['amount']}" if a["amount"] > 1 else ""
            if a["kind"] == "painted terrain":
                amount = f" ({a['amount']} cells)"
            out.append(f"- day {a['day']}: {what}{amount}")
    else:
        out += ["", "## What the user did with the sandbox tools", "- Nothing: the run was left to itself."]

    # The built-in narrator's notes.
    # Every major note, but only the latest of each recurring minor one (learning, alphas).
    log = world.narrator.log
    latest_minor = {(i["species"], i["icon"]): i["id"] for i in log if i["level"] == "minor"}
    notes = [i for i in log if i["level"] == "major"
             or (i["level"] == "minor" and latest_minor[(i["species"], i["icon"])] == i["id"])][-NOTES:]
    if notes:
        out += ["", "## Notes from the built-in rule-based narrator (useful leads, but shallow; check them against "
                    "the tables)"]
        for i in notes:
            text = i["text"] if len(i["text"]) <= 320 else i["text"][:317] + "..."
            out.append(f"- day {i['day']} {i['title']}: {text}")

    text = "\n".join(out)
    png = render_map(world, focus)
    meta = {"days": round(days_now, 1), "rows": len(rows), "deaths_logged": sum(len(d) for d in ledger.last_deaths),
            "extinctions": [{"species": e["species"], "day": e["day"], "recolonised_day": e["recolonised_day"]}
                            for e in ledger.extinctions],
            "focus": {"species": focus["species"], "day": focus["day"]} if focus else None,
            "chars": len(text), "tick": world.tick, "super": bool(cfg.super_animals)}
    return {"text": text, "png": png, "meta": meta}


# ---------------------------------------------------------------- the Deaths & births charts

def ledger_series(world) -> dict:
    """Daily series: births and deaths (by cause) per day, breeding adults and food per head."""
    rows = world.ledger.rows
    causes = world.ledger.causes
    vv = vole_value(world)
    out = {"tick": [], "day": [], "cover": [], "voles": [], "causes": list(causes),
           "species": {sp.name: {"n": [], "births": [], "deaths": [], "causes": {c: [] for c in causes},
                                 "af": [], "am": [], "food": [], "territories": [], "paired": [], "floaters": []}
                       for sp in world.species}}
    for prev, row in zip(rows, rows[1:]):
        span = max(1.0, row["day"] - prev["day"])
        out["tick"].append(row["tick"])
        out["day"].append(row["day"])
        out["cover"].append(row["cover"])
        out["voles"].append(row["voles"])
        for si, sp in enumerate(world.species):
            s = out["species"][sp.name]
            r = row["species"][si]
            d = world.ledger.delta(row, prev, si)
            s["n"].append(r["n"])
            s["births"].append(round(d["births"] / span, 2))
            s["deaths"].append(round(sum(d["deaths"].values()) / span, 2))
            for c in causes:
                s["causes"][c].append(round(d["deaths"][c] / span, 2))
            s["af"].append(r["af"])
            s["am"].append(r["am"])
            if sp.diet == "plants":
                s["food"].append(row["cover"])
            else:
                prey = sum(row["species"][p]["n"] for p in world.prey_of[si]) + (row["voles"] * vv if sp.eats_voles else 0)
                s["food"].append(round(prey / r["n"], 1) if r["n"] else None)
            for k in ("territories", "paired", "floaters"):
                s[k].append(r.get(k))
    return out
