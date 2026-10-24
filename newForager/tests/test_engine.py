import base64
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import World, build_config  # noqa: E402
from engine import movement  # noqa: E402
from engine.animal import GOMPERTZ_SHAPE, Act  # noqa: E402
from engine.coats import fox_coat, rabbit_coat  # noqa: E402
from engine.config import FOX, RABBIT  # noqa: E402
from engine.genetics import MARKER, Genome, GenomeSpec  # noqa: E402
from engine.vegetation import Vegetation  # noqa: E402


def small_world(seed=7, **settings):
    cfg = build_config({"initial": {"rabbit": 60, "fox": 6}, **settings})
    return World(replace(cfg, width=64, height=40), seed=seed)


def open_plane(rng, blocked=lambda x, y: False):
    return SimpleNamespace(rng=rng, walkable=lambda x, y: not blocked(x, y))


def bessel_ratio(kappa):
    """I1(kappa) / I0(kappa), the mean cosine of a von Mises turning angle."""
    theta = np.linspace(0, math.pi, 20001)
    w = np.exp(kappa * np.cos(theta))
    return np.trapezoid(w * np.cos(theta), theta) / np.trapezoid(w, theta)


def with_gene(gs, genome, symbol, a, b):
    gene = next(g for g in gs.sp.genes if g.symbol == symbol)
    i = gs.gene_at[symbol]
    genome.mat[i] = gene.alleles.index(a)
    genome.pat[i] = gene.alleles.index(b)
    return genome


# ---------------------------------------------------------------- whole-world properties

def test_same_seed_gives_identical_runs():
    a, b = small_world(seed=3), small_world(seed=3)
    for _ in range(300):
        a.step()
        b.step()
    fa, fb = a.frame(), b.frame()
    assert fa["animals"] == fb["animals"]
    assert a.history == b.history


def test_animals_never_stand_on_blocked_ground():
    w = small_world(seed=11)
    for _ in range(500):
        w.step()
        for a in w.animals:
            assert w.walkable(a.x, a.y), f"{a.sp.name} #{a.id} at ({a.x:.2f}, {a.y:.2f}) is on blocked ground"


def test_every_walkable_cell_has_a_path_to_water():
    w = small_world(seed=5)
    t = w.terrain
    walk = np.frombuffer(bytes(t.walk), dtype=np.uint8).reshape(t.height, t.width).astype(bool)
    dist = np.array(t.shore_dist).reshape(t.height, t.width)
    assert (dist[walk] >= 0).all()
    assert (dist[~walk] == -1).all()


def test_clock_runs_through_days_and_seasons():
    w = small_world()
    tpd, dpy = w.cfg.ticks_per_day, w.cfg.days_per_year
    seasons, lights = set(), []
    for _ in range(tpd * dpy):
        w.step()
        seasons.add(w.env["season"])
        lights.append(w.light)
    assert seasons == {"spring", "summer", "autumn", "winter"}
    assert min(lights) == 0.0 and max(lights) == 1.0
    assert w.env["year"] == 2


def test_breeding_is_seasonal():
    w = small_world()
    fox = [i for i, s in enumerate(w.species) if s.name == "fox"][0]
    rabbit = [i for i, s in enumerate(w.species) if s.name == "rabbit"][0]
    year = w.cfg.ticks_per_day * w.cfg.days_per_year
    rabbit_in, fox_in = 0, 0
    for _ in range(year):
        w.tick += 1
        w._environment()
        rabbit_in += w.in_season[rabbit]
        fox_in += w.in_season[fox]
    assert 0.5 * year < rabbit_in < 0.6 * year
    assert 0.15 * year < fox_in < 0.25 * year


def test_history_since_returns_only_new_samples():
    w = small_world()
    for _ in range(120):
        w.step()
    tick = w.history["tick"][2]
    newer = w.history_since(tick)
    assert newer["tick"] == w.history["tick"][3:]
    assert len(newer["species"]["rabbit"]["genes"]["A"]["freq"]) == len(newer["tick"])


# ---------------------------------------------------------------- movement

@pytest.mark.parametrize("kappa", [2.0, 10.0, 40.0])
def test_turning_angles_follow_von_mises(kappa):
    rng = random.Random(1)
    world = open_plane(rng)
    animal = SimpleNamespace(x=0.0, y=0.0, heading=0.0, avoid_side=1)
    cosines = []
    for _ in range(20000):
        before = animal.heading
        movement.step(animal, world, animal.heading, 0.1, kappa)
        cosines.append(math.cos(animal.heading - before))
    assert np.mean(cosines) == pytest.approx(bessel_ratio(kappa), abs=0.01)


def test_blocked_step_slides_along_the_shore():
    world = open_plane(random.Random(0), blocked=lambda x, y: x >= 5.0)
    animal = SimpleNamespace(x=4.8, y=5.0, heading=0.0, avoid_side=1)
    assert movement.step(animal, world, 0.0, 0.5, math.inf) == 0.5
    assert animal.x < 5.0 and abs(animal.y - 5.0) > 0.1


# ---------------------------------------------------------------- plants

def test_vegetation_grows_logistically_to_capacity():
    w = small_world()
    veg = Vegetation(w.terrain, replace(w.cfg, plant_growth=0.05), np.random.default_rng(0))
    land = veg.K > 0
    veg.B[land] = 0.01 * veg.K[land]
    for _ in range(600):
        veg.step()
        assert (veg.B <= veg.K + 1e-12).all() and (veg.B >= 0).all()
    assert (veg.B[~land] == 0).all()
    assert (veg.B[land] / veg.K[land]).min() > 0.99


# ---------------------------------------------------------------- genetics

def test_mendelian_segregation_gives_one_two_one():
    gs, rng = GenomeSpec(RABBIT), random.Random(1)
    parents = [with_gene(gs, gs.founder(rng, male=m), "A", "A", "a") for m in (False, True)]
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(8000):
        child = gs.child(parents[0], parents[1], rng, mutation=0.0)
        counts[gs.express(child, rng).genotype["A"].count("a")] += 1
    assert counts[0] / 8000 == pytest.approx(0.25, abs=0.02)
    assert counts[1] / 8000 == pytest.approx(0.50, abs=0.02)
    assert counts[2] / 8000 == pytest.approx(0.25, abs=0.02)


def test_sex_ratio_is_even():
    gs, rng = GenomeSpec(RABBIT), random.Random(2)
    mum, dad = gs.founder(rng, male=False), gs.founder(rng, male=True)
    males = sum(gs.child(mum, dad, rng, 0.0).male for _ in range(6000))
    assert males / 6000 == pytest.approx(0.5, abs=0.02)


def test_recombination_follows_haldane_map_distance():
    gs, rng = GenomeSpec(RABBIT), random.Random(3)
    # Two marker loci on the same chromosome, as far apart as possible.
    best = None
    for start, end in gs.ranges:
        markers = [i for i in range(start, end) if gs.kinds[i] == MARKER]
        if len(markers) >= 2:
            pair = (markers[0], markers[-1])
            if best is None or gs.positions[pair[1]] - gs.positions[pair[0]] > gs.positions[best[1]] - gs.positions[best[0]]:
                best = pair
    i, j = best
    d = gs.positions[j] - gs.positions[i]
    genome = Genome([0] * gs.n, [1] * gs.n, False)
    recombinant = 0
    trials = 20000
    for _ in range(trials):
        g = gs.gamete(genome, rng, mutation=0.0)
        recombinant += g[i] != g[j]
    haldane = 0.5 * (1 - math.exp(-2 * d))
    assert recombinant / trials == pytest.approx(haldane, abs=0.02)


def test_parent_offspring_regression_recovers_heritability():
    gs, rng = GenomeSpec(RABBIT), random.Random(4)
    h2 = RABBIT.traits["mass"].heritability
    mid, kids = [], []
    for _ in range(4000):
        mum, dad = gs.founder(rng, male=False), gs.founder(rng, male=True)
        pm = math.log(gs.express(mum, rng).traits["mass"])
        pd = math.log(gs.express(dad, rng).traits["mass"])
        child = gs.child(mum, dad, rng, 0.0)
        mid.append((pm + pd) / 2)
        kids.append(math.log(gs.express(child, rng).traits["mass"]))
    slope = np.polyfit(mid, kids, 1)[0]
    assert slope == pytest.approx(h2, abs=0.1)


def test_platinum_homozygotes_are_lethal():
    gs, rng = GenomeSpec(FOX), random.Random(5)
    genome = with_gene(gs, gs.founder(rng, male=False), "W", "Wp", "Wp")
    assert gs.express(genome, rng).lethal
    genome = with_gene(gs, genome, "W", "Wp", "w")
    expr = gs.express(genome, rng)
    assert not expr.lethal and gs.coats[expr.coat].name.startswith("Platinum")


def test_coat_genetics_follows_dominance_and_epistasis():
    wild = {"A": ("A", "A"), "B": ("B", "B"), "C": ("C", "C"), "D": ("D", "D"), "En": ("en", "en")}
    assert rabbit_coat(wild).name == "Chestnut agouti"
    assert rabbit_coat({**wild, "A": ("A", "a")}).name == "Chestnut agouti"  # a is recessive
    assert rabbit_coat({**wild, "A": ("a", "a"), "B": ("b", "b")}).name == "Chocolate"
    assert rabbit_coat({**wild, "A": ("a", "a"), "D": ("d", "d")}).name == "Blue"
    assert rabbit_coat({**wild, "C": ("c", "c"), "A": ("a", "a")}).name == "Albino"  # epistatic
    assert rabbit_coat({**wild, "C": ("ch", "c")}).name.startswith("Himalayan")
    assert rabbit_coat({**wild, "En": ("En", "en")}).name.startswith("Broken")
    charlie = rabbit_coat({**wild, "En": ("En", "En")})
    assert charlie.name.startswith("Charlie") and charlie.vigor < 1
    assert fox_coat({"S": ("S", "s"), "P": ("P", "P"), "W": ("w", "w")}).name == "Cross"
    assert fox_coat({"S": ("s", "s"), "P": ("p", "p"), "W": ("w", "w")}).name == "Pearl"


def test_inbreeding_exposes_recessive_load():
    gs, rng = GenomeSpec(RABBIT), random.Random(6)
    carrier = gs.founder(rng, male=False)
    for i in gs.load_at:
        carrier.mat[i], carrier.pat[i] = 1, 0
    selfed = [gs.express(gs.child(carrier, carrier, rng, 0.0), rng).load_hom for _ in range(400)]
    # A quarter of each locus becomes homozygous when a carrier is mated with itself.
    assert np.mean(selfed) == pytest.approx(len(gs.load_at) / 4, rel=0.15)


def test_birth_conserves_energy_and_inherits_genomes():
    w = small_world()
    mother = next(a for a in w.animals if a.sp.name == "rabbit" and a.adult and a.female)
    father = next(a for a in w.animals if a.sp.name == "rabbit" and a.adult and not a.female)
    mother.energy = mother.max_energy
    w.mate(mother, father)
    before = mother.energy
    w.give_birth(mother)
    kids = w._born
    assert kids and all(k.mother is mother and k.father_id == father.id for k in kids)
    # The litter's first feed comes straight after birth; milk is 80% efficient,
    # so the only energy lost is a quarter of what was passed on as milk.
    after = mother.energy + sum(k.energy for k in kids)
    assert after <= before
    assert before - after <= 0.25 * sum(k.energy for k in kids) + 1e-9


def test_gompertz_hazard_gives_the_configured_median_lifespan():
    w = small_world()
    a = next(x for x in w.animals if x.sp.name == "rabbit")
    L = a.lifespan
    cumulative = sum(a.gompertz * math.exp(GOMPERTZ_SHAPE * t / L) for t in range(1, int(L) + 1))
    assert math.exp(-cumulative) == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------- behaviour and sandbox

def test_territories_exclude_same_sex_neighbours():
    w = small_world()
    si = next(i for i, s in enumerate(w.species) if s.name == "fox")
    for a in list(w.animals):
        if a.si == si:
            w.kill(a, "removed")
    w._flush()
    x, y = w.nearest_walkable(20, 20)
    ids = [w.spawn(si, x, y, 1, "Red")[0] for _ in range(6)]
    holders = [w.by_id[i] for i in ids if w.by_id[i].territorial]
    assert holders
    spacing = w.species[si].territory
    for a in holders:
        for b in holders:
            if a is not b and a.female == b.female:
                assert math.hypot(a.home_x - b.home_x, a.home_y - b.home_y) >= spacing


def test_dispersing_foxes_pair_up_with_a_lone_mate():
    w = flat_world()
    vixen = put(w, "fox", 20.5, 20.5)
    vixen.female = True
    w._holders[_species(w, "fox")] = {vixen.id: vixen}
    vixen.home_x, vixen.home_y, vixen.territorial = 20.5, 20.5, True
    dog = put(w, "fox", 40.5, 20.5)
    dog.female, dog.territorial = False, False
    w._holders[_species(w, "fox")].pop(dog.id, None)
    assert w.settle(dog, 30.0)
    assert (dog.home_x, dog.home_y) == (vixen.home_x, vixen.home_y)  # they share her territory


def test_gene_editing_changes_the_coat():
    w = small_world()
    rabbit = next(a for a in w.animals if a.sp.name == "rabbit")
    w.edit_gene(rabbit, "C", ("c", "c"))
    assert w.genomes[rabbit.si].coats[rabbit.coat].name == "Albino"
    assert w.inspect(rabbit.id)["coat"] == "Albino"


def test_painting_water_moves_stranded_animals():
    w = small_world()
    a = w.animals[0]
    w.paint(a.x, a.y, 3, "water")
    assert w.walkable(a.x, a.y)


def test_api_round_trip():
    import app as webapp

    client = webapp.app.test_client()
    info = client.get("/api/world").get_json()
    n = info["width"] * info["height"]
    assert len(base64.b64decode(info["terrain"]["types"])) == n
    assert info["species"][0]["coats"] and info["species"][0]["genes"]
    frame = client.post("/api/step", json={"ticks": 3}).get_json()
    assert frame["tick"] >= 3 and frame["animals"] and "env" in frame
    uid = frame["animals"][0][0]
    detail = client.post("/api/step", json={"ticks": 0, "selected": uid}).get_json()["selected"]
    assert detail["id"] == uid and detail["chromosomes"]

    spawned = client.post("/api/tool", json={"tool": "spawn", "species": "fox", "x": 80, "y": 50, "count": 2, "preset": "Silver"}).get_json()
    assert len(spawned["ids"]) == 2
    assert client.get(f"/api/animal/{spawned['ids'][0]}").get_json()["coat"].endswith("Silver") \
        or client.get(f"/api/animal/{spawned['ids'][0]}").get_json()["coat"] == "Silver"
    assert client.post("/api/tool", json={"tool": "paint", "terrain": "water", "x": 10, "y": 10, "radius": 2}).status_code == 200
    assert client.post("/api/tool", json={"tool": "event", "event": "plague", "species": "rabbit"}).status_code == 200
    assert client.post("/api/tool", json={"tool": "event", "event": "volcano"}).status_code == 400

    edited = client.post("/api/lab", json={"action": "edit", "id": uid, "gene": "A", "alleles": ["a", "a"]})
    if edited.status_code == 200 and edited.get_json()["species"] == "rabbit":
        assert edited.get_json()["genes"][0]["raw"] == ["a", "a"]
    clone = client.post("/api/lab", json={"action": "clone", "id": spawned["ids"][0]}).get_json()
    assert "id" in clone
    assert client.post("/api/lab", json={"action": "breed", "id": spawned["ids"][0], "partner": spawned["ids"][0]}).status_code == 400

    assert client.get("/api/history?since=-1").get_json()["tick"]
    assert client.post("/api/tool", json={"tool": "repopulate"}).status_code == 200
    reset = client.post("/api/reset", json={"seed": 5, "settings": {"water_fraction": 0.2, "initial": {"fox": 3}}})
    assert reset.status_code == 200 and reset.get_json()["settings"]["initial"]["fox"] == 3
    assert client.post("/api/step", json={"ticks": "lots"}).status_code == 400


# ---------------------------------------------------------------- narrator

def _run_days(w, days):
    for _ in range(int(days * w.cfg.ticks_per_day)):
        w.step()


def test_narrator_explains_a_crash_with_its_real_cause():
    w = small_world(seed=4, initial={"rabbit": 120, "fox": 0}, immigration=0)
    _run_days(w, 1.1)
    rabbits = [a for a in w.animals if a.sp.name == "rabbit"]
    for a in rabbits[: int(len(rabbits) * 0.7)]:
        w.kill(a, "removed")
    w._flush()
    _run_days(w, 1.1)
    falls = [i for i in w.narrator.log if i["species"] == "rabbit" and i["title"].startswith("Rabbits fell")]
    assert falls, [i["title"] for i in w.narrator.log]
    assert "removal by you" in falls[-1]["text"] and "You removed" in falls[-1]["text"]


def test_narrator_blames_the_user_for_alleles_lost_to_removal():
    w = small_world(seed=8, initial={"rabbit": 0, "fox": 0}, immigration=0)
    si = next(i for i, s in enumerate(w.species) if s.name == "rabbit")
    x, y = w.nearest_walkable(30, 20)
    w.spawn(si, x, y, 12, "Wild type")
    victim = w.animals[0]
    w.edit_gene(victim, "B", ("B", "b"))  # a single copy of a rare allele
    w.narrator.observe_now()
    old = w.narrator.snaps[-1]
    w.kill(victim, "removed")
    w._flush()
    w.tick += 1
    new = w.narrator.snapshot()
    w.narrator.snaps.append(new)
    gene = next(g for g in w.species[si].genes if g.symbol == "B")
    item = w.narrator._explain_allele(si, gene, 1, old["species"][si]["freqs"]["B"][1], 0.0, True, False,
                                      w.narrator._merged(si, old, new), old, new, False)
    assert "vanished" in item["title"] and "you removed" in item["text"]
    assert "selection is likely" not in item["text"]


def test_drift_is_the_explanation_for_small_changes_in_small_populations():
    w = small_world(seed=8)
    si = next(i for i, s in enumerate(w.species) if s.name == "fox")
    narrator = w.narrator
    narrator.observe_now()
    old = narrator.snaps[-1]
    _run_days(w, 3)
    new = narrator.snapshot()
    narrator.snaps.append(new)
    gene = next(g for g in w.species[si].genes if g.symbol == "S")
    p0 = 0.30
    item = narrator._explain_allele(si, gene, 1, p0, p0 - 0.05, False, False, narrator._merged(si, old, new), old, new, False)
    # With only a handful of foxes, a 5-point change is well within drift's reach.
    assert "Chance alone could easily do this" in item["text"]


def test_report_explains_every_species():
    w = small_world(seed=2)
    _run_days(w, 1.5)
    items = w.narrator.report()
    species = {i["species"] for i in items}
    assert {"rabbit", "fox"} <= species
    assert all(i["text"] and i["title"] for i in items)
    assert not any(i["level"] == "report" for i in w.narrator.log)


def test_insight_endpoints():
    import app as webapp

    client = webapp.app.test_client()
    client.post("/api/reset", json={"seed": 3})
    client.post("/api/step", json={"ticks": 60})
    assert isinstance(client.get("/api/insights").get_json(), list)
    assert client.get("/api/explain").get_json()
    frame = client.post("/api/step", json={"ticks": 1, "insights_since": 0}).get_json()
    assert "insights" in frame


# ---------------------------------------------------------------- behaviour

from engine.animal import Cache  # noqa: E402


def _species(w, name):
    return next(i for i, s in enumerate(w.species) if s.name == name)


def _clear(w, name):
    for a in list(w.animals):
        if a.sp.name == name:
            w.kill(a, "removed")
    w._flush()


def test_young_hide_in_the_nest_and_are_nursed_on_visits():
    w = small_world(seed=12, initial={"rabbit": 30, "fox": 0}, immigration=0)
    mother = next(a for a in w.animals if a.sp.name == "rabbit" and a.adult and a.female)
    father = next(a for a in w.animals if a.sp.name == "rabbit" and a.adult and not a.female)
    mother.energy = mother.max_energy
    w.mate(mother, father)
    w.give_birth(mother)
    w._flush()
    kids = [k for k in mother.litter]
    assert kids and all(k.act == Act.NEST and k.concealed for k in kids)
    for k in kids:
        k.energy = 0.2 * k.max_energy
    mother.next_nurse = w.tick
    mother.x, mother.y = mother.nest_x, mother.nest_y
    fed = False
    for _ in range(40):
        w.step()
        if all(not k.alive for k in kids):
            break
        if any(k.alive and k.energy > 0.5 * k.max_energy for k in kids):
            fed = True
            break
    assert fed


@pytest.mark.parametrize("brains", ["rules", "neural"])
def test_prey_freezes_far_from_a_predator_and_flees_when_it_closes_in(brains):
    w = small_world(seed=9, brains=brains)
    rabbit = next(a for a in w.animals if a.sp.name == "rabbit" and a.adult and a.mother is None)
    fox = next(a for a in w.animals if a.sp.name == "fox")
    rabbit.home_x, rabbit.home_y = rabbit.x, rabbit.y
    rabbit.traits["boldness"] = rabbit.traits["freeze"] = 1.0  # an average temperament
    rabbit.alert_until = w.tick + 5  # already warned, so detection is certain
    for dist, expected in ((rabbit.vision - 0.5, Act.FREEZE), (1.0, Act.FLEE)):
        fox.x, fox.y = rabbit.x + dist, rabbit.y
        fox.last_move = 0.0
        if not w.walkable(fox.x, fox.y):
            pytest.skip("random placement put the fox on blocked ground")
        w._rebuild_buckets()
        w._think()
        rabbit._decide(w)
        assert rabbit.act == expected


def test_a_fleeing_rabbit_thumps_and_alerts_its_neighbours():
    w = small_world(seed=3)
    rabbits = [a for a in w.animals if a.sp.name == "rabbit" and a.mother is None]
    a, b = rabbits[0], rabbits[1]
    b.x, b.y = a.x + 1.0, a.y
    if not w.walkable(b.x, b.y):
        pytest.skip("blocked ground")
    w._rebuild_buckets()
    w.thump(a)
    assert b.alert_until > w.tick


def test_close_kin_do_not_mate():
    w = small_world(seed=5)
    mum = next(a for a in w.animals if a.sp.name == "rabbit" and a.female)
    son = next(a for a in w.animals if a.sp.name == "rabbit" and not a.female)
    son.mother_id = mum.id
    assert w.related(mum, son) and w.related(son, mum)
    other = next(a for a in w.animals if a.sp.name == "rabbit" and not a.female and a is not son)
    other.mother_id = other.father_id = None
    mum.mother_id = mum.father_id = None
    assert not w.related(mum, other)


def test_foxes_cache_leftovers_and_come_back_for_them():
    w = small_world(seed=6)
    fox = next(a for a in w.animals if a.sp.name == "fox" and a.adult)
    fox.partner = None
    fox.energy = 0.94 * fox.max_energy
    fox.act, fox.carcass = Act.EAT, 200.0
    fox._eat(w)
    assert fox.act == Act.CACHE and fox.carried > 0
    for _ in range(60):
        if fox.act != Act.CACHE:
            break
        fox._cache(w)
    assert any(isinstance(c, Cache) and c.owner == fox.id for c in w.caches)
    fox.x += 5.0 if w.walkable(fox.x + 5.0, fox.y) else 0.0
    assert w.find_cache(fox) is not None


def test_fresh_grass_supplies_water():
    w = small_world(seed=2)
    rabbit = next(a for a in w.animals if a.sp.name == "rabbit" and a.adult)
    w.veg.B[int(rabbit.y), int(rabbit.x)] = w.veg.K[int(rabbit.y), int(rabbit.x)] or 1.0
    rabbit.energy = 0.3 * rabbit.max_energy
    rabbit.water = 0.3 * rabbit.max_water
    rabbit.act = Act.EAT
    w.forage_moisture = 1.0
    for _ in range(20):
        rabbit._eat(w)
    assert rabbit.water > 0.3 * rabbit.max_water


def test_huddling_in_a_shared_burrow_saves_heat():
    w = small_world(seed=2)
    w.cold, w.heat = 1.0, 0.0
    a = next(x for x in w.animals if x.sp.name == "rabbit")
    exposed = a.thermal_cost(w)
    a.concealed = True
    w.huddle = {(int(a.x), int(a.y)): 1}
    alone = a.thermal_cost(w)
    w.huddle = {(int(a.x), int(a.y)): 4}
    huddled = a.thermal_cost(w)
    assert exposed > alone > huddled


# ---------------------------------------------------------------- end of the run

def test_run_ends_when_a_species_dies_out_and_can_resume():
    w = small_world(seed=4, immigration=0)
    w.step()
    _clear(w, "fox")
    w.step()
    assert w.finished and w.finished["species"] == "fox"
    tick = w.tick
    w.step()
    assert w.tick == tick  # nothing moves once the run has ended
    w.resume()
    w.step()
    assert w.tick == tick + 1 and w.finished is None


def test_runs_can_be_set_to_continue_through_extinctions():
    w = small_world(seed=4, immigration=0, end_on_extinction=False)
    w.step()
    _clear(w, "fox")
    for _ in range(3):
        w.step()
    assert w.finished is None


def test_summary_breaks_down_the_run():
    w = small_world(seed=2)
    for _ in range(w.cfg.ticks_per_day):
        w.step()
    s = w.summary()
    assert s["story"] and s["headline"]
    names = {sp["name"] for sp in s["species"]}
    assert names == {"rabbit", "fox"}
    rabbit = next(sp for sp in s["species"] if sp["name"] == "rabbit")
    assert rabbit["peak"] >= rabbit["low"] and rabbit["births"] >= 0 and rabbit["traits"]


def test_run_endpoints():
    import app as webapp

    client = webapp.app.test_client()
    client.post("/api/reset", json={"seed": 3})
    assert client.get("/api/summary").get_json()["story"]
    r = client.post("/api/run", json={"end_on_extinction": False}).get_json()
    assert r["end_on_extinction"] is False
    assert client.post("/api/run", json={"resume": True}).status_code == 200


# ---------------------------------------------------------------- navigation and target choice

from engine.navigation import find_path  # noqa: E402
from engine.terrain import GRASS, WATER  # noqa: E402


def flat_world(seed=1, walls=()):
    """A small all-grass world; `walls` are (x0, y0, x1, y1) boxes of water."""
    w = small_world(seed=seed, initial={"rabbit": 0, "fox": 0}, immigration=0)
    t = w.terrain
    t.types[:] = GRASS
    for x0, y0, x1, y1 in walls:
        t.types[y0:y1, x0:x1] = WATER
    t.recompute()
    w.veg.refresh_capacity(t)
    w._index_land()
    return w


def put(w, name, x, y):
    si = _species(w, name)
    a = w.spawn(si, x, y, 1, None)
    animal = w.by_id[a[0]]
    animal.x, animal.y = x, y
    animal.home_x, animal.home_y = x, y
    return animal


def test_astar_goes_round_walls_and_reports_dead_ends():
    w = flat_world(walls=[(30, 5, 32, 35)])  # a wall with a gap above y=5 and below y=35
    found = find_path(w.terrain.walk, w.width, w.height, 25.5, 20.5, 36.5, 20.5)
    assert found is not None
    path, length = found
    assert length > 11 and all(w.walkable(x, y) for x, y in path)
    # A ring of water round the goal makes it unreachable.
    w2 = flat_world(walls=[(40, 10, 50, 12), (40, 18, 50, 20), (40, 10, 42, 20), (48, 10, 50, 20)])
    assert find_path(w2.terrain.walk, w2.width, w2.height, 20.5, 15.5, 45.5, 15.5) is None
    # A detour longer than the budget counts as unreachable.
    assert find_path(w.terrain.walk, w.width, w.height, 25.5, 20.5, 36.5, 20.5, max_cost=12.0) is None


def test_animals_walk_round_obstacles_to_reach_home():
    w = flat_world(walls=[(30, 5, 32, 35)])
    rabbit = put(w, "rabbit", 26.5, 20.5)
    rabbit.home_x, rabbit.home_y = 36.5, 20.5
    rabbit.mother = None
    for _ in range(400):
        if math.hypot(rabbit.home_x - rabbit.x, rabbit.home_y - rabbit.y) < 0.6:
            break
        rabbit._toward(w, rabbit.home_x, rabbit.home_y, 0.5)
        assert w.walkable(rabbit.x, rabbit.y)
    assert math.hypot(rabbit.home_x - rabbit.x, rabbit.home_y - rabbit.y) < 0.6


def test_foxes_ignore_prey_only_reachable_by_a_long_detour():
    # A long lake between them: the way round is far longer than the gap.
    w = flat_world(walls=[(20, 2, 24, 38)])
    fox = put(w, "fox", 18.5, 20.5)
    rabbit = put(w, "rabbit", 26.5, 20.5)
    w._rebuild_buckets()
    for _ in range(40):
        assert w.find_prey(fox) is None or w.find_prey(fox)[0] is not rabbit


def test_foxes_route_round_a_small_obstacle_to_prey():
    w = flat_world(walls=[(21, 18, 23, 23)])
    fox = put(w, "fox", 18.5, 20.5)
    rabbit = put(w, "rabbit", 25.5, 20.5)
    w._rebuild_buckets()
    found = None
    for _ in range(50):
        found = w.find_prey(fox)
        if found:
            break
    assert found and found[0] is rabbit and found[1] > 7.0


def test_stalking_fox_switches_to_a_much_easier_target():
    w = flat_world()
    fox = put(w, "fox", 10.5, 20.5)
    far = put(w, "rabbit", 10.5 + 0.9 * fox.vision, 20.5)
    near = put(w, "rabbit", 13.0, 20.5)
    w._rebuild_buckets()
    fox._set_target(far)
    for _ in range(60):
        fox._reconsider(w, far)
        if fox.target is near:
            break
    assert fox.target is near


def test_chasing_fox_stays_on_the_fleeing_rabbit_unless_another_is_within_reach():
    w = flat_world()
    fox = put(w, "fox", 10.5, 20.5)
    quarry = put(w, "rabbit", 14.5, 20.5)
    bystander = put(w, "rabbit", 10.5, 24.0)  # closer to the fox, but not within lunging range
    quarry.act = Act.FLEE
    w._rebuild_buckets()
    fox._set_target(quarry)
    for _ in range(60):
        fox._reconsider(w, quarry)
    assert fox.target is quarry
    bystander.x, bystander.y = 11.5, 21.0  # now right under its nose
    w._rebuild_buckets()
    for _ in range(60):
        fox._reconsider(w, quarry)
        if fox.target is bystander:
            break
    assert fox.target is bystander


# ---------------------------------------------------------------- brains, personality and memory

from engine import brain  # noqa: E402


def test_batched_forward_pass_matches_each_network_alone():
    rng = np.random.default_rng(1)
    vecs = rng.normal(0, 0.3, (5, brain.N_WEIGHTS)).astype(np.float32)
    x = rng.uniform(0, 1, (5, brain.N_IN)).astype(np.float32)
    table = brain.BrainTable(capacity=2)  # forces the table to grow
    rows = [table.alloc(v) for v in vecs]
    batched, _ = brain.forward(x, table.w1[rows], table.w2[rows])
    for k, v in enumerate(vecs):
        w1, w2 = brain.unpack(v)
        hidden = np.tanh(x[k] @ w1)
        assert np.allclose(batched[k], np.append(hidden, 1.0) @ w2, atol=1e-5)


def test_choices_respect_the_mask_and_favour_the_current_option():
    scores = np.array([[0.5, 0.45, 2.0, 0, 0, 0, 0, 0]], dtype=np.float32)
    mask = np.array([[True, True, False, True, True, False, False, False]])
    assert brain.choose(scores.copy(), mask, np.array([-1]))[0] == brain.EAT
    assert brain.choose(scores.copy(), mask, np.array([brain.DRINK]))[0] == brain.DRINK  # within the hysteresis margin


def test_founders_brains_imitate_the_instincts():
    w = small_world(seed=5, brains="rules")
    xs, masks = [], []
    for _ in range(120):
        w.step()
        for a in w.animals:
            if a.inputs is not None and a.mother is None:
                xs.append(a.inputs + [1.0])
                masks.append(a.mask)
    x, mask = np.array(xs, dtype=np.float32), np.array(masks)
    none = np.full(len(x), -1)
    instinct = brain.squash(brain.rule_scores(x))
    target = brain.choose(instinct, mask, none)
    clear = brain.margin(instinct, mask) >= brain.CLEAR
    report = brain.training_report()
    for name in ("rabbit", "fox"):
        assert report[name]["agreement"] > 0.9
    # Founders' genomes, variation included, still decide like their instincts
    # whenever the instincts have a clear preference.
    gs = w.genomes[_species(w, "rabbit")]
    for seed in range(5):
        vec = gs.express(gs.founder(random.Random(seed), male=True), random.Random(seed)).brain
        w1, w2 = brain.unpack(vec)
        scores, _ = brain.forward(x, np.broadcast_to(w1, (len(x),) + w1.shape), np.broadcast_to(w2, (len(x),) + w2.shape))
        assert np.mean(brain.choose(scores, mask, none)[clear] == target[clear]) > 0.93


def test_brain_wiring_is_inherited_from_both_parents():
    gs = GenomeSpec(RABBIT)
    rng = random.Random(4)
    mum, dad = gs.founder(rng, male=False), gs.founder(rng, male=True)
    kid = gs.child(mum, dad, rng, mutation=0.0)
    for i in gs.brain_at:
        assert kid.mat[i] in (mum.mat[i], mum.pat[i])
        assert kid.pat[i] in (dad.mat[i], dad.pat[i])
    expr = gs.express(kid, rng)
    assert np.allclose(expr.brain, [kid.mat[i] + kid.pat[i] for i in gs.brain_at], atol=1e-6)


def test_brain_slots_are_recycled_when_animals_die():
    w = small_world(seed=2)
    si = _species(w, "rabbit")
    a = next(x for x in w.animals if x.si == si)
    row = a.brain_row
    w.kill(a, "removed")
    assert a.brain_row == -1 and row in w.brains[si].free
    for _ in range(600):
        w.step()
    assert len(w.brains[si].w1) <= 4 * max(64, max(w.counts()))


@pytest.mark.parametrize("brains", ["rules", "neural"])
def test_both_decision_modes_are_deterministic(brains):
    a, b = small_world(seed=4, brains=brains), small_world(seed=4, brains=brains)
    for _ in range(200):
        a.step()
        b.step()
    assert a.frame()["animals"] == b.frame()["animals"]
    assert a.counts()[0] > 0


def test_bold_rabbits_let_a_fox_come_closer_before_fleeing():
    w = flat_world()
    fox = put(w, "fox", 20.0, 20.5)
    acts = {}
    for boldness in (0.6, 2.0):
        rabbit = put(w, "rabbit", 21.6, 20.5)
        rabbit.traits["boldness"], rabbit.traits["freeze"] = boldness, 1.0
        rabbit.alert_until = w.tick + 5
        fox.last_move = 0.0
        w._rebuild_buckets()
        w._think()
        rabbit._decide(w)
        acts[boldness] = rabbit.act
        w.kill(rabbit, "removed")
        w._flush()
    assert acts[0.6] == Act.FLEE and acts[2.0] == Act.FREEZE


def test_appetite_sets_how_hungry_an_animal_gets_before_eating():
    w = flat_world()
    choices = {}
    for appetite in (1.0, 2.0):
        rabbit = put(w, "rabbit", 20.5, 20.5)
        rabbit.traits["appetite"] = appetite
        rabbit.energy = 0.8 * rabbit.max_energy
        rabbit.water = rabbit.max_water
        rabbit.adult = False  # not ready to mate
        w._rebuild_buckets()
        rabbit.sense(w)
        x = np.array([rabbit.inputs + [1.0]], dtype=np.float32)
        choices[appetite] = int(brain.choose(brain.rule_scores(x), np.array([rabbit.mask]), np.array([-1]))[0])
        w.kill(rabbit, "removed")
        w._flush()
    assert choices[1.0] == brain.EXPLORE and choices[2.0] == brain.EAT


def test_memories_follow_rescorla_wagner_and_fade():
    w = flat_world()
    rabbit = put(w, "rabbit", 20.5, 20.5)
    rabbit.traits["learning"] = 0.5
    key = rabbit.region(w)
    rabbit.learn(w, rabbit.danger_mem, key, 1.0, 1.0)
    assert rabbit.recall(w, rabbit.danger_mem, key, 1.0) == pytest.approx(0.5)
    rabbit.learn(w, rabbit.danger_mem, key, 1.0, 1.0)
    assert rabbit.recall(w, rabbit.danger_mem, key, 1.0) == pytest.approx(0.75)
    w.tick += w.cfg.ticks_per_day  # one half-life later
    assert rabbit.recall(w, rabbit.danger_mem, key, 1.0) == pytest.approx(0.375)
    slow = put(w, "rabbit", 20.5, 20.5)
    slow.traits["learning"] = 0.1
    slow.learn(w, slow.danger_mem, key, 1.0, 1.0)
    assert slow.recall(w, slow.danger_mem, key, 1.0) == pytest.approx(0.1)


def test_rabbits_avoid_grazing_where_they_met_a_fox():
    w = flat_world()
    rabbit = put(w, "rabbit", 20.5, 20.5)
    w.veg.B[:] = 0.0
    w.veg.B[20, 24] = w.veg.B[20, 16] = 0.8 * w.veg.capacity  # two patches, equally far either side
    w.veg.B[20, 24] *= 1.05  # the east one is slightly better...
    assert w.find_plant(rabbit) == (24, 20)
    rabbit.danger_mem[rabbit.region(w, 24.5, 20.5)] = (1.0, w.tick)  # ...but a fox was met there
    assert w.find_plant(rabbit) == (16, 20)


def test_escaping_a_predator_makes_a_rabbit_warier():
    w = flat_world()
    rabbit = put(w, "rabbit", 20.5, 20.5)
    rabbit.experience = 0.0
    rabbit.act = Act.FLEE
    w._rebuild_buckets()
    rabbit.sense(w)  # no predator in sight any more: it got away
    assert rabbit.experience > 0.0


def test_weaned_young_inherit_their_mothers_food_map():
    w = small_world(seed=12, initial={"rabbit": 30, "fox": 0}, immigration=0)
    mother = next(a for a in w.animals if a.sp.name == "rabbit" and a.adult and a.female)
    father = next(a for a in w.animals if a.sp.name == "rabbit" and a.adult and not a.female)
    mother.energy = mother.max_energy
    mother.food_mem = {7: (0.9, w.tick)}
    w.mate(mother, father)
    w.give_birth(mother)
    w._flush()
    kid = mother.litter[0]
    kid.age = int(kid.weaning) + 1
    kid.update(w)
    assert kid.mother is None and 7 in kid.food_mem


def test_inspector_shows_the_brain_and_memory():
    w = small_world(seed=6)
    w.step()
    a = next(x for x in w.animals if x.mother is None)
    d = w.inspect(a.id)
    assert len(d["brain"]["inputs"]) == brain.N_IN and len(d["brain"]["w1"]) == brain.N_IN
    assert d["brain"]["choice"] in range(brain.N_OUT)
    assert {"food", "danger", "experience"} <= set(d["memory"])
    info = w.describe()["brain"]
    assert info["mode"] == "neural" and len(info["options"]) == brain.N_OUT


def test_narrator_reports_brains_drifting_from_instinct():
    w = small_world(seed=6)
    w.step()
    tpd = w.cfg.ticks_per_day
    narr = w.narrator
    item = None
    for agree in (0.95, 0.6):  # the first check only sets the founders' baseline
        narr._brain_acc[0][2] = w.tick - 2 * tpd
        w.choice_pairs[0].clear()
        w.decisions[0] = 1000
        w.choice_pairs[0][(brain.EAT, brain.EXPLORE)] = round(1000 * (1 - agree))
        item = narr._brains(0)
    assert item is not None and "instinct" in item["text"] and "95%" in item["text"]


# ---------------------------------------------------------------- raising young

from engine.animal import PROVISION_SATED  # noqa: E402


def _fox_family(w):
    vixen = put(w, "fox", 20.5, 20.5)
    dog = put(w, "fox", 22.5, 20.5)
    for a, female in ((vixen, True), (dog, False)):
        a.female = female
        a.energy = a.max_energy
    w.mate(vixen, dog)
    w.give_birth(vixen)
    w._flush()
    return vixen, dog, [k for k in vixen.litter if k.alive]


def test_growing_young_keep_their_condition():
    w = flat_world()
    _, _, cubs = _fox_family(w)
    cub = cubs[0]
    cub.energy = 0.6 * cub.max_energy
    before = cub.energy
    grown = cub.mass
    cub.age = int(cub.weaning) - 1
    cub._allometry()
    assert cub.mass > grown
    # Its store grew with its body, less a small share (1/30) of the cost of growing.
    assert 0.55 < cub.energy / cub.max_energy < 0.6
    assert cub.energy > 2 * before


def test_a_vixen_with_four_cubs_needs_only_a_few_times_her_usual_food():
    w = flat_world()
    vixen, _, cubs = _fox_family(w)
    tpd = w.cfg.ticks_per_day
    own = vixen.upkeep * tpd
    # Milk the litter needs from birth to weaning if kept topped up (80% efficient).
    milk = 0.0
    for cub in cubs:
        cub.energy = 0.9 * cub.max_energy
        for _ in range(int(cub.weaning)):
            cub.age += 1
            if cub.age % 10 == 0:
                cub._allometry()
            cub.energy -= cub.upkeep
            milk += 0.9 * cub.max_energy - cub.energy
            cub.energy = 0.9 * cub.max_energy
    per_day = milk / 0.8 / (cubs[0].weaning / tpd)
    # Real lactating mammals need roughly 2-4x their usual intake at peak.
    assert 1.5 < (own + per_day) / own < 4.5


def test_parents_carry_part_of_a_kill_home_to_their_young():
    w = flat_world()
    vixen, dog, _ = _fox_family(w)
    dog.x, dog.y = 30.5, 20.5
    dog.energy = 0.5 * dog.max_energy
    dog.carcass = 320.0
    dog.act = Act.EAT
    for _ in range(40):
        dog._eat(w)
        if dog.act != Act.EAT:
            break
    assert dog.act == Act.CACHE and dog.cache_den == vixen.id
    assert dog.energy < 0.8 * dog.max_energy and dog.carried > 20
    assert dog.energy >= PROVISION_SATED * dog.max_energy - dog.intake


def test_young_out_of_the_den_eat_what_their_parents_bring():
    w = flat_world()
    vixen, _, cubs = _fox_family(w)
    cub = cubs[0]
    cub.age = int(cub.weaning * cub.sp.emerge) + 1
    cub.energy = 0.3 * cub.max_energy
    w.add_cache(vixen, 200.0, vixen.id)
    larder = w.caches[-1]
    w._rebuild_buckets()
    cub._nest(w)
    assert cub.energy > 0.3 * cub.max_energy and larder.energy < 200.0


def test_a_mother_finishes_her_stalk_before_going_back_to_nurse():
    w = flat_world()
    vixen, _, _ = _fox_family(w)
    rabbit = put(w, "rabbit", 26.5, 20.5)
    vixen.x, vixen.y = 22.5, 20.5
    vixen.energy = 0.6 * vixen.max_energy
    vixen.act = Act.HUNT
    vixen._set_target(rabbit)
    vixen.next_nurse = w.tick
    w._rebuild_buckets()
    w._think()
    vixen._decide(w)
    assert vixen.act == Act.HUNT
    vixen._set_target(None)
    vixen._decide(w)
    assert vixen.act == Act.VISIT


# ---------------------------------------------------------------- lifetime learning

def test_rewarded_choices_become_more_likely():
    rng = np.random.default_rng(0)
    vec = brain.initial_weights("rabbit").astype(np.float32)
    table = brain.BrainTable(capacity=1)
    row = table.alloc(vec)
    x = np.zeros((1, brain.N_IN), dtype=np.float32)
    x[0, 0], x[0, 10], x[0, 15] = 0.28, 1.0, 1.0  # a little hungry, by day: a close call between eating and not
    mask = np.array([[True, True, False, True, True, False, False, False]])
    none = np.array([-1])

    def p_eat():
        s, _ = brain.forward(x, table.w1[[row]], table.w2[[row]])
        z = np.where(mask, s, -np.inf) / brain.TAU
        p = np.exp(z - z.max())
        return float(p[0, brain.EAT] / p.sum())

    before = p_eat()
    for _ in range(300):
        s, h = brain.forward(x, table.w1[[row]], table.w2[[row]])
        choice, probs = brain.policy(s, mask, none, rng)
        value = table.value([row], h)
        # Eating is rewarded, anything else is not.
        reward = np.array([0.05 if table.primed[row] and last == brain.EAT else 0.0], dtype=np.float32) \
            if table.primed[row] else np.zeros(1, dtype=np.float32)
        table.learn([row], x, h, value, reward, np.ones(1, dtype=np.float32), choice, probs, mask)
        last = int(choice[0])
    assert p_eat() > before + 0.1


def test_learned_wiring_relaxes_back_towards_instinct():
    table = brain.BrainTable(capacity=1)
    row = table.alloc(brain.initial_weights("fox").astype(np.float32))
    table.w2[row] += 0.5
    x = np.zeros((1, brain.N_IN), dtype=np.float32)
    mask = np.ones((1, brain.N_OUT), dtype=bool)
    for _ in range(2000):
        s, h = brain.forward(x, table.w1[[row]], table.w2[[row]])
        table.learn([row], x, h, table.value([row], h), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32),
                    np.array([brain.EXPLORE]), np.full((1, brain.N_OUT), 1 / brain.N_OUT), mask)
    assert np.abs(table.w2[row] - table.i2[row]).max() < 0.5 * 0.75


def test_animals_rewire_from_experience_but_do_not_pass_it_on():
    w = small_world(seed=3)
    for _ in range(400):
        w.step()
    si = _species(w, "rabbit")
    table = w.brains[si]
    adults = [a for a in w.animals if a.si == si and a.mother is None and a.adult]
    changed = [a for a in adults if np.abs(table.w2[a.brain_row] - table.i2[a.brain_row]).max() > 1e-3]
    assert changed  # brains change during life
    parent = changed[0]
    gs = w.genomes[si]
    kid = gs.child(parent.genome, parent.genome, random.Random(1), mutation=0.0)
    inborn = gs.express(kid, random.Random(1)).brain
    w1, w2 = brain.unpack(inborn)
    # The young start from wiring built from genes alone, not from what the parent learned.
    for i in gs.brain_at[:20]:
        assert kid.mat[i] in (parent.genome.mat[i], parent.genome.pat[i])
    assert not np.allclose(w2, table.w2[parent.brain_row], atol=1e-4)


def test_meeting_an_urgent_need_is_the_most_rewarding():
    w = flat_world()
    rabbit = put(w, "rabbit", 20.5, 20.5)
    rabbit.energy, rabbit.water = 0.4 * rabbit.max_energy, 0.4 * rabbit.max_water
    w._rebuild_buckets()
    rabbit.sense(w)
    rabbit.water += 0.3 * rabbit.max_water  # drink when thirsty
    rabbit.sense(w)
    thirsty_drink = rabbit.reward
    rabbit.energy, rabbit.water = 0.4 * rabbit.max_energy, 0.95 * rabbit.max_water
    rabbit.sense(w)
    rabbit.water = rabbit.max_water  # drink when not thirsty
    rabbit.energy -= 0.0
    rabbit.sense(w)
    assert thirsty_drink > 0.05 > rabbit.reward >= 0.0


def test_a_near_miss_hurts():
    w = flat_world()
    rabbit = put(w, "rabbit", 20.5, 20.5)
    w._rebuild_buckets()
    rabbit.sense(w)
    rabbit.reward_events -= 1.0  # what a predator's missed lunge does
    rabbit.sense(w)
    assert rabbit.reward < -0.5


def test_no_learning_when_switched_off():
    w = small_world(seed=3, learning=False)
    for _ in range(200):
        w.step()
    table = w.brains[_species(w, "rabbit")]
    rows = [a.brain_row for a in w.animals if a.si == 0]
    assert np.allclose(table.w2[rows], table.i2[rows])


def test_young_and_quick_learners_are_more_plastic():
    w = flat_world()
    a = put(w, "rabbit", 20.5, 20.5)
    a.traits["learning"] = 0.5
    adult = a.plasticity()
    a.adult = False
    assert a.plasticity() == pytest.approx(adult * brain.YOUNG_PLASTICITY)
    a.traits["learning"] = 0.25
    a.adult = True
    assert a.plasticity() == pytest.approx(1.0)


def test_founders_habits_match_the_instincts():
    h = brain.thresholds(*[m[None] for m in brain.unpack(brain.initial_weights("rabbit").astype(np.float32))], True)
    assert 0.15 <= h["eat"][0] <= 0.4 and 0.15 <= h["drink"][0] <= 0.4
    assert 0.8 <= h["flee"][0] <= 1.25


def test_the_brain_tab_can_show_what_was_learned():
    w = small_world(seed=6)
    for _ in range(120):
        w.step()
    a = next(x for x in w.animals if x.mother is None and x.si == 0)
    b = w.inspect(a.id)["brain"]
    for key in ("born_w1", "born_w2", "change", "differs", "why", "habits", "value", "surprise", "log"):
        assert key in b
    assert len(b["why"]) == brain.N_IN and set(b["habits"]) >= {"eat", "drink", "flee"}
    assert {"learned", "eat_born", "eat_now"} <= set(w.history["species"]["rabbit"]["brain"])


# ---------------------------------------------------------------- fox families

def _fox_world():
    w = flat_world()
    w.env["year_fraction"] = 0.3  # summer: not the dispersal season
    return w


def test_young_foxes_stay_with_their_family_until_autumn():
    w = _fox_world()
    vixen, _, cubs = _fox_family(w)
    w._holders[_species(w, "fox")][vixen.id] = vixen
    vixen.territorial = True
    cub = cubs[0]
    cub.mother = None
    cub.age = math.ceil(cub.maturity) - 1
    cub.update(w)
    if cub.alive:
        assert cub.adult and cub.natal and not cub.territorial
        w.env["year_fraction"] = 0.6  # autumn
        cub.age = (cub.age // 100 + 1) * 100 - 1
        cub.update(w)
        assert not cub.natal


def test_a_helper_inherits_her_mothers_territory():
    w = _fox_world()
    vixen, _, cubs = _fox_family(w)
    si = _species(w, "fox")
    w._holders[si][vixen.id] = vixen
    vixen.territorial = True
    daughter = cubs[0]
    daughter.female, daughter.mother, daughter.adult = True, None, True
    daughter.helper_of = vixen
    home = (vixen.home_x, vixen.home_y)
    w.kill(vixen, "old age")
    assert daughter.territorial and (daughter.home_x, daughter.home_y) == home and daughter.helper_of is None


def test_a_challenger_that_wins_takes_over_the_territory():
    w = _fox_world()
    si = _species(w, "fox")
    resident = put(w, "fox", 20.5, 20.5)
    challenger = put(w, "fox", 22.5, 20.5)
    resident.female = challenger.female = False
    w._holders[si] = {resident.id: resident}
    resident.territorial, challenger.territorial = True, False
    resident.home_x, resident.home_y = 20.5, 20.5
    # A much bigger, well-fed challenger against a starving resident.
    challenger.mass, challenger.energy = 3 * resident.mass, challenger.max_energy
    resident.energy = 0.1 * resident.max_energy
    for _ in range(20):
        w.contest(challenger, resident, 4.0)
        if challenger.territorial:
            break
        challenger.energy = challenger.max_energy
    assert challenger.territorial and not resident.territorial
    assert (challenger.home_x, challenger.home_y) == (20.5, 20.5)


def test_territories_shrink_where_prey_is_plentiful():
    w = flat_world()
    si = _species(w, "fox")
    for i in range(200):
        put(w, "rabbit", 12.5 + (i % 20) * 0.8, 12.5 + (i // 20) * 0.8)
    w._rebuild_buckets()
    rich = w.territory_spacing(si, 20.0, 18.0)
    poor = w.territory_spacing(si, 55.0, 30.0)
    assert rich < w.species[si].territory < poor


def test_a_critical_need_overrides_what_was_learned():
    w = flat_world()
    rabbit = put(w, "rabbit", 20.5, 20.5)
    rabbit.water = 0.15 * rabbit.max_water
    rabbit.energy = 0.9 * rabbit.max_energy
    w._rebuild_buckets()
    rabbit.sense(w)
    allowed = [brain.OPTIONS[k] for k, ok in enumerate(rabbit.mask) if ok]
    assert allowed == ["drink"]


# ---------------------------------------------------------------- simulation clock

def _wait_for(client, condition, timeout=60.0):
    import time as _time
    end = _time.time() + timeout
    while _time.time() < end:
        status = client.get("/api/clock").get_json()
        if condition(status):
            return status
        _time.sleep(0.05)
    raise AssertionError("clock did not get there in time")


def test_skipping_ahead_gives_exactly_the_same_world_as_stepping():
    import app as webapp

    client = webapp.app.test_client()
    client.post("/api/clock", json={"running": False})
    client.post("/api/reset", json={"seed": 11, "settings": {"initial": {"rabbit": 40, "fox": 4}}})
    start = client.get("/api/clock").get_json()["tick"]
    client.post("/api/clock", json={"skip_days": 0.25})
    status = _wait_for(client, lambda s: not s["skip"])
    assert status["tick"] == start + 60 and not status["running"]
    served = client.post("/api/frame", json={"plants": False}).get_json()
    w = World(build_config({"initial": {"rabbit": 40, "fox": 4}}), seed=11)
    for _ in range(status["tick"]):
        w.step()
    assert served["animals"] == w.frame(plants=False)["animals"]


def test_the_clock_plays_and_pauses():
    import app as webapp

    client = webapp.app.test_client()
    client.post("/api/reset", json={"seed": 12, "settings": {"initial": {"rabbit": 30, "fox": 3}}})
    t0 = client.post("/api/clock", json={"running": True, "speed": 10}).get_json()["tick"]
    _wait_for(client, lambda s: s["tick"] >= t0 + 10)
    paused = client.post("/api/clock", json={"running": False}).get_json()
    import time as _time
    _time.sleep(0.3)
    later = client.get("/api/clock").get_json()
    assert not later["running"] and later["tick"] - paused["tick"] <= 1
    frame = client.post("/api/frame", json={}).get_json()
    assert frame["clock"]["tick"] == frame["tick"]


# ---------------------------------------------------------------- fruit, blind newborns, gestation, injuries, courtship

from engine.animal import BIRTH_MASS, VISION_FLOOR  # noqa: E402
from engine.vegetation import FRUIT_CAP, Fruit  # noqa: E402


def test_newborns_are_blind_until_their_eyes_open():
    w = flat_world()
    _, _, cubs = _fox_family(w)
    cub = cubs[0]
    assert cub.sight() == 0.0 and cub.vision == VISION_FLOOR
    cub.age = int(cub.sp.eyes_open * cub.weaning) + 20
    cub._allometry()
    assert 0.0 < cub.sight() < 1.0 and VISION_FLOOR < cub.vision < cub.traits["vision"]
    cub.age = int(cub.weaning) + 1
    cub._allometry()
    assert cub.vision == pytest.approx(cub.traits["vision"])


def test_longer_pregnancies_give_bigger_young():
    sizes = {}
    for g in (0.9, 1.15):
        w = flat_world()
        vixen = put(w, "fox", 20.5, 20.5)
        dog = put(w, "fox", 22.5, 20.5)
        vixen.female, dog.female = True, False
        vixen.traits["gestation"] = g
        vixen.energy = vixen.max_energy
        w.mate(vixen, dog)
        w.give_birth(vixen)
        w._flush()
        kids = [k for k in vixen.litter if k.alive]
        if not kids:
            pytest.skip("no young survived to birth")
        sizes[g] = kids[0].birth_frac
    assert sizes[1.15] == pytest.approx(BIRTH_MASS * 1.15 ** 3)
    assert sizes[1.15] > sizes[0.9]


def test_injuries_slow_an_animal_and_heal():
    w = flat_world()
    fox = put(w, "fox", 20.5, 20.5)
    fast = fox.speed
    fox.hurt(w, 0.4)
    assert fox.speed < fast and fox.reward_events < 0
    fox.energy = fox.max_energy
    for _ in range(600):
        fox.energy = fox.max_energy
        fox.water = fox.max_water
        fox.update(w)
        if not fox.alive:
            pytest.skip("died of other causes")
    assert fox.injury < 0.4


def test_females_turn_down_weak_and_injured_males_and_vixens_are_choosier():
    w = flat_world()
    vixen, dog = put(w, "fox", 20.5, 20.5), put(w, "fox", 22.5, 20.5)
    doe, buck = put(w, "rabbit", 20.5, 22.5), put(w, "rabbit", 22.5, 22.5)
    vixen.female = doe.female = True
    dog.female = buck.female = False
    for m in (dog, buck):
        m.energy = 0.9 * m.max_energy
        m.vigor = 1.0
    healthy = (w.acceptance(vixen, dog), w.acceptance(doe, buck))
    for m in (dog, buck):
        m.injury = 0.6
        m.energy = 0.4 * m.max_energy
    impaired = (w.acceptance(vixen, dog), w.acceptance(doe, buck))
    assert healthy[0] > impaired[0] and healthy[1] > impaired[1]
    assert impaired[0] < impaired[1]  # vixens are choosier than does


def test_a_stronger_rival_usually_drives_a_weaker_male_off():
    w = flat_world()
    strong, weak = put(w, "rabbit", 20.5, 20.5), put(w, "rabbit", 21.5, 20.5)
    wins = 0
    for _ in range(200):
        strong.energy, weak.energy = strong.max_energy, 0.3 * weak.max_energy
        weak.injury = 0.5
        strong.injury = 0.0
        wins += not w.mating_contest(weak, strong)
    assert wins > 150


def test_fruit_ripens_from_late_summer_and_is_gone_by_winter():
    assert Fruit.ripeness(0.1, True) == 0.0  # spring
    assert Fruit.ripeness(0.55, True) == pytest.approx(1.0)  # early autumn
    assert Fruit.ripeness(0.85, True) == 0.0  # winter


def test_foxes_eat_ripe_fruit():
    w = flat_world()
    w.fruit.bush[:] = False
    w.fruit.F[:] = 0.0
    w.fruit.bush[20, 24] = True
    w.fruit.F[20, 24] = FRUIT_CAP
    fox = put(w, "fox", 20.5, 20.5)
    fox.energy = 0.5 * fox.max_energy
    fox.skill = 0.2  # a young fox: fruit first
    fox.act = Act.HUNT
    w._rebuild_buckets()
    for _ in range(40):
        fox._hunt(w) if fox.act == Act.HUNT else fox._eat(w)
        if w.hunts[_species(w, "fox")]["fruit"]:
            break
    assert w.hunts[_species(w, "fox")]["fruit"] == 1 and w.fruit.F[20, 24] < FRUIT_CAP


def test_the_memory_map_shows_caches_den_territory_and_marks():
    w = flat_world()
    vixen, dog, _ = _fox_family(w)
    si = _species(w, "fox")
    vixen.territorial = True
    w._holders[si][vixen.id] = vixen
    w.add_cache(vixen, 120.0, vixen.id)
    vixen.marks.append((25.0, 20.0, w.tick))
    m = w.inspect(vixen.id)["memory"]
    assert m["caches"] and m["caches"][0][3]  # food carried home for the young
    assert m["den"] is not None and m["territory"] is not None and m["marks"]


# ---------------------------------------------------------------- pair territories, forays, voles

from engine.rodents import VOLE_CAP, Voles  # noqa: E402


def _hold(w, a, x, y, female):
    a.female = female
    a.home_x, a.home_y = x, y
    a.territorial = True
    w._holders[_species(w, "fox")][a.id] = a


def test_a_widowed_territory_is_filled_by_a_newcomer():
    w = flat_world()
    si = _species(w, "fox")
    w._holders[si] = {}
    vixen = put(w, "fox", 20.5, 20.5)
    _hold(w, vixen, 20.5, 20.5, True)
    # Another male's territory nearby would have blocked the old same-sex rule.
    neighbour = put(w, "fox", 40.5, 20.5)
    _hold(w, neighbour, 40.5, 20.5, False)
    floater = put(w, "fox", 60.5, 20.5)
    floater.female, floater.territorial = False, False
    w._holders[si].pop(floater.id, None)
    assert w.settle(floater, 60.0)
    assert (floater.home_x, floater.home_y) == (20.5, 20.5)
    assert w.territory_mates(vixen) == [floater]


def test_new_territories_keep_their_distance_from_every_other():
    w = flat_world()
    si = _species(w, "fox")
    w._holders[si] = {}
    vixen = put(w, "fox", 20.5, 20.5)
    _hold(w, vixen, 20.5, 20.5, True)
    male = put(w, "fox", 25.5, 20.5)
    male.female, male.territorial = False, False
    w._holders[si].pop(male.id, None)
    # He may join her territory, but not found a separate one right next to it.
    assert not w.claim_territory(male, 25.5, 20.5)
    assert w.claim_territory(male, 20.5, 20.5)


def test_lone_neighbours_pair_up():
    w = flat_world()
    si = _species(w, "fox")
    w._holders[si] = {}
    vixen, dog = put(w, "fox", 20.5, 20.5), put(w, "fox", 30.5, 20.5)
    _hold(w, vixen, 20.5, 20.5, True)
    _hold(w, dog, 30.5, 20.5, False)
    assert w.pair_up(dog)
    assert (dog.home_x, dog.home_y) == (20.5, 20.5)


def test_dog_foxes_range_further_for_mates_than_vixens():
    w = flat_world()
    vixen, dog = put(w, "fox", 20.5, 20.5), put(w, "fox", 22.5, 20.5)
    vixen.female, dog.female = True, False
    assert w.mate_reach(dog) > w.mate_reach(vixen) == vixen.sp.mate_range


def test_voles_breed_in_summer_and_dwindle_in_winter():
    w = flat_world()
    v = Voles(w.veg)
    v.V[:] = 0.2 * VOLE_CAP
    start = v.total()
    for _ in range(300):
        v.step(w.veg, breeding=True)
    summer = v.total()
    for _ in range(300):
        v.step(w.veg, breeding=False)
    assert summer > start and v.total() < summer


def test_foxes_catch_voles_by_mousing():
    w = flat_world()
    fox = put(w, "fox", 20.5, 20.5)
    w.voles.V[:] = VOLE_CAP
    fox.act = Act.HUNT
    fox.energy = 0.6 * fox.max_energy
    w._rebuild_buckets()
    si = _species(w, "fox")
    for _ in range(80):
        if fox.act == Act.HUNT:
            fox._hunt(w)
        else:
            fox._eat(w)
            fox.act = Act.HUNT
        if w.hunts[si]["vole"]:
            break
    assert w.hunts[si]["vole"] >= 1


# ---------------------------------------------------------------- prey switching, super animals, movement

from engine.animal import SEARCH_IMAGE_FLOOR  # noqa: E402
from engine.config import supercharge  # noqa: E402


def test_a_foxs_eye_for_rabbits_fades_when_it_meets_none_and_returns_when_it_does():
    w = flat_world()
    fox = put(w, "fox", 20.5, 20.5)
    fox.search_image = 1.0
    for _ in range(1000):
        fox.energy, fox.water = fox.max_energy, fox.max_water
        fox.update(w)
    assert fox.search_image < 0.6 and fox.search_image >= SEARCH_IMAGE_FLOOR
    fox.x, fox.y = 20.5, 20.5
    put(w, "rabbit", 21.5, 20.5)  # right under its nose: always noticeable
    w._rebuild_buckets()
    faded = fox.search_image
    for _ in range(20):
        if w.find_prey(fox):
            break
    assert fox.search_image > faded


def test_super_animals_have_higher_ceilings_and_crowned_alphas():
    from engine.config import RABBIT
    s = supercharge(RABBIT)
    assert s.traits["mass"].hi > RABBIT.traits["mass"].hi and s.traits["learning"].hi <= 0.95
    w = small_world(seed=4, super_animals=True)
    for _ in range(40):
        w.step()
    assert any(w.alphas)
    a = w.by_id[next(i for al in w.alphas for i in al)]
    assert w.inspect(a.id)["alpha"] in (1, 2, 3)
    assert w.frame()["animals"][[x[0] for x in w.frame()["animals"]].index(a.id)][6] & 128


def test_a_long_stride_cannot_hop_over_water():
    w = flat_world(walls=[(21, 0, 22, 40)])  # a one-cell-wide stream at x = 21
    rabbit = put(w, "rabbit", 20.4, 20.5)
    for _ in range(50):
        rabbit.x, rabbit.y = 20.4, 20.5
        movement.step(rabbit, w, 0.0, 2.5, math.inf)
        assert rabbit.x < 21.0


# ---------------------------------------------------------------- the ledger, the AI analyst, settings

import io  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import urllib.error  # noqa: E402
import zlib  # noqa: E402
from collections import Counter  # noqa: E402

from engine.animal import EMIGRATE_AFTER, TERRITORY_RETRY  # noqa: E402
from engine.config import settings_of  # noqa: E402
from engine.report import evidence, ledger_series  # noqa: E402


def test_the_ledger_keeps_a_daily_diary_that_adds_up():
    w = small_world(seed=2, end_on_extinction=False)
    tpd = w.cfg.ticks_per_day
    for _ in range(3 * tpd):
        w.step()
    rows = w.ledger.rows
    assert [r["day"] for r in rows] == [0, 1, 2, 3]
    for si in range(len(w.species)):
        steps = [w.ledger.delta(b, a, si) for a, b in zip(rows, rows[1:])]
        assert sum(d["births"] for d in steps) == w.births[si]
        assert sum((d["deaths"] for d in steps), Counter()) == +w.deaths[si]
        assert rows[-1]["species"][si]["n"] == w.counts()[si]
    assert w.ledger.parent_traits[0][-1] == w.births[0]  # each young counts its parents once
    assert w.ledger.dead_traits[0][:, -1].sum() == sum(w.deaths[0].values())


def test_an_extinction_is_recorded_with_its_last_deaths_and_a_recolonisation():
    w = small_world(seed=4, end_on_extinction=False, immigration=0)
    foxes = [a for a in w.animals if a.si == 1]
    for a in foxes:
        w.kill(a, "removed")
    w._flush()
    w.step()
    (ext,) = w.ledger.extinctions
    assert ext["species"] == "fox" and len(ext["last_deaths"]) == len(foxes)
    assert ext["last_deaths"][-1]["cause"] == "removed" and ext["recolonised_day"] is None
    w.spawn(1, 30, 20, 2)
    w.step()
    assert ext["recolonised_day"] == w.env["day"]


def test_sandbox_actions_are_logged_and_brush_strokes_merged():
    w = flat_world()
    a = w.paint(10, 10, 2, "water")
    b = w.paint(13, 10, 2, "water")
    painted = [x for x in w.ledger.actions if x["kind"] == "painted terrain"]
    assert len(painted) == 1 and painted[0]["amount"] == a + b and painted[0]["detail"] == "water"
    w.trigger("drought")
    assert w.ledger.actions[-1]["kind"] == "event" and w.ledger.actions[-1]["detail"] == "drought"


def test_the_ai_brief_covers_the_run_and_includes_a_map_picture():
    w = small_world(seed=3, end_on_extinction=False, immigration=0)
    for _ in range(2 * w.cfg.ticks_per_day):
        w.step()
    for a in [a for a in w.animals if a.si == 1]:
        w.kill(a, "removed")
    w._flush()
    w.step()
    brief = evidence(w, "extinction", "fox")
    text = brief["text"]
    for heading in ("## How this world works", "## Totals for the whole run", "## Timeline of the whole run",
                    "## How they evolved", "foxes to die before the extinction", "## The map, by region",
                    "## What the user did"):
        assert heading in text, heading
    assert brief["meta"]["focus"]["species"] == "fox" and brief["meta"]["extinctions"][0]["day"] == 3
    png = brief["png"]
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (4 * w.width, 4 * w.height)
    idat_len = struct.unpack(">I", png[33:37])[0]
    assert len(zlib.decompress(png[41:41 + idat_len])) == height * (1 + 3 * width)
    series = ledger_series(w)
    assert len(series["tick"]) == len(w.ledger.rows) - 1
    assert sum(series["species"]["rabbit"]["births"]) == w.births[0]


def test_env_file_parsing_and_a_missing_key(tmp_path, monkeypatch):
    import ai
    f = tmp_path / ".env"
    f.write_text('# comment\nANTHROPIC_API_KEY="sk-test"\nexport AI_MODEL=claude-x  # cheaper\nEMPTY=\n', encoding="utf-8")
    assert ai.read_env(f) == {"ANTHROPIC_API_KEY": "sk-test", "AI_MODEL": "claude-x", "EMPTY": ""}
    monkeypatch.setattr(ai, "ENV_PATH", f)
    assert ai.settings()["key"] == "sk-test" and ai.settings()["model"] == "claude-x"
    f.write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AI_MODEL", raising=False)
    assert ai.settings()["key"] == "" and ai.settings()["model"] == ai.DEFAULT_MODEL


class _FakeResponse:
    def __init__(self, lines):
        self.lines = [line.encode() for line in lines]

    def __iter__(self):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_ai_client_reads_claudes_streamed_answer(monkeypatch):
    import ai
    events = [
        "event: message_start",
        'data: {"type":"message_start","message":{"model":"claude-test","usage":{"input_tokens":1234}}}', "",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Foxes "}}',
        'data: {"type":"ping"}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"starved."}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
        'data: {"type":"message_stop"}',
    ]
    seen = {}

    def fake_urlopen(req, timeout):
        seen.update(url=req.full_url, headers=dict(req.header_items()), body=json.loads(req.data))
        return _FakeResponse(events)

    monkeypatch.setattr(ai.urllib.request, "urlopen", fake_urlopen)
    conf = {"key": "sk-test", "model": "claude-test", "base_url": "https://example.invalid"}
    out = list(ai.stream(conf, "system", [{"role": "user", "content": "hi"}], 100))
    assert "".join(v for k, v in out if k == "text") == "Foxes starved."
    assert out[-1] == ("done", {"model": "claude-test", "input_tokens": 1234, "output_tokens": 5, "stop_reason": "end_turn"})
    assert seen["url"] == "https://example.invalid/v1/messages" and seen["body"]["stream"] is True
    assert seen["headers"]["X-api-key"] == "sk-test" and seen["headers"]["Anthropic-version"] == ai.API_VERSION

    def rejected(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":{"message":"invalid x-api-key"}}'))

    monkeypatch.setattr(ai.urllib.request, "urlopen", rejected)
    with pytest.raises(ai.AIError, match="rejected"):
        list(ai.stream(conf, "system", [], 100))


def test_the_ai_endpoint_streams_an_answer_and_reuses_it(monkeypatch):
    import ai
    import app as webapp
    client = webapp.app.test_client()
    client.post("/api/reset", json={"seed": 3, "settings": {"width": 64, "height": 40, "initial": {"rabbit": 30, "fox": 3}}})
    for _ in range(5):
        client.post("/api/step", json={"ticks": 60})
    monkeypatch.setattr(ai, "settings", lambda: {"key": "", "model": "m", "base_url": "x"})
    res = client.post("/api/ai", json={"kind": "summary"})
    assert res.status_code == 400 and res.get_json()["setup"]

    calls = []

    def fake_analyse(conf, kind, brief, species=None, question=None, finished=False):
        calls.append(kind)
        yield "text", "Rabbits "
        yield "text", "boomed."
        yield "done", {"model": "claude-test", "stop_reason": "end_turn"}

    monkeypatch.setattr(ai, "settings", lambda: {"key": "sk-test", "model": "claude-test", "base_url": "x"})
    monkeypatch.setattr(ai, "analyse", fake_analyse)
    lines = [json.loads(x) for x in client.post("/api/ai", json={"kind": "question", "question": "Why?"}).get_data(as_text=True).splitlines()]
    assert "".join(x.get("text", "") for x in lines) == "Rabbits boomed." and lines[-1]["done"]
    again = [json.loads(x) for x in client.post("/api/ai", json={"kind": "question", "question": "why?"}).get_data(as_text=True).splitlines()]
    assert again[-1].get("cached") and calls == ["question"]
    eid = lines[0]["meta"]["evidence"]
    assert "## Timeline of the whole run" in client.get(f"/api/ai/evidence/{eid}.txt").get_data(as_text=True)
    assert client.get(f"/api/ai/evidence/{eid}.png").data[:4] == b"\x89PNG"
    assert client.post("/api/ai", json={"kind": "extinction", "species": "fox"}).status_code == 400
    assert client.get("/api/ledger").get_json()["species"]["rabbit"]["n"]


def test_emigration_can_be_switched_off():
    for emigration in (True, False):
        w = flat_world()
        w.cfg = replace(w.cfg, emigration=emigration)
        fox = put(w, "fox", 20.5, 20.5)
        w._holders[fox.si].pop(fox.id, None)
        fox.territorial, fox.natal, fox.helper_of = False, False, None
        fox.failed_claims = EMIGRATE_AFTER
        w.settle = lambda *args, **kw: False  # nowhere to settle
        for _ in range(TERRITORY_RETRY + 2):
            fox.energy, fox.water = fox.max_energy, fox.max_water
            fox.update(w)
            if not fox.alive:
                break
        assert (fox.cause == "emigrated") == emigration


def test_super_animals_young_start_with_their_parents_know_how():
    w = small_world(seed=5, super_animals=True, inherit_learning=0.5, initial={"rabbit": 0, "fox": 0}, immigration=0)
    assert w.teaching and not small_world(seed=5).teaching
    ids = w.spawn(0, 30, 20, 8)
    mother = next(w.by_id[i] for i in ids if w.by_id[i].female)
    father = next(w.by_id[i] for i in ids if not w.by_id[i].female)
    table = w.brains[0]
    table.w1[mother.brain_row] += 0.1  # what she has learned in life
    mother.experience = 0.8
    w.mate(mother, father)
    mother.energy = mother.max_energy
    w.give_birth(mother)
    w._flush()
    young = [w.by_id[i] for i in mother.children_ids if i in w.by_id]
    assert young
    for y in young:
        # Half of the parents' average know-how (hers +0.1, his 0) is built into the wiring it starts with.
        assert np.allclose(table.c1[y.brain_row], 0.025, atol=1e-6)
        assert np.allclose(table.w1[y.brain_row], table.i1[y.brain_row])
        assert y.experience == pytest.approx(0.4)


def test_map_size_is_a_setting_with_safe_limits():
    cfg = build_config({"width": 10, "height": 999})
    assert (cfg.width, cfg.height) == (40, 160)
    w = World(build_config({"width": 48, "height": 32, "initial": {"rabbit": 16, "fox": 2}}), seed=1)
    for _ in range(100):
        w.step()
    assert w.terrain.types.shape == (32, 48) and settings_of(w.cfg)["width"] == 48


def test_the_urge_to_mate_builds_in_season_is_sated_by_mating_and_fades_in_pregnancy():
    from engine.animal import REWARD_MATING
    w = small_world(seed=5, initial={"rabbit": 0, "fox": 0}, immigration=0)
    assert w.in_season[0]  # rabbits breed from the start of spring
    (fid,) = w.spawn(0, 30, 20, 1)
    female = w.by_id[fid]
    if not female.female:
        female.female = True
    for _ in range(200):  # alone, so she can't mate: the urge builds
        female.energy, female.water = female.max_energy, female.max_water
        female.update(w)
    assert female.urge > 0.6
    # It is felt, but costs the body nothing: her reserves were only topped up, never drained by it.
    w._rebuild_buckets()
    female.sense(w)
    male = w.by_id[w.spawn(0, 30.5, 20, 1)[0]]
    male.female = False
    w.mate(female, male)
    assert female.urge == 0.0 and male.urge == 0.0
    female.sense(w)
    assert female.reward > REWARD_MATING  # the pleasure of mating plus relief from the urge
    female.urge = 0.8
    for _ in range(80):  # pregnant now: the hormones fall and the urge fades (half-life about 35 ticks)
        female.energy, female.water = female.max_energy, female.max_water
        female.update(w)
    assert female.pregnant and female.urge < 0.2


# ---------------------------------------------------------------- climates, camouflage and the winter moult

from engine.climate import BIOMES, CLIMATES  # noqa: E402
from engine.coats import WINTER_LUM  # noqa: E402
from engine.terrain import SAND, WALKABLE  # noqa: E402


def climate_world(name, seed=3, **settings):
    cfg = build_config({"climate": name, "width": 64, "height": 40, "initial": {"rabbit": 20, "fox": 2}, **settings})
    return World(cfg, seed=seed)


def test_a_temperate_map_is_the_same_everywhere():
    w = climate_world("temperate")
    assert w.climate.uniform and not w.climate.offset.any()
    assert w.climate.growth(0.7, -0.5, 0) == 0.7  # plain seasonal growth, no per-cell field
    assert w.cold_heat(10, 10) == (w.cold, w.heat)


def test_climates_set_snow_and_heat_where_they_should():
    arctic, desert = climate_world("arctic"), climate_world("desert")
    assert arctic.climate.snow_at(-1.0, 30, 20) > 0.9  # deep winter snow
    assert arctic.climate.snow_at(1.0, 30, 20) == 0.0  # but a snow-free summer
    assert desert.climate.snow_at(-1.0, 30, 20) < 0.3  # too warm and dry for much snow
    assert desert.climate.temperature(1.0, 30, 20) > 1.4  # a scorching summer
    land = WALKABLE[desert.terrain.types]
    sand = (desert.terrain.types == SAND)[land].mean()
    assert 0.5 < sand < 0.9  # mostly sand, with patches of scrub to graze


def test_a_mixed_map_runs_from_cold_north_to_hot_dry_south():
    w = climate_world("mixed", width=120, height=80)
    off, dry = w.climate.offset, w.climate.dry
    assert off[:10].mean() < off[-10:].mean() - 1.0
    assert dry[-10:].mean() > dry[:10].mean() + 0.6
    found = set(BIOMES[k] for k in np.unique(w.climate.biomes(w.terrain)))
    assert {"tundra", "desert"} <= found


def test_climates_bring_their_landscape_unless_told_otherwise():
    assert build_config({"climate": "desert"}).water_fraction == CLIMATES["desert"]["terrain"]["water"] / 100
    assert build_config({"climate": "desert", "water_fraction": 0.2}).water_fraction == 0.2
    assert build_config({"climate": "volcano"}).climate == "temperate"


def test_camouflage_depends_on_coat_against_the_ground_under_the_animal():
    from engine.terrain import FOREST
    w = climate_world("arctic", initial={"rabbit": 0, "fox": 0}, immigration=0)
    w.light, w.moult = 1.0, 0.0
    rabbit = w.by_id[w.spawn(0, 30, 20, 1)[0]]
    rabbit.moults, rabbit.last_move = False, 0.0
    coat = {c.name: i for i, c in enumerate(w.genomes[0].coats)}
    cx, cy = int(rabbit.x), int(rabbit.y)

    def seen(name, kind=GRASS, temp=1.0, brown=0.0):
        rabbit.coat = coat[name]
        w.terrain.types[cy, cx] = kind
        w.veg.B[cy, cx] = w.veg.K[cy, cx]
        w.season_temp, w.grass_brown = temp, brown
        return w.visibility(rabbit)

    # Snow hides white and shows up dark; summer grass hides the wild agouti best;
    # dark leaf litter hides black.
    assert seen("Albino", temp=-1.0) < seen("Chestnut agouti", temp=-1.0) < seen("Black", temp=-1.0)
    assert seen("Chestnut agouti") < min(seen("Black"), seen("Albino"))
    assert seen("Black", FOREST) < seen("Chestnut agouti", FOREST) < seen("Albino", FOREST)
    # Moving gives it away; in the dark colour hardly matters.
    still = seen("Chestnut agouti")
    rabbit.last_move = rabbit.speed
    assert w.visibility(rabbit) > still
    rabbit.last_move, w.light = 0.0, 0.0
    assert w.visibility(rabbit) == pytest.approx(1.0)


def test_foxes_and_rabbits_see_an_orange_fox_much_like_green_grass():
    # Dichromats: orange and green differ mostly in lightness to them, so a red fox
    # is far better hidden in grass than a white or a black animal would be.
    from engine.coats import tone
    from engine.world import CHROMA, GROUND
    import math
    gap = lambda c: math.hypot(tone(c)[0] - GROUND["lush"][0], CHROMA * (tone(c)[1] - GROUND["lush"][1]))  # noqa: E731
    red, black, white = gap("#d9622b"), gap("#2b2727"), gap("#f4f1ea")
    assert red < 0.2 and red < black < white


def test_winter_white_animals_moult_by_day_length():
    w = climate_world("arctic", initial={"rabbit": 0, "fox": 0}, immigration=0)
    gs = w.genomes[0]
    rabbit = w.by_id[w.spawn(0, 30, 20, 1, "Winter white")[0]]
    assert rabbit.moults and not w.by_id[w.spawn(0, 30, 20, 1, "Wild type")[0]].moults
    w.moult = 1.0
    assert w.coat_lum_now(rabbit) == pytest.approx(WINTER_LUM)
    w.moult = 0.0
    assert w.coat_lum_now(rabbit) == pytest.approx(gs.coats[rabbit.coat].lum)
    tpd, dpy = w.cfg.ticks_per_day, w.cfg.days_per_year
    w.tick = int(10.5 * tpd)  # midwinter
    w._environment()
    assert w.env["moult"] > 0.99
    w.tick = int(4.5 * tpd)  # midsummer
    w._environment()
    assert w.env["moult"] == 0.0


def test_the_cold_costs_more_in_the_arctic_than_in_a_temperate_winter():
    costs = []
    for name in ("temperate", "arctic"):
        w = climate_world(name, initial={"rabbit": 0, "fox": 0}, immigration=0)
        rabbit = w.by_id[w.spawn(0, 30, 20, 1)[0]]
        rabbit.concealed = False
        w.season_temp, w.cold, w.heat = -0.5, 0.5, 0.0
        costs.append(rabbit.thermal_cost(w))
    assert costs[1] > costs[0] > 1.0
