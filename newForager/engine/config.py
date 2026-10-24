"""Simulation parameters.

The forms of the models and their scaling exponents come from the ecological
literature (see README.md). Absolute constants are calibrated for a watchable
timescale: a day is 240 ticks and a year is compressed to 12 days, so seasons
and generations play out in minutes rather than months.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True, kw_only=True)
class Trait:
    """A polygenic (quantitative) trait built from many small-effect loci."""

    mean: float
    cv: float  # phenotypic coefficient of variation among founders
    heritability: float  # share of the founders' variance that is additive genetic (h^2)
    lo: float
    hi: float
    loci: int = 5
    label: str = ""
    unit: str = ""


@dataclass(frozen=True, kw_only=True)
class Gene:
    """A Mendelian locus with discrete alleles."""

    symbol: str
    name: str
    alleles: tuple[str, ...]  # in order of dominance: earlier alleles mask later ones
    freqs: tuple[float, ...]  # founding allele frequencies
    labels: tuple[str, ...] = ()  # display names, if different from the allele symbols
    incomplete: bool = False  # heterozygotes are intermediate instead of dominant
    lethal: str | None = None  # allele that is lethal to homozygous embryos
    note: str = ""


@dataclass(frozen=True, kw_only=True)
class Species:
    name: str
    icon: str
    color: str
    diet: str  # "plants" or "meat"
    prey: tuple[str, ...] = ()
    initial: int

    # Genetics.
    traits: dict[str, Trait]
    genes: tuple[Gene, ...]
    coat: str  # which coat-colour model in coats.py
    chromosomes: int = 6  # autosome pairs
    chromosome_length: float = 1.0  # Morgans; the expected number of crossovers per meiosis
    load_loci: int = 24  # loci where recessive deleterious mutations can arise
    load_freq: float = 0.02  # founding frequency of deleterious alleles at each
    load_s: float = 0.25  # fitness cost of a homozygous deleterious locus
    load_h: float = 0.1  # dominance of deleterious alleles (0 = fully recessive)
    markers: int = 10  # neutral microsatellite-like loci
    marker_alleles: int = 8

    # Life history at the reference mass, scaled by (mass / ref_mass) ** 0.25.
    ref_mass: float
    lifespan: float
    maturity: float
    gestation: float
    refractory: float  # minimum ticks between a female's conceptions
    weaning: float
    breeding: tuple[float, float]  # breeding season as fractions of the year (0 = start of spring)

    # Daily rhythm: idle animals rest from this many hours after sunrise
    # until this many hours before sunset.
    rest_after_sunrise: float
    rest_before_sunset: float

    # Parental care. Young stay hidden in the nest or den and are nursed on
    # visits; they come out to explore at `emerge` x weaning age.
    nurse_interval: float = 240.0  # ticks between a mother's nursing visits
    nurse_ticks: int = 2  # length of one nursing visit
    emerge: float = 0.7
    nest_prep: float = 40.0  # ticks before birth that a mother heads home to her nest
    provisions: bool = False  # fathers bring food to the den

    # Dispersal: multipliers on the mean distance by sex, and the chance a
    # daughter stays in her natal home (female philopatry).
    male_dispersal: float = 1.0
    female_dispersal: float = 1.0
    philopatry: float = 0.0

    # Energetics, in arbitrary energy units.
    energy_per_kg: float
    bmr: float  # basal metabolic rate = bmr * M^0.75 (Kleiber 1932)
    cot: float  # cost of transport per cell = cot * M^0.68 (Taylor et al. 1982)
    vision_cost: float  # upkeep at the founding mean vision; scales with radius^2
    muscle_cost: float  # upkeep at the founding mean speed; scales with speed^2
    stamina_cost: float = 0.0  # upkeep at the founding mean stamina; scales with stamina^2
    learning_cost: float = 0.0  # upkeep of a fast-learning brain at the founding mean rate; scales with rate^2
    gestation_cost: float
    intake: float  # max ingestion per tick = intake * M^0.75
    assimilation: float
    investment: float  # fraction of the mother's reserves passed to the litter at birth

    # Water balance.
    water_per_kg: float
    water_loss: float
    drink_rate: float

    # Locomotion.
    cruise: float
    stamina: float  # ticks of sprinting, unless stamina is a heritable trait
    stamina_regen: float
    kappa_explore: float
    kappa_ars: float
    ars_duration: float
    home_radius: float
    dispersal: float
    mate_range: float
    territory: float = 0.0
    hazard: float = 0.0  # extra background mortality per tick (diseases such as mange, accidents)
    eyes_open: float = 0.3  # newborns are blind until this fraction of the weaning age
    choosiness: float = 0.5  # how strongly females turn down males in poor shape (0 = not at all)
    eats_fruit: bool = False  # forages for wild fruit in season
    eats_voles: bool = False  # hunts small rodents by mousing (see rodents.py)
    # Territory size follows food (the resource dispersion hypothesis, Macdonald
    # 1983): new territories are spaced `territory` apart at this prey density
    # (per 1000 land cells), closer where prey are richer, wider where poorer.
    territory_prey: float = 0.0
    # Family groups: young stay with their parents until the dispersal season
    # (fractions of the year), and some daughters stay on as non-breeding helpers.
    dispersal_season: tuple[float, float] | None = None
    helper_chance: float = 0.0  # chance a daughter stays as a helper where food is plentiful
    max_helpers: int = 0

    # Herbivores.
    half_saturation: float = 0.3
    graze_min: float = 0.0
    giving_up: float = 0.0

    # Predators.
    capture_radius: float = 0.0
    capture_success: float = 0.0
    rush_distance: float = 0.0
    stalk: float = 1.0  # stalking pace as a fraction of cruising pace
    caches: bool = False  # buries surplus food to eat later

    # As prey.
    carcass_energy: float = 0.0
    burrow_reach: float = 0.0
    fid: float = 0.0  # flight initiation distance in cells (0 = always flee at once)
    fid_per_cell: float = 0.0  # extra flight distance per cell away from the burrow
    thumps: bool = False  # drums an alarm that alerts neighbours


RABBIT = Species(
    name="rabbit",
    icon="🐇",
    color="#efe6d2",
    diet="plants",
    initial=160,
    traits={
        "mass": Trait(mean=1.6, cv=0.1, heritability=0.5, lo=0.5, hi=5.0, label="Body mass", unit="kg"),
        "speed": Trait(mean=0.85, cv=0.08, heritability=0.4, lo=0.25, hi=2.0, label="Top speed", unit="cells/tick"),
        "vision": Trait(mean=6.0, cv=0.1, heritability=0.4, lo=1.5, hi=20.0, label="Vision", unit="cells"),
        "litter": Trait(mean=4.0, cv=0.15, heritability=0.3, lo=1.0, hi=12.0, label="Litter size", unit="young"),
        "vigilance": Trait(mean=1.0, cv=0.15, heritability=0.35, lo=0.3, hi=3.0, label="Vigilance", unit="×"),
        "insulation": Trait(mean=1.0, cv=0.12, heritability=0.5, lo=0.4, hi=2.5, label="Fur insulation", unit="×"),
        # Personality (heritable behavioural tendencies; Reale et al. 2007).
        "boldness": Trait(mean=1.0, cv=0.2, heritability=0.3, lo=0.3, hi=3.0, loci=4, label="Boldness", unit="×"),
        "freeze": Trait(mean=1.0, cv=0.2, heritability=0.3, lo=0.2, hi=3.0, loci=4, label="Freezing tendency", unit="×"),
        "appetite": Trait(mean=1.0, cv=0.15, heritability=0.3, lo=0.4, hi=2.5, loci=4, label="Appetite", unit="×"),
        "libido": Trait(mean=1.0, cv=0.15, heritability=0.3, lo=0.3, hi=2.5, loci=4, label="Mating drive", unit="×"),
        "learning": Trait(mean=0.25, cv=0.3, heritability=0.3, lo=0.02, hi=0.9, loci=4, label="Learning rate", unit=""),
        # Longer pregnancies give bigger, better-provisioned young but cost the mother more.
        "gestation": Trait(mean=1.0, cv=0.04, heritability=0.3, lo=0.85, hi=1.2, loci=4, label="Gestation length", unit="×"),
    },
    genes=(
        Gene(symbol="A", name="Agouti", alleles=("A", "a"), freqs=(0.8, 0.2),
             note="a/a removes the banded agouti hairs, giving a solid (self) coat"),
        Gene(symbol="B", name="Brown", alleles=("B", "b"), freqs=(0.85, 0.15),
             note="b/b turns black pigment chocolate"),
        Gene(symbol="C", name="Colour", alleles=("C", "cchd", "ch", "c"), freqs=(0.86, 0.05, 0.04, 0.05),
             labels=("C", "cᶜʰᵈ", "cʰ", "c"),
             note="the albino series: chinchilla > Himalayan > albino"),
        Gene(symbol="D", name="Dilute", alleles=("D", "d"), freqs=(0.85, 0.15),
             note="d/d dilutes black to blue and chocolate to lilac"),
        Gene(symbol="En", name="English spotting", alleles=("en", "En"), freqs=(0.93, 0.07), incomplete=True,
             note="En/en is broken-patterned; En/En (charlie) is mostly white and less healthy"),
        Gene(symbol="Mo", name="Winter moult", alleles=("M", "m"), freqs=(0.03, 0.97),
             note="M (dominant) moults to a white winter coat, timed by day length, as mountain and snowshoe hares "
                  "do (Jones et al. 2018); a rare variant here, for selection to work on where snow lies long"),
    ),
    coat="rabbit",
    ref_mass=1.6,
    lifespan=4000,
    maturity=400,
    gestation=100,
    refractory=250,
    weaning=100,
    breeding=(0.0, 0.55),
    rest_after_sunrise=2.0,
    rest_before_sunset=2.0,
    nurse_interval=30,
    nurse_ticks=2,
    emerge=0.7,
    nest_prep=40,
    male_dispersal=1.5,
    female_dispersal=0.6,
    philopatry=0.6,
    energy_per_kg=100,
    bmr=0.1,
    cot=0.15,
    vision_cost=0.02,
    muscle_cost=0.02,
    learning_cost=0.01,
    gestation_cost=1.3,
    intake=2.5,
    assimilation=0.5,
    investment=0.25,
    water_per_kg=60,
    water_loss=0.1,
    drink_rate=0.06,
    cruise=0.4,
    stamina=25,
    stamina_regen=0.3,
    kappa_explore=30,
    kappa_ars=3,
    ars_duration=60,
    home_radius=15,
    dispersal=20,
    mate_range=15,
    half_saturation=0.3,
    graze_min=0.15,
    giving_up=0.06,
    carcass_energy=200,
    eyes_open=0.4,  # kits open their eyes at about 10 days, weaned at about 25
    choosiness=0.4,  # rabbits: mostly male rank decides who mates (von Holst et al.); females are less choosy
    burrow_reach=8.0,
    fid=2.0,
    fid_per_cell=0.25,
    thumps=True,
)

FOX = Species(
    name="fox",
    icon="🦊",
    color="#e0662f",
    diet="meat",
    prey=("rabbit",),
    initial=14,
    traits={
        "mass": Trait(mean=6.0, cv=0.1, heritability=0.5, lo=1.5, hi=20.0, label="Body mass", unit="kg"),
        "speed": Trait(mean=0.75, cv=0.08, heritability=0.4, lo=0.2, hi=2.0, label="Top speed", unit="cells/tick"),
        "vision": Trait(mean=10.0, cv=0.1, heritability=0.4, lo=3.0, hi=30.0, label="Vision", unit="cells"),
        "litter": Trait(mean=4.0, cv=0.15, heritability=0.3, lo=1.0, hi=10.0, label="Litter size", unit="young"),
        "stamina": Trait(mean=60.0, cv=0.12, heritability=0.4, lo=15.0, hi=200.0, label="Stamina", unit="ticks"),
        "insulation": Trait(mean=1.0, cv=0.12, heritability=0.5, lo=0.4, hi=2.5, label="Fur insulation", unit="×"),
        "appetite": Trait(mean=1.0, cv=0.15, heritability=0.3, lo=0.4, hi=2.5, loci=4, label="Appetite", unit="×"),
        "libido": Trait(mean=1.0, cv=0.15, heritability=0.3, lo=0.3, hi=2.5, loci=4, label="Mating drive", unit="×"),
        "persistence": Trait(mean=1.0, cv=0.2, heritability=0.3, lo=0.3, hi=3.0, loci=4, label="Chase persistence", unit="×"),
        "learning": Trait(mean=0.25, cv=0.3, heritability=0.3, lo=0.02, hi=0.9, loci=4, label="Learning rate", unit=""),
        "gestation": Trait(mean=1.0, cv=0.04, heritability=0.3, lo=0.85, hi=1.2, loci=4, label="Gestation length", unit="×"),
    },
    genes=(
        Gene(symbol="S", name="Silver", alleles=("S", "s"), freqs=(0.85, 0.15), incomplete=True,
             note="S/S red, S/s cross fox, s/s silver"),
        Gene(symbol="P", name="Pearl", alleles=("P", "p"), freqs=(0.9, 0.1),
             note="p/p dilutes the coat to pastel or pearl"),
        Gene(symbol="W", name="Platinum", alleles=("w", "Wp"), freqs=(0.96, 0.04), incomplete=True,
             labels=("w", "Wᵖ"), lethal="Wp",
             note="Wᵖ/w gives white-blazed platinum foxes; Wᵖ/Wᵖ embryos die"),
        Gene(symbol="Mo", name="Winter white", alleles=("Mo", "mo"), freqs=(0.86, 0.14),
             labels=("Mo", "mo"),
             note="mo/mo (recessive) moults to a white winter coat, like the Arctic fox's white morph (MC1R; "
                  "Våge et al. 2005); red foxes lack it, so it is a rare variant here for selection to work on"),
    ),
    coat="fox",
    ref_mass=6.0,
    lifespan=8000,
    maturity=600,
    gestation=150,
    refractory=1500,
    weaning=250,
    breeding=(0.85, 0.05),
    rest_after_sunrise=1.0,
    rest_before_sunset=1.0,
    nurse_interval=20,
    nurse_ticks=4,
    emerge=0.4,
    nest_prep=80,
    provisions=True,
    male_dispersal=1.3,
    female_dispersal=0.8,
    energy_per_kg=100,
    bmr=0.075,
    cot=0.12,
    vision_cost=0.03,
    muscle_cost=0.02,
    stamina_cost=0.02,
    learning_cost=0.015,
    gestation_cost=1.25,
    intake=4.0,
    assimilation=0.8,
    investment=0.25,
    water_per_kg=60,
    water_loss=0.08,
    drink_rate=0.06,
    cruise=0.45,
    stamina=60,
    stamina_regen=0.5,
    kappa_explore=40,
    kappa_ars=4,
    ars_duration=80,
    home_radius=25,
    dispersal=30,
    mate_range=40,
    territory=28,
    eyes_open=0.25,  # cubs open their eyes at about two weeks, weaned at about seven
    choosiness=1.0,  # vixens rely on the male to feed them and the cubs, so they are choosy
    eats_fruit=True,
    eats_voles=True,
    # Wild red foxes lose 30-60% of adults a year (Devenish-Nelson et al. 2013), much
    # of it to sarcoptic mange and accidents (Soulsbury et al. 2007): about 20% a year here.
    hazard=6e-5,
    territory_prey=80.0,  # food per 1000 land cells (rabbit-equivalents, rabbits plus voles) for a normal-sized territory
    dispersal_season=(0.55, 0.85),  # autumn to early winter, before the mating season
    helper_chance=0.4,
    max_helpers=2,
    capture_radius=0.8,
    capture_success=0.85,  # x (0.75 + 0.25 x hunting skill): 0.8 for a typical experienced founder
    rush_distance=4.0,
    stalk=0.6,
    caches=True,
)


@dataclass(frozen=True, kw_only=True)
class SimConfig:
    width: int = 160
    height: int = 100
    water_fraction: float = 0.12
    forest_fraction: float = 0.12
    rock_fraction: float = 0.05
    terrain_scale: float = 32.0

    ticks_per_day: int = 240
    days_per_year: int = 12
    seasons: bool = True

    plant_growth: float = 0.0035
    plant_seeding: float = 0.002
    plant_capacity: float = 1.0
    plant_energy: float = 40.0

    end_on_extinction: bool = True  # stop the run when a species dies out
    brains: str = "neural"  # "neural": evolving neural networks; "rules": hand-written instincts
    learning: bool = True  # neural brains also learn from experience during each animal's life
    super_animals: bool = False  # for fun: much faster evolution, no natural ceilings on traits
    evolution_speed: float = 6.0  # super animals mode: how many times faster evolution runs
    learning_speed: float = 2.0  # super animals mode: how many times faster brains learn
    inherit_learning: float = 0.5  # super animals mode: share of the parents' know-how their young start with
    emigration: bool = True  # floaters that find no territory leave the area (off: they stay and keep trying)
    climate: str = "temperate"  # see engine/climate.py: temperate, arctic, desert, mountains, grassland, mixed
    mutation: float = 1.0  # multiplier on all mutation rates
    immigration: float = 0.001
    stats_interval: int = 20

    species: tuple[Species, ...] = field(default_factory=lambda: (RABBIT, FOX))


# Settings the web UI may change, with their allowed ranges.
TUNABLE = {
    "water_fraction": (0.03, 0.5),
    "forest_fraction": (0.0, 0.5),
    "rock_fraction": (0.0, 0.3),
    "plant_growth": (0.0005, 0.05),
    "mutation": (0.0, 20.0),
    "immigration": (0.0, 0.01),
    "evolution_speed": (1.0, 20.0),
    "learning_speed": (1.0, 10.0),
    "inherit_learning": (0.0, 0.8),
}


# Map size in cells (the premade maps in the New world dialog set these).
MAP_SIZE = {"width": (40, 240), "height": (30, 160)}


def build_config(settings: dict) -> SimConfig:
    """Return the default config with validated overrides from ``settings``."""
    cfg = SimConfig()
    changes = {}
    for key, (lo, hi) in TUNABLE.items():
        if key in settings:
            changes[key] = min(hi, max(lo, float(settings[key])))
    for key, (lo, hi) in MAP_SIZE.items():
        if key in settings:
            changes[key] = int(min(hi, max(lo, int(float(settings[key])))))
    for key in ("seasons", "end_on_extinction", "learning", "super_animals", "emigration"):
        if key in settings:
            changes[key] = bool(settings[key])
    if settings.get("brains") in ("neural", "rules"):
        changes["brains"] = settings["brains"]
    from .climate import CLIMATES  # local import: climate imports terrain, which needs nothing from here
    if settings.get("climate") in CLIMATES:
        changes["climate"] = settings["climate"]
        # The climate's typical landscape, unless the settings say otherwise.
        for key, value in CLIMATES[settings["climate"]]["terrain"].items():
            if f"{key}_fraction" not in settings:
                changes[f"{key}_fraction"] = value / 100
    initial = settings.get("initial", {})
    species = tuple(
        replace(s, initial=min(2000, max(0, int(initial[s.name])))) if s.name in initial else s
        for s in cfg.species
    )
    return replace(cfg, species=species, **changes)


def settings_of(cfg: SimConfig) -> dict:
    out = {key: getattr(cfg, key) for key in TUNABLE}
    out["width"], out["height"] = cfg.width, cfg.height
    out["seasons"] = cfg.seasons
    out["end_on_extinction"] = cfg.end_on_extinction
    out["brains"] = cfg.brains
    out["learning"] = cfg.learning
    out["super_animals"] = cfg.super_animals
    out["emigration"] = cfg.emigration
    out["climate"] = cfg.climate
    out["initial"] = {s.name: s.initial for s in cfg.species}
    return out


# ---------------------------------------------------------------- super animals mode
# Not realistic, just for fun: mutation is far more frequent, inherited
# variation is larger and more heritable, and traits have no natural ceiling,
# so natural selection can push animals far beyond anything real. How much
# faster is up to the user (the evolution speed, 1-20x, default 6x). Brains
# can learn faster too (1-10x, default 2x), and young start life with part of
# what their parents learned (brain wiring, hunting skill, wariness), so
# know-how builds up over the generations as well as genes.

SUPER_CEILING = {"speed": 1.25, "vision": 1.6}  # these stay moderate so the engine stays sound; others x3


def super_spread(speed: float) -> float:
    """Extra inherited variation (and heritability) at an evolution speed: 1.5x at the default 6x."""
    return min(2.0, 1.0 + 0.1 * (speed - 1.0))


def supercharge(sp: Species, speed: float = 6.0) -> Species:
    spread = super_spread(speed)
    traits = {}
    for name, t in sp.traits.items():
        hi = 0.95 if name == "learning" else t.hi * SUPER_CEILING.get(name, 3.0)
        traits[name] = replace(t, hi=round(hi, 3), lo=round(t.lo * 0.5, 3), cv=round(t.cv * spread, 3),
                               heritability=round(min(0.9, t.heritability * spread), 2))
    return replace(sp, traits=traits)
