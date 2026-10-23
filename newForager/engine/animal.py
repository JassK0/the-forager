"""Individual animals: physiology, decisions and actions.

Every tick an animal pays its metabolic costs, may die, then chooses one
action. Fleeing a visible predator overrides everything (Lima & Dill 1990).
Otherwise the most pressing need wins: hunger, thirst or, when it is in good
condition and in season, mating. During its species' rest hours an animal
with no urgent need goes home and rests (rabbits underground), giving the
daily activity rhythm. Feeding and drinking bouts carry on until the animal
is sated, which stops it dithering between needs.

Behaviour follows field studies of rabbits and red foxes:

* Absentee maternal care: rabbit kits stay hidden in the nest and their
  mother visits once a day to nurse for a few minutes; fox cubs stay in the
  den, are nursed often, and come out to play at about four weeks.
* Pregnant females head home to give birth in their burrow or den.
* Prey that spot a distant predator freeze (motion is what gives them away)
  and only bolt once it comes within their flight initiation distance, which
  grows the farther they are from the burrow (Ydenberg & Dill 1986). Rabbits
  that bolt drum an alarm that puts nearby rabbits on alert.
* Fear lingers after a scare and keeps rabbits feeding close to their burrow
  (the landscape of fear; Laundre et al. 2001). Feeding is interrupted by
  scanning bouts, less often in a group (the group-size effect).
* Animals walk straight to their goals when they can and plan a route
  around water, trees and rock when they cannot (A*, navigation.py).
* Foxes pick prey by route length weighted by vulnerability (young,
  isolated and unwatchful prey are easier), switch while stalking if a
  clearly better target appears, and once a chase starts stay on the fleeing
  animal unless another comes within lunging range.
* Foxes stalk slowly and low, then rush. They bury surplus food and come
  back for it, pilfer other foxes' caches, sometimes kill when not hungry,
  and fathers carry food to the den for the vixen and cubs.
* Territorial foxes scent-mark their range and chase off same-sex intruders;
  floaters steer clear of occupied territories.
* Males disperse farther than females; many rabbit daughters stay in their
  mother's warren. Animals avoid mating with close kin, and females are more
  willing to mate with males in good condition.
* Animals sharing a burrow huddle, which saves heat; burrows are also
  buffered against cold and heat. Fresh plants and prey supply water, so
  animals on lush pasture rarely need to drink.
* Young animals play and adults groom between other activities.

Decisions are made by an evolvable neural network or by hand-written
instincts (brain.py): each tick the animal senses its state and surroundings
(`sense`), a choice is made for the whole species at once (World._think), and
`_decide` turns it into an action. Memory and learning shape the inputs:
animals remember where food was good and where they met danger (Rescorla &
Wagner 1972 learning with forgetting), copy their mother's food map at
weaning, grow warier with each encounter they survive, and young foxes get
better at hunting with practice. Learning rate is a gene.

An animal's traits are not set directly: they are expressed from its genome
at birth (see genetics.py), so everything here acts on inherited variation.
"""

import math
from enum import IntEnum

from . import brain, movement

LN2 = math.log(2.0)
GOMPERTZ_SHAPE = 5.0  # larger = senescent deaths more tightly clustered around the lifespan
MAKEHAM = 2e-5  # age-independent hazard per tick (disease, accidents); species can add their own
CRITICAL_NEED = 0.75  # a deficit this severe narrows the choice to meeting it (an emergency instinct)

# When to eat, drink, mate and rest is decided in brain.py; these are the
# instincts the decision layer does not override.
URGENT = 0.55  # deficit that interrupts chores (caching, grooming, play) and nesting before birth
SATED = 0.95
CONDITION = 0.5  # minimum reserves (fraction of capacity) for breeding

# Predator detection: falls off with distance, doubles if the predator runs,
# drops while the prey feeds head-down, in darkness and against a stalking
# predator, and rises with the predator's contrast against the background.
DETECT = 0.35
HEAD_DOWN = 0.4
STALK_DETECT = 0.6
SCAN_DETECT = 1.5  # detection while rearing up to scan
SCAN = 0.12  # chance per feeding tick of a scanning bout, at vigilance 1
NAIVE = 0.6  # detection by a completely inexperienced animal, relative to an experienced one
REGION = 6  # resolution of the memory maps, in cells
FOOD_MEMORY_DAYS = 2.0  # half-life of a remembered food patch (plants regrow, prey moves on)
DANGER_MEMORY_DAYS = 1.0  # half-life of a remembered scare
MEMORY_SIZE = 32  # places remembered per map; the faintest memory is forgotten first
RECALL = 0.2  # remembered value worth travelling back for
# Rewards for lifetime learning (on top of the homeostatic reward of meeting needs).
REWARD_MATING = 0.4  # x mating drive
DRIVE_N, DRIVE_M = 4.0, 3.0  # exponents of the homeostatic drive
PAIN_LUNGE = 1.0  # a predator's lunge that just missed
FEAR_COST = 0.03  # per tick, x threat, while a predator is in sight
PAIN_CHASED = 0.2  # driven off by a territory holder
FIGHT_WON, FIGHT_LOST = 0.3, 0.4
RECENT_SITUATIONS = 24  # situations remembered for showing how its decisions have changed
BRAIN_LOG_EVERY = 30  # ticks between entries in an animal's learning record
FEAR_DECAY = 0.995  # per tick; fear halves in roughly half a day
THUMP_RADIUS = 6.0
ALERT_TICKS = 15
HIDE_MEAN = 30.0  # mean ticks a prey animal stays down its burrow after a scare
HUNT_STAMINA = 0.3  # fraction of stamina a predator wants before starting a stalk
ABORT_DISTANCE = 1.5  # x rush distance: give up on prey that bolts while farther than this
OPPORTUNISM = 0.15  # chance a sated predator attacks prey that wanders very close

# Thermoregulation: extra upkeep in cold weather, reduced by insulation, and
# in hot weather, increased by it. The two balance near insulation 1.
COLD_COST = 0.25
HEAT_COST = 0.3
BURROW_COLD = 0.5  # burrows halve the cold...
BURROW_HEAT = 0.3  # ...and keep most of the heat out
HUDDLE = 0.3  # each extra body in the burrow cuts the remaining cold cost

PLANT_WATER = 12.0  # water per unit of fresh plant biomass eaten
MEAT_WATER = 0.25  # water per unit of energy eaten as meat

BIRTH_MASS = 0.08  # newborn mass as a fraction of adult mass
WEANING_MASS = 0.35  # mass at weaning
GROWTH_COST = 0.2  # energy spent building each unit of new tissue, x its stored energy
# The calendar compresses growth about 30x (a fox cub is weaned in a day, not
# seven weeks), but food is still found at real daily rates. Charging growth at
# the compressed rate would make a vixen with four cubs need ~10x her normal
# intake; real lactating mammals need 2-4x. So growth is paid for at the real
# daily rate: young keep their condition as they grow, and 1/30 of the cost of
# the new tissue and the bigger energy store is charged.
GROWTH_COMPRESSION = 30.0
LACTATION_EFFICIENCY = 0.8  # fraction of the mother's energy that reaches her young as milk
NURSE_FLOOR = 0.2  # mothers stop nursing when their own reserves fall below this
TERRITORY_RETRY = 100  # ticks between a floater's attempts to claim a vacant territory
EMIGRATE_AFTER = 4  # failed searches after which a floater leaves the map (the area is full)
DEFEND_CHANCE = 0.6
CONTEST_CONDITION = 0.4  # reserves a floater needs before it will fight a resident for its territory
CHASE_TICKS = 25
EVICT_TICKS = 20
MARK_CHANCE = 0.03
CACHE_MIN = 20.0  # smallest leftover worth burying
PROVISION_SATED = 0.7  # parents with young stop eating here and carry the rest of a kill to the den

REST_CHANCE = 0.008  # per tick, while idle
REST_MEAN = 25.0
PLAY_CHANCE = 0.02
GROOM_CHANCE = 0.005
DIRECTED_KAPPA = 40.0  # turning noise while heading for a goal
FLEE_KAPPA = 20.0  # escape runs zig-zag a little (protean behaviour)
TROT = 0.6  # fraction of top speed when sprinting is exhausted
STALL_LIMIT = 40  # ticks without closing on a target before giving up
IGNORE_TICKS = 120  # how long an abandoned target is ignored
MATE_RADIUS = 1.0
LUNGE_RECOVERY = 4  # ticks a predator needs after a missed capture


class Act(IntEnum):
    WANDER = 0
    REST = 1
    FORAGE = 2
    EAT = 3
    SEEK_WATER = 4
    DRINK = 5
    SEEK_MATE = 6
    FLEE = 7
    HUNT = 8
    HIDE = 9
    NEST = 10  # dependent young in the nest or den
    HOME = 11  # heading home to rest or give birth
    FREEZE = 12
    ALERT = 13  # warned by an alarm thump
    GROOM = 14
    PLAY = 15
    CACHE = 16  # burying food
    MARK = 17  # scent-marking the territory
    CHASE = 18  # driving off an intruder
    VISIT = 19  # mother visiting the nest to nurse


IDLE = (Act.WANDER, Act.REST, Act.GROOM, Act.PLAY, Act.MARK)


class Carrion:
    """The remains of an animal that died of something other than predation."""

    __slots__ = ("x", "y", "energy", "until", "si")

    def __init__(self, x, y, energy, until, si):
        self.x, self.y, self.energy, self.until, self.si = x, y, energy, until, si


class Cache(Carrion):
    """Food a predator has buried to come back to."""

    __slots__ = ("owner", "den")

    def __init__(self, x, y, energy, until, si, owner, den):
        super().__init__(x, y, energy, until, si)
        self.owner = owner
        self.den = den


class Animal:
    __slots__ = (
        "id", "sp", "si", "genome", "traits", "breeding", "genotype", "coat", "coat_lum", "vigor",
        "load", "load_hom", "het", "female", "generation", "mother_id", "father_id", "children_ids",
        "x", "y", "heading", "last_move", "avoid_side", "home_x", "home_y",
        "age", "adult", "alive", "cause", "born",
        "mass", "speed", "vision", "max_energy", "energy", "max_water", "water",
        "upkeep", "cot", "water_loss", "intake", "stamina", "max_stamina",
        "lifespan", "maturity", "gestation", "weaning", "gompertz", "mother", "territorial",
        "act", "target", "best_d", "stall", "ignore", "ignore_until", "search_cd",
        "threats", "ars", "rest_left", "stun", "carcass",
        "pregnant", "gestation_left", "mate_genome", "mate_traits", "mate_genotype", "mate_id", "mate_generation",
        "refractory", "partner", "nest_x", "nest_y", "litter", "next_nurse", "nursing",
        "fear", "alert_until", "scanning", "stalking", "group", "group_t", "flee_until", "chase_until",
        "carried", "cache_den", "dest", "concealed", "surplus", "path", "path_goal", "path_t", "route",
        "brain_row", "planned", "rule_choice", "inputs", "mask", "scores", "hidden", "spotted", "threat", "charging",
        "food_mem", "danger_mem", "experience", "skill", "goal",
        "children", "kills", "failed_claims", "prey_idx", "predator_idx", "helper_of", "natal",
        "reward", "reward_events", "prev_drive", "surprise", "value_now", "wellbeing", "surprise_avg",
        "recent", "brain_log", "rewards_total",
    )

    def __init__(self, world, uid, si, genome, expr, x, y, *, age=0, generation=0,
                 mother_id=None, father_id=None):
        sp = world.species[si]
        rng = world.rng
        self.id = uid
        self.sp = sp
        self.si = si
        self.genome = genome
        self.traits = expr.traits
        self.breeding = expr.breeding
        self.genotype = expr.genotype
        self.coat = expr.coat
        self.coat_lum = world.genomes[si].coats[expr.coat].lum
        self.vigor = expr.vigor
        self.load = expr.load
        self.load_hom = expr.load_hom
        self.het = expr.het
        self.female = not genome.male
        self.generation = generation
        self.mother_id = mother_id
        self.father_id = father_id
        self.children_ids = []

        self.x, self.y = x, y
        self.heading = rng.uniform(-math.pi, math.pi)
        self.last_move = 0.0
        self.avoid_side = 1 if rng.random() < 0.5 else -1
        self.home_x, self.home_y = x, y

        # Biological times follow the quarter-power law (Lindstedt & Calder 1981).
        q = (self.traits["mass"] / sp.ref_mass) ** 0.25
        self.lifespan = sp.lifespan * q
        self.maturity = sp.maturity * q
        self.gestation = sp.gestation * q
        self.weaning = sp.weaning * q
        self.mother = None  # set while the animal is a dependent, nursing juvenile
        # Territorial species breed only once they hold a territory (Brown 1969).
        self.territorial = sp.territory <= 0
        # Gompertz hazard a*exp(b*age/lifespan), with a set so half of all
        # animals that escape other causes are dead by `lifespan`.
        self.gompertz = GOMPERTZ_SHAPE * LN2 / (self.lifespan * math.expm1(GOMPERTZ_SHAPE))

        self.age = age
        self.born = world.tick - age
        self.adult = age >= self.maturity
        self.alive = True
        self.cause = None
        self.vision = self.traits["vision"]
        self.max_stamina = self.traits.get("stamina", sp.stamina)
        self._allometry()
        self.energy = 0.8 * self.max_energy
        self.water = 0.8 * self.max_water
        self.stamina = self.max_stamina

        self.act = Act.WANDER
        self.target = None
        self.best_d = math.inf
        self.stall = 0
        self.ignore = None
        self.ignore_until = 0
        self.search_cd = 0
        self.threats = ()
        self.ars = 0
        self.rest_left = 0.0
        self.stun = 0
        self.carcass = 0.0

        self.pregnant = False
        self.gestation_left = 0.0
        self.mate_genome = None
        self.mate_traits = None
        self.mate_genotype = None
        self.mate_id = None
        self.mate_generation = 0
        self.refractory = 0.0
        self.partner = None
        self.nest_x, self.nest_y = x, y
        self.litter = []
        self.next_nurse = 0
        self.nursing = 0
        self.fear = 0.0
        self.alert_until = 0
        self.scanning = False
        self.stalking = False
        self.group = 0
        self.group_t = 0
        self.flee_until = 0
        self.chase_until = 0
        self.carried = 0.0
        self.cache_den = None
        self.dest = None
        self.concealed = False
        self.surplus = False
        self.path = None
        self.path_goal = None
        self.path_t = 0
        self.route = None
        # Decision-making (filled in by World._think each tick).
        self.brain_row = world.brains[si].alloc(expr.brain) if world.brains is not None else -1
        self.planned = None
        self.rule_choice = None
        self.inputs = None
        self.mask = None
        self.scores = None
        self.hidden = None
        self.spotted = False
        self.threat = 0.0
        self.charging = False
        # Memory and experience. Founders arrive as experienced adults.
        self.food_mem = {}
        self.danger_mem = {}
        self.experience = rng.uniform(0.6, 0.9) if self.adult else 0.0
        self.skill = rng.uniform(0.6, 0.9) if self.adult else 0.0
        self.goal = None
        self.children = 0
        self.kills = 0
        self.failed_claims = 0
        self.helper_of = None  # the breeding female this animal helps (a subordinate in her family)
        # Lifetime learning: rewards, prediction errors and a record for the inspector.
        self.reward = 0.0
        self.reward_events = 0.0
        self.prev_drive = None
        self.surprise = 0.0
        self.value_now = 0.0
        self.wellbeing = 0.0
        self.surprise_avg = 0.0
        self.rewards_total = 0.0
        self.recent = []
        self.brain_log = []
        self.natal = False  # grown up but still living with its family until the dispersal season
        self.prey_idx = world.prey_of[si]
        self.predator_idx = world.predators_of[si]

    # ------------------------------------------------------------------ physiology

    def _growth(self) -> float:
        """Fraction of adult mass: fast growth on milk, then slower to maturity."""
        if self.adult:
            return 1.0
        if self.age < self.weaning:
            return BIRTH_MASS + (WEANING_MASS - BIRTH_MASS) * self.age / self.weaning
        return WEANING_MASS + (1 - WEANING_MASS) * (self.age - self.weaning) / max(1.0, self.maturity - self.weaning)

    def _allometry(self) -> None:
        """Recompute mass-dependent rates as the animal grows."""
        sp, t = self.sp, self.traits
        frac = self._growth()
        m = t["mass"] * frac
        m75 = m ** 0.75
        ref = sp.traits
        if getattr(self, "mass", None) and m > self.mass:
            # Young keep their condition as they grow, paying for the new tissue
            # and the bigger store at the real (uncompressed) daily rate.
            fill = self.energy / self.max_energy * sp.energy_per_kg * (m - self.mass)
            cost = fill + (m - self.mass) * sp.energy_per_kg * GROWTH_COST
            self.energy += fill - cost / GROWTH_COMPRESSION
        self.mass = m
        self.max_energy = sp.energy_per_kg * m
        self.max_water = sp.water_per_kg * m
        upkeep = (
            sp.bmr * m75
            + sp.vision_cost * frac ** 0.75 * (t["vision"] / ref["vision"].mean) ** 2
            + sp.muscle_cost * frac * (t["speed"] / ref["speed"].mean) ** 2
        )
        if "stamina" in t:
            upkeep += sp.stamina_cost * frac * (t["stamina"] / ref["stamina"].mean) ** 2
        if "learning" in t:
            # Fast-learning brains are expensive to run (the expensive-brain hypothesis).
            upkeep += sp.learning_cost * frac ** 0.75 * (t["learning"] / ref["learning"].mean) ** 2
        # Genetic load: low-vigour animals are metabolically inefficient.
        self.upkeep = upkeep * (1.0 + 0.5 * (1.0 - self.vigor))
        self.cot = sp.cot * m ** 0.68
        self.water_loss = sp.water_loss * m75
        self.intake = sp.intake * m75
        # Maximum sprint speed scales roughly as M^0.17 (Garland 1983).
        self.speed = t["speed"] * frac ** 0.17

    def ready_to_mate(self, w) -> bool:
        return (
            self.adult
            and self.territorial
            and not self.pregnant
            and self.refractory <= 0
            and w.in_season[self.si]
            and self.energy > CONDITION * self.max_energy
            and self.water > CONDITION * self.max_water
        )

    def thermal_cost(self, w) -> float:
        ins = self.traits["insulation"]
        cold, heat = w.cold, w.heat
        if self.concealed:
            # Underground: buffered from the weather and warmed by burrow-mates.
            mates = w.huddle.get((int(self.x), int(self.y)), 1)
            cold *= BURROW_COLD / (1.0 + HUDDLE * (mates - 1))
            heat *= BURROW_HEAT
        return 1.0 + COLD_COST * cold / ins + HEAT_COST * heat * max(0.0, ins - 0.6)

    def update(self, w) -> None:
        sp = self.sp
        self.age += 1
        if not self.adult:
            if self.age >= self.maturity:
                self.adult = True
                self._allometry()
                if sp.dispersal_season and not w.dispersal_time(self.si):
                    self.natal = True  # stays with the family until the dispersal season
                else:
                    self._leave_home(w)
            elif self.age % 10 == 0:
                self._allometry()

        burn = self.upkeep * self.thermal_cost(w)
        if self.pregnant:
            burn *= sp.gestation_cost
        self.energy -= burn
        self.water -= self.water_loss * (1.0 + 0.5 * w.heat * (BURROW_HEAT if self.concealed else 1.0))
        if self.energy <= 0.0:
            w.kill(self, "starvation")
            return
        if self.water <= 0.0:
            w.kill(self, "dehydration")
            return
        r = w.rng.random()
        background = MAKEHAM + sp.hazard + w.plague[self.si]
        if r < background:
            w.kill(self, "disease")
            return
        if r < background + self.gompertz * math.exp(GOMPERTZ_SHAPE * self.age / self.lifespan):
            w.kill(self, "old age")
            return

        if self.mother is not None:
            m = self.mother
            if m.alive and self.age < self.weaning:
                if m.act == Act.VISIT and m.nursing > 0 and abs(m.x - self.x) + abs(m.y - self.y) < 2.0:
                    self._suckle(m)
            else:
                # Weaned, or orphaned: from now on it fends for itself, starting
                # with what it picked up from its mother about where food is.
                if m.alive:
                    self.food_mem = dict(m.food_mem)
                    self.danger_mem = dict(m.danger_mem)
                self.mother = None
                if self.act == Act.NEST:
                    self.concealed = False
                    self._switch(Act.WANDER)
        elif self.adult and not self.territorial and self.age % TERRITORY_RETRY == 0:
            if self.natal:
                if w.dispersal_time(self.si):
                    self._leave_home(w)
            elif self.helper_of is not None:
                if not self.helper_of.alive or not self.helper_of.territorial:
                    self.helper_of = None  # her family broke up: she becomes a floater
            # Floaters keep prospecting for a vacancy or a lone mate's territory;
            # if the whole neighbourhood stays full they leave the area, so
            # territories cap the local density.
            elif not w.settle(self, 2.0 * sp.dispersal):
                self.failed_claims += 1
                if self.failed_claims >= EMIGRATE_AFTER:
                    w.kill(self, "emigrated")
                    return
        if self.pregnant:
            self.gestation_left -= 1
            if self.gestation_left <= 0:
                w.give_birth(self)
        if self.refractory > 0:
            self.refractory -= 1
        if self.ars > 0:
            self.ars -= 1
        if self.fear > 0.01:
            self.fear *= FEAR_DECAY
        if self.stamina < self.max_stamina:
            self.stamina = min(self.max_stamina, self.stamina + sp.stamina_regen)
        self.scanning = False
        self.stalking = False
        if self.stun > 0:
            self.stun -= 1
            self.last_move = 0.0
            return

        self._decide(w)
        if self.alive:  # a territorial fight can end in death
            _ACTIONS[self.act](self, w)

    def _suckle(self, mother) -> None:
        """Lactation: the mother tops up her young's reserves at a cost to her own."""
        spare = mother.energy - NURSE_FLOOR * mother.max_energy
        need = 0.9 * self.max_energy - self.energy
        if need > 0.0 and spare > 0.0:
            give = min(need, spare * LACTATION_EFFICIENCY)
            self.energy += give
            mother.energy -= give / LACTATION_EFFICIENCY
            # Caring for young is itself rewarding, offsetting what it costs her.
            mother.reward_events += give / LACTATION_EFFICIENCY / mother.max_energy
        spare = mother.water - NURSE_FLOOR * mother.max_water
        need = 0.9 * self.max_water - self.water
        if need > 0.0 and spare > 0.0:
            give = min(need, spare)
            self.water += give
            mother.water -= give

    def _leave_home(self, w) -> None:
        """At the dispersal season: sons leave; some daughters stay on as helpers,
        more often where food is plentiful and the family has room."""
        self.natal = False
        sp = self.sp
        mum = w.by_id.get(self.mother_id)
        if (self.female and sp.max_helpers and mum is not None and mum.territorial
                and w.helpers_of(mum) < sp.max_helpers
                and w.rng.random() < sp.helper_chance * min(1.0, w.prey_richness(self.si, mum.home_x, mum.home_y))):
            self.helper_of = mum
            self.home_x, self.home_y = mum.home_x, mum.home_y
            w.social[self.si]["stayed as helper"] += 1
            return
        self._disperse(w)

    def _disperse(self, w) -> None:
        """Natal dispersal: males travel farther; many daughters stay in their natal home."""
        sp, rng = self.sp, w.rng
        if self.female and sp.philopatry and sp.territory <= 0 and rng.random() < sp.philopatry:
            return
        mean = sp.dispersal * (sp.female_dispersal if self.female else sp.male_dispersal)
        if sp.territory > 0:
            # Settle with an unpaired mate or in a vacancy within dispersal range.
            w.settle(self, 2.0 * mean, here=False)
            return
        for _ in range(8):
            d = rng.expovariate(1.0 / mean)
            a = rng.uniform(-math.pi, math.pi)
            hx, hy = self.x + d * math.cos(a), self.y + d * math.sin(a)
            if w.walkable(hx, hy):
                self.home_x, self.home_y = hx, hy
                return

    # ------------------------------------------------------------------ decisions

    def _switch(self, act) -> None:
        self.act = act
        self.target = None
        self.best_d = math.inf
        self.stall = 0
        self.carcass = 0.0
        self.surplus = False
        self.concealed = False
        self.path = None
        self.route = None

    def _spots(self, w, threats) -> bool:
        """Whether the animal notices any of the predators in its field of view this tick."""
        if self.act in (Act.EAT, Act.DRINK) and not self.scanning:
            posture = HEAD_DOWN
        elif self.scanning or self.act in (Act.FREEZE, Act.ALERT):
            posture = SCAN_DETECT
        else:
            posture = 1.0
        base = DETECT * posture * self.traits.get("vigilance", 1.0) * (0.5 + 0.5 * w.light)
        base *= NAIVE + (1.0 - NAIVE) * self.experience  # experience teaches what danger looks like
        miss = 1.0
        for d2, p in threats:
            q = base * w.visibility(p) * (1.0 - math.sqrt(d2) / self.vision)
            if p.last_move > p.speed * p.sp.cruise * 1.05:
                q *= 2.0
            elif p.stalking:
                q *= STALK_DETECT
            miss *= 1.0 - min(max(q, 0.0), 1.0)
        return w.rng.random() >= miss

    # ------------------------------------------------------------------ memory

    def region(self, w, x=None, y=None) -> int:
        x = self.x if x is None else x
        y = self.y if y is None else y
        return int(x) // REGION + (int(y) // REGION) * w.region_cols

    def recall(self, w, memory: dict, key: int, half_life_days: float) -> float:
        entry = memory.get(key)
        if entry is None:
            return 0.0
        value, when = entry
        return value * 0.5 ** ((w.tick - when) / (half_life_days * w.cfg.ticks_per_day))

    def learn(self, w, memory: dict, key: int, outcome: float, half_life_days: float, weight: float = 1.0) -> None:
        """Rescorla-Wagner update: move the expectation towards what happened."""
        old = self.recall(w, memory, key, half_life_days)
        if key not in memory and len(memory) >= MEMORY_SIZE:
            del memory[min(memory, key=lambda k: self.recall(w, memory, k, half_life_days))]
        memory[key] = (old + weight * self.traits.get("learning", 0.25) * (outcome - old), w.tick)

    def gain_experience(self, amount: float) -> None:
        self.experience += amount * self.traits.get("learning", 0.25) * (1.0 - self.experience)

    def _remembered_place(self, w, memory: dict) -> tuple[float, float] | None:
        """The most promising remembered place to go back to, away from known danger."""
        here = self.region(w)
        best, best_score = None, 0.0
        for key, _ in memory.items():
            if key == here:
                continue
            value = self.recall(w, memory, key, FOOD_MEMORY_DAYS)
            if value < RECALL:
                continue
            cx = (key % w.region_cols + 0.5) * REGION
            cy = (key // w.region_cols + 0.5) * REGION
            d = math.hypot(cx - self.x, cy - self.y)
            if d > 45.0:
                continue
            score = value * (1.0 - self.recall(w, self.danger_mem, key, DANGER_MEMORY_DAYS)) / (1.0 + d / 10.0)
            if score > best_score:
                best, best_score = (cx, cy), score
        if best is not None and not w.walkable(*best):
            best = w.nearest_walkable(*best, limit=3)
        return best

    def _go_to_remembered(self, w, memory: dict) -> bool:
        """Travel towards a remembered good spot. Returns False if there is none."""
        if self.goal is None or math.hypot(self.goal[0] - self.x, self.goal[1] - self.y) < 2.0:
            self.goal = self._remembered_place(w, memory)
        if self.goal is None:
            return False
        if self._toward(w, self.goal[0], self.goal[1], self.speed * self.sp.cruise) is None:
            self.learn(w, memory, self.region(w, *self.goal), 0.0, FOOD_MEMORY_DAYS)
            self.goal = None
            return False
        return True

    # ------------------------------------------------------------------ sensing

    def sense(self, w) -> None:
        """Gather this tick's inputs for the decision layer (called by World._think)."""
        sp = self.sp
        was_spotted = self.spotted
        self.spotted = False
        self.threat = 0.0
        self.charging = False
        if self.predator_idx and not self.concealed and self.mother is None:
            threats = w.query(self.predator_idx, self.x, self.y, self.vision)
            if threats:
                alerted = self.act in (Act.FLEE, Act.FREEZE, Act.ALERT) or self.alert_until > w.tick
                if alerted or self._spots(w, threats):
                    if not was_spotted:
                        # A new encounter: remember the place, and learn what predators look like.
                        self.gain_experience(0.3)
                        self.learn(w, self.danger_mem, self.region(w), 1.0, DANGER_MEMORY_DAYS)
                    self.spotted = True
                    self.threats = threats
                    self.fear = 1.0
                    nearest = math.sqrt(min(d2 for d2, _ in threats))
                    self.charging = any(p.last_move > p.speed * p.sp.cruise * 1.05 for _, p in threats)
                    # Flight initiation distance: farther from the burrow, and for
                    # shyer animals, the predator is not allowed as close.
                    d_home = math.hypot(self.home_x - self.x, self.home_y - self.y)
                    fid = (sp.fid + sp.fid_per_cell * min(15.0, d_home)) / self.traits.get("boldness", 1.0)
                    self.threat = min(3.0, fid / max(nearest, 0.1)) if sp.fid > 0 else 3.0
            elif self.act == Act.FLEE and self.flee_until <= w.tick:
                self.gain_experience(1.0)  # got away: a lesson learned

        hunger = 1.0 - self.energy / self.max_energy
        thirst = 1.0 - self.water / self.max_water
        here = self.region(w)
        if sp.diet == "plants":
            food = float(w.veg.B[int(self.y), int(self.x)]) / w.veg.capacity
        else:
            food = self.recall(w, self.food_mem, here, FOOD_MEMORY_DAYS)
        if w.tick >= self.group_t:
            self.group = len(w.query((self.si,), self.x, self.y, 3.0, exclude=self))
            self.group_t = w.tick + 10
        ready = self.ready_to_mate(w)
        alarm = self.alert_until > w.tick
        self.inputs = [
            min(1.5, hunger * self.traits.get("appetite", 1.0)),
            thirst,
            self.traits.get("libido", 1.0) if ready else 0.0,
            self.threat,
            1.0 if self.charging else 0.0,
            1.0 if alarm else 0.0,
            self.fear,
            self.recall(w, self.danger_mem, here, DANGER_MEMORY_DAYS),
            min(1.0, food),
            min(2.0, math.hypot(self.home_x - self.x, self.home_y - self.y) / sp.home_radius),
            w.light,
            1.0 if w.resting[self.si] else 0.0,
            self.traits.get("freeze", 0.0),
            min(self.group, 8) / 8.0,
            0.0 if self.adult else 1.0,
        ]
        threatened = self.spotted
        # Taking cover (standing alert, slipping back to the burrow) is an option
        # once danger is seen or an alarm is heard, but an alarm alone does not
        # interrupt a nursing visit.
        cover = threatened or (alarm and self.act != Act.VISIT)
        self.mask = [True, True, ready, True, True, threatened, threatened, cover]
        if max(hunger, thirst) > CRITICAL_NEED:
            # Emergency instinct: a critical need narrows the choice to meeting it
            # (or escaping danger), whatever the brain has learned.
            urgent = brain.DRINK if thirst >= hunger else brain.EAT
            self.mask = [m and k in (urgent, brain.FLEE, brain.FREEZE, brain.HIDE) for k, m in enumerate(self.mask)]

        # Reward since the last decision: needs met (drive reduced), plus pleasure
        # and pain from events, plus the unpleasantness of a predator in sight.
        # Homeostatic drive (Keramati & Gutkin 2014): D = (sum deficit^n)^(1/m), n > m,
        # so an urgent need dominates and meeting it is the most rewarding.
        h, t = min(1.0, max(0.0, hunger)), min(1.0, max(0.0, thirst))
        drive = (h ** DRIVE_N + t ** DRIVE_N) ** (1.0 / DRIVE_M)
        r = (self.prev_drive - drive) if self.prev_drive is not None else 0.0
        r += self.reward_events - FEAR_COST * self.threat
        self.prev_drive = drive
        self.reward_events = 0.0
        self.reward = r
        self.rewards_total += r
        if (w.tick + self.id) % 10 == 0:
            self.recent.append((self.inputs + [1.0], self.mask))
            if len(self.recent) > RECENT_SITUATIONS:
                del self.recent[0]

    def plasticity(self) -> float:
        """How fast this animal learns: its learning-rate gene, and faster when young."""
        p = self.traits.get("learning", 0.25) / 0.25
        return p * (brain.YOUNG_PLASTICITY if not self.adult else 1.0)

    # ------------------------------------------------------------------ decisions

    def _respond(self, w, plan: int) -> None:
        """Carry out an anti-predator choice."""
        if plan == brain.FLEE:
            if self.act != Act.FLEE:
                self._switch(Act.FLEE)
                if self.sp.thumps:
                    w.thump(self)
        elif plan == brain.FREEZE:
            if self.act != Act.FREEZE:
                self._switch(Act.FREEZE)
        elif self.act != Act.ALERT:
            self._switch(Act.ALERT)

    def _decide(self, w) -> None:
        sp = self.sp
        act = self.act
        if act == Act.HIDE:
            # Stay underground until the timer runs out or a need becomes critical.
            if self.rest_left > 0 and self.energy > 0.2 * self.max_energy and self.water > 0.2 * self.max_water:
                self.concealed = True
                return
            self._switch(Act.WANDER)
        if act == Act.FLEE and self.flee_until > w.tick:
            return  # being chased off by a rival
        if self.mother is not None:
            if self.act != Act.NEST:
                self._switch(Act.NEST)
            return

        plan = self.planned if self.planned is not None else brain.EXPLORE
        if self.predator_idx:
            if plan in (brain.FLEE, brain.FREEZE, brain.HIDE) and (self.spotted or self.alert_until > w.tick):
                self._respond(w, plan)
                return
            if self.act in (Act.FLEE, Act.FREEZE, Act.ALERT):
                self.threats = ()
                self._switch(Act.WANDER)

        act = self.act
        hunger = 1.0 - self.energy / self.max_energy
        thirst = 1.0 - self.water / self.max_water
        calm = hunger < URGENT and thirst < URGENT
        if act == Act.VISIT and hunger < 0.85 and thirst < 0.85:
            return
        if act in (Act.CACHE, Act.CHASE, Act.MARK, Act.GROOM, Act.PLAY) and calm:
            return
        # Instincts that the brain does not override: nursing and nesting. A
        # mother finishes a stalk, chase or meal first.
        busy = act == Act.EAT or (act == Act.HUNT and isinstance(self.target, Animal))
        if self.litter and w.tick >= self.next_nurse and hunger < 0.8 and thirst < 0.8 and not busy:
            self.litter = [k for k in self.litter if k.alive and k.mother is self]
            if self.litter:
                self._switch(Act.VISIT)
                return
        if self.pregnant and self.gestation_left <= sp.nest_prep and calm:
            at_home = math.hypot(self.home_x - self.x, self.home_y - self.y) < 0.8
            if act not in (Act.HOME, Act.HIDE, Act.REST) or (act == Act.REST and not at_home):
                self._switch(Act.HOME)
            return
        # Bouts carry on until done.
        if act == Act.EAT and hunger > 1.0 - SATED and thirst < 0.8:
            return
        if act == Act.DRINK and self.water < self.max_water and hunger < 0.8:
            return
        if act == Act.HUNT and self.surplus and isinstance(self.target, Animal) and self.target.alive:
            return

        # The decision layer's choice.
        if plan == brain.EAT:
            want = Act.FORAGE if sp.diet == "plants" else Act.HUNT
        elif plan == brain.DRINK:
            want = Act.SEEK_WATER
        elif plan == brain.MATE:
            want = Act.SEEK_MATE
        elif plan == brain.REST:
            if act not in (Act.HOME, Act.REST, Act.HIDE):
                self._switch(Act.HOME)
            return
        else:
            want = None
        if want is not None:
            if want != act:
                self._switch(want)
            return
        if act not in IDLE or (act == Act.REST and self.rest_left > REST_MEAN * 4):
            self._switch(Act.WANDER)
        if sp.territory > 0 and self.territorial and self.adult and self._patrol(w):
            return
        if self.prey_idx and self.act in (Act.WANDER, Act.REST) and (w.tick + self.id) % 5 == 0 \
                and self.stamina >= 0.9 * self.max_stamina:
            # Opportunistic (surplus) killing: prey that blunders close is taken anyway.
            found = w.find_prey(self, radius=sp.rush_distance)
            prey = found[0] if found else None
            if prey is not None and w.rng.random() < OPPORTUNISM:
                self._switch(Act.HUNT)
                self._set_target(prey)
                self.surplus = True

    def _patrol(self, w) -> bool:
        """Territory holders scent-mark their range and chase off same-sex intruders."""
        sp = self.sp
        core = sp.territory / 2
        if (w.tick + self.id) % 8 == 0:
            for d2, a in w.query((self.si,), self.x, self.y, self.vision, exclude=self):
                if not a.adult or a.female != self.female or a is self.partner or a.mother_id == self.id \
                        or a.natal or a.helper_of is self:
                    continue  # family is tolerated
                trespassing = not a.territorial or math.hypot(a.home_x - self.home_x, a.home_y - self.home_y) >= sp.territory
                if trespassing and math.hypot(a.x - self.home_x, a.y - self.home_y) < core \
                        and a.act != Act.CHASE and w.rng.random() < DEFEND_CHANCE:
                    self._switch(Act.CHASE)
                    self.target = a
                    self.chase_until = w.tick + CHASE_TICKS
                    w.evict(a, self, d2)
                    return True
        d_home = math.hypot(self.home_x - self.x, self.home_y - self.y)
        if self.act == Act.WANDER and 0.5 * core < d_home < 1.1 * core and w.rng.random() < MARK_CHANCE:
            self._switch(Act.MARK)
            self.rest_left = 3
            return True
        return False

    # ------------------------------------------------------------------ movement helpers

    def _move(self, w, heading, speed, kappa, sprint=False, limit=math.inf) -> None:
        if sprint and self.stamina >= 1.0:
            self.stamina -= 1.0
            speed = self.speed
        moved = movement.step(self, w, heading, min(speed, limit), kappa)
        self.last_move = moved
        self.energy -= self.cot * moved

    def _range(self, w, pace) -> None:
        """Correlated random walk, tortuous after a meal and tethered to a home range."""
        sp = self.sp
        desired = self.heading
        dx, dy = self.home_x - self.x, self.home_y - self.y
        d = math.hypot(dx, dy)
        # Fear shrinks the range: scared (and shy) animals keep close to their refuge.
        radius = sp.home_radius * max(0.3, 1.0 - 0.5 * self.fear / self.traits.get("boldness", 1.0))
        settled = self.territorial or not self.adult or self.natal or self.helper_of is not None
        if d > radius and settled:
            pull = min(1.0, (d - radius) / radius)
            desired = movement.blend(desired, math.atan2(dy, dx), pull)
        elif sp.territory > 0 and not settled:
            if self.failed_claims and self.energy > CONTEST_CONDITION * self.max_energy:
                # No vacancy to be found: head for a resident's core and challenge it.
                rival = w.nearest_rival(self, 2.0 * sp.dispersal)
                if rival is not None:
                    desired = movement.blend(desired, math.atan2(rival.home_y - self.y, rival.home_x - self.x), 0.5)
            else:
                # Floaters steer away from the scent marks of territory holders.
                away = w.territory_push(self)
                if away is not None:
                    desired = movement.blend(desired, away, 0.6)
        if self.danger_mem:
            # Steer away from places where danger was met recently.
            ax = self.x + math.cos(desired) * REGION
            ay = self.y + math.sin(desired) * REGION
            if self.recall(w, self.danger_mem, self.region(w, ax, ay), DANGER_MEMORY_DAYS) > 0.4:
                desired = movement.blend(desired, desired + math.pi, 0.6)
        kappa = sp.kappa_ars if self.ars > 0 else sp.kappa_explore
        # Step lengths vary from tick to tick (gamma distributed, mean 1).
        self._move(w, desired, self.speed * pace * w.rng.gammavariate(4.0, 0.25), kappa)

    def _set_target(self, target) -> None:
        self.target = target
        self.best_d = math.inf
        self.stall = 0
        self.path = None
        self.route = None

    def _ignored(self, w):
        return self.ignore if w.tick < self.ignore_until else None

    def _progress(self, w, d, limit=STALL_LIMIT) -> bool:
        """Track whether the animal is closing on its target; give up if not."""
        if d < self.best_d - 0.05:
            self.best_d = d
            self.stall = 0
            return True
        self.stall += 1
        if self.stall <= limit:
            return True
        self.ignore = self.target
        self.ignore_until = w.tick + IGNORE_TICKS
        self._set_target(None)
        return False

    def _toward(self, w, tx, ty, speed, sprint=False):
        """Head for (tx, ty), planning a route if the straight line is blocked.

        Returns the remaining route length, or None if there is no route of
        reasonable length (the caller should then give up on this goal).
        """
        d = math.hypot(tx - self.x, ty - self.y)
        if d < 0.5 or w.clear_path(self.x, self.y, tx, ty):
            self.path = None
            self._move(w, math.atan2(ty - self.y, tx - self.x), speed, DIRECTED_KAPPA, sprint=sprint, limit=d)
            return d
        path, goal = self.path, self.path_goal
        if path is None or w.tick >= self.path_t or goal is None or abs(goal[0] - tx) + abs(goal[1] - ty) > 2.0:
            found = w.find_path(self.x, self.y, tx, ty, max(40.0, 3.0 * d))
            if found is None:
                self.path = None
                return None
            path = self.path = found[0]
            self.path_goal = (tx, ty)
            self.path_t = w.tick + 15  # replan now and then: goals move
        # Cut corners: skip waypoints that are already in plain view.
        while len(path) > 1 and w.clear_path(self.x, self.y, path[1][0], path[1][1]):
            path.pop(0)
        wx, wy = path[0]
        if len(path) > 1 and math.hypot(wx - self.x, wy - self.y) < 0.4:
            path.pop(0)
            wx, wy = path[0]
        step = math.hypot(wx - self.x, wy - self.y)
        self._move(w, math.atan2(wy - self.y, wx - self.x), speed, DIRECTED_KAPPA, sprint=sprint,
                   limit=step if len(path) == 1 else math.inf)
        return step + sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))

    def _approach(self, w, tx, ty, speed, sprint=False) -> bool:
        """Move towards the current target; give it up if it can't be reached."""
        d = self._toward(w, tx, ty, speed, sprint)
        if d is None:
            self.ignore, self.ignore_until = self.target, w.tick + IGNORE_TICKS
            self._set_target(None)
            return False
        return self._progress(w, d)

    # ------------------------------------------------------------------ actions

    def _wander(self, w) -> None:
        r = w.rng.random()
        if r < REST_CHANCE:
            self.act = Act.REST
            self.rest_left = w.rng.expovariate(1.0 / REST_MEAN)
            self.last_move = 0.0
        elif not self.adult and r < REST_CHANCE + PLAY_CHANCE and self.energy > 0.6 * self.max_energy:
            self.act = Act.PLAY
            self.rest_left = 6 + 10 * w.rng.random()
            self._play(w)
        elif r < REST_CHANCE + PLAY_CHANCE + GROOM_CHANCE:
            self.act = Act.GROOM
            self.rest_left = w.rng.expovariate(1.0 / 8.0)
            self.last_move = 0.0
        else:
            self._range(w, self.sp.cruise)

    def _rest(self, w) -> None:
        self.last_move = 0.0
        self.rest_left -= 1
        if self.rest_left <= 0:
            self.act = Act.WANDER

    def _groom(self, w) -> None:
        self._rest(w)

    def _mark(self, w) -> None:
        self._rest(w)

    def _play(self, w) -> None:
        """Bursts of running and jumping: young animals practising motor skills."""
        self.rest_left -= 1
        if self.rest_left <= 0:
            self.act = Act.WANDER
        self._move(w, self.heading, self.speed * 0.7, 1.0)

    def _go_home(self, w) -> None:
        d = math.hypot(self.home_x - self.x, self.home_y - self.y)
        if d >= 0.6:
            r = self._toward(w, self.home_x, self.home_y, self.speed * self.sp.cruise)
            if r is not None and self._progress(w, r):
                return
        # Arrived, or can't get there: rest here.
        self.last_move = 0.0
        stay = w.rest_remaining[self.si]
        if self.pregnant:
            stay = max(stay, self.gestation_left + 2)
        if self.sp.burrow_reach > 0.0 and d < 0.6:
            self.act = Act.HIDE
            self.x, self.y = self.home_x, self.home_y
            self.concealed = True
        else:
            self._switch(Act.REST)
        self.rest_left = max(stay, 1)

    def _forage(self, w) -> None:
        sp = self.sp
        B = w.veg.B
        if B[int(self.y), int(self.x)] >= sp.graze_min:
            self._switch(Act.EAT)
            self._eat(w)
            return
        t = self.target
        if t is not None and B[t[1], t[0]] < sp.graze_min:
            t = self.target = None
        if t is None:
            if self.search_cd > 0:
                self.search_cd -= 1
            else:
                t = w.find_plant(self)
                if t is None:
                    self.search_cd = 3
                    self.learn(w, self.food_mem, self.region(w), 0.0, FOOD_MEMORY_DAYS, 0.5)
                else:
                    self._set_target(t)
                    self.goal = None
        if t is None:
            # Nothing in sight: head back to a remembered good patch, else search.
            if not self._go_to_remembered(w, self.food_mem):
                self._range(w, sp.cruise)
            return
        self._approach(w, t[0] + 0.5, t[1] + 0.5, self.speed * sp.cruise)

    def _eat(self, w) -> None:
        sp = self.sp
        self.last_move = 0.0
        if self.carcass > 0.0:
            take = min(self.intake, self.carcass)
            self.carcass -= take
            self.energy = min(self.max_energy, self.energy + take * sp.assimilation)
            self.water = min(self.max_water, self.water + take * MEAT_WATER)
            # Parents feeding young eat their share and carry the rest home.
            full = PROVISION_SATED if self._den_to_feed(w) is not None else SATED
            if self.carcass <= 0.0 or self.energy >= full * self.max_energy:
                leftover = self.carcass
                self._switch(Act.REST)
                self.rest_left = w.rng.expovariate(1.0 / REST_MEAN)
                if sp.caches and leftover >= CACHE_MIN:
                    self._start_cache(w, leftover)
            return

        # Scanning bouts: rear up and look around instead of eating. More
        # vigilant and more frightened animals scan more; groups scan less.
        if w.tick >= self.group_t:
            self.group = len(w.query((self.si,), self.x, self.y, 3.0, exclude=self))
            self.group_t = w.tick + 10
        danger = self.recall(w, self.danger_mem, self.region(w), DANGER_MEMORY_DAYS) if self.danger_mem else 0.0
        scan = SCAN * self.traits.get("vigilance", 1.0) * (1.0 + self.fear + danger) / (1.0 + 0.25 * min(self.group, 8))
        if w.rng.random() < scan:
            self.scanning = True
            return

        cx, cy = int(self.x), int(self.y)
        B = w.veg.B
        b = float(B[cy, cx])
        if b < sp.giving_up:
            # Patch exhausted: search nearby, where more food is likely.
            self._switch(Act.FORAGE)
            self.ars = sp.ars_duration
            return
        per_biomass = w.plant_energy * sp.assimilation
        # Holling type II functional response: intake saturates with biomass.
        bite = (self.intake / w.plant_energy) * b / (b + sp.half_saturation)
        bite = min(bite, b, (self.max_energy - self.energy) / per_biomass)
        B[cy, cx] = b - bite
        self.energy += bite * per_biomass
        self.learn(w, self.food_mem, self.region(w), min(1.0, b / w.veg.capacity), FOOD_MEMORY_DAYS, 0.3)
        # Fresh green plants are mostly water; dry winter forage much less so.
        self.water = min(self.max_water, self.water + bite * PLANT_WATER * w.forage_moisture)
        if self.energy >= SATED * self.max_energy:
            self._switch(Act.WANDER)
            self.ars = sp.ars_duration

    def has_young(self, w) -> bool:
        """Whether any of this animal's offspring are still immature."""
        for c in self.children_ids[-10:]:
            kid = w.by_id.get(c)
            if kid is not None and not kid.adult:
                return True
        return False

    def _den_to_feed(self, w):
        """The mother whose den this animal provisions (itself, or its mate), if any."""
        if not self.sp.provisions:
            return None
        if self.female:
            return self if self.has_young(w) else None
        mate = self.partner
        return mate if mate is not None and mate.alive and mate.has_young(w) else None

    def _start_cache(self, w, energy) -> None:
        """Carry leftovers to the den while there are young to feed, or bury them nearby."""
        mother = self._den_to_feed(w)
        if mother is not None:
            dest, den = (mother.nest_x, mother.nest_y), mother.id
        else:
            a, r = w.rng.uniform(-math.pi, math.pi), w.rng.uniform(1.0, 3.0)
            dest = (self.x + r * math.cos(a), self.y + r * math.sin(a))
            if not w.walkable(*dest):
                dest = (self.x, self.y)
            den = None
        self._switch(Act.CACHE)
        self.carried = energy
        self.cache_den = den
        self.dest = dest

    def _cache(self, w) -> None:
        if self.dest is None:  # digging
            self.last_move = 0.0
            self.rest_left -= 1
            if self.rest_left <= 0:
                w.add_cache(self, self.carried, self.cache_den)
                self.carried = 0.0
                self._switch(Act.WANDER)
            return
        tx, ty = self.dest
        if math.hypot(tx - self.x, ty - self.y) >= 0.7:
            r = self._toward(w, tx, ty, self.speed * self.sp.cruise)
            if r is not None and self._progress(w, r):
                return
        self.dest = None  # arrived, or can't get there: dig here
        self.rest_left = 3
        self.last_move = 0.0

    def _scavenge(self, w, food) -> None:
        """Walk to a carcass or cache and eat it."""
        d = math.hypot(food.x - self.x, food.y - self.y)
        if d < 0.8:
            self.carcass = food.energy
            food.energy = 0.0
            w.hunts[self.si]["cache" if isinstance(food, Cache) else "scavenged"] += 1
            self.act = Act.EAT
            self.target = None
            self.last_move = 0.0
        else:
            self._approach(w, food.x, food.y, self.speed * self.sp.cruise)

    def _hunt(self, w) -> None:
        sp = self.sp
        prey = self.target
        if isinstance(prey, Carrion):
            if prey.energy > 0.0:
                self._scavenge(w, prey)
                return
            prey = None
            self._set_target(None)
        if prey is not None:
            dx, dy = prey.x - self.x, prey.y - self.y
            if not prey.alive or prey.concealed or dx * dx + dy * dy > (1.25 * self.vision) ** 2:
                prey = None
                self._set_target(None)
                if self.surplus:
                    self._switch(Act.WANDER)
                    return
        if prey is None:
            if self.search_cd > 0:
                self.search_cd -= 1
            else:
                food = w.find_cache(self) or w.find_carrion(self)
                if food is not None:
                    self._set_target(food)
                    self._scavenge(w, food)
                    return
                if self.stamina >= HUNT_STAMINA * self.max_stamina:
                    found = w.find_prey(self)
                    prey = found[0] if found else None
                if prey is None:
                    self.search_cd = 2
                    self.learn(w, self.food_mem, self.region(w), 0.0, FOOD_MEMORY_DAYS, 0.2)
                else:
                    self._set_target(prey)
                    self.goal = None
        elif self.search_cd > 0:
            self.search_cd -= 1
        else:
            self.search_cd = 3
            self._reconsider(w, prey)
            prey = self.target
        if prey is None:
            # No prey in sight: return to a remembered hunting ground, else search.
            if not self._go_to_remembered(w, self.food_mem):
                self._range(w, sp.cruise)
            return

        persistence = self.traits.get("persistence", 1.0)
        dx, dy = prey.x - self.x, prey.y - self.y
        d = math.hypot(dx, dy)
        if d <= sp.capture_radius:
            w.pounce(self)
            self.skill += 0.3 * self.traits.get("learning", 0.25) * (1.0 - self.skill)
            # Young foxes are clumsy hunters and improve with practice.
            if w.rng.random() < sp.capture_success * (0.75 + 0.25 * self.skill):
                self.skill += self.traits.get("learning", 0.25) * (1.0 - self.skill)
                self.learn(w, self.food_mem, self.region(w), 1.0, FOOD_MEMORY_DAYS)
                self.carcass = prey.mass * prey.sp.carcass_energy
                self.kills += 1
                w.hunts[self.si]["surplus kill" if self.surplus else "kill"] += 1
                w.kill(prey, "predation")
                self.act = Act.EAT
                self.target = None
                self.surplus = False
            else:
                w.hunts[self.si]["missed lunge"] += 1
                prey.reward_events -= PAIN_LUNGE
                self.stun = LUNGE_RECOVERY
                self.stamina = max(0.0, self.stamina - 5.0)
            self.last_move = 0.0
            return

        # Stalk slowly and low, then rush once close. Prey that spots the
        # stalker early has too big a head start, so the attempt is abandoned.
        if prey.act == Act.FLEE and d > ABORT_DISTANCE * persistence * sp.rush_distance:
            w.hunts[self.si]["spotted"] += 1
            self.ignore, self.ignore_until = prey, w.tick + IGNORE_TICKS
            self._set_target(None)
            self._range(w, sp.cruise)
            return
        rush = d < sp.rush_distance or prey.act == Act.FLEE
        if rush and self.stamina < 1.0:
            w.hunts[self.si]["exhausted"] += 1
            self.ignore, self.ignore_until = prey, w.tick + IGNORE_TICKS
            self._set_target(None)
            self._range(w, sp.cruise)
            return
        pace = self.speed * sp.cruise * (1.0 if rush else sp.stalk)
        self.stalking = not rush
        if not w.clear_path(self.x, self.y, prey.x, prey.y):
            # The prey is round an obstacle: follow a planned route, unless the
            # detour is too long to be worth it.
            r = self._toward(w, prey.x, prey.y, pace, sprint=rush)
            if r is None or r > 1.5 * d + 6.0:
                w.hunts[self.si]["unreachable"] += 1
                self.ignore, self.ignore_until = prey, w.tick + IGNORE_TICKS
                self._set_target(None)
                return
            self.route = r
            if not self._progress(w, r, STALL_LIMIT * persistence):
                w.hunts[self.si]["outrun"] += 1
            return
        self.route = d
        if not self._progress(w, d, STALL_LIMIT * persistence):
            w.hunts[self.si]["outrun"] += 1
            self._range(w, sp.cruise)
            return
        # Aim ahead of the prey to intercept it rather than tail-chase.
        lead = min(d / max(self.speed, 1e-6), 8.0) * prey.last_move
        tx = prey.x + math.cos(prey.heading) * lead
        ty = prey.y + math.sin(prey.heading) * lead
        self._move(w, math.atan2(ty - self.y, tx - self.x), pace, DIRECTED_KAPPA, sprint=rush)

    def _reconsider(self, w, prey) -> None:
        """Switch to a clearly better target.

        While stalking, a fox weighs every prey animal it notices and switches
        if one is much easier to reach. Once a chase is on it stays with the
        fleeing animal, switching only to one that comes within lunging range.
        """
        d = math.hypot(prey.x - self.x, prey.y - self.y)
        chasing = prey.act == Act.FLEE or d < self.sp.rush_distance
        found = w.find_prey(self, radius=2.0 if chasing else None, skip=prey)
        if found is None:
            return
        other, cost = found
        current = (self.route or d) * w.vulnerability(prey)
        if cost < (1.0 if chasing else 0.6) * current:
            w.hunts[self.si]["switched target"] += 1
            self._set_target(other)

    def _flee(self, w) -> None:
        # Run directly away from the threats, each weighted by 1/distance.
        fx = fy = 0.0
        for d2, p in self.threats:
            if d2 > 1e-9 and p.alive:
                fx += (self.x - p.x) / d2
                fy += (self.y - p.y) / d2
        heading = math.atan2(fy, fx) if (fx or fy) else self.heading

        # Bolt for the burrow instead if it is close and not behind the predator.
        reach = self.sp.burrow_reach
        if reach > 0.0:
            bx, by = self.home_x - self.x, self.home_y - self.y
            db = math.hypot(bx, by)
            if db < 0.6:
                self.act = Act.HIDE
                self.x, self.y = self.home_x, self.home_y
                self.rest_left = w.rng.expovariate(1.0 / HIDE_MEAN)
                self.concealed = True
                self.last_move = 0.0
                return
            if db < reach and (bx * fx + by * fy) > -0.3 * db * math.hypot(fx, fy):
                self._move(w, math.atan2(by, bx), self.speed * TROT, DIRECTED_KAPPA, sprint=True, limit=db)
                return
        self._move(w, heading, self.speed * TROT, FLEE_KAPPA, sprint=True)

    def _freeze(self, w) -> None:
        self.last_move = 0.0
        self.scanning = True

    def _alert(self, w) -> None:
        """Warned by a thump: stand tall and look, or slip back towards the burrow."""
        self.scanning = True
        d = math.hypot(self.home_x - self.x, self.home_y - self.y)
        if self.sp.burrow_reach > 0.0 and d < 12.0:
            if d < 0.6:
                self.act = Act.HIDE
                self.x, self.y = self.home_x, self.home_y
                self.rest_left = w.rng.expovariate(1.0 / HIDE_MEAN)
                self.concealed = True
                self.last_move = 0.0
                return
            self._toward(w, self.home_x, self.home_y, self.speed * TROT)
        else:
            self.last_move = 0.0

    def _hide(self, w) -> None:
        self.last_move = 0.0
        self.rest_left -= 1
        self.concealed = True

    def _nest(self, w) -> None:
        """Dependent young stay in the nest; older ones play around its entrance."""
        self.last_move = 0.0
        if self.age < self.weaning * self.sp.emerge:
            self.x, self.y = self.nest_x, self.nest_y
            self.concealed = True
            return
        if self.predator_idx and w.query(self.predator_idx, self.x, self.y, min(self.vision, 4.0)):
            # Danger: bolt back into the nest.
            self.x, self.y = self.nest_x, self.nest_y
            self.concealed = True
            return
        self.concealed = False
        if self.energy < 0.7 * self.max_energy:
            # Older young eat the prey their parents carry back to the den.
            for c in w.caches:
                if c.den == self.mother_id and c.energy > 0.0 and abs(c.x - self.x) + abs(c.y - self.y) < 4.0:
                    take = min(c.energy, self.intake, (self.max_energy - self.energy) / self.sp.assimilation)
                    c.energy -= take
                    self.energy += take * self.sp.assimilation
                    break
        d = math.hypot(self.nest_x - self.x, self.nest_y - self.y)
        if d > 2.0:
            self._toward(w, self.nest_x, self.nest_y, self.speed * 0.5)
        elif w.rng.random() < 0.15:
            self._move(w, self.heading, self.speed * 0.4, 1.5)

    def _visit(self, w) -> None:
        """A mother's nursing visit: go to the nest, nurse briefly, leave."""
        if self.nursing > 0:
            self.nursing -= 1
            self.last_move = 0.0
            self.concealed = self.sp.burrow_reach > 0.0
            if self.nursing <= 0:
                self.next_nurse = w.tick + self.sp.nurse_interval
                self._switch(Act.WANDER)
            return
        d = math.hypot(self.nest_x - self.x, self.nest_y - self.y)
        if d < 0.8:
            self.x, self.y = self.nest_x, self.nest_y
            self.nursing = self.sp.nurse_ticks
            self.last_move = 0.0
            self.concealed = self.sp.burrow_reach > 0.0
        else:
            r = self._toward(w, self.nest_x, self.nest_y, self.speed * self.sp.cruise * 1.3)
            if r is None or not self._progress(w, r):
                self.next_nurse = w.tick + 30
                self._switch(Act.WANDER)

    def _chase(self, w) -> None:
        a = self.target
        if a is None or not a.alive or w.tick >= self.chase_until:
            self._switch(Act.WANDER)
            return
        r = self._toward(w, a.x, a.y, self.speed * TROT, sprint=True)
        if r is None or r < 1.5:
            self._switch(Act.WANDER)

    def _seek_water(self, w) -> None:
        if w.terrain.is_shore(int(self.x), int(self.y)):
            self.act = Act.DRINK
            self._drink(w)
            return
        heading = w.terrain.water_heading(self.x, self.y)
        if heading is None:
            self._range(w, self.sp.cruise)
        else:
            self._move(w, heading, self.speed * self.sp.cruise, DIRECTED_KAPPA)

    def _drink(self, w) -> None:
        self.last_move = 0.0
        self.water = min(self.max_water, self.water + self.sp.drink_rate * self.max_water)
        if self.water >= self.max_water:
            self._switch(Act.WANDER)

    def _seek_mate(self, w) -> None:
        mate = self.target
        if mate is not None:
            dx, dy = mate.x - self.x, mate.y - self.y
            if not mate.alive or not mate.ready_to_mate(w) or dx * dx + dy * dy > (1.25 * self.sp.mate_range) ** 2:
                mate = None
                self._set_target(None)
        if mate is None:
            if self.search_cd > 0:
                self.search_cd -= 1
            else:
                mate = w.find_mate(self)
                if mate is None:
                    self.search_cd = 3
                else:
                    self._set_target(mate)
        if mate is None:
            self._range(w, self.sp.cruise)
            return
        d = math.hypot(mate.x - self.x, mate.y - self.y)
        if d <= MATE_RADIUS:
            male = mate if self.female else self
            # Female choice: males in good condition, and bigger males, are
            # more likely to be accepted (sexual selection).
            appeal = (male.energy / male.max_energy) * (male.mass / male.sp.ref_mass) ** 0.5
            if w.rng.random() < min(1.0, max(0.2, 0.35 + 0.65 * appeal)):
                w.mate(self, mate)
            else:
                w.rejections[self.si] += 1
                for a, b in ((self, mate), (mate, self)):
                    a.ignore, a.ignore_until = b, w.tick + IGNORE_TICKS
                    a._set_target(None)
            return
        self._approach(w, mate.x, mate.y, self.speed * self.sp.cruise)


_ACTIONS = {
    Act.WANDER: Animal._wander,
    Act.REST: Animal._rest,
    Act.FORAGE: Animal._forage,
    Act.EAT: Animal._eat,
    Act.SEEK_WATER: Animal._seek_water,
    Act.DRINK: Animal._drink,
    Act.SEEK_MATE: Animal._seek_mate,
    Act.FLEE: Animal._flee,
    Act.HUNT: Animal._hunt,
    Act.HIDE: Animal._hide,
    Act.NEST: Animal._nest,
    Act.HOME: Animal._go_home,
    Act.FREEZE: Animal._freeze,
    Act.ALERT: Animal._alert,
    Act.GROOM: Animal._groom,
    Act.PLAY: Animal._play,
    Act.CACHE: Animal._cache,
    Act.MARK: Animal._mark,
    Act.CHASE: Animal._chase,
    Act.VISIT: Animal._visit,
}
