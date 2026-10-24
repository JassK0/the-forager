# Forager

A 3D ecosystem sandbox in the spirit of Sebastian Lague's *Simulating an
Ecosystem*. Rabbits and foxes live, forage, hunt, breed and evolve on a voxel
landscape with days, nights and seasons. Every animal carries a diploid
genome, and what you see (coat colour, body size, speed, vision, temperament,
even the wiring of its brain) is expressed from it.

The simulation engine is Python (numpy for the landscape), served by Flask.
The browser renders it with Three.js.

## Running it

From the repository root, with the virtual environment active:

```powershell
pip install -r newForager/requirements.txt
cd newForager
python app.py            # then open http://127.0.0.1:5000
```

The page loads Three.js and its fonts from public CDNs, so the browser needs
internet access the first time.

### The AI analyst (optional)

To have Claude explain your runs, put a Claude API key in `newForager/.env`
(copy `.env.example` if the file is missing):

```
ANTHROPIC_API_KEY=sk-ant-...
AI_MODEL=claude-opus-5-5     # optional; claude-sonnet-5-5 is cheaper and faster
```

`.env` is in `.gitignore`, so the key is never committed. The server reads the
file on every request, so there is no need to restart after adding the key.
Nothing extra needs installing: `ai.py` calls the Claude Messages API with
Python's standard library.

### Controls

| | |
|---|---|
| Orbit / pan / zoom | left-drag / right-drag / scroll (with a painting tool: right-drag orbits) |
| Move the camera | WASD or the arrow keys |
| Inspect an animal | click it (Select tool, `1`) |
| Play / pause | Space |
| Tools | `1`–`8`: select, add animals, grow plants, water, grassland, trees, rock, remove |
| Thought bubbles | `B` |
| Follow the selected animal | `F` |

**Speed:** at 1× the world runs 4 ticks per second, so a day lasts a minute and
animals move at a natural pace. The client interpolates between simulation
states, so motion stays smooth at 60 fps at any speed. *Max* runs as fast as
the Python engine allows, which is about 150–200 ticks per second.

**The inspector** shows an animal's needs, life stage, traits (phenotype vs
breeding value), Mendelian genotype, and a chromosome-by-chromosome view of
its genome. It also lists the animal's family, and has lab tools: follow,
clone, breed with another animal, gene editing and removal.

**Events** trigger droughts, blooms, cold snaps and plagues. **Repopulate**
replaces the animals on the current landscape, and **New world** generates a
new map.

**New world** has premade maps: the standard meadow, a small valley (80 × 50,
about four fox territories), a tiny test plot (48 × 32, one fox pair, for
watching a few animals closely), a lake district, open plains and wild
uplands. Each sets the size, landscape, starting numbers and immigration
(scaled to the length of the edge, where newcomers arrive). You can change
any of them afterwards. On the tiny plot, numbers are so small that chance
rules and either species can die out within a year or two, so that preset
keeps the run going through extinctions and newcomers refill it. **Emigration** can
be switched off for a closed population: foxes that find no territory then stay
on as floaters instead of leaving.

**The Data drawer** opens on the **🤖 AI analyst**:
- **At a glance:** plain numbers for the last 3 days: counts and trend, adult
  females and males, territories, births and deaths with the main cause, and
  warnings such as "no adult females: they can't breed". No AI is needed.
- **Ask:** *Summarize this run*, *Why did the foxes die out?* (one button per
  recorded extinction), or any question in your own words. Claude's answer
  streams in, citing the days and numbers that prove each point.
- **Evidence:** each answer links to exactly what Claude was given (*See the
  evidence*) and the map it looked at.
- **What Claude reads:** the run's daily ledger (births, deaths by cause,
  adults by sex, territories, condition, food, weather); the last deaths
  before each extinction (who, how old, what of, where, in what condition);
  the traits of the parents and of each cause's victims (natural selection);
  how traits and learned behaviour changed; your sandbox actions; the
  narrator's notes; and a rendered map with a region table.
- **Cost:** each answer sends about 30–60 KB of records, roughly 10–20K tokens,
  to Anthropic's API and uses your key's credit. Asking the same thing again at
  the same moment reuses the saved answer.

The end-of-run summary is written the same way, with the built-in notes folded
underneath. Extinction pop-ups have an **Ask AI why** button. Without a key,
everything else works as before, and the summary uses the built-in notes.

The other tabs have charts for:
- populations and the predator–prey phase plane;
- deaths and births: born vs died per day, what killed them, adults of each
  sex old enough to breed, and food per head;
- trait means with standard deviation, against the mean breeding value;
- allele frequencies, with observed vs Hardy–Weinberg heterozygosity;
- neutral genetic diversity and genetic load.

**CSV** exports everything.

**💡 Why?** opens the narrator, which explains what is happening and why. When
something notable happens (a population crash or boom, an allele spreading or
vanishing, a trait evolving, genetic diversity draining away, an event ending,
a new season) a pop-up explains it. **Explain right now** writes a fresh report
on the last day.

The narrator's explanations come from what the simulation recorded, by fixed
rules rather than an AI model:
- **Population changes** are broken down by recorded cause of death and
  births, with the food, weather and predator context behind them.
- **Allele changes** are tested against the drift expected at the current
  effective population size, sd = √(p(1−p)·t / 2Ne) (Wright 1931). The
  narrator also checks whether carriers were over-represented among animals
  dying of a given cause, or under-represented among parents.
- **Trait evolution** is explained with the breeder's equation, R = h²S
  (Lush 1937), using the measured selection differential between parents and
  the average adult. Victims are compared with survivors to name the pressure.

All of this lives in `engine/narrator.py`.

URL options: `?zoom=0.5` starts closer, `?inspect=fox&tab=genome` opens on a
fox's genome, `?data=genes` opens the drawer on a tab, and `?why=explain` opens
the narrator with a fresh report.

### Without the browser

```powershell
python headless.py --years 5 --seed 3 --out run.csv
python headless.py --years 5 --brains rules   # fixed instincts instead of evolving brains
python -m pytest tests
python -m tools.train_brains                  # re-train the founders' brains (about a minute)
```

## Time

A day is 240 ticks, and a year is compressed to 12 days (3 per season), so
seasons and generations pass in minutes. Within a day, physiology runs on
roughly realistic time (a rabbit starves in about two days). Life history is
scaled to the compressed year: rabbits live about a year and a half and breed
from spring to late summer, and foxes breed once a year in late winter.

### Speed

The simulation runs on its own clock in the server, in a background thread.
The browser never waits for it: a few times a second it fetches the latest
state and moves the animals smoothly in between. Speed only changes how
quickly ticks are computed, never what happens in them. Every tick is the
same calculation, so a run is identical at any speed and reproducible from
its seed.

- **1×–25×** ask for a fixed rate (1× = 4 ticks a second). **Max** runs as
  fast as the computer allows. The chip beside the speeds shows the speed
  actually reached. It turns amber when the computer can't keep up with the
  requested speed, which happens with many hundreds of animals.
- **⏩ Skip ahead** simulates a day, a season or 1–25 years flat out, with a
  progress bar, then carries on as before. It gives exactly the same result
  as watching (this is tested). As a rough guide, with 300–450 animals it
  runs at about 20× real time, or about 35 seconds per simulated year; more
  animals make it slower.
- For the longest runs without the browser, `headless.py` runs the same
  engine and writes the time series to CSV.

Optimisations to the engine are checked against recorded reference runs: a
change must reproduce them exactly, tick for tick.

## Genetics (`engine/genetics.py`, `engine/coats.py`)

- **Genome.** Six pairs of autosomes carry about 80 loci at map positions in
  Morgans, plus XY sex chromosomes.
- **Meiosis.** Crossovers per chromosome follow a Poisson distribution with
  mean equal to the map length (Haldane 1919), so linked loci tend to be
  inherited together. The tests check recombination fractions against
  Haldane's map function.
- **Quantitative traits.** Body mass, speed, vision, litter size, vigilance,
  stamina and fur insulation are each built from five additive loci, with
  developmental noise set to a chosen heritability. Personality traits
  (below) and gestation length are built the same way from four loci each.
- **Gestation length** (both species, about ±4% between individuals, as in
  real populations) is a trade-off. A longer pregnancy produces bigger young
  (birth mass grows with its cube, since most fetal growth comes late) and a
  larger share of the mother's reserves, so they start life with more in
  hand. In exchange the mother carries the cost of pregnancy for longer. The tests recover h² from
  parent–offspring regression. Every trait has a cost:
  - Larger animals need more food.
  - Speed, vision and stamina carry upkeep costs.
  - Vigilance costs feeding time.
  - Insulation helps in winter but overheats in summer, so selection on it
    fluctuates with the seasons.
- **Mendelian coat genes.**
  - Rabbits: Agouti, Brown, the albino series (C > cᶜʰᵈ > cʰ > c), Dilute,
    and English spotting. The En/En "charlie" genotype carries a health cost.
  - Foxes: silver (red, cross and silver phases), pearl, and platinum.
    Platinum is lethal when homozygous.
  - Dominance and epistasis follow the real genes (albino masks everything).
    About 40 rabbit and 12 fox phenotypes are possible.
  - **Winter white (Mo).** A rare variant that moults to a white coat from
    mid-autumn to early spring. It is dominant in rabbits, as in mountain and
    snowshoe hares (Jones et al. 2018), and recessive in foxes, as in the
    Arctic fox's white morph (MC1R; Våge et al. 2005). Red foxes don't really
    carry it; it is here so that selection has something to work on where snow
    lies long. The moult is timed by day length, not by snow, so a white coat
    can be mismatched when snow is late or absent, as with real hares in a
    warming climate (Mills et al. 2013).
- **Camouflage.** How readily an animal is spotted depends on how different
  its coat looks from the ground right under it, to the eye that is watching.
  It works both ways: foxes spot prey by it, and rabbits spot stalking foxes.
  - **Their eyes:** foxes and rabbits are dichromats, red-green colour-blind
    (Jacobs 1993). They judge lightness, and yellow against blue at half
    weight. So an orange fox looks to a rabbit much like green grass, which is
    why red foxes are camouflaged at all.
  - **The ground's colours:**

    | Ground | Colour |
    |---|---|
    | Growing grass | mid-dark green |
    | Dead grass (autumn, winter, dry country) | pale straw |
    | Grazed earth | brown |
    | Leaf litter under trees | dark |
    | Sand | pale |
    | Rock | grey |
    | Snow | white |

  - **Movement gives it away:** camouflage works while an animal keeps still.
    A running rabbit is easy to see whatever its colour (Stevens & Merilaita
    2009). At night colour hardly matters.
  - **What it favours:** each colour has its place.
    - Wild agouti brown on grass and earth
    - Black and chocolate in the woods
    - Grey on rock
    - Pale fawn on sand and dead grass
    - White on snow
    - Red foxes in grass and dead bracken

    So selection on coat colour differs from place to place and season to
    season. The narrator and the AI analyst compare how much the animals
    foxes caught stood out against the ground with how much the living do.
- **Genetic load.** Mostly recessive deleterious mutations arise and are
  purged by selection. Inbreeding exposes them as reduced vigour, causing
  stillbirths and costlier metabolism.
- **Neutral markers.** Microsatellite-like loci track drift and diversity.
- **Brain loci.** One locus per connection of the neural network (200 in
  all), spread over the chromosomes. Each connection's strength is the sum of
  its two alleles, so brains recombine and mutate like any polygenic trait.
- **Mutation rates** are far above natural ones so that evolution is visible
  within a session. You can scale them in New world.

## Climates (`engine/climate.py`)

Choose a climate under New world:

| Climate | What it is like |
|---|---|
| 🌳 Temperate meadow | mild seasons: snowy winters, green summers (the original world) |
| ❄️ Arctic tundra | long snowy winters, a short summer for plants to grow, few trees |
| 🏜️ Desert | scorching summers, cool winters with little snow, patchy scrub and oases round a few waterholes |
| ⛰️ Mountains | green valleys under cold, snowy heights, with a tree line |
| 🌾 Grassland plains | open, dry-ish plains with warm summers and little cover |
| 🌍 Mixed | a cold, snowy north grading into a hot, dry south, with cold mountain tops |

The Mixed map puts several regions side by side, so you can watch the same
species adapt differently to each, a cline.

Every cell has its own temperature (the season's, plus the place's offset,
colder with altitude) and dryness. These set:
- **snow cover:** where it is cold, and not too dry for snow to fall;
- **plant growth:** how much plants grow (less in the cold and the dry, where
  sand also leaves only patches of scrub) and when (frost stops growth, and
  summer heat makes desert plants go dormant);
- **water from food:** how much animals get from what they eat;
- **cold and heat costs:** what each animal pays where it stands. Thick fur
  and burrows help in the cold; in the heat, thick fur hurts and water is lost
  faster;
- **the ground's colour,** and so camouflage.

What to watch:
- **Arctic:** insulation and winter-white coats should spread.
- **Desert:** thinner fur, and coats matching the sand.
- **Mixed:** differences between the north and south of the same map.

The inspector shows where an animal lives ("📍 Tundra · bitterly cold · deep
snow · coat well hidden against the ground"). The AI analyst gets a
region-by-region table of conditions and adaptation: biome, temperature,
snow, and each species' winter-white share, camouflage gap and fur.

Each climate suggests its own landscape and starting numbers, scaled to how
much the land can feed. In testing over 3 years, both species persisted in
every climate on most seeds. The Arctic and the desert are deliberately
harsh: numbers run low, and chance and immigration matter more there.

## Brains, personality and memory (`engine/brain.py`)

Every tick, each independent animal senses its situation and chooses one of
eight options: eat (hunt, for foxes), drink, find a mate, rest, explore,
flee, freeze or take cover. Its existing skills then carry the choice out:
route planning, stalking, grazing, nursing and caching are unchanged. Two
decision-makers are available, chosen under New world → Decision-making:

- **Evolving neural networks** (default). A small network per animal:
  15 senses plus a bias → 8 tanh hidden neurons → one score per option. The
  highest-scoring available option wins, with a small bonus for the option
  already being pursued so animals don't dither. The 200 weights are genes,
  so behaviour evolves by natural selection and drift (Huse, Strand & Giske
  1999; Eliassen et al. 2016). A whole species is evaluated in one batched
  numpy calculation per layer, which costs about the same as the rules.
- **Fixed instincts.** The hand-written rules the model started with,
  written as utility functions (`rule_scores`).

The senses are hunger, thirst, mating drive, threat (predator distance
relative to the flight distance), whether a predator is charging, an alarm
heard, fear, remembered danger and food here, distance from home, daylight,
rest hours, freezing tendency, company and youth.

**Founders copy the instincts.** A random network would starve on day one, so
`tools/train_brains.py` records the situations animals meet in rules-mode runs,
adds random situations covering every input's range, and fits the network to
the instinct utilities (behaviour cloning, masked MSE, Adam, plain numpy).
Training adds random noise to the weights, so the result keeps working when
genetic variation nudges them (robustness of this kind is called
canalisation). The trained brains pick the instinct's choice in 100% of
clear-cut decisions (about 96% including close calls), and founders with
genetic variation do so in over 99%. Weights are saved to
`engine/brain_weights.json`.

**How far brains have evolved.** In neural mode the instinct choice is still
computed every tick as a reference. The 🧠 Brains chart shows how often brains
agree with it, and the narrator reports the commonest changes of mind
("where instinct would have them rest, they now eat"). Only clear-cut
decisions count, where the instincts prefer one option by a margin of at
least 0.1: close calls (eat now or drink first?) flip with tiny differences
in wiring, like personal quirks.

**Personality genes** set the gains on the senses, the way hormones modulate
drives, so they matter in both modes. Each starts at the old hand-set value
(1×) and varies between animals (Réale et al. 2007):

| Gene | Species | Effect |
|---|---|---|
| Boldness | rabbit | Divides the flight distance: bold rabbits let a fox come closer before fleeing, and a scare shrinks their range less. |
| Freezing tendency | rabbit | How strongly a rabbit prefers freezing to fleeing. |
| Appetite | both | Multiplies felt hunger, so hungrier animals start eating sooner. |
| Mating drive | both | How much mating takes priority over eating and drinking. |
| Chase persistence | fox | How long a fox keeps chasing without gaining, and how far ahead a bolting rabbit may get before it gives up. |
| Learning rate | both | The learning rate of every memory, below. A fast-learning brain costs upkeep (the expensive-brain hypothesis), so selection weighs learning against its cost (the Baldwin effect). |

**Memory and learning.** Each animal keeps two rough maps of 6×6-cell
regions, updated by the Rescorla–Wagner rule (expectation += rate × (outcome
− expectation)) and fading with time:

- *Food* (half-life 2 days): rabbits learn how good the grazing was where
  they ate, and foxes where they killed. With nothing in sight, they head
  back to the best remembered spot nearby, away from known danger. Places
  that turn out to be empty are marked down.
- *Danger* (half-life 1 day): every new predator encounter marks the place.
  Rabbits steer their wandering away from it, value its grazing less, and
  scan more while feeding there.
- *Wariness* replaces the old "young animals are naive" rule. Detection
  starts at 60% of an experienced animal's and grows with every predator
  spotted and every escape. Young foxes likewise start as clumsy hunters
  (75% of the catch rate) and improve with each pounce and kill.
- At weaning, young copy their mother's food and danger maps (social
  learning of where to feed).

**Lifetime learning (reinforcement learning).** Genes only set the wiring an
animal is born with. During its life every animal rewires its own network from
experience, the way the vertebrate basal ganglia are thought to learn
(actor–critic reinforcement learning; Barto 1995; Joel, Niv & Ruppin 2002).
It can be switched off under New world.

- *What feels good or bad.* Reward is homeostatic (Keramati & Gutkin 2014):
  the drive is D = (hunger⁴ + thirst⁴ + (0.8 × urge)⁴)^⅓, and reward is how
  much an action reduces it. An urgent need therefore dominates: water when
  parched is far more rewarding than a snack when nearly full. A predator's
  lunge that just missed, seeing a predator close, being chased off or losing
  a fight feel bad. Nursing young is rewarding to a mother.
- *The urge to mate.* As in real mammals, sexual motivation is switched on by
  breeding hormones, not learned (Pfaus 2009).
  - **Builds up:** in the breeding season, an adult breeder's urge to mate
    builds over about half a day, faster with a high mating-drive gene. Foxes
    must hold a territory, since subordinates don't breed.
  - **Sated:** mating relieves the urge and also feels good in itself.
    Pregnancy and the end of the season let it fade.
  - **Felt, but harmless:** the brain feels how aroused it is, and while the
    urge is strong, food matters a little less. Unlike hunger, it never harms
    the body.
  - **Why it matters:** without it, fast-learning brains in Super animals mode
    learned that searching for a mate usually goes unrewarded, unlearned
    mating, and passed the habit on.
- *Predicting and being surprised.* A critic (one more output of the network)
  learns to predict future wellbeing. The gap between what happened and what
  it expected, the reward prediction error, plays the role of dopamine
  (Schultz, Dayan & Montague 1997).
- *Rewiring.* The prediction error strengthens or weakens the connections
  behind recent choices, back-propagated through both layers, with
  eligibility traces so that a meal still credits the decision to go looking
  for food a while before.
- *Trying things.* Choices are slightly noisy (a softmax), so animals
  occasionally break habit and can discover that something else pays.
- *Forgetting and inheritance.* Learned changes relax back towards the inborn
  wiring (half-life about two weeks) unless experience keeps reinforcing
  them, and they are not passed on: young start from their genes. Juveniles
  learn 1.5× faster, and the learning-rate gene scales it, so evolution can
  favour good learners (the Baldwin effect; Hinton & Nowlan 1987).

A critical need still overrides learning. When hunger or thirst passes 75%,
the choice narrows to meeting that need (or escaping danger), like a
hard-wired emergency response. Without this, rabbits that learned in summer
to get their water from fresh grass kept doing so in winter, when the forage
is dry, and died of thirst.

Does it make them smarter? In a controlled test, half of each population
could learn and half could not, living in the same worlds (3 seeds × 2
years). Learning rabbits raised about 11–14% more young in two worlds and
about the same in the third, with similar lifespans: they starve less, but
graze more and so meet more foxes. Learning foxes lived about 35–40% longer
in the two worlds with enough foxes to compare. In practice rabbits learn to
graze almost continuously and to drink a little later, and to flee a little
earlier after near misses. Foxes learn to start hunting sooner.

Select an animal and open its 🧠 Brain tab to see how it thinks:

- *Right now:* what it wants to do, the senses that pushed that choice up
  or down ("because of ▲ hunger ▼ daylight"), what it expects, and how
  surprised it was by the last outcome.
- *The network:* view it as it is **now** (signal flowing through it), as it
  was **born with** (its genes), or **what it learned** (connections
  strengthened or weakened by experience).
- *What it has learned:* when it starts eating, drinking and fleeing,
  compared with the inborn wiring tested on the same situations; options it
  now prefers more or less; and how often it now decides differently.
- *Learning over its life:* charts of how much it has rewired, its wellbeing
  and its surprise (prediction error) by age.
- *Senses, option scores, memories, wariness and hunting skill.* "Show
  memory on map" paints its remembered food (green) and danger (red). It also
  shows what it does that leaves a mark: the food it has buried (orange
  diamonds), food carried home for its young (pale diamonds), its den, its
  territory, its recent scent marks (yellow cones), and a line to its mate.

Data → 🧠 Brains shows the population picture: agreement with the founders'
instincts (evolution), how often animals decide differently from how they
were born (learning), and inborn versus learned habits over time.

## Super animals mode (just for fun)

Tick "🦸 Super animals" under New world to see what selection does with the
brakes off. It is not realistic. Three sliders set how far it goes:

- **Evolution speed** (1–20×, default 6×): mutation is that many times more
  frequent. Inherited variation is larger and more heritable, 1.5× at 6× and
  up to 2× at 11× and above.
- **Brain learning speed** (1–10×, default 2×): brains rewire that much faster
  from rewards and punishments.
- **Know-how passed to young** (0–80%, default 50%): young start life with
  that share of their parents' know-how on top of their genes. That is their
  learned brain wiring (including what they themselves were taught), the
  mother's hunting skill and her wariness. Know-how therefore builds up over
  the generations as well as genes, so animals can find and keep ways of
  doing things their instincts never had. In real animals, learning is not
  inherited.

Traits also have no natural ceiling: ×3, with speed and vision allowed a
little more so movement and senses stay sound.

Every few hours of simulated time the three strongest adults of each species
are crowned 👑 alphas: they wear a gold crown on the map, and the inspector
shows their prowess relative to the average founder (speed, senses, size,
health, condition, experience and learning). The narrator announces each new
top animal. Everything else, including costs, trade-offs and natural
selection, works as normal. The end-of-run report (with a key) explains how
each species evolved and why, from the selection records: which traits the
parents had, and which the victims of each cause had.

## Ending a run

With **🏁 End when a species dies out** on (the default, switchable in the
💡 Why? panel or in New world), the run stops the moment a species goes
extinct and shows a breakdown:
- The story of the run.
- Populations over time.
- Per-species peaks, lows, boom-and-bust cycles and causes of death.
- How traits and allele frequencies evolved.
- Genetic diversity.
- A hall of fame: the oldest animal, the most offspring, the top hunter.
- A timeline of key moments.

From there you can keep running, release a new population or generate a new
world. Turn the option off to run indefinitely. **📜 Run summary** shows the
same breakdown at any time.

## Behaviour, from field studies

Each behaviour below is modelled because it changes who survives and
reproduces. None of it is decoration:

| Behaviour | What it does in the model | Source |
|---|---|---|
| Absentee maternal care | Rabbit kits stay hidden in the nest. The doe returns briefly to nurse and is otherwise away. Kits emerge near the nest before weaning. | [Rabbit maternal behaviour](https://www.intechopen.com/chapters/80083): does nurse about once a day for about 3 minutes |
| Den-raised cubs, fathers provide | Fox cubs stay in the den and come out at about 40% of weaning age. While there are young, males and the vixen eat their share of a kill (to 70% full) and carry the rest to the den. Cubs that have come out eat it, as do newly independent young nearby. A vixen finishes a stalk or a meal before going back to nurse. | [Discover Wildlife](https://www.discoverwildlife.com/animal-facts/mammals/understand-fox-behaviour), [Wildlife Online](https://www.wildlifeonline.me.uk/animals/article/red-fox-behaviour-helpers-in-fox-society) |
| Food caching, surplus killing | Foxes bury leftovers, remember their caches, pilfer others' caches by smell, and sometimes kill when not hungry. | [Wildlife Online](https://www.wildlifeonline.me.uk/animals/article/red-fox-diet-killing-to-excess-caching), [Sklepkovych & Montevecchi 1996](https://journalhosting.ucalgary.ca/index.php/arctic/article/view/64254) |
| Freeze, then flee | Prey that spot a distant predator freeze (still prey are harder to notice) and bolt once it comes within flight distance. That distance grows the farther they are from the burrow (economic escape, Ydenberg & Dill 1986). | [Tobajas et al. 2024](https://link.springer.com/article/10.1007/s42991-024-00398-3) |
| Alarm thumping | A fleeing rabbit drums an alarm, and neighbours stop, scan and head for their burrows. | [Petplan: rabbit behaviour](https://www.petplan.co.uk/pet-information/rabbit/advice/fearful-rabbit-behaviour-explained/) |
| Landscape of fear | After a scare, rabbits feed closer to their burrow and range less. Fear fades over about half a day. | [Tobajas et al. 2024](https://link.springer.com/article/10.1007/s42991-024-00398-3) |
| Vigilance and group size | Feeding is interrupted by scanning bouts (rearing up). Vigilant genotypes and frightened rabbits scan more; rabbits in groups scan less. | [Villafuerte & Moreno 1997](https://link.springer.com/doi/10.1007/BF00171099) |
| Huddling and burrow climate | Burrows buffer cold and heat, and every extra animal sharing a burrow cuts the cost of keeping warm. | [Huddling review](https://pmc.ncbi.nlm.nih.gov/articles/PMC3306396), [rabbit pup huddles](https://pmc.ncbi.nlm.nih.gov/articles/PMC5994089) |
| Water from food | Fresh plants and prey supply water, so animals on lush pasture rarely drink. Dry winter forage supplies much less. | [Armenteros et al. 2015](https://museucienciesjournals.cat/en/abc/issue/38-2-2015/use-of-water-troughs-by-wild-rabbits-oryctolagus-cuniculus-in-a-farmland-area-of-north-west-spain), [Cooke 1982](https://publish.csiro.au/WR/WR9820247) |
| Male-biased dispersal, warrens | Young males disperse farther. Most rabbit daughters stay in their mother's warren, so related females share burrows. | [Natal dispersal in rabbits](https://ueaeprints.uea.ac.uk/53034/), [Webb et al.](https://research-portal.uea.ac.uk/en/publications/fine-scale-genetic-structuring-in-a-natural-population-of-europea/) |
| Pair territories, scent marking and defence | Red foxes hold territories as pairs. A fox that loses its mate keeps the territory until a newcomer of the missing sex fills it. Dispersers join a lone mate or prospect for a vacancy, and leave only when the area is full. Residents mark the edge of their range and chase off same-sex intruders, and floaters steer away from occupied territories. Contests are ritualised and rarely fatal. A floater that finds no vacancy may challenge a resident. Bigger, better-fed animals usually win, and residents have an owner's advantage (Maynard Smith & Parker 1976). A winning challenger takes the territory, and the loser becomes a floater. About 2% of mating-season fights end in a fatal injury. | [Giuggioli et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3053310), [Folia Zoologica 2015](https://bioone.org/journals/Folia-Zoologica/volume-64/issue-1/fozo.v64.i1.a5.2015/Ontogenic-patterns-of-scent-marking-in-red-foxes-Vulpes-vulpes/10.25225/fozo.v64.i1.a5.2015.full) |
| Stalk, then chase | Foxes creep slowly and low (harder to spot). If the rabbit bolts, they switch to a bounding chase of that rabbit and finish with a pounce. | [Texas Parks & Wildlife](https://tpwd.texas.gov/publications/nonpwdpubs/introducing_mammals/red_foxes/), [Wildlife Online](https://www.wildlifeonline.me.uk/animals/article/red-fox-diet-hunting-strategies-behaviour) |
| Target choice and switching | Foxes pick prey by route length weighted by vulnerability: young ×0.7, alone ×0.85, watching the fox ×1.3. While stalking, they switch if a target is much easier (under 60% of the cost). Once chasing, they stay on the fleeing animal unless another comes within lunging range. | Young prey favoured: [Annear et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10038972/). Edge or isolated prey attacked more: [selfish-herd test](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11285717/). Mid-attack switching when prey are close together: [fringe-lipped bats](https://cupola.gettysburg.edu/biofac/95) |
| Route planning | Animals walk straight to goals when they can. When water, trees or rock block the way, they plan a route (A*). Prey reachable only by a long detour is ignored. | |
| Kin recognition | Parents, offspring and siblings don't mate. | Pusey & Wolf 1996 |
| Mate choice and male rivalry | A female judges a courting male's condition, size, genetic health (vigour) and injuries. Vixens also weigh his hunting skill and the richness of his territory, because males feed the vixen and cubs. A typical healthy male is always accepted, and a weak, sick or injured one is often refused. Vixens are choosier than does. When a rival male is near the female, the two compete, and the bigger, fitter, unhurt male usually wins. In rabbits this rank competition matters more than female choice. | Foxes: heavier males hold larger territories and father more cubs ([Iossa et al. 2008](https://bioone.org/journals/Journal-of-Mammalogy/volume-89/issue-6/07-MAMM-A-405.1/Body-mass-territory-size-and-life-history-tactics-in-a/10.1644/07-MAMM-A-405.1.full)). Rabbits: separate rank orders for each sex, settled by fights, drive breeding success (von Holst, Rödel et al.) |
| Injuries | Lost fights and a predator's near miss can injure an animal. An injury slows it, makes it a worse hunter and a less attractive mate, and costs energy to heal (a moderate one heals in about three days). About a quarter of severe injuries leave a permanent limp. | |
| Mating forays | In the mating season, territorial dog foxes search twice as far for receptive vixens, beyond their own territory, so lone or neighbouring vixens still get mated. Residents chase off intruding males, which sometimes leads to fights. | Extra-territorial movements of males in the breeding season: Soulsbury et al. 2011, *Journal of Mammalogy* |
| Voles and mousing | Foxes listen for and pounce on voles in rough grass, their staple prey in much of their range. Vole numbers breed in summer and dwindle in winter. | |
| Prey switching (search image) | A fox that has met rabbits lately is quick to notice them. One that hasn't loses its "eye" for rabbits (half-life about a day, down to 40%), notices them less, and mouses more instead. A rabbit right under its nose is still taken. So predation eases on scarce rabbits: they are held low after a crash but not wiped out. | Search images: Tinbergen 1960. A type III functional response of foxes to rabbits: [Pech et al. 1995](https://www.publish.csiro.au/wr/WR9950055). Evidence is mixed: Finnish foxes behaved as plain opportunists ([Dell'Arte et al. 2007](https://research.utu.fi/converis/portal/detail/Publication/31209874?lang=en_GB)) |
| Fruit | Brambles and wild fruit grow at woodland edges and ripen from late summer, peaking in early autumn and gone by winter. Foxes eat fruit when they see no prey. Young foxes, still clumsy hunters, fill up on fruit first. | Fruit was about 23% of autumn stomach contents in one Italian study, and foxes eat it widely in summer and autumn ([Ferro et al. 1987](https://www.storianaturale.org/anp/PDF%20ANP/08_1987_Ferro%20Ferro%20De%20Santis%20Boano_Alimentazione%20autunnale%20ed%20invernale%20della%20Volpe%20nel%20Cuneese%20.pdf); [Hungarian vineyard study](https://ojs.mtak.hu/index.php/actazool/article/view/935)) |
| Blind newborns | Cubs and kits are born blind. Fox cubs' eyes open at about a quarter of the way to weaning (about two weeks), rabbits' at about 40% (about 10 days), and sight then develops until weaning. | [Wildlife Online: cub development](https://www.wildlifeonline.me.uk/animals/article/red-fox-breeding-growth-development-of-cubs) |
| Learned wariness, play, grooming | Inexperienced animals are slower to recognise danger and grow warier with each encounter. The young play and adults groom between other activities. | Predator recognition is learned in many prey species: [Griffin 2004](https://doi.org/10.3758/BF03196002) |
| Personality | Boldness, freezing, appetite, mating drive and chase persistence vary between individuals and are heritable. | [Réale et al. 2007](https://doi.org/10.1111/j.1469-185X.2007.00010.x); [Dochtermann et al. 2015](https://doi.org/10.1098/rspb.2014.2201) |

**Time-scale note:** the year is compressed to 12 days, so life history runs
about 30× faster than the day/night cycle. Rabbit kits are weaned within
half a day, so a once-a-day nursing visit would come too late. Nursing
visits are scaled to that compressed life history instead: every 30 ticks
for rabbits and 20 for foxes, with kits fed straight after birth.

The energy cost of growth is scaled the same way. Young grow about 30×
faster than in life, but food is still found at real daily rates, so
charging growth on the compressed calendar would make a vixen with three
cubs need almost 10× her usual food (2.6 rabbits a day). No fox could hunt
that much, and cubs starved among plenty of rabbits. Instead, young keep
their condition as they grow and pay 1/30 of the cost of new tissue and the
bigger energy store, so a nursing vixen needs about 2.4× her usual food. Real
lactating mammals need about 2–4× at peak (Oftedal 1984; Thompson 1992).

## Ecology and behaviour (`engine/animal.py`, `engine/world.py`)

- **Movement:** a stride longer than one cell is checked along its whole
  length, so fast animals can't hop over a narrow stream or wall of rock.
- **Movement models:**
  - Correlated random walks with von Mises turning angles (Kareiva & Shigesada 1983).
  - Area-restricted search after feeding (Kareiva & Odell 1987).
  - Biased random walks towards goals (Codling et al. 2008).
  - Home ranges with natal dispersal.
- **Energetics:**
  - Kleiber metabolism (∝ M^0.75).
  - Cost of transport ∝ M^0.68.
  - Quarter-power life history.
  - Seasonal thermoregulation, lactation, and condition-dependent litter size.
- **Plants:** logistic growth with a seasonal growth rate, seed rain onto bare
  ground, and fertility set by soil moisture and distance to water.
- **Feeding:** a Holling type II functional response, rate-maximising patch
  choice (Charnov 1976) and giving-up densities.
- **Predation:**
  - Probabilistic detection that depends on distance, light, vigilance and
    coat contrast (Lima & Dill 1990).
  - Alarm spreading between neighbours (Pulliam 1973).
  - Stalk-and-rush hunting limited by stamina.
  - Burrows as a prey refuge.
  - Scavenging of animals that died of other causes.
- **Daily and yearly rhythms:** rabbits rest underground in the middle of the
  day, foxes rest at their dens, and both breed in season.
- **Fox territories:** adults need a territory to breed. A territory belongs
  to a pair: it has one centre, held by at most one male and one female, and
  new territories keep their distance from all others. When a fox loses its
  mate, the territory becomes a vacancy for the missing sex. Lone neighbours
  of opposite sexes pair up (the male moves in with the vixen).
  - **Finding a place:** a disperser first looks for a lone mate's territory
    within about twice its dispersal distance, otherwise for space to found
    one.
  - **Floaters** keep prospecting, after a first failure across the whole
    area, and may challenge residents.
  - **Emigration:** they leave only if no territory anywhere is missing their
    sex and there is no room for a new one (Brown 1969). Territories
    therefore cap predator density.
  - **Size follows food** (the resource dispersion hypothesis, Macdonald
    1983): rabbits plus voles. New territories are packed up to 0.6× closer
    where food is plentiful and spread up to 1.6× wider where it is scarce.
    The food level for a normal-sized territory is set so that rabbits
    outnumber foxes several times over. With denser foxes, foxes living on
    voles ate rabbits down to a handful in long runs (hyperpredation).
- **Voles, the foxes' staple prey:** small rodents are far too many to
  simulate one by one, so each cell holds a vole density instead.
  - **Population:** it grows towards a capacity set mostly by the land and
    partly by the current grass, breeds from spring to mid-autumn, declines
    over winter and spreads between cells.
  - **Mousing:** when a fox sees no larger prey it walks slowly, listening,
    and pounces. Its catch rate rises with vole density (a Holling type II
    response), and each catch removes a vole worth about 1/50 of a rabbit.
  - **Why it matters:** voles let foxes ride out rabbit crashes instead of
    starving. Without them, well-paired foxes overshot their rabbit prey and
    died out in testing.
- **Fox families:** young foxes stay with their parents through the summer
  and leave in the autumn dispersal season, before the mating season. Some
  daughters (up to two per family, more likely where food is plentiful) stay
  on as non-breeding helpers (Macdonald 1979). They do not breed while their
  mother holds the territory, they carry food to her cubs, and the oldest
  inherits the territory if she dies. How much helpers really help is debated
  in the field literature. Infanticide by strangers is reported only
  anecdotally in red foxes, so it is not modelled.
- **Mortality:** Gompertz–Makeham senescence, starvation, dehydration,
  predation, disease and (rarely) territorial fights. Foxes have extra
  background mortality, about 20% a year, standing in for sarcoptic mange and
  accidents. Wild red foxes lose 30–60% of adults a year (Devenish-Nelson et
  al. 2013; Soulsbury et al. 2007). Occasional immigrants bring in gene flow.

## What to expect

Rabbits follow an annual cycle: few in spring, peaking in autumn, with a
winter die-off. Foxes peak after their spring litters and track rabbit
numbers, capped by territories. Results differ between seeds, and that
variation is real model behaviour. All model *forms* and scaling *exponents*
come from the literature above. The *constants* in `engine/config.py` are
tuned for a watchable timescale, not calibrated to field data.

## Layout

```
engine/
  config.py      species, genes and world parameters
  genetics.py    genome layout, meiosis, mutation, expression
  coats.py       coat-colour genetics and presets
  terrain.py     biomes, heights, editing, water flow field
  climate.py     climates: local temperature, dryness, snow, biomes, plant productivity
  vegetation.py  seasonal logistic plant growth; wild fruit
  rodents.py     voles as a density field, the foxes' staple prey
  movement.py    correlated random walks and obstacle avoidance
  navigation.py  A* route planning around water, trees and rock
  brain.py       senses → neural network → choice; instinct utilities
  brain_weights.json  founders' trained brains (from tools/train_brains.py)
  animal.py      physiology, sensing, memory, decisions and actions
  world.py       clock, weather, sandbox tools, statistics
  narrator.py    evidence-based explanations of what happened and why
  ledger.py      the run's daily diary: births, deaths by cause, last deaths, extinctions, selection
  report.py      the AI analyst's brief (tables + map image) and the Deaths & births series
ai.py            the AI analyst: .env settings, prompts, streaming Claude API client
.env.example     template for .env (your API key; .env itself is git-ignored)
app.py           Flask API
headless.py      command-line runs and CSV export
tools/train_brains.py  behaviour cloning of the instincts into the founders' brains
static/js/       Three.js scene, voxel animals, UI and charts
tests/           pytest suite
```
