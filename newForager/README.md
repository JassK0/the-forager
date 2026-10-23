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

**The Data drawer** has charts for:
- populations and the predator–prey phase plane;
- trait means with standard deviation, against the mean breeding value;
- allele frequencies, with observed vs Hardy–Weinberg heterozygosity;
- neutral genetic diversity and genetic load.

**CSV** exports everything.

**💡 Why?** opens the narrator, which explains what is happening and why. When
something notable happens (a population crash or boom, an allele spreading or
vanishing, a trait evolving, genetic diversity draining away, an event ending,
a new season) a pop-up explains it. **Explain right now** writes a fresh report
on the last day.

The explanations come from what the simulation actually recorded, not from an
AI model:
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
  (below) are built the same way from four loci each. The tests recover h² from
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
- **Camouflage.** A coat's contrast with the ground changes how readily it is
  spotted, so coat colour is under real natural selection. Dark rabbits stand
  out on winter snow, and white ones on summer grass.
- **Genetic load.** Mostly recessive deleterious mutations arise and are
  purged by selection. Inbreeding exposes them as reduced vigour, causing
  stillbirths and costlier metabolism.
- **Neutral markers.** Microsatellite-like loci track drift and diversity.
- **Brain loci.** One locus per connection of the neural network (200 in
  all), spread over the chromosomes. Each connection's strength is the sum of
  its two alleles, so brains recombine and mutate like any polygenic trait.
- **Mutation rates** are far above natural ones so that evolution is visible
  within a session. You can scale them in New world.

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
  the drive is D = (hunger⁴ + thirst⁴)^⅓, and reward is how much an action
  reduces it. An urgent need therefore dominates: water when parched is far
  more rewarding than a snack when nearly full. Mating feels good. A
  predator's lunge that just missed, seeing a predator close, being chased off
  or losing a fight feel bad. Nursing young is rewarding to a mother.
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
  memory on map" paints its remembered food (green) and danger (red).

Data → 🧠 Brains shows the population picture: agreement with the founders'
instincts (evolution), how often animals decide differently from how they
were born (learning), and inborn versus learned habits over time.

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
| Pair territories, scent marking and defence | Red foxes hold territories as pairs. Dispersers join a lone mate or prospect for a vacancy, and leave only when the area is full. Residents mark the edge of their range and chase off same-sex intruders, and floaters steer away from occupied territories. Contests are ritualised and rarely fatal. A floater that finds no vacancy may challenge a resident. Bigger, better-fed animals usually win, and residents have an owner's advantage (Maynard Smith & Parker 1976). A winning challenger takes the territory, and the loser becomes a floater. About 2% of mating-season fights end in a fatal injury. | [Giuggioli et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3053310), [Folia Zoologica 2015](https://bioone.org/journals/Folia-Zoologica/volume-64/issue-1/fozo.v64.i1.a5.2015/Ontogenic-patterns-of-scent-marking-in-red-foxes-Vulpes-vulpes/10.25225/fozo.v64.i1.a5.2015.full) |
| Stalk, then chase | Foxes creep slowly and low (harder to spot). If the rabbit bolts, they switch to a bounding chase of that rabbit and finish with a pounce. | [Texas Parks & Wildlife](https://tpwd.texas.gov/publications/nonpwdpubs/introducing_mammals/red_foxes/), [Wildlife Online](https://www.wildlifeonline.me.uk/animals/article/red-fox-diet-hunting-strategies-behaviour) |
| Target choice and switching | Foxes pick prey by route length weighted by vulnerability: young ×0.7, alone ×0.85, watching the fox ×1.3. While stalking, they switch if a target is much easier (under 60% of the cost). Once chasing, they stay on the fleeing animal unless another comes within lunging range. | Young prey favoured: [Annear et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10038972/). Edge or isolated prey attacked more: [selfish-herd test](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11285717/). Mid-attack switching when prey are close together: [fringe-lipped bats](https://cupola.gettysburg.edu/biofac/95) |
| Route planning | Animals walk straight to goals when they can. When water, trees or rock block the way, they plan a route (A*). Prey reachable only by a long detour is ignored. | |
| Kin recognition, mate choice | Parents, offspring and siblings don't mate. Females accept males in better condition and of larger size more readily. | Pusey & Wolf 1996; Andersson 1994 |
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

- **Movement:**
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
- **Fox territories:** adults need a territory to breed, and territories
  are held by pairs: a dispersing fox first looks for a territory holder of
  the opposite sex without a mate and joins it, otherwise prospects for a
  vacancy within about twice its dispersal distance. Floaters keep
  prospecting, may challenge residents, and emigrate only when the
  neighbourhood stays full (Brown 1969), so territories cap predator
  density. Territory size follows food (the resource dispersion hypothesis,
  Macdonald 1983): new territories are packed up to 0.6× closer where rabbits
  are plentiful and spread up to 1.6× wider where they are scarce, so fox
  numbers track their prey.
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
  vegetation.py  seasonal logistic plant growth
  movement.py    correlated random walks and obstacle avoidance
  navigation.py  A* route planning around water, trees and rock
  brain.py       senses → neural network → choice; instinct utilities
  brain_weights.json  founders' trained brains (from tools/train_brains.py)
  animal.py      physiology, sensing, memory, decisions and actions
  world.py       clock, weather, sandbox tools, statistics
  narrator.py    evidence-based explanations of what happened and why
app.py           Flask API
headless.py      command-line runs and CSV export
tools/train_brains.py  behaviour cloning of the instincts into the founders' brains
static/js/       Three.js scene, voxel animals, UI and charts
tests/           pytest suite
```
