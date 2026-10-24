"""Genomes: diploid chromosomes, meiosis with crossing over, mutation and expression.

Each species has a fixed genome layout (``GenomeSpec``): a set of autosomes,
each carrying loci at map positions measured in Morgans. There are four kinds
of locus:

* Quantitative trait loci (QTL). Each allele carries an effect on the log scale
  of a polygenic trait such as body mass, and effects add up across loci. A
  developmental-noise term on top sets the heritability, so selection on the
  phenotype only partly reaches the genes (Falconer & Mackay 1996).
* Mendelian genes: discrete alleles with dominance, used for coat colour.
  These include one allele that is lethal when homozygous.
* Deleterious loci: mostly recessive mutations that lower vigour. Mutation
  keeps creating them and selection keeps purging them, so they sit at a
  mutation-selection balance. Their homozygosity rises with inbreeding, which
  produces inbreeding depression (Charlesworth & Willis 2009).
* Neutral markers: microsatellite-like loci with stepwise mutation, used to
  measure genetic diversity and drift.
* Brain loci: one per weight of the animal's neural network (brain.py). The
  weight is the sum of the two alleles, so brains are inherited, recombined
  and mutated like any polygenic trait.

Gametes are made by meiosis. On each chromosome the number of crossovers is
Poisson with mean equal to its map length (Haldane 1919), so nearby loci tend
to be inherited together (linkage). Sex is XY: mothers always pass an X, and
fathers an X or a Y with equal probability.

Mutation rates are far higher than in nature (real rates are about 1e-5 per
gene per generation) so that evolution is visible within minutes.
"""

import bisect
import itertools
import math
import random
import zlib
from dataclasses import dataclass

import numpy as np

from . import brain
from .coats import COAT_MODELS, PRESETS, Coat
from .config import Species

QTL, GENE, LOAD, MARKER, BRAIN = range(5)
KIND_NAMES = ("qtl", "gene", "load", "marker", "brain")

QTL_MUTATION = 0.01  # per locus per gamete
QTL_MUTATION_SD = 0.35  # new effect = old + N(0, this x founding allelic SD)
GENE_MUTATION = 2e-4
LOAD_MUTATION = 5e-4  # functional -> deleterious
LOAD_REVERSION = 5e-5
MARKER_MUTATION = 2e-3
BRAIN_MUTATION = 0.01  # per weight locus per gamete
BRAIN_MUTATION_SD = 0.06  # trained weights average about 0.2 in size
BRAIN_FOUNDER_SD = 0.012  # standing variation among founders' brains (per allele)


class Genome:
    """Two haplotypes (maternal and paternal) plus the sex chromosomes."""

    __slots__ = ("mat", "pat", "male")

    def __init__(self, mat: list, pat: list, male: bool):
        self.mat = mat
        self.pat = pat
        self.male = male

    def copy(self) -> "Genome":
        return Genome(list(self.mat), list(self.pat), self.male)


@dataclass
class Expression:
    traits: dict[str, float]
    breeding: dict[str, float]  # genetic value as a multiplier on the species mean
    genotype: dict[str, tuple[str, str]]
    coat: int  # index into GenomeSpec.coats
    vigor: float
    lethal: bool
    load: int  # number of deleterious alleles carried
    load_hom: int  # loci homozygous for a deleterious allele
    het: int  # heterozygous neutral markers
    brain: np.ndarray  # neural-network weights


def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, rng.random()
    while p > limit:
        k += 1
        p *= rng.random()
    return k


class GenomeSpec:
    """The genome layout of one species and the operations on its genomes."""

    def __init__(self, sp: Species):
        self.sp = sp
        layout_rng = random.Random(zlib.crc32(sp.name.encode()))
        loci = [(QTL, t) for t, spec in sp.traits.items() for _ in range(spec.loci)]
        loci += [(GENE, i) for i in range(len(sp.genes))]
        loci += [(LOAD, k) for k in range(sp.load_loci)]
        loci += [(MARKER, k) for k in range(sp.markers)]
        loci += [(BRAIN, k) for k in range(brain.N_WEIGHTS)]

        chromosomes = [[] for _ in range(sp.chromosomes)]
        for locus in loci:
            chromosomes[layout_rng.randrange(sp.chromosomes)].append(
                (layout_rng.uniform(0.0, sp.chromosome_length), locus))
        self.kinds: list[int] = []
        self.keys: list = []
        self.positions: list[float] = []
        self.ranges: list[tuple[int, int]] = []
        self.chrom_positions: list[list[float]] = []
        for chrom in chromosomes:
            chrom.sort()
            start = len(self.kinds)
            for pos, (kind, key) in chrom:
                self.kinds.append(kind)
                self.keys.append(key)
                self.positions.append(pos)
            self.ranges.append((start, len(self.kinds)))
            self.chrom_positions.append([pos for pos, _ in chrom])
        self.n = len(self.kinds)

        self.qtl = {t: [i for i in range(self.n) if self.kinds[i] == QTL and self.keys[i] == t] for t in sp.traits}
        self.gene_at = {sp.genes[self.keys[i]].symbol: i for i in range(self.n) if self.kinds[i] == GENE}
        self.load_at = [i for i in range(self.n) if self.kinds[i] == LOAD]
        self.marker_at = [i for i in range(self.n) if self.kinds[i] == MARKER]
        # Brain loci in weight order, and the founders' (instinct-trained) weights.
        brain_loci = sorted((self.keys[i], i) for i in range(self.n) if self.kinds[i] == BRAIN)
        self.brain_at = [i for _, i in brain_loci]
        self.brain_init = brain.initial_weights(sp.name)

        # Split each trait's founding variance into additive genetic and
        # environmental parts on the log scale.
        self.allele_sd, self.env_sd, self.log_shift, self.genetic_shift = {}, {}, {}, {}
        for t, spec in sp.traits.items():
            total = math.log(1.0 + spec.cv ** 2)
            vg = spec.heritability * total
            self.allele_sd[t] = math.sqrt(vg / (2 * spec.loci))
            self.env_sd[t] = math.sqrt(total - vg)
            self.log_shift[t] = -0.5 * total  # keeps the founding mean at spec.mean
            self.genetic_shift[t] = -0.5 * vg  # keeps the founding mean breeding value at 1

        self.mutation_rate = []
        for kind in self.kinds:
            self.mutation_rate.append({QTL: QTL_MUTATION, GENE: GENE_MUTATION, LOAD: LOAD_MUTATION,
                                       MARKER: MARKER_MUTATION, BRAIN: BRAIN_MUTATION}[kind])

        # Precompute every coat phenotype so the client gets one shared table.
        model = COAT_MODELS[sp.coat]
        self.coats: list[Coat] = []
        coat_index: dict[str, int] = {}
        self._coat_of: dict[tuple, int] = {}
        pairs = [list(itertools.combinations_with_replacement(range(len(g.alleles)), 2)) for g in sp.genes]
        for combo in itertools.product(*pairs):
            genotype = {g.symbol: (g.alleles[a], g.alleles[b]) for g, (a, b) in zip(sp.genes, combo)}
            coat = model(genotype)
            if coat.name not in coat_index:
                coat_index[coat.name] = len(self.coats)
                self.coats.append(coat)
            self._coat_of[combo] = coat_index[coat.name]

    # ------------------------------------------------------------------ creation

    def founder(self, rng: random.Random, male: bool, preset: str | None = None) -> Genome:
        """A genome drawn from the founding population's allele frequencies."""
        sp = self.sp
        forced = {}
        if preset:
            for symbol, pair in PRESETS[sp.coat].get(preset, {}).items():
                gene = next(g for g in sp.genes if g.symbol == symbol)
                forced[symbol] = [gene.alleles.index(a) for a in pair]
            if preset in PRESETS[sp.coat]:
                # Named colour presets are otherwise wild type (the commonest allele) at every colour gene.
                for g in sp.genes:
                    wild = max(range(len(g.alleles)), key=lambda k: g.freqs[k])
                    forced.setdefault(g.symbol, [wild, wild])
        haps = ([], [])
        for kind, key in zip(self.kinds, self.keys):
            for h in (0, 1):
                if kind == QTL:
                    haps[h].append(rng.gauss(0.0, self.allele_sd[key]))
                elif kind == GENE:
                    g = sp.genes[key]
                    if g.symbol in forced:
                        haps[h].append(forced[g.symbol][h])
                    else:
                        haps[h].append(rng.choices(range(len(g.alleles)), g.freqs)[0])
                elif kind == LOAD:
                    haps[h].append(1 if rng.random() < sp.load_freq else 0)
                elif kind == MARKER:
                    haps[h].append(rng.randrange(sp.marker_alleles))
                else:
                    # Each allele carries half of the instinct-trained weight, plus variation.
                    haps[h].append(0.5 * self.brain_init[key] + rng.gauss(0.0, BRAIN_FOUNDER_SD))
        return Genome(haps[0], haps[1], male)

    def gamete(self, genome: Genome, rng: random.Random, mutation: float) -> list:
        """One recombinant, possibly mutated, haploid set of autosomes."""
        out = []
        for (start, end), positions in zip(self.ranges, self.chrom_positions):
            if rng.random() < 0.5:
                strands = (genome.mat, genome.pat)
            else:
                strands = (genome.pat, genome.mat)
            crossovers = _poisson(rng, self.sp.chromosome_length)
            if crossovers == 0:
                out.extend(strands[0][start:end])
                continue
            cuts = sorted(bisect.bisect_left(positions, rng.uniform(0.0, self.sp.chromosome_length))
                          for _ in range(crossovers))
            strand, at = 0, start
            for cut in cuts:
                out.extend(strands[strand][at:start + cut])
                at = start + cut
                strand ^= 1
            out.extend(strands[strand][at:end])
        if mutation > 0.0:
            self._mutate(out, rng, mutation)
        return out

    def _mutate(self, hap: list, rng: random.Random, scale: float) -> None:
        sp = self.sp
        rates = self.mutation_rate
        for i in range(self.n):
            if rng.random() >= rates[i] * scale:
                continue
            kind = self.kinds[i]
            if kind == QTL:
                hap[i] += rng.gauss(0.0, QTL_MUTATION_SD * self.allele_sd[self.keys[i]])
            elif kind == GENE:
                n_alleles = len(sp.genes[self.keys[i]].alleles)
                hap[i] = (hap[i] + rng.randrange(1, n_alleles)) % n_alleles
            elif kind == LOAD:
                if hap[i] == 0:
                    hap[i] = 1
                elif rng.random() < LOAD_REVERSION / LOAD_MUTATION:
                    hap[i] = 0
            elif kind == MARKER:
                hap[i] = min(sp.marker_alleles - 1, max(0, hap[i] + (1 if rng.random() < 0.5 else -1)))
            else:
                hap[i] += rng.gauss(0.0, BRAIN_MUTATION_SD)

    def child(self, mother: Genome, father: Genome, rng: random.Random, mutation: float) -> Genome:
        return Genome(self.gamete(mother, rng, mutation), self.gamete(father, rng, mutation), rng.random() < 0.5)

    # ------------------------------------------------------------------ expression

    def coat_index(self, genome: Genome) -> int:
        combo = tuple(
            (min(genome.mat[i], genome.pat[i]), max(genome.mat[i], genome.pat[i]))
            for i in (self.gene_at[g.symbol] for g in self.sp.genes)
        )
        return self._coat_of[combo]

    def express(self, genome: Genome, rng: random.Random) -> Expression:
        sp = self.sp
        mat, pat = genome.mat, genome.pat
        traits, breeding = {}, {}
        for t, spec in sp.traits.items():
            g = sum(mat[i] + pat[i] for i in self.qtl[t])
            value = spec.mean * math.exp(g + rng.gauss(0.0, self.env_sd[t]) + self.log_shift[t])
            traits[t] = min(spec.hi, max(spec.lo, value))
            breeding[t] = math.exp(g + self.genetic_shift[t])

        genotype, lethal = {}, False
        for g in sp.genes:
            i = self.gene_at[g.symbol]
            pair = (g.alleles[mat[i]], g.alleles[pat[i]])
            genotype[g.symbol] = pair
            if g.lethal is not None and pair[0] == pair[1] == g.lethal:
                lethal = True

        load = hom = 0
        for i in self.load_at:
            a, b = mat[i], pat[i]
            load += a + b
            if a and b:
                hom += 1
        het_load = load - 2 * hom
        coat = self.coat_index(genome)
        vigor = ((1 - sp.load_s) ** hom) * ((1 - sp.load_h * sp.load_s) ** het_load) * self.coats[coat].vigor
        het = sum(1 for i in self.marker_at if mat[i] != pat[i])
        weights = np.array([mat[i] + pat[i] for i in self.brain_at], dtype=np.float32)
        return Expression(traits, breeding, genotype, coat, vigor, lethal, load, hom, het, weights)

    # ------------------------------------------------------------------ editing and display

    def set_gene(self, genome: Genome, symbol: str, alleles: tuple[str, str]) -> None:
        """Gene editing: overwrite both copies of one Mendelian gene."""
        gene = next(g for g in self.sp.genes if g.symbol == symbol)
        i = self.gene_at[symbol]
        genome.mat[i] = gene.alleles.index(alleles[0])
        genome.pat[i] = gene.alleles.index(alleles[1])

    def describe_genome(self, genome: Genome) -> list[list[dict]]:
        """Per-chromosome locus list for the genome viewer."""
        sp = self.sp
        out = []
        for (start, end) in self.ranges:
            chrom = []
            for i in range(start, end):
                kind, key = self.kinds[i], self.keys[i]
                a, b = genome.mat[i], genome.pat[i]
                if kind == QTL:
                    entry = {"kind": "qtl", "label": key, "a": round(a / self.allele_sd[key], 2),
                             "b": round(b / self.allele_sd[key], 2)}
                elif kind == GENE:
                    g = sp.genes[key]
                    labels = g.labels or g.alleles
                    entry = {"kind": "gene", "label": g.symbol, "a": labels[a], "b": labels[b]}
                elif kind == LOAD:
                    entry = {"kind": "load", "label": f"L{key + 1}", "a": a, "b": b}
                elif kind == MARKER:
                    entry = {"kind": "marker", "label": f"M{key + 1}", "a": a, "b": b}
                else:
                    entry = {"kind": "brain", "label": f"W{key + 1}", "a": round(a, 2), "b": round(b, 2)}
                entry["pos"] = round(self.positions[i], 3)
                chrom.append(entry)
            out.append(chrom)
        return out
