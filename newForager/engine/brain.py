"""Decision-making: an evolvable neural network, and the instincts it starts from.

Every tick each free animal chooses one of eight options (eat, drink, mate,
rest, explore, flee, freeze, hide). Its low-level skills (route planning,
stalking, nursing, burying food) then carry the choice out.

The choice is made by a small neural network (15 inputs + bias -> 8 tanh
hidden units -> one score per option). Its 200 weights are genes: each weight
is the sum of a maternal and a paternal allele, shuffled by meiosis and
occasionally mutated, so behaviour evolves by natural selection like any other
trait (Huse, Strand & Giske 1999; Eliassen et al. 2016).

A random network would starve on day one, so founders' weights are trained
beforehand (tools/train_brains.py) to imitate hand-written instincts: the
utility functions in `rule_scores` below. The same functions can also drive
the animals directly ("rules" mode), and in neural mode they are kept as a
reference, so the simulation can report how far evolved brains have drifted
from their ancestors' instincts.

Personality genes enter as gains on the inputs (hunger x appetite, mating
drive x libido (and how aroused the animal is), predator distance relative to a boldness-scaled flight
distance, freezing tendency), the way hormones modulate drives, so they matter
in both modes.

Lifetime learning. Genes only set the wiring an animal is born with. From
then on each animal learns from its own experience by reinforcement, the way
the vertebrate basal ganglia are thought to (actor-critic learning; Barto
1995; Joel, Niv & Ruppin 2002):

* Reward is homeostatic (Keramati & Gutkin 2014): food when hungry and water
  when thirsty feel good in proportion to how much they reduce the drive.
  In the breeding season, breeding hormones build an urge to mate that counts
  in the same drive, so mating (which also feels good in itself) relieves it.
  Being lunged at by a predator, seeing one close, or losing a fight feels bad.
* A critic (an extra output of the network) learns to predict future
  wellbeing. The difference between what happened and what it predicted, the
  reward prediction error, plays the role of dopamine (Schultz, Dayan &
  Montague 1997).
* The prediction error strengthens or weakens the connections that led to the
  recent choices, using eligibility traces so that a meal still credits the
  decision to go foraging a while before.
* Choices are slightly noisy (softmax), so animals occasionally try something
  other than their habit and can discover that it pays.
* Learned changes slowly fade back towards the inborn wiring unless they keep
  being reinforced. Young animals learn faster, and the learning-rate gene
  scales it. What is learned is not inherited, but the ability to learn is,
  so evolution and learning interact (the Baldwin effect; Hinton & Nowlan 1987).
"""

import json
import zlib
from pathlib import Path

import numpy as np

INPUTS = (
    "hunger", "thirst", "mating drive", "threat", "predator charging", "alarm heard", "fear",
    "danger here", "food here", "distance from home", "daylight", "rest time", "freezing tendency",
    "company", "youth",
)
OPTIONS = ("eat", "drink", "mate", "rest", "explore", "flee", "freeze", "hide")
EAT, DRINK, MATE, REST, EXPLORE, FLEE, FREEZE, HIDE = range(len(OPTIONS))
N_IN = len(INPUTS) + 1  # plus a constant bias input
N_HID = 8
N_OUT = len(OPTIONS)
N_WEIGHTS = N_IN * N_HID + (N_HID + 1) * N_OUT
HYSTERESIS = 0.1  # bonus for the option already being pursued, so animals don't dither

# Lifetime reinforcement learning.
GAMMA = 0.98  # discount per tick: wellbeing about 50 ticks (5 hours) ahead still counts
TRACE = 0.9  # eligibility-trace decay (lambda)
TAU = 0.05  # decision noise (softmax temperature)
ACTOR_RATE = 0.02  # at the founding mean learning gene
CRITIC_RATE = 0.1  # normalised step: the share of each prediction error corrected at once
HIDDEN_RATE = 0.3  # hidden-layer connections learn at this fraction of the output layer's rate
FORGET = 0.0002  # per tick, learned weights relax towards the inborn ones (half-life about 14 days)
MAX_STEP = 0.02  # largest change to one connection in one tick
YOUNG_PLASTICITY = 1.5  # juveniles learn faster
CLEAR = 0.1  # instinct margin above which a decision counts as clear-cut (not a close call)

HUNGRY = 0.3  # instinct: start eating at this (appetite-scaled) hunger
URGENT = 0.55  # ...and this during rest hours
MATE_PRIORITY = 0.45
REST_PULL = 0.3  # how strongly rest hours pull animals home; needs must exceed URGENT to override it

WEIGHTS_FILE = Path(__file__).with_name("brain_weights.json")


def unpack(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A flat weight vector as (input->hidden, hidden+bias->output) matrices."""
    w1 = vec[: N_IN * N_HID].reshape(N_IN, N_HID)
    w2 = vec[N_IN * N_HID:].reshape(N_HID + 1, N_OUT)
    return w1, w2


def forward(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched forward pass. x: (n, N_IN); w1: (n, N_IN, N_HID); w2: (n, N_HID+1, N_OUT)."""
    hidden = np.tanh(np.einsum("ni,nih->nh", x, w1))
    hidden1 = np.concatenate([hidden, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    return np.einsum("nh,nho->no", hidden1, w2), hidden


def rule_scores(x) -> np.ndarray:
    """Hand-written instincts as utilities, one per option (higher = preferred).

    These reproduce the original rule-based behaviour: flee once a predator
    is inside the flight distance, freeze when it is farther off (more so for
    freezers), take cover when an alarm is heard, eat and drink when need
    passes a threshold (higher during rest hours), mate when in condition and
    not too needy, rest at home in rest hours, otherwise explore.

    x: (n, >= 15) inputs, in the order of INPUTS. Returns (n, N_OUT).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    hunger, thirst, mate, threat, charging, alarm, fear, danger = x[:, :8].T
    resting = x[:, 11] > 0.5
    freeze = x[:, 12]
    # During rest hours everything is measured against resting, with a clear margin
    # over wandering so that small genetic variation does not flip it.
    need_t = np.where(resting, URGENT - REST_PULL, HUNGRY)
    out = np.empty((len(x), N_OUT), dtype=np.float32)
    out[:, EAT] = hunger - need_t
    out[:, DRINK] = thirst - need_t
    out[:, MATE] = np.where((mate > 0) & ~resting, MATE_PRIORITY * mate - np.maximum(hunger, thirst) + 0.15, -1.0)
    out[:, REST] = np.where(resting, REST_PULL, -0.3)
    out[:, EXPLORE] = 0.0
    out[:, FLEE] = 2.0 + 4.0 * (threat - 1.0) + 4.0 * charging
    out[:, FREEZE] = 0.5 + freeze * (0.5 + threat)  # equals flee at the flight distance for an average freezer
    out[:, HIDE] = np.where((alarm > 0.5) & (threat <= 0), 0.8, np.where(threat > 0, 0.3 + 0.5 * danger, -1.0))
    return out


def squash(scores: np.ndarray) -> np.ndarray:
    """Order-preserving compression of instinct utilities, used as training targets
    so that very large flee scores don't dominate the fit. Near zero it is the
    identity, so the margins that matter for everyday choices are kept."""
    return 2.0 * np.tanh(scores / 2.0)


def margin(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """How much the best available option beats the runner-up, per row."""
    top2 = np.partition(np.where(mask, scores, -np.inf), -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def choose(scores: np.ndarray, mask: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Masked argmax with a bonus for the current option. All arrays are batched."""
    s = np.where(mask, scores, -np.inf)
    rows = np.arange(len(s))
    has_current = current >= 0
    s[rows[has_current], current[has_current]] += HYSTERESIS
    return np.argmax(s, axis=1)


def policy(scores: np.ndarray, mask: np.ndarray, current: np.ndarray,
           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Noisy choice: sample from a softmax over the available options (with the
    habit bonus for the current one). Returns the choices and the probabilities."""
    s = np.where(mask, scores, -np.inf).astype(np.float64)
    rows = np.arange(len(s))
    has_current = current >= 0
    s[rows[has_current], current[has_current]] += HYSTERESIS
    z = s / TAU
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    choice = np.argmax(z + rng.gumbel(size=z.shape), axis=1)  # Gumbel-max = a softmax sample
    return choice, p


class BrainTable:
    """Struct-of-arrays storage for one species' brains, so a whole species can
    be evaluated and can learn with one batched calculation per step.

    Each row holds an animal's current wiring (w1, w2), the wiring it was born
    with (i1, i2), its critic (v, predicting future wellbeing) and the
    eligibility traces used by learning. In Super animals mode young are born
    with part of their parents' know-how on top of their genes (c1, c2, which
    are included in i1, i2)."""

    ARRAYS = ("w1", "w2", "i1", "i2", "c1", "c2", "v", "e1", "e2", "hprev", "vprev", "primed")

    def __init__(self, capacity: int = 256):
        self.w1 = np.zeros((capacity, N_IN, N_HID), dtype=np.float32)
        self.w2 = np.zeros((capacity, N_HID + 1, N_OUT), dtype=np.float32)
        self.i1 = np.zeros_like(self.w1)
        self.i2 = np.zeros_like(self.w2)
        self.c1 = np.zeros_like(self.w1)
        self.c2 = np.zeros_like(self.w2)
        self.v = np.zeros((capacity, N_HID + 1), dtype=np.float32)
        self.e1 = np.zeros_like(self.w1)
        self.e2 = np.zeros_like(self.w2)
        self.hprev = np.zeros_like(self.v)
        self.vprev = np.zeros(capacity, dtype=np.float32)
        self.primed = np.zeros(capacity, dtype=bool)
        self.free = list(range(capacity - 1, -1, -1))

    def alloc(self, vec: np.ndarray) -> int:
        if not self.free:
            n = len(self.w1)
            for name in self.ARRAYS:
                arr = getattr(self, name)
                setattr(self, name, np.concatenate([arr, np.zeros_like(arr)]))
            self.free = list(range(2 * n - 1, n - 1, -1))
        row = self.free.pop()
        w1, w2 = unpack(vec)
        self.w1[row] = self.i1[row] = w1
        self.w2[row] = self.i2[row] = w2
        self.c1[row] = 0.0
        self.c2[row] = 0.0
        self.v[row] = 0.0
        self.e1[row] = 0.0
        self.e2[row] = 0.0
        self.hprev[row] = 0.0
        self.vprev[row] = 0.0
        self.primed[row] = False
        return row

    def release(self, row: int) -> None:
        self.free.append(row)

    def know_how(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """Everything in an animal's wiring that isn't from its genes: what it was
        taught at birth plus what it has learned since."""
        return (self.w1[row] - self.i1[row] + self.c1[row], self.w2[row] - self.i2[row] + self.c2[row])

    def teach(self, row: int, k1: np.ndarray, k2: np.ndarray) -> None:
        """Start a newborn with know-how on top of its genes."""
        self.c1[row] = k1
        self.c2[row] = k2
        self.i1[row] += k1
        self.i2[row] += k2
        self.w1[row] = self.i1[row]
        self.w2[row] = self.i2[row]

    def value(self, rows, hidden: np.ndarray) -> np.ndarray:
        """The critic's prediction of future wellbeing."""
        h1 = np.concatenate([hidden, np.ones((len(hidden), 1), dtype=hidden.dtype)], axis=1)
        return np.einsum("nh,nh->n", h1, self.v[rows])

    def learn(self, rows, x: np.ndarray, hidden: np.ndarray, value: np.ndarray, reward: np.ndarray,
              plasticity: np.ndarray, choice: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """One actor-critic step for a batch of animals. Returns the prediction errors.

        The error for the step just finished is r + gamma V(now) - V(before). It
        updates the critic and the decision weights along their eligibility
        traces; then the traces take in the choice just made."""
        rows = np.asarray(rows)
        delta = np.clip(reward + GAMMA * value - self.vprev[rows], -1.0, 1.0) * self.primed[rows]
        # Critic: normalised TD(0), which is stable for any step below 2.
        hp = self.hprev[rows]
        self.v[rows] += (CRITIC_RATE * delta / (np.einsum("nh,nh->n", hp, hp) + 1e-6))[:, None] * hp
        a = plasticity * delta
        # (Worked in place to avoid temporary arrays; the arithmetic is unchanged.)
        e1, e2 = self.e1[rows], self.e2[rows]
        step2 = (ACTOR_RATE * a)[:, None, None] * e2
        np.clip(step2, -MAX_STEP, MAX_STEP, out=step2)
        step1 = (ACTOR_RATE * HIDDEN_RATE * a)[:, None, None] * e1
        np.clip(step1, -MAX_STEP, MAX_STEP, out=step1)
        w1, w2 = self.w1[rows], self.w2[rows]
        pull = self.i2[rows]
        pull -= w2
        pull *= FORGET
        pull += step2
        w2 += pull
        pull = self.i1[rows]
        pull -= w1
        pull *= FORGET
        pull += step1
        w1 += pull
        self.w1[rows], self.w2[rows] = w1, w2
        # Traces: decay, then add how the chosen option's probability depends on each weight.
        g = -probs.astype(np.float32)
        g[np.arange(len(g)), choice] += 1.0
        g = np.where(mask, g, 0.0) / TAU
        h1 = np.concatenate([hidden, np.ones((len(hidden), 1), dtype=hidden.dtype)], axis=1)
        decay = GAMMA * TRACE
        e2 *= decay
        e2 += h1[:, :, None] * g[:, None, :]
        self.e2[rows] = e2
        gh = np.einsum("no,nho->nh", g, w2[:, :N_HID, :]) * (1.0 - hidden ** 2)
        e1 *= decay
        e1 += x[:, :, None] * gh[:, None, :]
        self.e1[rows] = e1
        self.hprev[rows] = h1
        self.vprev[rows] = value
        self.primed[rows] = True
        return delta


def attributions(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, option: int) -> np.ndarray:
    """How much each input pushed one option's score up or down right now
    (gradient x input), for explaining a single decision."""
    hidden = np.tanh(x @ w1)
    grad = w1 @ (w2[:N_HID, option] * (1.0 - hidden ** 2))
    return x * grad


def _sweep(w1s: np.ndarray, w2s: np.ndarray, base: np.ndarray, col: int, levels: np.ndarray,
           mask: np.ndarray, target: int) -> np.ndarray:
    """For each network, the first level of input `col` at which it picks `target`."""
    x = np.repeat(base[None, :], len(levels), axis=0)
    x[:, col] = levels
    hidden = np.tanh(np.einsum("li,kih->klh", x, w1s))
    h1 = np.concatenate([hidden, np.ones(hidden.shape[:2] + (1,), dtype=hidden.dtype)], axis=2)
    scores = np.einsum("klh,kho->klo", h1, w2s)
    choice = np.argmax(np.where(mask, scores, -np.inf), axis=2)
    hit = choice == target
    first = np.argmax(hit, axis=1)
    return np.where(hit.any(axis=1), levels[first], np.nan)


def thresholds(w1s: np.ndarray, w2s: np.ndarray, prey: bool) -> dict:
    """Behavioural fingerprints of a batch of networks, probed in a calm daytime
    situation: the hunger at which they start eating, the thirst at which they
    start drinking, and (for prey) how close a predator may come, as a multiple
    of the flight distance, before they flee rather than freeze."""
    base = np.zeros(N_IN, dtype=np.float32)
    base[1], base[8], base[9], base[10], base[12], base[13], base[15] = 0.05, 0.5, 0.3, 1.0, 1.0, 0.25, 1.0
    calm = np.array([True, True, False, True, True, False, False, False])
    out = {
        "eat": _sweep(w1s, w2s, base, 0, np.linspace(0.0, 1.0, 41, dtype=np.float32), calm, EAT),
        "drink": _sweep(w1s, w2s, base, 1, np.linspace(0.0, 1.0, 41, dtype=np.float32), calm, DRINK),
    }
    if prey:
        scared = base.copy()
        scared[6] = 1.0  # fear
        threat = np.linspace(0.2, 3.0, 57, dtype=np.float32)
        at = _sweep(w1s, w2s, scared, 3, threat, np.ones(N_OUT, dtype=bool) & ~np.eye(N_OUT, dtype=bool)[MATE], FLEE)
        out["flee"] = 1.0 / at  # flees when the predator is within this x the flight distance
    return out


_initial: dict | None = None


def initial_weights(species: str) -> np.ndarray:
    """The founders' trained weights, or a weak random network if none are saved."""
    global _initial
    if _initial is None:
        _initial = json.loads(WEIGHTS_FILE.read_text()) if WEIGHTS_FILE.exists() else {}
    if species in _initial and len(_initial[species]["weights"]) == N_WEIGHTS:
        return np.array(_initial[species]["weights"], dtype=np.float64)
    rng = np.random.default_rng(zlib.crc32(species.encode()))
    return rng.normal(0.0, 0.1, N_WEIGHTS)


def training_report() -> dict:
    """How closely the founders' brains match the instincts they were trained on."""
    initial_weights("rabbit")
    return {k: {kk: vv for kk, vv in v.items() if kk != "weights"} for k, v in (_initial or {}).items()}
