"""Train the founders' brains to imitate the hand-written instincts.

    python -m tools.train_brains            # from the newForager folder
    python -m tools.train_brains --days 6 --seeds 4

A randomly wired network would starve on day one, so before any evolution the
founding generation's brains are fitted to the instinct utilities in
engine/brain.py (behaviour cloning). Training data come from two sources:

* situations real animals meet, recorded from simulations driven by the
  instincts ("rules" mode), so the network is accurate where it matters, and
* random situations covering the full range of every input, including rare
  ones (a charging fox, a very bold or very hungry animal), so that mutated
  personalities and unusual states still get sensible choices.

The network is fitted by masked mean-squared error on the (squashed) instinct
utilities with Adam, in plain numpy. Random noise is added to the weights at
every training step, so the fitted network keeps working when founders'
genetic variation and later mutations nudge its weights (robustness to small
perturbations, which biologists call canalisation). The weights and how often
the trained network picks the same option as the instincts are saved to
engine/brain_weights.json, which genetics.py reads to build founders' genomes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import World, brain, build_config  # noqa: E402
from engine.genetics import BRAIN_FOUNDER_SD  # noqa: E402

N_RAW = len(brain.INPUTS)


def record(days: float, seeds: int, every: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Inputs and option masks met by animals in rules-mode simulations."""
    samples: dict[str, list] = {}
    for seed in range(seeds):
        world = World(build_config({"brains": "rules", "end_on_extinction": False}), seed=1000 + seed)
        for _ in range(int(days * world.cfg.ticks_per_day)):
            world.step()
            for a in world.animals:
                if a.inputs is None or a.mother is not None:
                    continue
                # Rare animals (predators) are recorded every tick, common ones less often.
                if world.tick % (1 if a.sp.prey else every) == 0:
                    samples.setdefault(a.sp.name, []).append((a.inputs, a.mask))
        print(f"  seed {seed}: {', '.join(f'{k} {len(v)}' for k, v in samples.items())} samples so far", flush=True)
    return {name: (np.array([x for x, _ in rows], dtype=np.float64), np.array([m for _, m in rows], dtype=bool))
            for name, rows in samples.items()}


def synthetic(n: int, prey: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random situations spanning the range of every input."""
    x = np.zeros((n, N_RAW))
    x[:, 0] = rng.uniform(0.0, 1.5, n)  # hunger x appetite
    x[:, 1] = rng.uniform(0.0, 1.0, n)  # thirst
    ready = rng.random(n) < 0.4
    x[:, 2] = np.where(ready, rng.uniform(0.3, 2.5, n), 0.0)  # mating drive
    x[:, 9] = rng.uniform(0.0, 2.0, n)  # distance from home
    x[:, 10] = rng.uniform(0.0, 1.0, n)  # daylight
    x[:, 11] = rng.random(n) < 0.35  # rest time
    x[:, 13] = rng.uniform(0.0, 1.0, n)  # company
    x[:, 14] = rng.random(n) < 0.2  # youth
    spotted = np.zeros(n, dtype=bool)
    alarm = np.zeros(n, dtype=bool)
    if prey:
        spotted = rng.random(n) < 0.5
        x[:, 3] = np.where(spotted, rng.uniform(0.05, 3.0, n), 0.0)  # threat
        x[:, 4] = spotted & (rng.random(n) < 0.3)  # charging
        alarm = rng.random(n) < 0.3
        x[:, 5] = alarm
        x[:, 6] = np.where(spotted, 1.0, rng.uniform(0.0, 1.0, n))  # fear
        x[:, 7] = rng.uniform(0.0, 1.0, n) ** 2  # danger here
        x[:, 8] = rng.uniform(0.0, 1.0, n)  # food here
        x[:, 12] = rng.uniform(0.2, 3.0, n)  # freezing tendency
    else:
        x[:, 8] = rng.uniform(0.0, 1.0, n) ** 2
    mask = np.ones((n, brain.N_OUT), dtype=bool)
    mask[:, brain.MATE] = ready
    mask[:, brain.FLEE] = spotted
    mask[:, brain.FREEZE] = spotted
    mask[:, brain.HIDE] = spotted | alarm
    return x, mask


def with_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1))], axis=1)


def agreement(w1, w2, x, mask, clear_only: bool = False) -> float:
    if not len(x):
        return float("nan")
    scores, _ = brain.forward(x, np.broadcast_to(w1, (len(x),) + w1.shape), np.broadcast_to(w2, (len(x),) + w2.shape))
    none = np.full(len(x), -1)
    instinct = brain.squash(brain.rule_scores(x))
    same = brain.choose(scores, mask, none) == brain.choose(instinct, mask, none)
    if clear_only:
        same = same[brain.margin(instinct, mask) >= brain.CLEAR]
    return float(np.mean(same))


def founders_agreement(w1, w2, x, mask, rng, copies: int = 20) -> float:
    """Clear-cut agreement of founders, whose weights vary around the trained ones."""
    sd = BRAIN_FOUNDER_SD * np.sqrt(2.0)  # a weight is the sum of two alleles
    return float(np.mean([agreement(w1 + rng.normal(0, sd, w1.shape), w2 + rng.normal(0, sd, w2.shape), x, mask, True)
                          for _ in range(copies)]))


def fit(x: np.ndarray, mask: np.ndarray, epochs: int, rng: np.random.Generator,
        noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Masked MSE regression of the network onto the instinct utilities, with Adam
    and weight noise."""
    target = brain.squash(brain.rule_scores(x)).astype(np.float64)
    w1 = rng.normal(0.0, 1.0 / np.sqrt(brain.N_IN), (brain.N_IN, brain.N_HID))
    w2 = rng.normal(0.0, 1.0 / np.sqrt(brain.N_HID + 1), (brain.N_HID + 1, brain.N_OUT))
    params = [w1, w2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    batch = 512
    step = 0
    n = len(x)
    for epoch in range(epochs):
        lr = 0.01 * 0.5 * (1.0 + np.cos(np.pi * epoch / epochs)) + 2e-4  # cosine decay
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb, tb, mb = x[idx], target[idx], mask[idx]
            # Gradients are taken at randomly perturbed weights and applied to the clean ones.
            n1 = w1 + rng.normal(0.0, noise, w1.shape)
            n2 = w2 + rng.normal(0.0, noise, w2.shape)
            hidden = np.tanh(xb @ n1)
            h1 = np.concatenate([hidden, np.ones((len(idx), 1))], axis=1)
            out = h1 @ n2
            err = (out - tb) * mb
            total += float(np.sum(err ** 2))
            d_out = 2.0 * err / max(1, mb.sum())
            g2 = h1.T @ d_out
            d_hidden = (d_out @ n2[:-1].T) * (1.0 - hidden ** 2)
            g1 = xb.T @ d_hidden
            step += 1
            for p, g, mm, vv in zip(params, (g1, g2), m, v):
                mm *= b1
                mm += (1 - b1) * g
                vv *= b2
                vv += (1 - b2) * g * g
                p -= lr * (mm / (1 - b1 ** step)) / (np.sqrt(vv / (1 - b2 ** step)) + eps)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:4d}  loss {total / max(1, mask.sum()):.5f}", flush=True)
    return w1, w2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=float, default=6.0, help="simulated days per recording run")
    parser.add_argument("--seeds", type=int, default=3, help="recording runs")
    parser.add_argument("--every", type=int, default=3, help="record common species every N ticks")
    parser.add_argument("--synthetic", type=int, default=60000, help="random situations per species")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--noise", type=float, default=0.04, help="weight noise during training (robustness)")
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    start = time.perf_counter()
    print("Recording situations from instinct-driven simulations...")
    recorded = record(args.days, args.seeds, args.every)
    config = build_config({})
    out = {}
    for sp in config.species:
        name = sp.name
        real_x, real_m = recorded.get(name, (np.zeros((0, N_RAW)), np.zeros((0, brain.N_OUT), dtype=bool)))
        hunted = any(name in other.prey for other in config.species)
        syn_x, syn_m = synthetic(args.synthetic, hunted, rng)
        x = with_bias(np.concatenate([real_x, syn_x]))
        mask = np.concatenate([real_m, syn_m])
        print(f"Training the {name} brain on {len(real_x)} recorded + {len(syn_x)} random situations...")
        w1, w2 = fit(x, mask, args.epochs, rng, args.noise)
        threatened = real_m[:, brain.FLEE] if len(real_m) else np.zeros(0, dtype=bool)
        real = with_bias(real_x)
        stats = {
            "agreement": round(agreement(w1, w2, real, real_m), 4),
            "agreement_threatened": round(agreement(w1, w2, real[threatened], real_m[threatened]), 4)
            if threatened.any() else None,
            "agreement_random": round(agreement(w1, w2, with_bias(syn_x), syn_m), 4),
            "agreement_clear": round(agreement(w1, w2, real, real_m, True), 4),
            "founders_clear": round(founders_agreement(w1, w2, real[:20000], real_m[:20000], rng), 4),
            "recorded": int(len(real_x)),
            "random": int(len(syn_x)),
        }
        threat_note = ""
        if stats["agreement_threatened"] is not None:
            threat_note = f"{100 * stats['agreement_threatened']:.1f}% under threat, "
        print(f"  {name}: picks the instinct's choice {100 * stats['agreement']:.1f}% of the time "
              f"({threat_note}{100 * stats['agreement_random']:.1f}% in random situations); "
              f"clear-cut decisions {100 * stats['agreement_clear']:.1f}%, "
              f"founders with genetic variation {100 * stats['founders_clear']:.1f}%")
        stats["weights"] = [round(float(v), 5) for v in np.concatenate([w1.ravel(), w2.ravel()])]
        out[name] = stats
    brain.WEIGHTS_FILE.write_text(json.dumps(out))
    print(f"Saved {brain.WEIGHTS_FILE} in {time.perf_counter() - start:.0f} s.")


if __name__ == "__main__":
    main()
