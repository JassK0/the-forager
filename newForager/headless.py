"""Run the simulation without the web UI and write the time series to CSV.

    python headless.py --years 5 --seed 3 --out run.csv
"""

import argparse
import csv
import time

from engine import World, build_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None, help="CSV file for population, trait and allele time series")
    parser.add_argument("--report", type=int, default=3, help="status lines per simulated year")
    parser.add_argument("--brains", choices=("neural", "rules"), default="neural",
                        help="evolving neural networks, or fixed instincts")
    args = parser.parse_args()

    world = World(build_config({"brains": args.brains}), seed=args.seed)
    cfg = world.cfg
    year = cfg.ticks_per_day * cfg.days_per_year
    every = max(1, year // args.report)
    names = [s.name for s in world.species]
    start = time.perf_counter()
    for _ in range(int(args.years * year)):
        world.step()
        if world.tick % every == 0:
            env = world.env
            elapsed = time.perf_counter() - start
            counts = " ".join(f"{n}={c:<4}" for n, c in zip(names, world.counts()))
            deaths = " | ".join(
                f"{n}: " + ", ".join(f"{k} {v}" for k, v in world.deaths[i].most_common(3))
                for i, n in enumerate(names)
            )
            print(f"y{env['year']} {env['season']:<6} t={world.tick:<6} {counts} plants={world.veg.total():6.0f} "
                  f"{1000 * elapsed / world.tick:.1f}ms/t [{deaths}]", flush=True)
        if not world.animals:
            print(f"Everything died out at tick {world.tick}.")
            break

    if args.out:
        h = world.history
        header = ["tick", "plants"]
        columns = [h["tick"], h["plants"]]
        for name, row in h["species"].items():
            header += [f"{name}_count", f"{name}_juveniles", f"{name}_vigor", f"{name}_load",
                       f"{name}_marker_he", f"{name}_marker_ho"]
            columns += [row["count"], row["juveniles"], row["vigor"], row["load"],
                        row["diversity"]["he"], row["diversity"]["ho"]]
            for trait, series in row["traits"].items():
                for stat in ("mean", "sd", "genetic"):
                    header.append(f"{name}_{trait}_{stat}")
                    columns.append(series[stat])
            spec = next(s for s in world.species if s.name == name)
            for gene in spec.genes:
                freqs = row["genes"][gene.symbol]["freq"]
                for k, allele in enumerate(gene.alleles):
                    header.append(f"{name}_{gene.symbol}_{allele}_freq")
                    columns.append([None if f is None else f[k] for f in freqs])
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zip(*columns))
        print(f"Wrote {len(h['tick'])} samples to {args.out}")


if __name__ == "__main__":
    main()
