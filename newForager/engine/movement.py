"""Animal movement: biased correlated random walks with obstacle avoidance.

Each tick an animal turns by an angle drawn from a von Mises distribution
centred on a desired heading, then steps forward. Choosing the desired heading
and the concentration kappa gives the standard family of movement models:

* desired = current heading, large kappa: a correlated random walk with long,
  fairly straight runs, used for ranging (Kareiva & Shigesada 1983).
* desired = current heading, small kappa: tortuous area-restricted search that
  keeps an animal near a recently found resource (Kareiva & Odell 1987;
  Benhamou 1992).
* desired = bearing to a goal: a biased correlated random walk, i.e. goal
  directed movement with realistic noise (Codling, Plank & Benhamou 2008).

The persistence length of a correlated random walk is step / (1 - r) where
r = I1(kappa) / I0(kappa) is the mean cosine of the turning angle; for large
kappa that is about 2 * kappa * step. Steps here are a fraction of a cell, so
kappa has to be large (tens) for paths to look like real animal tracks rather
than jitter. That mismatch is the usual reason hand-rolled random walks look
wrong.
"""

import math

TAU = 2 * math.pi
_SWEEP = tuple(i * math.pi / 8 for i in range(1, 9))  # 22.5 deg steps up to 180 deg


def blend(a: float, b: float, t: float) -> float:
    """Circular interpolation from heading ``a`` towards ``b`` by fraction ``t``."""
    return math.atan2(
        (1 - t) * math.sin(a) + t * math.sin(b),
        (1 - t) * math.cos(a) + t * math.cos(b),
    )


def step(animal, world, desired: float, distance: float, kappa: float) -> float:
    """Turn towards ``desired`` with von Mises noise and move ``distance``.

    If the step would enter water or leave the map, headings increasingly far
    from the intended one are tried, starting on the side the animal last slid
    towards so that it follows a shoreline instead of dithering in a corner.
    Returns the distance actually moved.
    """
    if distance <= 0.0:
        return 0.0
    if kappa < math.inf:
        heading = world.rng.vonmisesvariate(desired % TAU, kappa)
    else:
        heading = desired
    x, y = animal.x, animal.y
    walkable = world.walkable

    nx, ny = x + math.cos(heading) * distance, y + math.sin(heading) * distance
    if walkable(nx, ny):
        animal.x, animal.y, animal.heading = nx, ny, math.remainder(heading, TAU)
        return distance

    side = animal.avoid_side
    for offset in _SWEEP:
        for s in (side, -side):
            a = heading + s * offset
            nx, ny = x + math.cos(a) * distance, y + math.sin(a) * distance
            if walkable(nx, ny):
                animal.avoid_side = s
                animal.x, animal.y, animal.heading = nx, ny, math.remainder(a, TAU)
                return distance

    animal.heading = math.remainder(heading + math.pi, TAU)
    return 0.0
