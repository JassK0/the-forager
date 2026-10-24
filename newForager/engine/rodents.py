"""Small rodents (field voles and mice): the staple prey of red foxes in much of
their range, and the reason foxes do not simply die out when rabbits are scarce.

They are far too numerous to simulate one by one, so each cell holds a vole
density instead. It grows logistically towards a carrying capacity set mostly
by how good the land is for grass and partly by the grass there now (heavy
grazing by rabbits leaves somewhat fewer). Voles breed from spring to
mid-autumn, decline over winter and spread slowly between neighbouring cells. Foxes catch them by "mousing":
listening as they walk slowly through the grass, then pouncing. Each catch
removes a vole from that cell.
"""

import numpy as np

VOLE_ENERGY = 6.0  # a vole (about 30 g) against about 320 for a rabbit carcass
VOLE_CAP = 4.0  # voles per cell on full-grown grass
VOLE_GROWTH = 0.003  # per tick at the height of the breeding season
VOLE_WINTER = 0.9996  # survival per tick outside the breeding season
VOLE_SPREAD = 0.02  # share of the difference with neighbouring cells mixed per tick
BREEDING = (0.02, 0.7)  # spring to mid-autumn, as fractions of the year (0 = start of spring)
HABITAT = 0.6  # share of vole capacity set by the land rather than by today's grass
VOLE_HALF = 1.0  # density at which a mousing fox catches at half its best rate
MOUSE_SUCCESS = 0.25  # best chance per tick of mousing to catch one


class Voles:
    def __init__(self, veg):
        self.V = 0.5 * VOLE_CAP * veg.B / veg.capacity
        self._nb = np.empty_like(self.V)

    @staticmethod
    def breeding(year_fraction: float, seasons: bool) -> bool:
        return not seasons or BREEDING[0] <= year_fraction < BREEDING[1]

    def step(self, veg, breeding: bool) -> None:
        V, nb = self.V, self._nb
        # Voles live in rough grass and field margins that rabbits never graze bare:
        # most of their capacity follows the land's potential, the rest the grass now.
        cap = VOLE_CAP * (HABITAT * veg.K + (1.0 - HABITAT) * veg.B) / veg.capacity
        if breeding:
            with np.errstate(divide="ignore", invalid="ignore"):
                V += VOLE_GROWTH * V * np.where(cap > 0, 1.0 - V / cap, -1.0)
        else:
            V *= VOLE_WINTER
        # Dispersal: a little mixing with the four neighbours.
        nb.fill(0.0)
        nb[1:] += V[:-1]
        nb[:-1] += V[1:]
        nb[:, 1:] += V[:, :-1]
        nb[:, :-1] += V[:, 1:]
        V += VOLE_SPREAD * (0.25 * nb - V)
        np.minimum(V, np.maximum(cap, 0.0) * 1.2, out=V)  # grazed-out ground holds few voles
        np.maximum(V, 0.0, out=V)

    def total(self) -> float:
        return float(self.V.sum())

    def near(self, x: float, y: float, r: float) -> float:
        """Voles within about r of (x, y) (a square window scaled to the circle)."""
        h, w = self.V.shape
        x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
        y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
        return float(self.V[y0:y1, x0:x1].sum()) * (np.pi / 4)
