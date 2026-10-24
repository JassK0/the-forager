"""Coat colour genetics: genotype to visible phenotype.

Rabbit loci follow the classical rabbit colour genes (Castle 1930; Searle
1968; Fontanesi 2021): Agouti, Brown, the albino series at C (full colour >
chinchilla > Himalayan > albino), Dilute, and English spotting, where En/En
"charlie" rabbits carry a health cost. Fox loci follow red fox colour phases:
silver (red, cross and silver phases), pearl dilution, and the platinum
white-marking allele, which is lethal when homozygous.

A coat's mean luminance sets how well it matches the background, so coat
colour is under natural selection: dark coats stand out on snow, white coats
on summer grass.

How the animals see colour. Foxes and rabbits are dichromats: like a person
with red-green colour blindness, they have only two kinds of cone, and tell
colours apart by lightness and along a yellow-blue axis (Jacobs 1993; Jacobs
et al. 1993 for the fox). So to them an orange fox and green grass differ
little, which is why a red fox is camouflaged from its prey. `luminance` is
the lightness (sRGB luma, roughly perceptual lightness) and `yellow_blue` the
yellow-blue opponent signal.
"""

from dataclasses import dataclass, field

PARTS = ("body", "head", "ears", "tail", "tailtip", "chest", "muzzle", "legs", "stripe", "eyes", "nose")


@dataclass(frozen=True)
class Coat:
    name: str
    colors: dict = field(hash=False)
    lum: float = 0.5  # mean luminance of the visible coat (0 black .. 1 white)
    vigor: float = 1.0  # fitness multiplier for genotypes with a health cost
    stripe: bool = False  # draw the dorsal stripe / shoulder cross


def _rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _hex(r: float, g: float, b: float) -> str:
    return "#%02x%02x%02x" % (round(r), round(g), round(b))


def mix(a: str, b: str, t: float) -> str:
    (r1, g1, b1), (r2, g2, b2) = _rgb(a), _rgb(b)
    return _hex(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def luminance(c: str) -> float:
    r, g, b = _rgb(c)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


def yellow_blue(c: str) -> float:
    """How yellow (+) or blue (-) a colour looks to a dichromat: long/middle-wave
    cones against short-wave ones."""
    r, g, b = _rgb(c)
    return ((r + g) / 2 - b) / 255


def tone(c: str) -> tuple[float, float]:
    """A colour as a dichromat sees it: (lightness, yellow-blue)."""
    return luminance(c), yellow_blue(c)


def _dominant(alleles: tuple[str, ...], pair: tuple[str, str]) -> str:
    return min(pair, key=alleles.index)


WHITE = "#f4f1ea"
PINK_EYE = "#d8546f"
DARK_EYE = "#1d140e"
NOSE = "#e39aa8"

_AGOUTI = {
    ("B", False): ("#8b6b4c", "Chestnut agouti"),
    ("b", False): ("#a0714b", "Chocolate agouti"),
    ("B", True): ("#9b948e", "Opal"),
    ("b", True): ("#c4aa96", "Lynx"),
}
_CHINCHILLA = {
    ("B", False): ("#8e8b89", "Chinchilla"),
    ("b", False): ("#8f7b6f", "Chocolate chinchilla"),
    ("B", True): ("#a8abb1", "Squirrel"),
    ("b", True): ("#baaeae", "Lilac chinchilla"),
}
_SELF = {
    ("B", False): ("#2b2727", "Black"),
    ("b", False): ("#5c3c2a", "Chocolate"),
    ("B", True): ("#6d7581", "Blue"),
    ("b", True): ("#aa9ba1", "Lilac"),
}


def rabbit_coat(g: dict[str, tuple[str, str]]) -> Coat:
    agouti = _dominant(("A", "a"), g["A"]) == "A"
    brown = _dominant(("B", "b"), g["B"])
    c = _dominant(("C", "cchd", "ch", "c"), g["C"])
    dilute = _dominant(("D", "d"), g["D"]) == "d"
    spots = g["En"].count("En")

    if c == "c":
        # Albino is epistatic: no pigment at all, whatever the other genes say.
        colors = dict(body=WHITE, head=WHITE, ears=WHITE, tail=WHITE, stripe=WHITE, eyes=PINK_EYE, nose=NOSE)
        return Coat("Albino", colors, luminance(WHITE))

    self_color, self_name = _SELF[(brown, dilute)]
    if c == "ch":
        # Himalayan: temperature-sensitive pigment only develops at the cold extremities.
        colors = dict(body=WHITE, head=WHITE, ears=self_color, tail=WHITE, stripe=WHITE,
                      eyes=PINK_EYE, nose=self_color)
        return Coat(f"Himalayan ({self_name.lower()})", colors,
                    0.85 * luminance(WHITE) + 0.15 * luminance(self_color))

    if agouti:
        color, name = (_CHINCHILLA if c == "cchd" else _AGOUTI)[(brown, dilute)]
        tail = "#efe9df"  # agouti rabbits have the white cottontail underside
    else:
        color, name = self_color, self_name
        tail = color
    colors = dict(body=color, head=color, ears=mix(color, "#000000", 0.18), tail=tail, stripe=color,
                  eyes=DARK_EYE, nose=NOSE)
    if spots == 1:
        colors.update(body=WHITE, head=WHITE)
        return Coat(f"Broken {name.lower()}", colors, 0.8 * luminance(WHITE) + 0.2 * luminance(color), stripe=True)
    if spots == 2:
        colors.update(body=WHITE, head=WHITE, tail=WHITE)
        return Coat(f"Charlie {name.lower()}", colors, 0.93 * luminance(WHITE) + 0.07 * luminance(color), vigor=0.85)
    return Coat(name, colors, luminance(color))


_FOX_BASE = ("#d9622b", "#b4542a", "#2f2e34")
_FOX_PEARL = ("#e5a874", "#c98d65", "#8f94a1")
_FOX_NAMES = ("Red", "Cross", "Silver")
_FOX_PEARL_NAMES = ("Pastel red", "Pastel cross", "Pearl")


def fox_coat(g: dict[str, tuple[str, str]]) -> Coat:
    silver = g["S"].count("s")
    pearl = _dominant(("P", "p"), g["P"]) == "p"
    platinum = g["W"].count("Wp") >= 1
    base = (_FOX_PEARL if pearl else _FOX_BASE)[silver]
    name = (_FOX_PEARL_NAMES if pearl else _FOX_NAMES)[silver]
    legs = "#5d5864" if pearl else "#262022"
    pale = mix(base, WHITE, 0.3)
    colors = dict(
        body=base, head=base, ears=legs, tail=base, tailtip=WHITE,
        chest=WHITE if silver < 2 else pale, muzzle=WHITE if silver < 2 else pale,
        legs=legs, stripe=legs, eyes="#e2b13a", nose="#151212",
    )
    lum = luminance(base)
    if platinum:
        colors.update(body=mix(base, WHITE, 0.62), head=WHITE, chest=WHITE, muzzle=WHITE, tail=mix(base, WHITE, 0.5))
        name = f"Platinum {name.lower()}"
        lum = 0.6 * luminance(colors["body"]) + 0.4 * luminance(WHITE)
    return Coat(name, colors, lum, stripe=silver == 1)


COAT_MODELS = {"rabbit": rabbit_coat, "fox": fox_coat}

# Seasonal moult to a white winter coat (the moult itself is timed by day length
# in world.py). Dominant in hares, recessive in Arctic foxes.
WINTER_WHITE = "#f1f3f6"
WINTER_LUM = luminance(WINTER_WHITE)
WINTER_YB = yellow_blue(WINTER_WHITE)


def coat_tone(coat: Coat) -> tuple[float, float]:
    """The coat's lightness (weighted over its parts) and the yellow-blue of its body."""
    return coat.lum, yellow_blue(coat.colors["body"])
MOULTS = {
    "rabbit": lambda g: "M" in g.get("Mo", ()),
    "fox": lambda g: g.get("Mo", ("Mo", "Mo")).count("mo") == 2,
}

PRESETS = {
    "rabbit": {
        "Wild type": {},
        "Black": {"A": ("a", "a")},
        "Chocolate": {"A": ("a", "a"), "B": ("b", "b")},
        "Blue": {"A": ("a", "a"), "D": ("d", "d")},
        "Lilac": {"A": ("a", "a"), "B": ("b", "b"), "D": ("d", "d")},
        "Chinchilla": {"C": ("cchd", "cchd")},
        "Himalayan": {"C": ("ch", "ch")},
        "Albino": {"C": ("c", "c")},
        "Broken": {"En": ("En", "en")},
        "Opal": {"D": ("d", "d")},
        "Winter white": {"Mo": ("M", "M")},
    },
    "fox": {
        "Red": {},
        "Cross": {"S": ("S", "s")},
        "Silver": {"S": ("s", "s")},
        "Pearl": {"S": ("s", "s"), "P": ("p", "p")},
        "Platinum": {"W": ("Wp", "w")},
    },
}
