import dataclasses
import functools
from typing import FrozenSet, Optional, List

from parsurf.tools._builder import AtLayer


def checkerboard_basis(c: complex) -> str:
    """Classifies a position as either X or Z in a checkerboard pattern.

    Each "square" in the checkerboard is 4 units wide and tall.
    """
    x = c.real // 4
    y = c.imag // 4
    if (x + y) % 2 == 0:
        return 'X'
    return 'Z'


@dataclasses.dataclass(frozen=True)
class HalfTile:
    """One half of a surface code tile. Two data qubits and their measurement qubit."""

    d0: complex
    d1: complex
    m: complex

    def parity_keys(self, *, layer: int) -> List[AtLayer]:
        return [
            AtLayer(('parity', self.d0, self.m), layer=layer),
            AtLayer(('parity', self.d1, self.m), layer=layer),
        ]


@dataclasses.dataclass(frozen=True)
class Tile:
    """A four body or two body stabilizer from a surface code.

    Divided into two halves. Each half has its own measurement qubit, to help with decomposing into
    two body measurements.
    """

    a: Optional[complex]
    b: Optional[complex]
    c: Optional[complex]
    d: Optional[complex]

    ua: complex
    ub: complex
    uc: complex
    ud: complex

    center: complex
    basis: str

    def um1(self) -> Optional[complex]:
        c = (self.ua + self.uc) / 2
        c = (c + self.center) / 2
        return c

    def um2(self) -> Optional[complex]:
        c = (self.ub + self.ud) / 2
        c = (c + self.center) / 2
        return c

    def m1(self) -> Optional[complex]:
        if self.a is None and self.c is None:
            return None
        return self.um1()

    def m2(self) -> Optional[complex]:
        if self.b is None and self.d is None:
            return None
        return self.um2()

    @functools.cached_property
    def measure_set(self) -> FrozenSet[complex]:
        """All measure qubits used by the tile."""
        return frozenset(t for t in [self.m1(), self.m2()] if t is not None)

    @functools.cached_property
    def used_set(self) -> FrozenSet[complex]:
        """All qubits used by the tile."""
        return self.data_set | self.measure_set

    @functools.cached_property
    def data_set(self) -> FrozenSet[complex]:
        """All data qubits used by the tile."""
        return frozenset(t for t in [self.a, self.b, self.c, self.d] if t is not None)


def surface_code_tiles(*, diam: int, flip_orientation: bool) -> List[Tile]:
    data_qubits = {
        x * 4 + 4j * y
        for x in range(diam)
        for y in range(diam)
    }

    tiles = []
    top_basis = 'Z'
    side_basis = 'X'
    for x in range(-1, diam):
        for y in range(-1, diam):
            tl = x*4 + 4j*y
            basis = checkerboard_basis(tl)

            # Omit tiles on the boundary that don't match the boundary type.
            if x in [-1, diam - 1] and basis != side_basis:
                continue
            if y in [-1, diam - 1] and basis != top_basis:
                continue

            # Pick the orientation that avoids bad hook errors.
            if (basis == 'Z') ^ flip_orientation:
                order = [tl, tl + 4j, tl + 4, tl + 4 + 4j]
            else:
                order = [tl, tl + 4, tl + 4j, tl + 4 + 4j]
            kept = [(d if d in data_qubits else None) for d in order]
            if all(d is None for d in kept):
                continue
            a, b, c, d = kept
            ua, ub, uc, ud = order
            tiles.append(Tile(
                a=a,
                b=b,
                c=c,
                d=d,
                ua=ua,
                ub=ub,
                uc=uc,
                ud=ud,
                basis=basis,
                center=tl + 2 + 2j,
            ))

    return tiles
