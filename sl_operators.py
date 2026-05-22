"""
Subjective Logic operators for binomial opinions.

An opinion is a 4-tuple (b, d, u, a) where:
  b  : belief mass        in [0, 1]
  d  : disbelief mass     in [0, 1]
  u  : uncertainty mass   in [0, 1]   (b + d + u = 1)
  a  : base rate          in [0, 1]   (prior probability of x)

Projected probability:  P(x) = b + a * u
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import random


@dataclass(frozen=True)
class Opinion:
    b: float
    d: float
    u: float
    a: float

    def __post_init__(self):
        # tolerate small floating-point drift but flag big inconsistencies
        s = self.b + self.d + self.u
        if not (0.999 <= s <= 1.001):
            raise ValueError(f"Opinion mass does not sum to 1: b+d+u={s}")
        for x in (self.b, self.d, self.u, self.a):
            if not (-1e-9 <= x <= 1 + 1e-9):
                raise ValueError(f"Opinion component out of [0,1]: {self}")

    @property
    def P(self) -> float:
        """Projected probability."""
        return self.b + self.a * self.u

    def as_tuple(self):
        return (self.b, self.d, self.u, self.a)


def random_opinion(rng: random.Random | None = None,
                   base_rate: float = 0.5) -> Opinion:
    """Generate a random valid binomial opinion."""
    if rng is None:
        rng = random
    # sample three non-negative numbers and normalize
    x = [rng.random() + 1e-3 for _ in range(3)]
    s = sum(x)
    b, d, u = x[0] / s, x[1] / s, x[2] / s
    return Opinion(b, d, u, base_rate)


# ------------------------------------------------------------------
# Trust discounting operators
# ------------------------------------------------------------------

def discount_TE(omega_referral: Opinion, omega_functional: Opinion) -> Opinion:
    """
    Two-Edge Path trust discounting.
    omega_referral : referral trust  A -> B   (omega^A_B)
    omega_functional : functional trust  B -> X   (omega^B_X)
    Returns omega^[A;B]_X.
    """
    P = omega_referral.P                         # projected probability of referral
    b = P * omega_functional.b
    d = P * omega_functional.d
    u = 1.0 - (b + d)
    a = omega_functional.a
    return Opinion(b, d, u, a)


def discount_RE(omega_AB: Opinion, omega_BC: Opinion) -> Opinion:
    """
    Referral-Edge Path trust discounting (the operator introduced
    in Section 4.6 of the chapter).  Both inputs are referral opinions.
    """
    bA, dA, uA, aA = omega_AB.as_tuple()
    bB, dB, uB, aB = omega_BC.as_tuple()

    b = bA * bB
    d = bA * dB
    u = 1.0 - (b + d)

    denom = 1.0 - bA * (bB + dB)
    if abs(denom) < 1e-12:
        # degenerate case: keep base-rate of the second opinion
        a = aB
    else:
        a = ((bA + uA * aA) * (bB + uB * aB) - bA * bB) / denom
        # numerical safety
        a = min(max(a, 0.0), 1.0)
    return Opinion(b, d, u, a)


def discount_path(opinions: list[Opinion]) -> Opinion:
    """
    Discount a series of opinions along a path.
    The last opinion is treated as the functional trust; all earlier ones
    are referral trust.  We use RE for the referral chain and TE for the
    final step (consistent with the chapter's framework).

    For a path with only two opinions this reduces to TE.
    """
    if len(opinions) == 0:
        raise ValueError("Need at least one opinion")
    if len(opinions) == 1:
        return opinions[0]
    if len(opinions) == 2:
        return discount_TE(opinions[0], opinions[1])
    # combine all referral opinions with RE, then apply TE with the functional
    referral = opinions[0]
    for op in opinions[1:-1]:
        referral = discount_RE(referral, op)
    return discount_TE(referral, opinions[-1])


# ------------------------------------------------------------------
# Belief fusion operators
# ------------------------------------------------------------------

def fuse_cumulative(op1: Opinion, op2: Opinion) -> Opinion:
    """Cumulative belief fusion of two opinions over the same variable."""
    b1, d1, u1, a1 = op1.as_tuple()
    b2, d2, u2, a2 = op2.as_tuple()

    if u1 < 1e-12 and u2 < 1e-12:
        # both dogmatic - take simple weighted limit (use 0.5/0.5)
        b = 0.5 * b1 + 0.5 * b2
        d = 0.5 * d1 + 0.5 * d2
        u = 0.0
        a = 0.5 * a1 + 0.5 * a2
        return Opinion(b, d, u, a)

    denom = u1 + u2 - u1 * u2
    if denom < 1e-12:
        # fall back to averaging
        return fuse_average(op1, op2)
    b = (b1 * u2 + b2 * u1) / denom
    d = (d1 * u2 + d2 * u1) / denom
    u = (u1 * u2) / denom

    if abs(u1 - 1) < 1e-12 and abs(u2 - 1) < 1e-12:
        a = 0.5 * (a1 + a2)
    else:
        denom_a = u1 + u2 - 2 * u1 * u2
        if denom_a < 1e-12:
            a = 0.5 * (a1 + a2)
        else:
            a = (a1 * u2 + a2 * u1 - (a1 + a2) * u1 * u2) / denom_a
            a = min(max(a, 0.0), 1.0)
    # numerical clean-up
    s = b + d + u
    return Opinion(b / s, d / s, u / s, a)


def fuse_average(op1: Opinion, op2: Opinion) -> Opinion:
    """Averaging belief fusion."""
    b1, d1, u1, a1 = op1.as_tuple()
    b2, d2, u2, a2 = op2.as_tuple()

    if u1 < 1e-12 and u2 < 1e-12:
        b = 0.5 * (b1 + b2)
        d = 0.5 * (d1 + d2)
        u = 0.0
        a = 0.5 * (a1 + a2)
        return Opinion(b, d, u, a)

    denom = u1 + u2
    if denom < 1e-12:
        return op1
    b = (b1 * u2 + b2 * u1) / denom
    d = (d1 * u2 + d2 * u1) / denom
    u = (2 * u1 * u2) / denom
    a = 0.5 * (a1 + a2)
    s = b + d + u
    return Opinion(b / s, d / s, u / s, a)


def fuse_many(opinions: Iterable[Opinion], operator: str = "cumulative") -> Opinion:
    """Fuse multiple opinions sequentially with the given fusion operator."""
    fuse = fuse_cumulative if operator == "cumulative" else fuse_average
    it = iter(opinions)
    try:
        result = next(it)
    except StopIteration:
        raise ValueError("fuse_many: no opinions to fuse")
    for op in it:
        result = fuse(result, op)
    return result
