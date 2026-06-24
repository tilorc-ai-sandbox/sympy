"""Implementation of :class:`BooleanSemiring` class."""
from __future__ import annotations

from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public


class BooleanElement(DomainElement):
    """An element of the boolean semiring {False, True}.

    Addition is OR, multiplication is AND.
    """

    __slots__ = ('_val',)

    def __init__(self, val):
        self._val = bool(val)

    def parent(self):
        return BOOL

    def __bool__(self):
        return self._val

    def __int__(self):
        return int(self._val)

    def __eq__(self, other):
        if isinstance(other, BooleanElement):
            return self._val == other._val
        return NotImplemented

    def __hash__(self):
        return hash(self._val)

    def __repr__(self):
        return repr(self._val)

    def __str__(self):
        return str(self._val)

    def __add__(self, other):
        if isinstance(other, BooleanElement):
            return BooleanElement(self._val or other._val)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, int):
            return BooleanElement(bool(other) or self._val)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, BooleanElement):
            return BooleanElement(self._val and other._val)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int):
            return BooleanElement(bool(other) and self._val)
        return NotImplemented

    def __pow__(self, n):
        if n == 0:
            return BooleanElement(True)
        return BooleanElement(self._val)

    def __pos__(self):
        return self

    def __abs__(self):
        return self


@public
class BooleanSemiring(Domain):
    """The boolean semiring domain ``BOOL``.

    Represents the set ``{False, True}`` with OR as addition and AND as
    multiplication. This is a semiring — it has no additive inverse, so it
    is not a ring or field.

    The primary use case is boolean matrix operations such as graph
    reachability and transitive closure.

    Examples
    ========

    >>> from sympy.polys.domains import BOOL
    >>> from sympy import S
    >>> T = BOOL(True)
    >>> F = BOOL(False)
    >>> T + F   # OR
    True
    >>> T * F   # AND
    False
    >>> BOOL.from_sympy(S.true)
    True
    >>> BOOL.to_sympy(T)
    True

    For matrix operations:

    >>> from sympy.polys.matrices import DomainMatrix
    >>> F, T = BOOL(False), BOOL(True)
    >>> A = DomainMatrix([[F, T, F], [F, F, T], [F, F, F]], (3, 3), BOOL)
    >>> (A ** 2)[0, 2]  # 2-step reachability from node 0 to node 2
    True

    See Also
    ========

    Domain
    """

    rep = 'BOOL'
    alias = 'BOOL'
    dtype = BooleanElement

    is_BooleanSemiring = True
    is_Exact = True
    is_Numerical = True
    is_Simple = True

    is_Ring = False
    is_Field = False
    has_assoc_Ring = False
    has_assoc_Field = False

    def __init__(self):
        """Allow instantiation of this domain."""

    @property
    def zero(self):
        return BooleanElement(False)

    @property
    def one(self):
        return BooleanElement(True)

    def __eq__(self, other):
        if isinstance(other, BooleanSemiring):
            return True
        return NotImplemented

    def __hash__(self):
        return hash('BOOL')

    def new(self, val):
        return BooleanElement(val)

    def __call__(self, val):
        return BooleanElement(val)

    def of_type(self, element):
        return isinstance(element, BooleanElement)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object."""
        from sympy.logic.boolalg import BooleanTrue, BooleanFalse
        return BooleanTrue() if a._val else BooleanFalse()

    def from_sympy(self, a):
        """Convert a SymPy Boolean to ``BooleanElement``."""
        from sympy.logic.boolalg import BooleanTrue, BooleanFalse
        if isinstance(a, BooleanTrue):
            return BooleanElement(True)
        elif isinstance(a, BooleanFalse):
            return BooleanElement(False)
        raise CoercionFailed("expected S.true or S.false, got %s" % a)

    def from_ZZ(self, a, K0):
        """Convert an integer 0 or 1 to BooleanElement."""
        if a == 0:
            return BooleanElement(False)
        elif a == 1:
            return BooleanElement(True)
        raise CoercionFailed("expected 0 or 1, got %s" % a)

    def from_BooleanSemiring(self, a, K0):
        return a

    def is_zero(self, a):
        return not a._val

    def is_one(self, a):
        return a._val

    def get_ring(self):
        raise DomainError("there is no ring associated with %s" % self)

    def get_field(self):
        raise DomainError("there is no field associated with %s" % self)


from sympy.polys.polyerrors import DomainError  # noqa: E402

BOOL = BooleanSemiring()
