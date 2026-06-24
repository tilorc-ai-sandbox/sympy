"""Tests for the BooleanSemiring domain."""
from sympy.polys.domains import BOOL
from sympy.polys.domains.booleansemiring import BooleanElement, BooleanSemiring
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed, DomainError
from sympy import S
import pytest


# --- Domain identity ---

def test_bool_singleton():
    assert isinstance(BOOL, BooleanSemiring)
    assert BOOL == BooleanSemiring()
    assert hash(BOOL) == hash(BooleanSemiring())
    assert str(BOOL) == 'BOOL'
    assert repr(BOOL) == 'BOOL'


def test_bool_flags():
    assert BOOL.is_BooleanSemiring
    assert BOOL.is_Exact
    assert not BOOL.is_Ring
    assert not BOOL.is_Field


def test_bool_zero_one():
    assert BOOL.zero == BooleanElement(False)
    assert BOOL.one == BooleanElement(True)
    assert BOOL.is_zero(BOOL.zero)
    assert BOOL.is_one(BOOL.one)
    assert not BOOL.is_zero(BOOL.one)
    assert not BOOL.is_one(BOOL.zero)


# --- Element construction ---

def test_bool_call():
    T = BOOL(True)
    F = BOOL(False)
    assert isinstance(T, BooleanElement)
    assert isinstance(F, BooleanElement)
    assert bool(T) is True
    assert bool(F) is False


# --- Element arithmetic ---

def test_bool_add_is_or():
    T, F = BOOL(True), BOOL(False)
    assert T + T == T
    assert T + F == T
    assert F + T == T
    assert F + F == F


def test_bool_mul_is_and():
    T, F = BOOL(True), BOOL(False)
    assert T * T == T
    assert T * F == F
    assert F * T == F
    assert F * F == F


def test_bool_pow():
    T, F = BOOL(True), BOOL(False)
    assert T ** 0 == T
    assert T ** 3 == T
    assert F ** 0 == T
    assert F ** 3 == F


def test_bool_semiring_identities():
    T, F = BOOL(True), BOOL(False)
    # zero is additive identity
    assert T + F == T
    assert F + T == T
    # one is multiplicative identity
    assert T * T == T
    assert F * T == F
    # zero annihilates multiplication
    assert T * F == F
    assert F * F == F


def test_bool_element_eq_hash():
    assert BOOL(True) == BOOL(True)
    assert BOOL(False) == BOOL(False)
    assert BOOL(True) != BOOL(False)
    assert hash(BOOL(True)) == hash(BOOL(True))
    assert hash(BOOL(False)) != hash(BOOL(True))


def test_bool_element_str_repr():
    assert str(BOOL(True)) == 'True'
    assert repr(BOOL(True)) == 'True'
    assert str(BOOL(False)) == 'False'


# --- from_sympy / to_sympy ---

def test_from_sympy():
    assert BOOL.from_sympy(S.true) == BOOL(True)
    assert BOOL.from_sympy(S.false) == BOOL(False)


def test_to_sympy():
    assert BOOL.to_sympy(BOOL(True)) == S.true
    assert BOOL.to_sympy(BOOL(False)) == S.false


def test_from_sympy_invalid():
    with pytest.raises(CoercionFailed):
        BOOL.from_sympy(S.One)


def test_from_ZZ():
    from sympy.polys.domains import ZZ
    assert BOOL.from_ZZ(ZZ(0), ZZ) == BOOL(False)
    assert BOOL.from_ZZ(ZZ(1), ZZ) == BOOL(True)
    with pytest.raises(CoercionFailed):
        BOOL.from_ZZ(ZZ(2), ZZ)


# --- get_ring / get_field raise ---

def test_no_ring_or_field():
    with pytest.raises(DomainError):
        BOOL.get_ring()
    with pytest.raises(DomainError):
        BOOL.get_field()


# --- DomainMatrix operations ---

def _make_adj():
    F, T = BOOL(False), BOOL(True)
    # adjacency: 0→1, 1→2
    return DomainMatrix(
        [[F, T, F],
         [F, F, T],
         [F, F, F]], (3, 3), BOOL)


def test_domainmatrix_add():
    F, T = BOOL(False), BOOL(True)
    A = DomainMatrix([[T, F], [F, T]], (2, 2), BOOL)
    B = DomainMatrix([[F, T], [T, F]], (2, 2), BOOL)
    C = A + B
    assert C[0, 0].element == T
    assert C[0, 1].element == T
    assert C[1, 0].element == T
    assert C[1, 1].element == T


def test_domainmatrix_matmul():
    A = _make_adj()
    A2 = A.matmul(A)
    F, T = BOOL(False), BOOL(True)
    # 0 can reach 2 in 2 steps (0→1→2)
    assert A2[0, 2].element == T
    # 0 cannot reach 0 in 2 steps
    assert A2[0, 0].element == F


def test_domainmatrix_pow():
    A = _make_adj()
    A2 = A ** 2
    T, F = BOOL(True), BOOL(False)
    assert A2[0, 2].element == T
    assert A2[1, 2].element == F  # 1→2 in 2 steps? No: 1→2 but 2 has no outgoing edges


def test_domainmatrix_transpose():
    F, T = BOOL(False), BOOL(True)
    A = DomainMatrix([[F, T], [F, F]], (2, 2), BOOL)
    AT = A.transpose()
    assert AT[0, 0].element == F
    assert AT[0, 1].element == F
    assert AT[1, 0].element == T
    assert AT[1, 1].element == F
