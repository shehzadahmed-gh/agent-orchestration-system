import pytest

from app.graph.safe_math import UnsafeExpressionError, safe_calculate


def test_basic_arithmetic():
    assert safe_calculate("2 + 3") == 5
    assert safe_calculate("10 - 4") == 6
    assert safe_calculate("6 * 7") == 42
    assert safe_calculate("9 / 2") == 4.5


def test_operator_precedence_and_parentheses():
    assert safe_calculate("12 * (3 + 4) / 2") == 42.0
    assert safe_calculate("2 + 3 * 4") == 14


def test_power_and_unary_minus():
    assert safe_calculate("2 ** 10") == 1024
    assert safe_calculate("-5 + 10") == 5


def test_floor_div_and_mod():
    assert safe_calculate("7 // 2") == 3
    assert safe_calculate("7 % 2") == 1


def test_rejects_name_lookup():
    with pytest.raises(UnsafeExpressionError):
        safe_calculate("__import__('os').system('echo pwned')")


def test_rejects_function_call():
    with pytest.raises(UnsafeExpressionError):
        safe_calculate("print(1)")


def test_rejects_attribute_access():
    with pytest.raises(UnsafeExpressionError):
        safe_calculate("(1).__class__")


def test_rejects_garbage_syntax():
    with pytest.raises(UnsafeExpressionError):
        safe_calculate("this is not math")
