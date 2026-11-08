import pytest

from app.graph.safe_math import CalculationError, UnsafeExpressionError, safe_calculate


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


def test_division_by_zero_is_a_calculation_error():
    with pytest.raises(CalculationError, match="division by zero"):
        safe_calculate("5 / 0")


@pytest.mark.timeout(5)
@pytest.mark.parametrize("expression", ["9**9**9", "2**100000", "10**5000", "(10**1000)**10"])
def test_huge_powers_are_rejected_quickly(expression):
    # Without limits these would keep the worker busy for minutes or forever.
    with pytest.raises(CalculationError):
        safe_calculate(expression)


def test_reasonable_powers_still_work():
    assert safe_calculate("2 ** 100") == 2**100
    assert safe_calculate("1.5 ** -3") == pytest.approx(1.5**-3)


def test_repeated_multiplication_cannot_grow_without_bound():
    big = "*".join(["(10**300)"] * 6)  # 10**1800, over the size cap
    with pytest.raises(CalculationError):
        safe_calculate(big)


def test_float_overflow_is_reported():
    with pytest.raises(CalculationError):
        safe_calculate("10.0 ** 400")


def test_overlong_expression_is_rejected():
    with pytest.raises(CalculationError):
        safe_calculate("1+" * 150 + "1")
