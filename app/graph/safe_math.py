"""Arithmetic evaluator for the calculator tool.

The expression is parsed with `ast` and only numeric literals, + - * / // % **
and unary +/- are evaluated. Names, calls, attribute access and everything
else raise UnsafeExpressionError, so the tool can't run arbitrary code.

Python integers have no size limit, so an input like 9**9**9 would keep a
worker busy indefinitely. To prevent that the expression length, the exponent
and the size of every intermediate integer are capped.
"""

from __future__ import annotations

import ast
import operator

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1_000
MAX_INT_BITS = 4_096  # roughly 1,200 decimal digits

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculationError(ValueError):
    """The expression is valid input but can't be evaluated (e.g. 1/0, too large)."""


class UnsafeExpressionError(CalculationError):
    """The expression contains something other than plain arithmetic."""


def _check_size(value):
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise CalculationError("result is too large")
    return value


def _power(base, exponent):
    if abs(exponent) > MAX_EXPONENT:
        raise CalculationError(f"exponent larger than {MAX_EXPONENT} is not supported")
    # Reject before computing: |base ** exp| has at most bit_length(base) * exp bits.
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
        if base.bit_length() * exponent > MAX_INT_BITS:
            raise CalculationError("result is too large")
    return base**exponent


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpressionError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            return _check_size(_power(left, right))
        if type(node.op) in _BINOPS:
            return _check_size(_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval(node.operand))
    raise UnsafeExpressionError(f"unsupported syntax: {type(node).__name__}")


def safe_calculate(expression: str) -> int | float:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculationError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"could not parse expression: {exc.msg}") from exc
    try:
        return _eval(tree.body)
    except ZeroDivisionError as exc:
        raise CalculationError("division by zero") from exc
    except OverflowError as exc:
        raise CalculationError("result is too large") from exc
