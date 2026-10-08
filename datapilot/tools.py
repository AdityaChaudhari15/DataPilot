"""The 10 deterministic tools from v1, as plain functions (testable) + a LangChain adapter."""
from __future__ import annotations
from typing import Callable


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b. Raises an error if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


def calculate_average(values: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    if not values:
        raise ValueError("Cannot average an empty list.")
    return sum(values) / len(values)


def sum_values(values: list[float]) -> float:
    """Calculate the sum of a list of numbers."""
    if not values:
        raise ValueError("Cannot sum an empty list.")
    return sum(values)


def find_max(values: list[float]) -> float:
    """Find the maximum value in a list of numbers."""
    if not values:
        raise ValueError("Cannot find the maximum of an empty list.")
    return max(values)


def find_min(values: list[float]) -> float:
    """Find the minimum value in a list of numbers."""
    if not values:
        raise ValueError("Cannot find the minimum of an empty list.")
    return min(values)


def calculate_percentage(part: float, whole: float) -> float:
    """Calculate what percentage `part` is of `whole`."""
    if whole == 0:
        raise ValueError("Cannot compute a percentage of zero.")
    return part / whole * 100


def compare_values(a: float, b: float) -> str:
    """Compare two numbers and say which is greater."""
    if a > b:
        return "a is greater than b"
    if b > a:
        return "b is greater than a"
    return "a is equal to b"


PURE_TOOLS: dict[str, Callable] = {
    f.__name__: f for f in (add, subtract, multiply, divide, calculate_average, sum_values,
                            find_max, find_min, calculate_percentage, compare_values)
}


def build_langchain_tools() -> list:
    """Wrap the pure functions as LangChain tools (imported lazily so tests run without LangChain)."""
    from langchain_core.tools import StructuredTool
    return [StructuredTool.from_function(f, name=n, description=f.__doc__) for n, f in PURE_TOOLS.items()]
