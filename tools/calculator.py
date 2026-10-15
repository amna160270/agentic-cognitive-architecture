"""tools/calculator.py
Safe math/statistics evaluator. It never calls eval(): the expression is parsed
into an AST and only whitelisted node types are evaluated by our own code.
"""
import ast
import math
import operator
import statistics

from pydantic import BaseModel, Field

from tools.base_tool import BaseTool, ToolError

MAX_EXPR_LEN = 500


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Math expression, e.g. '1000 * (1 + 0.05) ** 3'")


def _safe_pow(a, b):
    # Stop things like 9 ** 9 ** 9 from freezing the agent.
    if isinstance(a, int) and isinstance(b, int) and b > 0:
        if a.bit_length() * b > 10_000:
            raise ToolError("exponent too large")
    elif abs(b) > 10_000:
        raise ToolError("exponent too large")
    return a ** b


_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: _safe_pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil,
    "mean": statistics.mean, "median": statistics.median,
    "stdev": statistics.stdev, "pstdev": statistics.pstdev,
    "variance": statistics.variance,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError("only numbers are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(item) for item in node.elts]
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"unknown name '{node.id}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolError("this function is not allowed")
        if node.keywords:
            raise ToolError("keyword arguments are not allowed")
        return _FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError(f"unsupported syntax: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluates a math or statistics expression safely. Supports + - * / // % **, "
        "sqrt, log, exp, sin, cos, round, min, max, sum, mean, median, stdev, pi, e."
    )
    args_schema = CalculatorArgs

    def execute(self, expression: str) -> str:
        if len(expression) > MAX_EXPR_LEN:
            raise ToolError(f"expression longer than {MAX_EXPR_LEN} characters")
        # Small models often write 2^3 meaning "2 to the power 3".
        expression = expression.strip().replace("^", "**")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ToolError(f"syntax error: {e.msg}")
        try:
            result = _eval(tree)
        except ZeroDivisionError:
            raise ToolError("division by zero")
        except (ArithmeticError, ValueError, TypeError, statistics.StatisticsError) as e:
            raise ToolError(f"{type(e).__name__}: {e}")
        if isinstance(result, complex):
            raise ToolError("result is a complex number")
        if isinstance(result, float):
            result = round(result, 10)
        return str(result)