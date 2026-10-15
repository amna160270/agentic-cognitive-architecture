import pytest

from tools.base_tool import ToolRegistry
from tools.calculator import CalculatorTool
from tools.file_storage import FileStorageTool


@pytest.fixture
def calc():
    return CalculatorTool()


# ---------- calculator ----------
def test_calc_basic(calc):
    assert calc.run(expression="2 + 3 * 4").output == "14"


def test_calc_compound_interest(calc):
    r = calc.run(expression="10000 * (1 + 0.075) ** 5")
    assert r.ok and float(r.output) == pytest.approx(14356.29326171875)


def test_calc_caret_means_power(calc):
    assert calc.run(expression="2^10").output == "1024"


def test_calc_functions_and_stats(calc):
    assert calc.run(expression="sqrt(16) + max(1, 5)").output == "9.0"
    assert calc.run(expression="mean([1, 2, 3, 4])").output == "2.5"
    assert calc.run(expression="stdev([2, 4, 4, 4, 5, 5, 7, 9])").ok


@pytest.mark.parametrize("bad", [
    "__import__('os').system('echo hi')",
    "open('secret.txt')",
    "eval('1+1')",
    "().__class__",
    "[x for x in range(3)]",
    "'abc' * 3",
    "lambda: 1",
])
def test_calc_rejects_unsafe_code(calc, bad):
    assert not calc.run(expression=bad).ok


def test_calc_division_by_zero(calc):
    r = calc.run(expression="1 / 0")
    assert not r.ok and "division by zero" in r.output


def test_calc_huge_power_blocked(calc):
    assert not calc.run(expression="9 ** 9 ** 9").ok


def test_calc_missing_argument(calc):
    r = calc.run()
    assert not r.ok and "Invalid arguments" in r.output


# ---------- registry ----------
def test_registry_describe_and_schema(calc):
    reg = ToolRegistry()
    reg.register(calc)
    text = reg.describe()
    assert "calculator" in text and "expression" in text
    schema = reg.schemas()[0]
    assert schema["name"] == "calculator"
    assert "expression" in schema["inputSchema"]["properties"]


def test_registry_unknown_tool_and_duplicate(calc):
    reg = ToolRegistry()
    reg.register(calc)
    r = reg.execute("hack_tool", {})
    assert not r.ok and "not found" in r.output
    with pytest.raises(ValueError):
        reg.register(CalculatorTool())


def test_registry_executes_tool(calc):
    reg = ToolRegistry()
    reg.register(calc)
    assert reg.execute("calculator", {"expression": "1+1"}).output == "2"


# ---------- file storage ----------
def test_file_storage_writes(tmp_path):
    tool = FileStorageTool(output_dir=str(tmp_path))
    r = tool.run(filename="report.md", content="# Hello")
    assert r.ok and (tmp_path / "report.md").read_text() == "# Hello"


@pytest.mark.parametrize("name", ["../evil.txt", "..\\evil.txt", "/etc/passwd", "C:\\evil.txt", ""])
def test_file_storage_blocks_escape(tmp_path, name):
    tool = FileStorageTool(output_dir=str(tmp_path / "out"))
    assert not tool.run(filename=name, content="x").ok
    assert not (tmp_path / "evil.txt").exists()