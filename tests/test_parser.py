import pytest

from core.deliberation import ParseError, parse_react_output


def test_clean_action():
    r = parse_react_output(
        'Thought: need math\nAction: calculator\nAction Input: {"expression": "2+3"}'
    )
    assert not r.is_final
    assert r.action.tool_name == "calculator"
    assert r.action.tool_input == {"expression": "2+3"}
    assert r.action.thought == "need math"


def test_final_answer():
    r = parse_react_output("Thought: done\nFinal Answer: The result is 5.")
    assert r.is_final and r.final_answer == "The result is 5."


def test_single_quotes_json():
    r = parse_react_output("Thought: x\nAction: calculator\nAction Input: {'expression': '2+3'}")
    assert r.action.tool_input == {"expression": "2+3"}


def test_trailing_comma():
    r = parse_react_output('Thought: x\nAction: calculator\nAction Input: {"expression": "1+1",}')
    assert r.action.tool_input == {"expression": "1+1"}


def test_missing_closing_brace():
    r = parse_react_output('Thought: x\nAction: calculator\nAction Input: {"expression": "1+1"')
    assert r.action.tool_input == {"expression": "1+1"}


def test_code_fences_around_json():
    raw = 'Thought: x\nAction: python_runner\nAction Input: ```json\n{"code": "print(1)"}\n```'
    assert parse_react_output(raw).action.tool_input == {"code": "print(1)"}


def test_multiline_code_with_raw_newline():
    raw = 'Thought: x\nAction: python_runner\nAction Input: {"code": "a = 1\nprint(a)"}'
    assert "print(a)" in parse_react_output(raw).action.tool_input["code"]


def test_nested_json_and_braces_in_string():
    raw = 'Thought: x\nAction: t\nAction Input: {"a": {"b": 1}, "s": "}{"}'
    assert parse_react_output(raw).action.tool_input == {"a": {"b": 1}, "s": "}{"}


def test_action_wins_over_hallucinated_final_answer():
    raw = 'Thought: x\nAction: calculator\nAction Input: {"expression": "1+1"}\nFinal Answer: 2'
    assert not parse_react_output(raw).is_final


def test_missing_action_input_raises():
    with pytest.raises(ParseError):
        parse_react_output("Thought: x\nAction: calculator")


def test_garbage_raises():
    with pytest.raises(ParseError):
        parse_react_output("I think the answer is probably 5.")


def test_unfixable_json_raises():
    with pytest.raises(ParseError):
        parse_react_output("Thought: x\nAction: calculator\nAction Input: not json at all")