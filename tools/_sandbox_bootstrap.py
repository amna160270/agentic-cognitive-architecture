"""tools/_sandbox_bootstrap.py
Runs INSIDE the child process started by SandboxedPythonRunner.
It restricts what the agent's code can import and which files it can open,
then executes the code.

Usage: python -I _sandbox_bootstrap.py <code_file> <workdir> <allowed_imports_json>
"""
import ast
import builtins
import json
import sys
import traceback
import types
from pathlib import Path

code_file, workdir, allowed_json = sys.argv[1], sys.argv[2], sys.argv[3]
ALLOWED = set(json.loads(allowed_json))
WORKDIR = Path(workdir).resolve()

_real_import = builtins.__import__
_real_open = builtins.open


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Only the agent's code uses this function (we give it as the agent's
    # __import__). Standard-library modules keep the real builtins, so they can
    # import whatever they need internally.
    if level == 0:
        root = name.split(".")[0]
        if root not in ALLOWED:
            raise ImportError(f"import of '{root}' is blocked by the sandbox")
    return _real_import(name, globals, locals, fromlist, level)


def safe_open(file, mode="r", *args, **kwargs):
    path = Path(file).resolve()
    if path != WORKDIR and WORKDIR not in path.parents:
        raise PermissionError("the sandbox only allows files inside the workspace folder")
    return _real_open(file, mode, *args, **kwargs)


BLOCKED_BUILTINS = {
    "eval", "exec", "compile", "input", "breakpoint",
    "exit", "quit", "help", "open", "__import__",
    "getattr", "setattr", "delattr", "vars", "globals", "locals",
}
# Attributes starting with "_" are how code reaches hidden modules (random._os)
# or object internals (().__class__). A few harmless ones are still allowed.
SAFE_PRIVATE_ATTRS = {
    "__init__", "__name__", "__doc__", "__str__", "__repr__", "__len__",
    "__eq__", "__enter__", "__exit__",
}


def find_forbidden_attribute(source):
    """Return an error message if the code touches a private attribute."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None  # let exec() report the syntax error in the normal way
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            if node.attr not in SAFE_PRIVATE_ATTRS:
                return (f"PermissionError: line {node.lineno}: access to "
                        f"'{node.attr}' is blocked by the sandbox")
    return None


safe_builtins = {k: v for k, v in vars(builtins).items() if k not in BLOCKED_BUILTINS}
safe_builtins["__import__"] = safe_import
safe_builtins["open"] = safe_open

code = _real_open(code_file, encoding="utf-8").read()
problem = find_forbidden_attribute(code)
if problem:
    print(problem, file=sys.stderr)
    sys.exit(1)
sys.argv = ["agent_code.py"]  # so unittest.main() does not read our arguments
# The agent's code becomes the real __main__ module, so unittest.main() finds
# the test classes defined in it.
main_module = types.ModuleType("__main__")
sys.modules["__main__"] = main_module
user_globals = main_module.__dict__
user_globals["__builtins__"] = safe_builtins

try:
    exec(compile(code, "<agent_code>", "exec"), user_globals)
except SystemExit:
    raise
except BaseException:
    etype, evalue, tb = sys.exc_info()
    # tb.tb_next skips this bootstrap frame so the agent only sees its own lines.
    traceback.print_exception(etype, evalue, tb.tb_next if tb else None)
    sys.exit(1)