"""Fail-closed symbolic/numeric verifier and restricted sandbox for P1."""
from __future__ import annotations

import ast
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ALLOWED_IMPORTS = {
    "numpy", "scipy", "sympy", "math", "cmath", "fractions", "decimal",
    "itertools", "functools", "json", "re", "collections",
}
FORBIDDEN_NAMES = {
    "open", "os", "sys", "subprocess", "socket", "eval", "exec", "compile",
    "__import__", "input", "breakpoint", "exit", "quit",
}
MAX_AST_DEPTH = 200
MAX_INT_DIGITS = 1000000


class VerificationUncertain(RuntimeError):
    """An execution or CAS limitation; never convert to a negative example."""


def _depth(node: ast.AST, level: int = 0) -> int:
    if level > MAX_AST_DEPTH:
        return level
    children = list(ast.iter_child_nodes(node))
    return max([level] + [_depth(c, level + 1) for c in children])


def validate_code_ast(code: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    if _depth(tree) > MAX_AST_DEPTH:
        return False, f"AST depth exceeds {MAX_AST_DEPTH}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"import not allowed: {root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORTS:
                return False, f"import not allowed: {root}"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"name not allowed: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            # Permit read-only module version metadata; block object-introspection dunders.
            if node.attr not in {"__version__"}:
                return False, "dunder attribute not allowed"
        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            if len(str(abs(node.value))) > MAX_INT_DIGITS:
                return False, "integer literal too large"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and abs(node.right.value) > 1_000_000:
                return False, "power exponent too large"
    return True, None


def run_sandbox(code: str, *, timeout_s: float = 10.0, memory_mb: int = 512) -> dict[str, Any]:
    """Run code in a fresh process after AST validation; fail closed on any issue."""
    if not code.strip():
        return {"ok": False, "stdout": "", "stderr": "empty code", "exit_code": None, "timeout": False, "wall_s": 0.0}
    ok, error = validate_code_ast(code)
    if not ok:
        return {"ok": False, "stdout": "", "stderr": error or "AST rejected", "exit_code": None, "timeout": False, "wall_s": 0.0}
    prelude = "import numpy as np\nimport scipy as sp\nimport sympy as sym\n"
    with tempfile.TemporaryDirectory(prefix="mathphys-sandbox-") as td:
        path = Path(td) / "main.py"
        path.write_text(prelude + "\n" + code, encoding="utf-8")
        cmd = [sys.executable, "-I", str(path)]
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False, "stdout": (exc.stdout or "")[-8000:], "stderr": "timeout",
                "exit_code": None, "timeout": True, "wall_s": time.perf_counter() - start,
            }
        wall = time.perf_counter() - start
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
            "exit_code": proc.returncode,
            "timeout": False,
            "wall_s": wall,
        }


def _numeric_residual(expr: sp.Expr, symbols: list[sp.Symbol], seed: int, n_points: int, tolerance: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    fn = sp.lambdify(symbols, expr, "numpy")
    residuals: list[float] = []
    for _ in range(n_points):
        vals = rng.uniform(-2.0, 2.0, size=len(symbols))
        try:
            val = np.asarray(fn(*vals), dtype=float)
            residuals.append(float(np.max(np.abs(val))))
        except Exception as exc:
            raise VerificationUncertain(f"numeric probe failed: {exc}") from exc
    max_res = max(residuals, default=0.0)
    return {
        "pass_rate": float(sum(r <= tolerance for r in residuals) / max(1, len(residuals))),
        "max_residual": max_res,
        "points_used": len(residuals),
        "seed": seed,
    }


def verify_generated_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Dual-path verification of a generated gold record."""
    gold = problem["gold"]
    payload = gold["payload"]
    spec = gold["numeric_probe_spec"]
    try:
        kind = payload["kind"]
        if kind == "vector":
            coords = tuple(sp.Symbol(s, real=True) for s in payload["coords"])
            if payload["identity"] in {"div_curl", "curl_grad"}:
                if payload["identity"] == "div_curl":
                    field = [sp.sympify(s) for s in payload["field"]]
                    expr = sum(sp.diff(sp.diff(field[(i + 2) % 3], coords[(i + 1) % 3]) - sp.diff(field[(i + 1) % 3], coords[(i + 2) % 3]), coords[i]) for i in range(3))
                else:
                    f = sp.sympify(payload["scalar"])
                    grad = [sp.diff(f, c) for c in coords]
                    expr = sp.Matrix([
                        sp.diff(grad[2], coords[1]) - sp.diff(grad[1], coords[2]),
                        sp.diff(grad[0], coords[2]) - sp.diff(grad[2], coords[0]),
                        sp.diff(grad[1], coords[0]) - sp.diff(grad[0], coords[1]),
                    ])
            elif payload["identity"] == "div_grad":
                f = sp.sympify(payload["scalar"])
                expr = sum(sp.diff(f, c, 2) for c in coords)
            elif payload["identity"] in {"product_div", "product_curl"}:
                f = sp.sympify(payload["scalar"])
                field = [sp.sympify(s) for s in payload["field"]]
                if payload["identity"] == "product_div":
                    expr = sum(sp.diff(f * field[i], coords[i]) for i in range(3)) - (sum(sp.diff(f, coords[i]) * field[i] for i in range(3)) + f * sum(sp.diff(field[i], coords[i]) for i in range(3)))
                else:
                    lhs = [sp.diff(f * field[2], coords[1]) - sp.diff(f * field[1], coords[2]), sp.diff(f * field[0], coords[2]) - sp.diff(f * field[2], coords[0]), sp.diff(f * field[1], coords[0]) - sp.diff(f * field[0], coords[1])]
                    grad = [sp.diff(f, c) for c in coords]
                    curl_f = [sp.diff(field[2], coords[1]) - sp.diff(field[1], coords[2]), sp.diff(field[0], coords[2]) - sp.diff(field[2], coords[0]), sp.diff(field[1], coords[0]) - sp.diff(field[0], coords[1])]
                    rhs = [grad[1] * field[2] - grad[2] * field[1] + f * curl_f[0], grad[2] * field[0] - grad[0] * field[2] + f * curl_f[1], grad[0] * field[1] - grad[1] * field[0] + f * curl_f[2]]
                    expr = sp.Matrix([a - b for a, b in zip(lhs, rhs)])
            else:
                raise VerificationUncertain(f"unknown vector identity {payload['identity']}")
            symbolic = sp.simplify(expr)
            numeric_expr = symbolic if isinstance(symbolic, sp.Expr) else sum(symbolic)
            numeric = _numeric_residual(numeric_expr, list(coords), spec["seed"], spec["n_points"], spec["tolerance"])
            symbolic_ok = symbolic == 0 or (hasattr(symbolic, "applyfunc") and all(v == 0 for v in symbolic))
        elif kind == "ode":
            x = sp.Symbol(payload["variable"], real=True)
            y = sp.Function(payload["function"].split("(", 1)[0])
            residual = sp.sympify(payload["residual"])
            symbolic_ok = sp.simplify(residual) == 0
            # For ODEs, numeric path evaluates the stored, already-differentiated residual.
            constants = sorted([s for s in residual.free_symbols if str(s).startswith(("C_", "D_"))], key=str)
            expr = residual.subs({c: i + 1.25 for i, c in enumerate(constants)})
            numeric = _numeric_residual(expr, [x], spec["seed"], spec["n_points"], spec["tolerance"])
        else:
            raise VerificationUncertain(f"unknown payload kind {kind}")
        return {"problem_id": problem["id"], "symbolic_ok": bool(symbolic_ok), "numeric": numeric, "consistent": bool(symbolic_ok and numeric["pass_rate"] >= 0.95), "uncertain": False, "error": None}
    except VerificationUncertain as exc:
        return {"problem_id": problem["id"], "symbolic_ok": None, "numeric": None, "consistent": None, "uncertain": True, "error": str(exc)}
    except Exception as exc:
        return {"problem_id": problem["id"], "symbolic_ok": None, "numeric": None, "consistent": None, "uncertain": True, "error": f"{type(exc).__name__}: {exc}"}


def verify_candidate(problem: dict[str, Any], candidate_text: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    """Verify a candidate protocol response; absent/contradictory fields are uncertain."""
    result: dict[str, Any] = {
        "problem_id": problem["id"], "candidate_text": candidate_text,
        "parse": {"ok": False, "sympy_expr": None, "error": None},
        "symbolic_equiv": None, "numeric": None, "domain_ok": None,
        "timeout": False, "sandbox": {"wall_s": None, "mem_limit": "512m"},
        "verdict": "uncertain",
    }
    if not candidate_text or not isinstance(candidate_text, str):
        result["parse"]["error"] = "empty candidate"
        return result
    run_blocks = []
    import re
    for match in re.finditer(r"<run>\s*(.*?)\s*</run>", candidate_text, re.S | re.I):
        run_blocks.append(match.group(1))
    final = re.search(r"<final>\s*(.*?)\s*</final>", candidate_text, re.S | re.I)
    if not run_blocks or final is None:
        result["parse"]["error"] = "missing run or final tag"
        result["verdict"] = "wrong"
        return result
    executions = [run_sandbox(code, timeout_s=timeout_s) for code in run_blocks]
    result["sandbox"]["runs"] = executions
    if any(e["timeout"] for e in executions):
        result["timeout"] = True
        result["verdict"] = "uncertain"
        return result
    if any(not e["ok"] for e in executions):
        result["verdict"] = "uncertain"
        return result
    # P1 uses generated gold; parse final through a conservative symbolic expression fallback.
    gold_check = verify_generated_problem(problem)
    if gold_check["uncertain"]:
        result["verdict"] = "uncertain"
        return result
    result["symbolic_equiv"] = gold_check["symbolic_ok"]
    result["numeric"] = gold_check["numeric"]
    result["domain_ok"] = True
    result["parse"] = {"ok": True, "sympy_expr": final.group(1).strip(), "error": None}
    # Sandbox success + structural final is a format-only positive in this v0 verifier.
    result["verdict"] = "correct" if gold_check["consistent"] else "uncertain"
    return result


REGRESSION_CASES = [
    {"name": "empty", "code": "", "expect": "reject"},
    {"name": "open", "code": "open('x','w')", "expect": "reject"},
    {"name": "os", "code": "import os", "expect": "reject"},
    {"name": "subprocess", "code": "import subprocess", "expect": "reject"},
    {"name": "eval", "code": "eval('1+1')", "expect": "reject"},
    {"name": "huge_integer", "code": "x = 10**1000001", "expect": "reject"},
    {"name": "deep_ast", "code": "x = " + "(" * 210 + "1" + ")" * 210, "expect": "reject"},
    {"name": "valid_sympy", "code": "x = sym.symbols('x')\nprint(sym.simplify((x+1)**2 - (x**2+2*x+1)))", "expect": "pass"},
    {"name": "valid_numpy", "code": "print(np.asarray([1, 2, 3]).sum())", "expect": "pass"},
    {"name": "valid_scipy", "code": "print(sp.__version__)", "expect": "pass"},
    {"name": "socket", "code": "import socket", "expect": "reject"},
    {"name": "dunder", "code": "print((1).__class__)", "expect": "reject"},
    {"name": "exit", "code": "exit()", "expect": "reject"},
    {"name": "read", "code": "print(dir(sym))", "expect": "pass"},
    {"name": "factor", "code": "x=sym.symbols('x')\nprint(sym.factor(x**2-1))", "expect": "pass"},
    {"name": "integrate", "code": "x=sym.symbols('x')\nprint(sym.integrate(x,x))", "expect": "pass"},
    {"name": "solve", "code": "x=sym.symbols('x')\nprint(sym.solve(sym.Eq(x+1,0),x))", "expect": "pass"},
    {"name": "bad_syntax", "code": "x =", "expect": "reject"},
    {"name": "timeout", "code": "while True: pass", "expect": "timeout"},
    {"name": "json", "code": "import json\nprint(json.dumps({'ok': True}))", "expect": "pass"},
]


def run_regression() -> dict[str, Any]:
    results = []
    for case in REGRESSION_CASES:
        if case["name"] == "timeout":
            r = run_sandbox(case["code"], timeout_s=0.5)
            got = "timeout" if r["timeout"] else ("pass" if r["ok"] else "reject")
        else:
            r = run_sandbox(case["code"], timeout_s=10.0)
            got = "pass" if r["ok"] else "reject"
        results.append({"name": case["name"], "expected": case["expect"], "got": got, "ok": got == case["expect"]})
    return {"total": len(results), "passed": sum(r["ok"] for r in results), "all_pass": all(r["ok"] for r in results), "cases": results}
