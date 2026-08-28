"""Deterministic P1 problem generators for the math-physics smoke pipeline."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any

import sympy as sp


GENERATOR_VERSION = "p1-v0.1"
DEFAULT_TOLERANCE = 1e-8


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _split_for(family: str, index: int, counts: dict[str, int]) -> str:
    """Use whole families for holdout; keep deterministic train/dev assignment."""
    if family in {"ode_characteristic", "vector_product_curl"}:
        return "holdout_family" if index < counts.get("holdout_family", 50) else "secret"
    if index < counts.get("train", 500):
        return "train"
    if index < counts.get("train", 500) + counts.get("dev", 100):
        return "dev"
    return "secret"


def _format_spec(kind: str, good: str, bad: str) -> dict[str, Any]:
    if kind == "vector":
        conventions = ["Cartesian 正交基", "分量用括号表示", "零向量写成 (0, 0, 0)"]
    else:
        conventions = ["显式写出积分常数 C", "先给通解再给验证残差", "不省略定义域条件"]
    return {
        "conventions": conventions,
        "example_good": good,
        "example_bad": bad,
    }


def _problem(
    *,
    problem_id: str,
    family: str,
    category: str,
    prompt: str,
    context: str,
    canonical: str,
    equiv_classes: list[str],
    format_spec: dict[str, Any],
    difficulty: int,
    seed: int,
    payload: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    return {
        "id": problem_id,
        "family": family,
        "method_family": family,
        "category": category,
        "prompt": prompt,
        "context": context,
        "gold": {
            "canonical_sympy": canonical,
            "accept_equiv_classes": equiv_classes,
            "numeric_probe_spec": {
                "n_points": 32,
                "tolerance": DEFAULT_TOLERANCE,
                "seed": seed,
            },
            "payload": payload,
        },
        "format_spec": format_spec,
        "difficulty": difficulty,
        "split": split,
        "provenance": {
            "generator_version": GENERATOR_VERSION,
            "params_seed": seed,
        },
    }


def _vector_field(rng: random.Random, prefix: str) -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, list[sp.Expr]]:
    x, y, z = sp.symbols(f"{prefix}x {prefix}y {prefix}z", real=True)
    basis = [x, y, z]
    choices: list[sp.Expr] = []
    for var in basis:
        coeff = rng.randint(1, 4)
        power = rng.randint(1, 3)
        kind = rng.randrange(3)
        if kind == 0:
            choices.append(coeff * var**power)
        elif kind == 1:
            choices.append(coeff * sp.sin(var))
        else:
            choices.append(coeff * sp.cos(var))
    return x, y, z, choices


def _grad(f: sp.Expr, coords: tuple[sp.Symbol, sp.Symbol, sp.Symbol]) -> list[sp.Expr]:
    return [sp.diff(f, c) for c in coords]


def _curl(v: list[sp.Expr], coords: tuple[sp.Symbol, sp.Symbol, sp.Symbol]) -> list[sp.Expr]:
    x, y, z = coords
    return [sp.diff(v[2], y) - sp.diff(v[1], z), sp.diff(v[0], z) - sp.diff(v[2], x), sp.diff(v[1], x) - sp.diff(v[0], y)]


def _div(v: list[sp.Expr], coords: tuple[sp.Symbol, sp.Symbol, sp.Symbol]) -> sp.Expr:
    return sum(sp.diff(component, coord) for component, coord in zip(v, coords))


def make_vector_problem(seed: int, index: int, family: str, split: str) -> dict[str, Any]:
    rng = random.Random(seed)
    x, y, z, field = _vector_field(rng, f"v{index}_")
    coords = (x, y, z)
    scalar = rng.randint(1, 4) * x**rng.randint(1, 2) + rng.randint(1, 4) * sp.sin(y) + rng.randint(1, 3) * z
    if family == "vector_div_curl":
        lhs = _div(_curl(field, coords), coords)
        expression = f"∇·(∇×F), F={tuple(field)}"
        payload = {"kind": "vector", "coords": [str(c) for c in coords], "field": [sp.sstr(e) for e in field], "identity": "div_curl"}
        method_text = "先计算旋度，再逐分量求散度"
        classes = ["scalar_zero", "zero_scalar"]
    elif family == "vector_curl_grad":
        lhs = _curl(_grad(scalar, coords), coords)
        expression = f"∇×(∇f), f={sp.sstr(scalar)}"
        payload = {"kind": "vector", "coords": [str(c) for c in coords], "scalar": sp.sstr(scalar), "identity": "curl_grad"}
        method_text = "先写梯度，再计算旋度"
        classes = ["zero_vector"]
    elif family == "vector_div_grad":
        lhs = _div(_grad(scalar, coords), coords)
        expression = f"∇·(∇f), f={sp.sstr(scalar)}"
        payload = {"kind": "vector", "coords": [str(c) for c in coords], "scalar": sp.sstr(scalar), "identity": "div_grad"}
        method_text = "先写梯度，再求散度并整理为 Laplacian"
        classes = ["laplacian"]
    elif family == "vector_product_div":
        f = scalar
        product = [f * component for component in field]
        rhs = sum(a * b for a, b in zip(_grad(f, coords), field)) + f * _div(field, coords)
        lhs = _div(product, coords) - rhs
        expression = f"∇·(fF)=∇f·F+f∇·F, f={sp.sstr(f)}, F={tuple(field)}"
        payload = {"kind": "vector", "coords": [str(c) for c in coords], "scalar": sp.sstr(f), "field": [sp.sstr(e) for e in field], "identity": "product_div"}
        method_text = "分别展开左侧散度和右侧乘积法则"
        classes = ["scalar_zero"]
    elif family == "vector_product_curl":
        f = scalar
        product = [f * component for component in field]
        rhs = [a * b + f * c for a, b, c in zip(_grad(f, coords), _curl(field, coords), [1, 1, 1])]
        # Correct vector product rule: grad(f) × F + f curl(F).
        grad_cross = [
            _grad(f, coords)[1] * field[2] - _grad(f, coords)[2] * field[1],
            _grad(f, coords)[2] * field[0] - _grad(f, coords)[0] * field[2],
            _grad(f, coords)[0] * field[1] - _grad(f, coords)[1] * field[0],
        ]
        rhs_vec = [a + f * b for a, b in zip(grad_cross, _curl(field, coords))]
        lhs_vec = _curl(product, coords)
        lhs = [sp.simplify(a - b) for a, b in zip(lhs_vec, rhs_vec)]
        expression = f"∇×(fF)=∇f×F+f∇×F, f={sp.sstr(f)}, F={tuple(field)}"
        payload = {"kind": "vector", "coords": [str(c) for c in coords], "scalar": sp.sstr(f), "field": [sp.sstr(e) for e in field], "identity": "product_curl"}
        method_text = "分别计算两个叉乘项并逐分量比较"
        classes = ["zero_vector"]
    else:
        raise ValueError(f"unknown vector family: {family}")

    canonical = sp.sstr(lhs)
    prompt_templates = [
        f"在 Cartesian 正交坐标中，验证 {expression} 恒等式。{method_text}，最后化简到规范形式。",
        f"请对 {expression} 做符号推导并检查恒等式成立。要求逐分量写出中间结果，最终给出规范形式。",
    ]
    prompt = prompt_templates[index % len(prompt_templates)]
    return _problem(
        problem_id=f"{family}-{index:06d}", family=family, category="simplify", prompt=prompt,
        context="变量均为实数；使用 Cartesian 正交基；除题面变量外不引入隐藏条件。",
        canonical=canonical, equiv_classes=classes,
        format_spec=_format_spec("vector", "逐分量推导并写成 (0, 0, 0) 或 0", "只写‘显然成立’而没有分量计算"),
        difficulty=1 + (index % 2), seed=seed, payload=payload, split=split,
    )


def make_ode_problem(seed: int, index: int, family: str, split: str) -> dict[str, Any]:
    rng = random.Random(seed)
    x = sp.symbols(f"x_{index}", real=True)
    y = sp.Function(f"y_{index}")
    C = sp.Symbol(f"C_{index}")
    if family == "ode_integrating_factor":
        p = rng.choice([1, 2, -1, 3])
        q = rng.choice([sp.Integer(1), x, x + 1, sp.exp(x)])
        ode = sp.Eq(sp.diff(y(x), x) + p * y(x), q)
        sol = sp.dsolve(ode, y(x)).rhs
        method = "integrating_factor"
        method_text = "使用积分因子法"
    elif family == "ode_separation":
        k = rng.choice([-2, -1, 1, 2, 3])
        n = rng.choice([0, 1, 2])
        sol = C * sp.exp(sp.Rational(k, n + 1) * x ** (n + 1))
        ode = sp.Eq(sp.diff(y(x), x), k * x**n * y(x))
        method = "separation"
        method_text = "分离变量后积分"
    elif family == "ode_characteristic":
        a = rng.choice([1, 2, 3])
        b = rng.choice([1, 2, 4])
        r1, r2 = -a, -b
        if r1 == r2:
            sol = (C + sp.Symbol(f"D_{index}") * x) * sp.exp(r1 * x)
        else:
            sol = C * sp.exp(r1 * x) + sp.Symbol(f"D_{index}") * sp.exp(r2 * x)
        ode = sp.Eq(sp.diff(y(x), x, 2) + (a + b) * sp.diff(y(x), x) + a * b * y(x), 0)
        method = "characteristic"
        method_text = "写特征方程并使用特征根"
    else:
        raise ValueError(f"unknown ODE family: {family}")

    residual = sp.simplify((ode.lhs - ode.rhs).subs(y(x), sol).doit())
    canonical = sp.sstr(residual)
    prompt_templates = [
        f"求解 {sp.sstr(ode)}，要求{method_text}，明确写出积分常数并验证代回残差。",
        f"请解微分方程 {sp.sstr(ode)}。先说明解法方法，再给出通解，并用符号代回检查。",
    ]
    payload = {
        "kind": "ode", "family": method, "variable": str(x), "function": str(y(x)),
        "ode": sp.sstr(ode), "solution": sp.sstr(sol), "residual": sp.sstr(residual),
        "parameters": {"constants": [str(c) for c in sol.free_symbols if str(c).startswith(("C_", "D_"))]},
    }
    return _problem(
        problem_id=f"{family}-{index:06d}", family=family, category="forward",
        prompt=prompt_templates[index % 2],
        context="变量为实数；通解中的积分常数独立；避开解表达式的奇点。",
        canonical=canonical, equiv_classes=["zero_residual"],
        format_spec=_format_spec("ode", "写出 y(x)=... 并保留所有积分常数", "只给一个数值初值解而不写通解"),
        difficulty=1 + (index % 2), seed=seed, payload=payload, split=split,
    )


VECTOR_FAMILIES = ["vector_div_curl", "vector_curl_grad", "vector_div_grad", "vector_product_div", "vector_product_curl"]
ODE_FAMILIES = ["ode_integrating_factor", "ode_separation", "ode_characteristic"]


def generate_dataset(*, train: int = 500, dev: int = 100, holdout: int = 50, secret: int = 50, seed: int = 20260828) -> list[dict[str, Any]]:
    """Generate exact split sizes; holdout is made from entire method families."""
    if min(train, dev, holdout, secret) < 0:
        raise ValueError("split sizes must be non-negative")
    rows: list[dict[str, Any]] = []
    families = VECTOR_FAMILIES + ODE_FAMILIES
    holdout_families = {"vector_product_curl", "ode_characteristic"}

    def add(n: int, split: str, offset: int) -> None:
        for j in range(n):
            i = offset + j
            if split == "holdout_family":
                held = sorted(holdout_families)
                family = held[j % len(held)]
            elif split in {"train", "dev"}:
                available = [f for f in families if f not in holdout_families]
                family = available[j % len(available)]
            else:
                family = families[i % len(families)]
            family_seed = seed + _stable_int(f"{family}:{split}") % 100000 + i * 17
            if family.startswith("vector_"):
                row = make_vector_problem(family_seed, i, family, split)
            else:
                row = make_ode_problem(family_seed, i, family, split)
            rows.append(row)

    # Train/dev use the remaining families; secret is generated separately and never reused.
    add(train, "train", 0)
    add(dev, "dev", train)
    add(holdout, "holdout_family", train + dev)
    add(secret, "secret", train + dev + holdout)
    expected = {"train": train, "dev": dev, "holdout_family": holdout, "secret": secret}
    actual = {name: sum(r["split"] == name for r in rows) for name in expected}
    if actual != expected:
        raise RuntimeError(f"split counts mismatch: {actual} != {expected}")
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
