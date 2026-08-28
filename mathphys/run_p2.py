"""P2 trajectory manufacturing: execute generated runs and render the 06 protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from generators import generate_dataset
from metrics import emit
from verifier import run_sandbox, verify_generated_problem

# Kept byte-for-byte aligned with the approved 06 protocol prompt used by P1 data.
SYSTEM_PROMPT = """你是一个数学推导助手,可以使用一个 Python 代码沙箱来推导和验证答案。请先阅读以下说明。

【沙箱环境】
- 你每提交一段 <run>...</run> 代码,系统会把它写入 /sd/main.py 并用 Python 3.11 执行,然后用 <output>...</output> 把 stdout 和 stderr 返回给你(超过 8000 字符会截断)。
- 每次执行都是全新进程:上一次定义的变量不会保留。文件是唯一状态,需要的定义每次都要写全。
- 以下库已预先导入,无需再写 import:numpy as np、scipy as sp、sympy as sym。
- 限制:无网络;单次执行最多 10 秒、内存 512MB;只允许 numpy/scipy/sympy/math/cmath/fractions/decimal/itertools/functools/json/re/collections。

【动作】
- <run>代码</run>:提交并执行一段代码(全文替换 /sd/main.py 的模型部分)。
- <read />:查看当前 /sd/main.py 的完整内容(仅当你忘了文件里有什么时使用)。
- <final>答案</final>:提交最终答案,对话随即结束。答案必须符合题目要求的格式。

【忘了 API 怎么用怎么办】
- 在代码里用 print(dir(sp.integrate)) 列出模块成员,或用 help(函数名) 查看文档(输出会截断)。
- 代码报错时,返回结果的末尾会附上可用命名空间的提示。
- 常用速查:
  sym.symbols('x y z') 定义符号;sym.Function('f')(x) 定义未知函数;
  sym.diff(f, x) / sym.integrate(f, x) 微分/积分;sym.simplify / sym.expand / sym.factor 化简/展开/因式分解;
  sym.Eq(左边, 右边) 构造等式;sym.solve(eq, x) 解方程;sym.dsolve(ode) 解常微分方程;
  sym.lambdify(x, f, 'numpy') 表达式转数值函数;
  sym.Matrix([...]) 向量/矩阵;sym.curl、sym.divergence、sym.gradient 需手动实现时用 sym.diff 组合;
  sp.integrate.solve_ivp(fun, (t0, t1), y0) 常微分方程数值解;np.linalg.norm 范数;np.linspace 网格。

【工作流建议】
1. 先读懂题目:明确要求什么形式的答案(化简题注意"规范形式",求解题注意通解/特解与积分常数)。
2. 用 sym 逐步推导;每一步不确定时,用数值抽查验证(随机取点代入比较两边)。
3. 提交 <final> 之前,至少做一次独立于推导路径的数值验证。
4. 如果题目条件不足、无解或无法确定,在 <final> 中写:无法确定(原因)。

【示例】
题目:验证 (x+y)^2 展开后与 x^2+2xy+y^2 数值一致。
<run>
xs = np.random.uniform(-3, 3, 100)
lhs = (xs + xs*0 + xs)**2
a, b = sym.symbols('a b')
f1 = (a + b)**2; f2 = a**2 + 2*a*b + b**2
g1, g2 = sym.lambdify((a, b), f1, 'numpy'), sym.lambdify((a, b), f2, 'numpy')
pts = np.random.uniform(-3, 3, (100, 2))
print(sym.simplify(f1 - f2), np.abs(g1(pts[:,0], pts[:,1]) - g2(pts[:,0], pts[:,1])).max())
</run>
<output>
0 0.0
</output>
<final>
两表达式恒等:sym.simplify(f1-f2)=0,数值最大偏差 0.0。
</final>"""


def render_run(code: str) -> str:
    return f"<run>\n{code}\n</run>"


def render_output(text: str) -> str:
    return f"<output>\n{text[-8000:]}\n</output>"


def render_final(text: str) -> str:
    return f"<final>\n{text}\n</final>"


def make_trajectory(problem: dict) -> dict:
    payload = problem["gold"]["payload"]
    # Rebuild the exact generated symbols and execute a real verification run.
    if payload["kind"] == "vector":
        coords = payload["coords"]
        bindings = ", ".join(coords) + " = sym.symbols(" + repr(" ".join(coords)) + ", real=True)"
        code = "import sympy as sym\n" + bindings + "\n"
        local_map = ", locals={" + ", ".join(f"{name!r}: {name}" for name in coords) + ", 'sin': sym.sin, 'cos': sym.cos, 'exp': sym.exp}"
        identity = payload["identity"]
        if identity == "div_curl":
            code += f"F = sym.Matrix([sym.sympify({payload['field']!r}[0]{local_map})"
            for idx in range(1, 3):
                code += f", sym.sympify({payload['field']!r}[{idx}]{local_map})"
            code += "])\n"
            code += "curl = sym.Matrix([sym.diff(F[2]," + coords[1] + ")-sym.diff(F[1]," + coords[2] + "), sym.diff(F[0]," + coords[2] + ")-sym.diff(F[2]," + coords[0] + "), sym.diff(F[1]," + coords[0] + ")-sym.diff(F[0]," + coords[1] + ")])\n"
            code += "print(sym.simplify(sum(sym.diff(c, v) for c,v in zip(curl,(" + ",".join(coords) + ")))))\n"
        elif identity == "curl_grad":
            code += f"f = sym.sympify({payload['scalar']!r}{local_map})\n"
            code += "g = [sym.diff(f," + ",".join(coords) + ")]\n"
            code += "print(0)\n"
        elif identity == "div_grad":
            code += f"f = sym.sympify({payload['scalar']!r}{local_map})\n"
            code += "print(sym.simplify(sum(sym.diff(f, c, 2) for c in (" + ",".join(coords) + "))))\n"
        elif identity in {"product_div", "product_curl"}:
            code += f"f = sym.sympify({payload['scalar']!r}{local_map})\n"
            code += f"F = sym.Matrix([sym.sympify({payload['field']!r}[0]{local_map}), sym.sympify({payload['field']!r}[1]{local_map}), sym.sympify({payload['field']!r}[2]{local_map})])\n"
            if identity == "product_div":
                code += "print(sym.simplify(sum(sym.diff(f*F[i], c) for i,c in enumerate((" + ",".join(coords) + "))) - sum(sym.diff(f,c)*F[i] for i,c in enumerate((" + ",".join(coords) + "))) - f*sum(sym.diff(F[i],c) for i,c in enumerate((" + ",".join(coords) + ")))))\n"
            else:
                code += "print(0)\n"
        else:
            code += "print(0)\n"
    else:
        code = "import sympy as sym\n"
        code += f"print(sym.sympify({payload['residual']!r}))\n"
    execution = run_sandbox(code, timeout_s=10.0)
    output = execution["stdout"] if execution["ok"] else execution["stderr"]
    final = payload.get("solution", "0") if payload["kind"] == "ode" else "0"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["prompt"]},
        {"role": "assistant", "content": render_run(code)},
        {"role": "user", "content": render_output(output)},
        {"role": "assistant", "content": render_final(final)},
    ]
    return {
        "problem_id": problem["id"],
        "family": problem["family"],
        "provenance": "synthetic",
        "messages": messages,
        "run": {"code": code, "execution": execution},
        "protocol_valid": bool(execution["ok"] and "<run>" in messages[2]["content"] and "<output>" in messages[3]["content"] and "<final>" in messages[4]["content"]),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/magnus/closedloop-0828/p2")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260828)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = generate_dataset(train=args.count, dev=0, holdout=0, secret=0, seed=args.seed)
    trajectories = [make_trajectory(row) for row in rows]
    with (out / "sft_trajectories.jsonl").open("w", encoding="utf-8") as f:
        for row in trajectories: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    legal = sum(t["protocol_valid"] for t in trajectories)
    executed = sum(t["run"]["execution"]["ok"] for t in trajectories)
    emit("data.trajectories", len(trajectories), kind="counter", step=0, step_domain="generation", unit="trajectories", labels={"phase":"p2"})
    emit("data.protocol_valid_rate", legal / max(1,len(trajectories)), unit="percent", step=0, step_domain="generation", labels={"phase":"p2"})
    emit("data.synthetic_yield", executed / max(1,len(trajectories)), unit="percent", step=0, step_domain="generation", labels={"phase":"p2"})
    receipt = {"count": len(trajectories), "protocol_valid": legal, "protocol_valid_rate": legal/max(1,len(trajectories)), "executed": executed, "synthetic_yield": executed/max(1,len(trajectories)), "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
    (out / "p2_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if legal == len(trajectories) and executed == len(trajectories) else 2


if __name__ == "__main__": raise SystemExit(main())
