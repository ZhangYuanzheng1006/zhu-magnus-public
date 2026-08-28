"""
0828-verify 任务 3 数据构造:200 条 06 协议格式冒烟数据。
- 消息结构严格按 docs/0827/06 §2:system(逐字复制 §5)→ user(题目)→ assistant(<run>代码</run>)→ user(<output>结果</output>)→ assistant(<final>答案</final>)
- 题目三类:求导 / 多项式展开化简 / 一阶方程求解,各约 70 条,每条 1–2 个 run 步
- 只对 assistant 段计 loss(SFT 时 completion-only mask)

用法:python 0828_make_smoke_data.py --out /data/magnus/smoke-0828/student/data.jsonl
"""
import argparse, json, random, hashlib, os, sympy as sym

# 06 §5 系统提示词全文(v1.0-rc1,中文版)——逐字复制
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
lhs = (xs + xs*0 + xs)**2  # 这里用同一组点比较两个表达式
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
    return f"<output>\n{text}\n</output>"


def render_final(text: str) -> str:
    return f"<final>\n{text}\n</final>"


def make_derivative(seed: int):
    """求导类:随机多项式/三角/指数组合,问导数。"""
    rng = random.Random(seed)
    x = sym.symbols("x")
    terms = []
    for _ in range(rng.randint(2, 3)):
        kind = rng.choice(["poly", "trig", "exp"])
        coeff = rng.randint(1, 5)
        if kind == "poly":
            deg = rng.randint(2, 4)
            terms.append(coeff * x ** deg)
        elif kind == "trig":
            terms.append(coeff * sym.sin(x) if rng.random() < 0.5 else coeff * sym.cos(x))
        else:
            terms.append(coeff * sym.exp(x))
    f = sum(terms)
    df = sym.diff(f, x)
    prompt = f"求函数 f(x) = {sym.sstr(f)} 的导数 f'(x),并化简。"
    code = (
        "x = sym.symbols('x')\n"
        f"f = {sym.srepr(f)}\n"
        "df = sym.diff(f, x)\n"
        "print(sym.simplify(df))\n"
    )
    output = str(sym.simplify(df))
    final = f"f'(x) = {sym.sstr(sym.simplify(df))}。"
    return prompt, [("run", code, output)], final


def make_expand(seed: int):
    """多项式展开/化简类。"""
    rng = random.Random(seed)
    x, y = sym.symbols("x y")
    # (ax + by)^n 或 (x+1)(x+c) 等
    form = rng.choice(["power", "product"])
    if form == "power":
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        n = rng.randint(2, 3)
        expr = (a * x + b * y) ** n
        prompt = f"展开并化简 ({a}*x + {b}*y)^{n},将结果写成标准多项式形式。"
    else:
        c = rng.randint(1, 4)
        expr = (x + 1) * (x + c)
        prompt = f"展开并化简 (x+1)(x+{c}),将结果写成标准多项式形式。"
    expanded = sym.expand(expr)
    code = (
        "x, y = sym.symbols('x y')\n"
        f"expr = {sym.srepr(expr)}\n"
        "print(sym.expand(expr))\n"
    )
    output = str(expanded)
    final = f"展开结果为 {sym.sstr(expanded)}。"
    return prompt, [("run", code, output)], final


def make_ode(seed: int):
    """一阶方程求解类(y' + p(x)y = q(x) 或可分离)。"""
    rng = random.Random(seed)
    x = sym.symbols("x")
    y = sym.Function("y")
    kind = rng.choice(["linear", "separable"])
    if kind == "linear":
        p = rng.choice([2, -1, 1])
        q = sym.exp(x) if rng.random() < 0.5 else x
        ode = sym.Eq(y(x).diff(x) + p * y(x), q)
        prompt = f"求解一阶线性微分方程 y' + {p} y = {sym.sstr(q)},给出通解。"
    else:
        # 可分离:y' = k * x^n * y^m,取 m=1 使解为 exp 形式
        k = rng.choice([1, -2, 3])
        n = rng.choice([1, 2])
        ode = sym.Eq(y(x).diff(x), k * x ** n * y(x))
        prompt = f"求解可分离微分方程 y' = {k} x^{n} y,给出通解。"
    sol = sym.dsolve(ode, y(x))
    code = (
        "x = sym.symbols('x')\n"
        "y = sym.Function('y')\n"
        f"ode = {sym.srepr(ode)}\n"
        "print(sym.dsolve(ode, y(x)))\n"
    )
    output = str(sol)
    final = f"通解为 {sym.sstr(sol.rhs)}。"
    return prompt, [("run", code, output)], final


GENERATORS = {
    "derivative": make_derivative,
    "expand": make_expand,
    "ode": make_ode,
}


def build_messages(prompt: str, steps, final: str):
    """06 协议消息序列:system → user(题目) → assistant(run) → user(output) → assistant(final)。"""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for kind, code, output in steps:
        msgs.append({"role": "assistant", "content": render_run(code)})
        msgs.append({"role": "user", "content": render_output(output)})
    msgs.append({"role": "assistant", "content": render_final(final)})
    return msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/magnus/smoke-0828/student/data.jsonl")
    p.add_argument("--count", type=int, default=200)
    args = p.parse_args()

    sys_prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    print("SYSTEM_PROMPT SHA256:", sys_prompt_hash)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    per = args.count // 3
    rows = []
    for name, gen in GENERATORS.items():
        for i in range(per):
            seed = 1000 + hash(name) % 1000 * 100 + i
            prompt, steps, final = gen(seed)
            msgs = build_messages(prompt, steps, final)
            rows.append({
                "id": f"smoke-{name}-{i:03d}",
                "category": name,
                "messages": msgs,
                "system_prompt_sha256": sys_prompt_hash,
            })

    # 补齐到 count
    i = 0
    while len(rows) < args.count:
        name = ["derivative", "expand", "ode"][i % 3]
        seed = 9000 + i
        prompt, steps, final = GENERATORS[name](seed)
        msgs = build_messages(prompt, steps, final)
        rows.append({
            "id": f"smoke-{name}-extra-{i:03d}",
            "category": name,
            "messages": msgs,
            "system_prompt_sha256": sys_prompt_hash,
        })
        i += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")
    from collections import Counter
    print("category dist:", dict(Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()
