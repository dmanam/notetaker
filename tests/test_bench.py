"""The benchmark scorer, and the shipped ground truth's self-consistency."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from bench_diagrams import score_case, summarise
from diagrams import normalise, normalise_style

# --- normalisation: two correct diagrams must not differ --------------------
assert normalise(r"$M_\infty$") == normalise(r"M_\infty") == normalise(r"{M_\infty}")
assert normalise(r"\mathrm{Pro}_{\mathbb N}(\mathrm{Fin})") == \
       normalise(r"\operatorname{Pro}_\mathbb{N}(Fin)")
assert normalise(r"\varprojlim_n S_n") == normalise(r"\lim_n S_n")
assert normalise(r"\widetilde N") == normalise(r"\tilde N")
# a node written "S \in Pro(Fin)" names the object S
assert normalise(r"S \in \mathrm{Pro}_{\mathbb N}(Fin)") == normalise("S")
assert normalise(r"\mathrm{Pro}_{\mathbb N}(Fin) \ni S") == normalise("S")
# \infty must survive: it starts with the letters of \in
assert normalise(r"M_\infty") == r"m_\infty", normalise(r"M_\infty")
# trailing sentence punctuation is not part of the name
assert normalise(r"\mathbb N\cup\{\infty\} .") == normalise(r"\mathbb N \cup \{\infty\}")
# \{ \} must leave no stray backslash
assert normalise(r"\mathbb N \cup \{\infty\}") == r"n\cup\infty"
# genuinely different objects stay different — the prime is significant
assert normalise("M_0") != normalise("M_1")
assert normalise("S'") != normalise("S")
assert normalise("S'") == normalise("$S'$")
assert normalise("") == normalise(None) == ""
print("normalisation: cosmetic differences collapse, real ones do not")

assert normalise_style(["two heads"]) == normalise_style(["twoheadrightarrow"])
assert normalise_style(["hook"]) == "mono"
assert normalise_style(["bend left=20"]) == "", "layout is not a style"
assert normalise_style(["dashed"]) == normalise_style(["dotted"])
print("style aliases collapse; layout options ignored")

# --- scoring ----------------------------------------------------------------
TRUTH = {
    "nodes": [r"M_\infty", "M_2", "M_1", "M_0",
              r"S_\infty", "S_2", "S_1", r"S \in \mathrm{Pro}_\mathbb{N}(Fin)"],
    "arrows": [
        {"from": r"S_\infty", "to": r"M_\infty", "style": ["dashed"]},
        {"from": "S_2", "to": "M_2", "style": ["dashed"]},
        {"from": "S_1", "to": "M_1", "style": ["dashed"]},
        {"from": r"S \in \mathrm{Pro}_\mathbb{N}(Fin)", "to": "M_0", "style": []},
        {"from": "M_1", "to": "M_0", "style": ["two heads"]},
    ],
}
case = {"id": "l3-b11", "board": 11, "truth": TRUTH}

# What the pipeline actually produced: right directions, S dropped, the given
# map re-hung on S_1.
PRODUCED = r"""\begin{tikzcd}
M_\infty \arrow[r] & \cdots \arrow[r] & M_2 \arrow[r] & M_1 \arrow[r, two heads] & M_0 \\
S_\infty \arrow[u, dashed, "\exists?"] \arrow[r] & \cdots \arrow[r] & S_2 \arrow[u, dashed] \arrow[r] & S_1 \arrow[u, dashed] \arrow[ur] & {}
\end{tikzcd}"""
r = score_case(case, PRODUCED)
assert r["nodes_missing"] == ["s"], r["nodes_missing"]   # the object is S
assert r["arrows_missing"] and r["lint"]
print(f"real output: {len(r['nodes_missing'])} missing node, "
      f"{len(r['arrows_missing'])} missing arrow, {len(r['lint'])} lint")

# A correct diagram scores clean whatever cosmetic choices it makes.
CORRECT = r"""\begin{tikzcd}
M_\infty \arrow[r] & M_2 \arrow[r] & M_1 \arrow[r, twoheadrightarrow] & M_0 \\
S_\infty \arrow[u, dotted] & S_2 \arrow[u, dotted] \arrow[l] & S_1 \arrow[u, dotted] & {\operatorname{Pro}_\mathbb{N}(Fin) \ni S} \arrow[u]
\end{tikzcd}"""
r2 = score_case(case, CORRECT)
assert not r2["nodes_missing"] and not r2["arrows_missing"], r2
print("a correct diagram in different spelling scores clean")

# A reversed arrow is named as such, not merely counted missing + invented.
FLIPPED = r"""\begin{tikzcd}
M_\infty \arrow[d, dotted] \arrow[r] & M_2 \arrow[r] & M_1 \arrow[r, twoheadrightarrow] & M_0 \\
S_\infty & S_2 \arrow[u, dotted] & S_1 \arrow[u, dotted] & {\operatorname{Pro}_\mathbb{N}(Fin) \ni S} \arrow[u]
\end{tikzcd}"""
r3 = score_case(case, FLIPPED)
assert r3["reversed"], r3
assert "infty" in r3["reversed"][0], f"\\infty truncated: {r3['reversed']}"
print(f"a reversed arrow is named: {r3['reversed']}")

# An empty answer is a total miss, not a crash.
r4 = score_case(case, "")
assert len(r4["nodes_missing"]) == r4["n_nodes"] > 0
assert not r4["attempted"]

# A board holds several diagrams; the run may draw more than one. The best
# match wins, so the case is not scored against a different diagram.
r5 = score_case(case, ["\\begin{tikzcd}X \\arrow[r] & Y\\end{tikzcd}", CORRECT])
assert not r5["nodes_missing"], "best match should have been picked"
assert score_case(case, [])["n_nodes"] == r4["n_nodes"]
print("best-match selection across several produced diagrams")

# "exact" demands COMPLETE ground truth — an arrow the truth omits reads as
# invented, which is the point.
SQUARE = {"id": "sq", "board": 1, "truth": {
    "nodes": ["A", "B", "C", "D"],
    "arrows": [{"from": "A", "to": "B", "style": []},
               {"from": "A", "to": "C", "style": ["hook"]},
               {"from": "B", "to": "D", "style": ["two heads"]},
               {"from": "C", "to": "D", "style": ["dashed"]}]}}
exact = score_case(SQUARE, r"""\begin{tikzcd}
A \arrow[r, "f"] \arrow[d, hook] & B \arrow[d, twoheadrightarrow] \\
C \arrow[r, dotted] & D
\end{tikzcd}""")
assert not any(exact[k] for k in ("nodes_missing", "nodes_extra",
                                  "arrows_missing", "arrows_extra",
                                  "style_mismatch")), exact
assert exact["attempted"]
print("a complete case scores exact")

# A style read wrong is a lesser error than an arrow that is not there.
styled = score_case(SQUARE, r"""\begin{tikzcd}
A \arrow[r] \arrow[d, hook] & B \arrow[d] \\ C \arrow[r, dotted] & D
\end{tikzcd}""")
assert styled["style_mismatch"] and not styled["arrows_missing"], styled
print("style mismatch is reported apart from missing arrows")

s = summarise([r, r2, r3, r4, exact])
# Only SQUARE is exact: this TRUTH lists five arrows of board 11 rather than
# all of them, so even the correct diagram registers the rest as invented.
# That is the rule working, and it is why shipped ground truth is complete.
assert s["cases"] == 5 and s["exact"] == 1, s
assert 0 < s["node_recall"] < 1 and s["reversed_arrows"] >= 1
print(f"summary: {s}")

# --- the shipped ground truth ------------------------------------------------
cases = json.loads((REPO / "bench" / "cases.json").read_text())["cases"]
for c in cases:
    tr = c["truth"]
    names = [normalise(n) for n in tr["nodes"]]
    assert len(names) == len(set(names)), f"{c['id']}: nodes collide: {names}"
    known = set(names)
    for a in tr["arrows"] + tr.get("uncertain", []):
        for end in ("from", "to"):
            assert normalise(a[end]) in known, \
                f"{c['id']}: arrow {end} {a[end]!r} is not one of the nodes"
    assert tr["arrows"], f"{c['id']} has no arrows"
print(f"{len(cases)} ground-truth cases are internally consistent")

print("\nALL OK")
