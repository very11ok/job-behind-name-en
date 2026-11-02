"""Figure generation: three-criteria signature trends + firm span + three-way clash + rule-family comparison."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(r"D:\harness office\research\jobtask-agent\out")
FIG = BASE / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sig = pd.read_csv(BASE / "results" / "signature_trend.csv")
firm = pd.read_csv(BASE / "results" / "firm_year_stats.csv")
clash = pd.read_csv(BASE / "results" / "clash_2019.csv")
var = pd.read_csv(BASE / "results" / "rules_variants_2019_llm.csv")
varlex = pd.read_csv(BASE / "results" / "rules_variants_2019_lex.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Signature trends
ax = axes[0, 0]
ax.plot(sig["year"], sig["mixed_E0xE2_share"], "o-", color="#c0392b", label="High/low exposure mix share (a)")
ax.plot(sig["year"], sig["size_mean"] / 10, "s--", color="#2980b9", label="Mean tasks per job / 10 (width)")
ax.axvline(2022, color="gray", ls=":", lw=1.5)
ax.text(2022.05, ax.get_ylim()[0] + 0.02, "Generative AI shock", fontsize=9, color="gray")
ax.set_title("(a) Complementarity re-bundling signature: newer jobs are wider and more mixed", fontsize=12)
ax.set_xlabel("Year"); ax.legend(fontsize=9); ax.grid(alpha=.3)

# (2) Firm-level panel span
ax = axes[0, 1]
fy = firm.groupby("year")[["mean_size", "union_per_job", "mean_evar"]].mean()
ax.plot(fy.index, fy["mean_size"], "o-", color="#8e44ad", label="Mean tasks per job within firm")
ax.plot(fy.index, fy["union_per_job"], "s-", color="#16a085", label="Firm task union / number of jobs")
ax.plot(fy.index, fy["mean_evar"] * 10, "^-", color="#e67e22", label="Within-job E variance x10")
ax.set_title("(b) Hierarchical unbundling signature: within-firm span has not narrowed (it widened instead)", fontsize=12)
ax.set_xlabel("Year"); ax.legend(fontsize=9); ax.grid(alpha=.3)

# (3) Three-way clash
ax = axes[1, 0]
x = range(len(clash))
w = 0.35
ax.bar([i - w / 2 for i in x], clash["AUC"] - 0.82, w, color="#34495e", label="AUC-0.82")
ax.bar([i + w / 2 for i in x], clash["lift@1%"] / 40, w, color="#e74c3c", label="lift@1% / 40")
ax.set_xticks(list(x)); ax.set_xticklabels(clash["model"], fontsize=10)
for i, r in clash.iterrows():
    ax.text(i - w / 2, r["AUC"] - 0.82 + 0.0008, f"{r['AUC']:.4f}", ha="center", fontsize=8)
    ax.text(i + w / 2, r["lift@1%"] / 40 + 0.0008, f"{r['lift@1%']:.2f}", ha="center", fontsize=8)
ax.set_title("Three-criteria semantics vs graph-only baseline: only LLM judgments add lift", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")

# (4) Rule-family comparison
ax = axes[1, 1]
# NOTE: score keys kept verbatim (matched against the `score` column of rules_variants_*.csv; functional data)
key = ["B4 = mod×sameE×coh", "B1 = mod×hom×cod×coh", "B5 = sameE×coh",
       "A2 = 1[E2×E0]×ctx_low", "A1 = gap×ctx_low"]
llm = var.set_index("score").reindex(key)
lex = varlex.set_index("score").reindex(key)
idx = range(len(key))
ax.barh([i + 0.18 for i in idx], llm["AUC"], 0.34, color="#2980b9", label="LLM judgment")
ax.barh([i - 0.18 for i in idx], lex["AUC"], 0.34, color="#95a5a6", label="Lexicon (control)")
ax.axvline(0.5, color="red", ls="--", lw=1)
ax.set_yticks(list(idx)); ax.set_yticklabels(key, fontsize=9)
ax.set_xlim(0.45, 0.60)
ax.set_title("Rule-family pairwise AUC: (b) works, (a) has no signal", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=.3, axis="x")

plt.tight_layout()
plt.savefig(FIG / "summary.png", dpi=160, bbox_inches="tight")
print("saved", FIG / "summary.png")
