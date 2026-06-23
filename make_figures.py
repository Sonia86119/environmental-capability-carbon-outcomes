#!/usr/bin/env python3
"""
Regenerate the three paper figures at publication quality (300 dpi) from the source data.

  figure1_analytical_design.png   Analytical design and interpretation boundaries
  figure2_coefficient_plot.png    Capability-score coefficients with 95% CIs (Table 3 M1-M4)
  figure3_quartile_crossclass.png Cross-classification of capability and Scope 1+2 intensity quartiles

USAGE:
  python make_figures.py --data-dir data --out-dir figures
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reproduce_paper3 import build, ols  # reuse the validated pipeline

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})
ACCENT, MUTED = "#1f6fb2", "#9aa0a6"

def _save(fig, base):
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")  # vector
    plt.close(fig)



def fig1(out):
    # Clean 4-stage flow diagram; explanatory text belongs in the caption, not the figure.
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12.0, 3.0))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    boxes = ["Resource-use and\nenvironmental-\ninnovation scores",
             "Disclosure\nselection\n(Scope 1+2,\nScope 3)",
             "Carbon outcomes\n(operational and\nvalue-chain;\ntotal, intensity)",
             "Evidence claim\n(alignment, weak\nalignment, or\ndivergence)"]
    w, h, y = 20.5, 64, 18
    xs = [2.0, 26.5, 51.0, 75.5]
    for x, txt in zip(xs, boxes):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.5",
                                    facecolor="#eef2f7", edgecolor="#33475b", lw=1.4))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11.0)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + w + 3.7, y + h / 2), xytext=(x + w + 0.6, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#33475b"))
    _save(fig, out)


def fig2(df, out):
    specs = [("M1 Scope 1+2 total", "ln_co2_total"),
             ("M2 Scope 1+2 intensity", "ln_co2_per_assets"),
             ("M3 Scope 3 total", "ln_co2_s3"),
             ("M4 Scope 3 intensity", "ln_co2_s3_assets")]
    rows = []
    for lab, dep in specs:
        m, _ = ols(df, dep, ["capability", "ln_mcap"])
        b, se = m.params["capability"], m.bse["capability"]
        rows.append((lab, b, 1.96 * se, (b - 1.96 * se > 0) or (b + 1.96 * se < 0)))
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ys = list(range(len(rows)))[::-1]  # M1 at top
    for y, (lab, b, ci, sig) in zip(ys, rows):
        c = ACCENT if sig else MUTED
        ax.errorbar(b, y, xerr=ci, fmt="o", ms=8, capsize=4, lw=1.8, color=c, ecolor=c)
    ax.axvline(0, color="#444", lw=1.0, ls="--")
    ax.set_yticks(ys[::-1]); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Capability-score coefficient (95% confidence interval)")
    ax.set_title("Capability-score association by carbon outcome", fontsize=14, pad=10)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.margins(y=0.18)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color=ACCENT, lw=0, label="95% CI excludes 0"),
                       Line2D([0], [0], marker="o", color=MUTED, lw=0, label="not significant")],
              loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout(); _save(fig, out)


def fig3(df, out):
    d = df.dropna(subset=["capability", "ln_co2_per_assets"]).copy()
    for v in ["capability", "ln_co2_per_assets"]:
        d[v] = d[v].clip(d[v].quantile(.01), d[v].quantile(.99))
    qc = pd.qcut(d["capability"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    qi = pd.qcut(d["ln_co2_per_assets"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    ct = pd.crosstab(qc, qi)
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    im = ax.imshow(ct.values, cmap="Blues", vmin=ct.values.min() - 15, vmax=ct.values.max() + 5)
    ax.set_xticks(range(4)); ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(4)); ax.set_yticklabels(ct.index)
    ax.set_xlabel("Scope 1+2 intensity quartile"); ax.set_ylabel("Capability-score quartile")
    for i in range(4):
        for j in range(4):
            v = ct.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > ct.values.mean() + 8 else "black", fontsize=13)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Firm count")
    ax.set_title("Capability vs Scope 1+2 intensity quartiles\n"
                 r"$\chi^2$ = 11.69, df = 9, p = 0.231; Cramer's V = 0.052", fontsize=12.5, pad=10)
    fig.tight_layout(); _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = build(a.data_dir)
    fig1(os.path.join(a.out_dir, "figure1_analytical_design"))
    fig2(df, os.path.join(a.out_dir, "figure2_coefficient_plot"))
    fig3(df, os.path.join(a.out_dir, "figure3_quartile_crossclass"))
    print("wrote 3 figures (PNG + PDF) to", a.out_dir)


if __name__ == "__main__":
    main()
