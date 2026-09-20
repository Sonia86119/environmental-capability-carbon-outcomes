"""Draw the paper's figures from the screening record and the results tables.

Input:
    data/sample_flow.csv
    results/t05_denominators.csv
    results/t06_components.csv
    results/t07_industry.csv
    results/t09_lagged.csv

Output (../paper3_overleaf/CSRM/source/figures/), each as a vector PDF, a
300 dpi PNG and a caption in a companion text file:
    figure2_sample_flow      screening steps from the raw extract to each sample
    figure3_denominators     the EMI score under the five intensity denominators
    figure4_components       the components of the measure against revenue intensity
    figure5_lagged           the 2021 score, and its change, against 2025 outcomes
    figure6_heterogeneity    the carbon-intensive split and the two interactions

Figure 1, the conceptual model, is drawn in TikZ and is not produced here.

Estimates are read from the results tables rather than re-estimated, so a
figure cannot disagree with the table it accompanies. Confidence intervals
are the coefficient plus and minus 1.96 standard errors, the standard errors
being clustered by country group as everywhere else in the paper. Series are
separated by marker and line style rather than by colour, so the figures
survive greyscale printing.

    python make_figures.py
    python make_figures.py --data-dir synthetic_data --results-dir synthetic_results \
        --figures-dir synthetic_results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

DATA = Path("data")
RESULTS = Path("results")
FIGURES = Path("..") / "paper3_overleaf" / "CSRM" / "source" / "figures"

WIDTH = 6.5                       # one journal column, used at \linewidth
INK = "0.15"
CRITICAL = 1.96

# Denominators in the order the results section discusses them: the three
# financial scaling variables first, then the two physical ones, so that the
# physical pair reads as one block.
DENOMINATORS = [("revenue", "Revenue", "financial"),
                ("assets", "Total assets", "financial"),
                ("ev", "Enterprise value", "financial"),
                ("employees", "Employees", "physical"),
                ("energy", "Energy use", "physical")]

MEASURES = [("rating", "EMI score (equal weights)"),
            ("rating_resource_use", "Resource-use theme"),
            ("rating_innovation", "Environmental-innovation theme"),
            ("rating_emissions_theme", "Emissions theme"),
            ("rating_three_theme", "Three-theme mean"),
            ("rating_pillar", "Environmental pillar"),
            ("rating_esg", "Full ESG score")]

OUTCOMES = {"ln_scope12": "Log Scope 1+2",
            "ln_scope3": "Log Scope 3",
            "ln_scope12_per_revenue": "Log Scope 1+2 / revenue",
            "ln_scope12_per_assets": "Log Scope 1+2 / assets",
            "ln_scope12_per_employees": "Log Scope 1+2 / employees",
            "ln_scope12_per_energy": "Log Scope 1+2 / energy"}

CAPTIONS = {
    "figure2_sample_flow":
        "Sample construction. The chain records every screening step from the "
        "raw extract to each estimation sample, with the firms removed at each "
        "step shown alongside. The five boxes at the foot are the intensity "
        "samples, each requiring Scope 1+2 emissions and the denominator named; "
        "the branch on the right is the lagged design, which matches firms "
        "rated in 2021 to a 2025 observation. Counts are firms unless stated "
        "otherwise.",
    "figure3_denominators":
        "The EMI score under five intensity denominators. Each estimate comes "
        "from a separate regression of log Scope 1+2 emissions per unit of the "
        "denominator on the EMI score and log market capitalisation, with "
        "country-group and sector fixed effects and standard errors clustered "
        "by country group; bars are 95 per cent confidence intervals. "
        "Sector-demeaned outcomes subtract the median of that intensity within "
        "the firm's sector and year. The shaded band holds the two physical "
        "denominators, and the number of firms in each regression is given on "
        "the right.",
    "figure4_components":
        "Alternative constructions of the measure against revenue intensity. "
        "Each estimate replaces the EMI score with the named measure in the "
        "intensity model of panel (a), holding the controls, fixed effects and "
        "clustering fixed; bars are 95 per cent confidence intervals on 1,554 "
        "firms. The emissions theme, marked by an open triangle, is the one "
        "component that carries the opposite sign. The first principal "
        "component of the two standardised themes is shown separately in panel "
        "(b) because it is on a standardised rather than a 0--100 scale.",
    "figure5_lagged":
        "The 2021 EMI score against 2025 outcomes. Panel (a) regresses each "
        "2025 outcome on the 2021 score with log market capitalisation and "
        "country-group and sector fixed effects; panel (b) adds the 2021--2025 "
        "change in the score to the 2021 level and plots the coefficient on the "
        "change term for every outcome. Bars are 95 per cent confidence intervals from standard "
        "errors clustered by country group, and the number of firms in each "
        "regression is given on the right.",
    "figure6_heterogeneity":
        "Industry heterogeneity. The split estimates the main model separately "
        "within the carbon-intensive sectors (Energy, Utilities, Basic "
        "Materials) and within all others, with country-group and sector fixed "
        "effects. The interaction rows retain both country-group and sector "
        "fixed effects, which absorb the level of the moderator but not its "
        "interaction with a score that varies within sectors; sector carbon "
        "intensity is the median log Scope 1+2 per unit of revenue within the "
        "firm's sector and year, computed without the firm itself. Bars are 95 per cent confidence intervals from "
        "standard errors clustered by country group, and the number of firms in "
        "each regression is given on the right."}


def configure() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "legend.frameon": False,
        "savefig.dpi": 300})


def save(fig: plt.Figure, name: str) -> None:
    """Write the vector copy, the raster copy and the caption together."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / ("%s.%s" % (name, suffix)),
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    (FIGURES / ("%s.txt" % name)).write_text(CAPTIONS[name] + "\n")
    print("  %-26s pdf, png, txt" % name)


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------

def box(ax, x, y, width, height, text, face="white", edge="solid",
        size=7.2) -> None:
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=face, edgecolor=INK, linewidth=0.8,
                                linestyle=edge))
    ax.text(x + width / 2.0, y + height / 2.0, text, ha="center", va="center",
            fontsize=size, linespacing=1.35)


def arrow(ax, start, end) -> None:
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.8,
                                shrinkA=0, shrinkB=0))


def estimate(ax, y, coef, se, marker="o", face=INK, style="-") -> None:
    """One point estimate with its 95 per cent interval, drawn at height y."""
    half = CRITICAL * se
    ax.plot([coef - half, coef + half], [y, y], linestyle=style, color=INK,
            linewidth=0.9, zorder=3)
    for end in (coef - half, coef + half):
        ax.plot([end, end], [y - 0.13, y + 0.13], linestyle="-", color=INK,
                linewidth=0.9, zorder=3)
    ax.plot([coef], [y], marker=marker, markersize=4.4, linestyle="none",
            color=INK, markerfacecolor=face, markeredgewidth=0.9, zorder=4)


def sample_sizes(ax, positions, counts, header="N") -> None:
    """Row counts printed against the right-hand edge of the plotting area."""
    transform = ax.get_yaxis_transform()
    for y, n in zip(positions, counts):
        ax.text(1.02, y, "{:,}".format(int(n)), transform=transform,
                va="center", ha="left", fontsize=7, clip_on=False)
    if header:
        ax.text(1.02, max(positions) + 0.75, header, transform=transform,
                va="center", ha="left", fontsize=7, style="italic", clip_on=False)


def coefficient_axes(ax, label="EMI score coefficient (95% CI)",
                     zero_width=0.8) -> None:
    ax.axvline(0.0, color=INK, linewidth=zero_width, linestyle=(0, (4, 2)),
               zorder=1)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="0.75")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.set_xlabel(label)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def figure2_sample_flow() -> None:
    flow = pd.read_csv(DATA / "sample_flow.csv")
    flow["step"] = flow["step"].str.strip()
    firms = dict(zip(flow["step"], flow["firms"]))
    years = dict(zip(flow["step"], flow["firm_years"]))
    count = lambda step: "{:,}".format(int(firms[step]))
    span = lambda step: "{:,}".format(int(years[step]))

    fig, ax = plt.subplots(figsize=(WIDTH, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(4, 100)
    ax.axis("off")

    left, width = 26.0, 36.0
    centre = left + width / 2.0
    height = 9.0
    chain = [
        (88.0, "Raw extract\n%s instruments, %s firm-years"
         % (count("Raw universe"), span("Raw universe"))),
        (74.0, "Operating firms\n%s" % count("Drop non-operating instruments")),
        (60.0, "Analysis window 2021–2026\n%s firms, %s firm-years"
         % (count("Resolve duplicate firm-years"),
            span("Resolve duplicate firm-years"))),
        (46.0, "Rated cross-section\n%s firms"
         % count("Cross-section: scored firms, latest year")),
        (32.0, "Scope 1+2 reported\n%s firms" % count("with Scope 1+2 emissions"))]

    for y, text in chain:
        box(ax, left, y, width, height, text)
    for (upper, _), (lower, _) in zip(chain, chain[1:]):
        arrow(ax, (centre, upper), (centre, lower + height))

    notes = [(83.0, "747 funds and index\nproducts removed"),
             (69.0, "121 rows outside the\nwindow; 6 duplicate\nfirm-years resolved"),
             (55.0, "both environmental\ntheme scores required"),
             (41.0, "both emission\ncomponents present")]
    for y, text in notes:
        ax.text(left - 3.0, y, text, ha="right", va="center", fontsize=6.6,
                style="italic", color="0.35", linespacing=1.35)

    # The five intensity samples fan out from the Scope 1+2 box.
    bus = 24.0
    arrow(ax, (centre, 32.0), (centre, bus + 0.6))
    panel_width, gap = 18.4, 2.0
    starts = [i * (panel_width + gap) for i in range(len(DENOMINATORS))]
    ax.plot([starts[0] + panel_width / 2.0, starts[-1] + panel_width / 2.0],
            [bus, bus], color=INK, linewidth=0.8)
    for start, (key, label, _) in zip(starts, DENOMINATORS):
        middle = start + panel_width / 2.0
        arrow(ax, (middle, bus), (middle, 20.0))
        box(ax, start, 11.0, panel_width, 9.0,
            "%s\n%s firms" % (label, count("with Scope 1+2 and %s" % key)),
            face="0.93", size=6.8)

    # The lagged design is a branch off the in-window sample, not of the
    # cross-section, since it matches on year rather than on rating recency.
    branch_left, branch_width = 68.0, 32.0
    box(ax, branch_left, 60.0, branch_width, 11.0,
        "Lagged sample\n2021 rating, 2025 outcome\n%s firms"
        % count("Lagged sample: 2021 rating to 2025 outcome"),
        edge=(0, (4, 2)))
    box(ax, branch_left, 44.0, branch_width, 11.0,
        "With a later\nScope 1+2 figure\n%s firms"
        % count("with a later Scope 1+2 figure"), edge=(0, (4, 2)))
    arrow(ax, (left + width, 65.5), (branch_left, 65.5))
    arrow(ax, (branch_left + branch_width / 2.0, 60.0),
          (branch_left + branch_width / 2.0, 55.0))

    save(fig, "figure2_sample_flow")


def figure3_denominators() -> None:
    rows = pd.read_csv(RESULTS / "t05_denominators.csv")
    rows = rows[rows["fixed_effects"] == "country_group, sector"].set_index("outcome")

    fig, ax = plt.subplots(figsize=(WIDTH, 4.5))
    positions, counts, ticks, labels, band = [], [], [], [], []
    y = 0.0
    for key, label, kind in DENOMINATORS:
        raw = rows.loc["ln_scope12_per_%s" % key]
        demeaned = rows.loc["ln_scope12_per_%s_sector_demeaned" % key]
        estimate(ax, y, raw["coef"], raw["se"], marker="o", face=INK, style="-")
        estimate(ax, y - 0.9, demeaned["coef"], demeaned["se"], marker="s",
                 face="white", style=(0, (4, 2)))
        if kind == "physical":
            band.append(y)
        positions += [y, y - 0.9]
        counts += [raw["n"], demeaned["n"]]
        ticks.append(y - 0.45)
        labels.append(label)
        y -= 2.2

    ax.axhspan(min(band) - 1.35, max(band) + 0.5, facecolor="0.92",
               edgecolor="none", zorder=0)
    ax.text(0.015, max(band) - 1.55, "physical denominators",
            transform=ax.get_yaxis_transform(), fontsize=7, style="italic",
            color="0.35", va="center")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y + 0.75, 1.0)
    coefficient_axes(ax)
    sample_sizes(ax, positions, counts)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=INK, markerfacecolor=INK,
               linestyle="-", linewidth=0.9, markersize=4.4, label="Raw intensity"),
        Line2D([], [], marker="s", color=INK, markerfacecolor="white",
               linestyle=(0, (4, 2)), linewidth=0.9, markersize=4.4,
               label="Sector-demeaned"),
        ],
        loc="upper right", fontsize=7.2, handlelength=2.4)
    save(fig, "figure3_denominators")


def figure4_components() -> None:
    rows = pd.read_csv(RESULTS / "t06_components.csv")
    rows = rows[(rows["outcome"] == "ln_scope12_per_revenue")
                & (rows["controls"] == "ln_mcap")].set_index("predictor")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.4), sharey=True,
                             gridspec_kw={"width_ratios": [2.7, 1.0],
                                          "wspace": 0.08})
    main, inset = axes
    positions = list(range(len(MEASURES) - 1, -1, -1))

    for y, (key, label) in zip(positions, MEASURES):
        row = rows.loc[key]
        emissions = key == "rating_emissions_theme"
        estimate(main, y, row["coef"], row["se"],
                 marker="v" if emissions else "o",
                 face="white" if emissions else INK)
        if emissions:
            main.text(row["coef"] + CRITICAL * row["se"] + 0.0006, y,
                      "opposite sign", fontsize=7, style="italic",
                      color="0.35", va="center")

    component = rows.loc["rating_pca"]
    estimate(inset, -1.0, component["coef"], component["se"], marker="D",
             face="white")

    main.set_yticks(positions + [-1.0])
    main.set_yticklabels([label for _, label in MEASURES]
                         + ["First principal component"])
    main.set_ylim(-1.9, len(MEASURES) - 0.4)
    coefficient_axes(main, "Coefficient on the 0–100 measure (95% CI)")
    coefficient_axes(inset, "Standardised scale (95% CI)")
    inset.tick_params(left=False)
    for axis in (main, inset):
        axis.axhline(-0.5, color="0.6", linewidth=0.6, linestyle=(0, (2, 2)),
                     zorder=1)
    main.text(0.0, 1.02, "(a)", transform=main.transAxes, va="bottom", fontsize=8)
    inset.text(0.0, 1.02, "(b)", transform=inset.transAxes, va="bottom", fontsize=8)
    inset.xaxis.set_major_locator(plt.MaxNLocator(3))
    save(fig, "figure4_components")


def figure5_lagged() -> None:
    rows = pd.read_csv(RESULTS / "t09_lagged.csv")
    dated = rows["controls"].fillna("").str.contains("dated")
    level = rows[(rows["predictor"] == "rating_base") & ~dated].set_index("outcome")
    change = rows[rows["predictor"] == "rating_change"].set_index("outcome")

    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.6),
                             gridspec_kw={"height_ratios": [3.0, 3.0],
                                          "hspace": 0.40})
    upper, lower = axes

    order = list(OUTCOMES)
    positions = list(range(len(order) - 1, -1, -1))
    for y, outcome in zip(positions, order):
        row = level.loc[outcome]
        estimate(upper, y, row["coef"], row["se"])
    upper.set_yticks(positions)
    upper.set_yticklabels([OUTCOMES[o] for o in order])
    upper.set_ylim(-0.7, len(order) - 0.3)
    coefficient_axes(upper, "2021 EMI score coefficient (95% CI)")
    sample_sizes(upper, positions, [level.loc[o, "n"] for o in order])
    upper.text(0.0, 1.03, "(a)", transform=upper.transAxes, va="bottom",
               fontsize=8)

    changed = [o for o in OUTCOMES if o in change.index]
    positions = list(range(len(changed) - 1, -1, -1))
    for y, outcome in zip(positions, changed):
        row = change.loc[outcome]
        estimate(lower, y, row["coef"], row["se"], marker="s", face="white",
                 style=(0, (4, 2)))
    lower.set_yticks(positions)
    lower.set_yticklabels([OUTCOMES[o] for o in changed])
    lower.set_ylim(-0.7, len(changed) - 0.3)
    coefficient_axes(lower, "Coefficient on the 2021–2025 change (95% CI)",
                     zero_width=1.3)
    sample_sizes(lower, positions, [change.loc[o, "n"] for o in changed], header="")
    lower.text(0.0, 1.06, "(b)", transform=lower.transAxes, va="bottom",
               fontsize=8)

    save(fig, "figure5_lagged")


def figure6_heterogeneity() -> None:
    rows = pd.read_csv(RESULTS / "t07_industry.csv")
    controls = rows["controls"].fillna("")

    specifications = [
        ("Carbon-intensive sectors only",
         (rows["predictor"] == "rating") & controls.str.contains("carbon intensive")),
        ("All other sectors",
         (rows["predictor"] == "rating") & controls.str.contains("other sectors")),
        ("EMI score × carbon-intensive",
         (rows["predictor"] == "rating_x_intensive")
         & controls.str.contains("sector effects")),
        ("EMI score × sector carbon intensity",
         (rows["predictor"] == "rating_x_sector_intensity")
         & controls.str.contains("sector effects"))]
    series = [("ln_scope12", "Log Scope 1+2", "o", INK, "-"),
              ("ln_scope12_per_revenue", "Log Scope 1+2 / revenue", "s", "white",
               (0, (4, 2)))]

    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    ticks, labels, positions, counts = [], [], [], []
    y = float(len(specifications) - 1)
    for label, selected in specifications:
        for offset, (outcome, _, marker, face, style) in zip((0.19, -0.19), series):
            row = rows[selected & (rows["outcome"] == outcome)].iloc[0]
            estimate(ax, y + offset, row["coef"], row["se"], marker=marker,
                     face=face, style=style)
            positions.append(y + offset)
            counts.append(row["n"])
        ticks.append(y)
        labels.append(label)
        y -= 1.0

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(specifications) - 0.4)
    coefficient_axes(ax, "Coefficient on the term named at the left (95% CI)")
    sample_sizes(ax, positions, counts)
    ax.legend(handles=[
        Line2D([], [], marker=marker, color=INK, markerfacecolor=face,
               linestyle=style, linewidth=0.9, markersize=4.4, label=label)
        for _, label, marker, face, style in series],
        loc="lower right", fontsize=7.2, handlelength=2.4)
    save(fig, "figure6_heterogeneity")


def main() -> int:
    global DATA, RESULTS, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="directory holding the screening record")
    parser.add_argument("--results-dir", type=Path, default=RESULTS,
                        help="directory holding the results tables")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES,
                        help="directory to write the figures to")
    arguments = parser.parse_args()
    DATA, RESULTS = arguments.data_dir, arguments.results_dir
    FIGURES = arguments.figures_dir

    configure()
    print("Writing to %s" % FIGURES)
    figure2_sample_flow()
    figure3_denominators()
    figure4_components()
    figure5_lagged()
    figure6_heterogeneity()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
