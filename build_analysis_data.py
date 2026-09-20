"""Build the analysis datasets from the LSEG extracts.

Inputs (not redistributable; regenerate with the data-access protocol):
    data/lseg_static.csv   one row per firm: classification, country, founding year
    data/lseg_panel.csv    one row per firm-year, five fiscal years

Outputs:
    data/analysis_panel.csv         firm-year, all constructed variables
    data/analysis_cross_section.csv one row per firm, most recent fiscal year
    data/analysis_lagged.csv        base-year predictors, later-year outcomes
    data/sample_flow.csv            attrition at every screening step
    data/screening_log.txt          human-readable record of the same

Every screening decision is recorded rather than applied silently, so the
sample flow reported in the paper is generated rather than transcribed.

    python build_analysis_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATA = Path("data")
STATIC_FILE = DATA / "lseg_static.csv"
PANEL_FILE = DATA / "lseg_panel.csv"

# Fiscal years retained. The extract requests five years; a small number of
# rows carry much older dates where a firm's reporting history is sparse.
FIRST_YEAR = 2021
LAST_YEAR = 2026

# Country fixed effects: a country with fewer than this many firms is pooled
# into a residual group. Alternatives are also written so the choice can be
# shown to be immaterial.
COUNTRY_CUTOFF = 25
COUNTRY_CUTOFF_ALTERNATIVES = (10, 50)

# Winsorising applied to constructed ratios, as a sensitivity variant.
WINSOR_LIMITS = (0.01, 0.99)

# Sectors conventionally treated as carbon intensive.
CARBON_INTENSIVE_SECTORS = {"Energy", "Utilities", "Basic Materials"}

# Lag structure for the dynamic test.
LAG_BASE_YEAR = 2021
LAG_OUTCOME_YEAR = 2025

# Source column names, kept in one place so a changed extract is a one-line fix.
COL = {
    "ric": "RIC",
    "date": "Date",
    "revenue": "Revenue",
    "net_income": "Net Income Incl Extra Before Distributions",
    "assets": "Total Assets",
    "debt": "Total Debt",
    "ev": "Enterprise Value (Daily Time Series)",
    "mcap": "Company Market Cap",
    "employees": "Full-Time Employees",
    "free_float": "Free Float (Percent)",
    "co2_total": "CO2 Equivalent Emissions Total",
    "scope1": "CO2 Equivalent Emissions Direct, Scope 1",
    "scope2": "CO2 Equivalent Emissions Indirect, Scope 2",
    "scope3": "CO2 Equivalent Emissions Indirect, Scope 3",
    "co2_method": "CO2 Estimation Method",
    "energy": "Energy Use Total",
    "water": "Water Withdrawal Total",
    "waste_total": "Waste Total",
    "waste_recycled": "Waste Recycled Total",
    "waste_hazardous": "Hazardous Waste",
    "waste_nonhazardous": "Non-Hazardous Waste",
    "pillar": "Environmental Pillar Score",
    "emissions_score": "Emissions Score",
    "resource_use": "Resource Use Score",
    "innovation": "Environmental Innovation Score",
    "esg": "ESG Score",
    "esgc": "ESG Combined Score",
    "reporting_scope": "ESG Reporting Scope",
    "gri": "GRI Report Guidelines",
    "sector": "TRBC Economic Sector Name",
    "business_sector": "TRBC Business Sector Name",
    "industry_group": "TRBC Industry Group Name",
    "industry": "TRBC Industry Name",
    "activity": "TRBC Activity Name",
    "country_hq": "Country of Headquarters",
    "country_exch": "Country of Exchange",
    "founded": "Organization Founded Year",
    "isin": "ISIN",
    "name": "Company Common Name",
}

FLOW: list[dict] = []


def record(step: str, detail: str, firms: int, rows: int | None = None) -> None:
    """Append one line to the sample-flow record."""
    FLOW.append({"step": step, "detail": detail, "firms": firms,
                 "firm_years": "" if rows is None else rows})
    tail = "" if rows is None else "  firm-years %6d" % rows
    print("  %-46s firms %5d%s" % (step, firms, tail))


# --------------------------------------------------------------------------
# Loading and screening
# --------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not STATIC_FILE.exists() or not PANEL_FILE.exists():
        raise SystemExit(
            "Missing %s or %s. These hold licensed LSEG data and are not\n"
            "distributed with the code; see the data-access protocol in README.md."
            % (STATIC_FILE, PANEL_FILE))
    static = pd.read_csv(STATIC_FILE)
    panel = pd.read_csv(PANEL_FILE)
    if "Instrument" in panel.columns:
        panel = panel.rename(columns={"Instrument": COL["ric"]})
    return static, panel



def assign_fiscal_year(panel: pd.DataFrame) -> pd.DataFrame:
    """Date every firm-year, including rows the extract leaves undated.

    The reporting date travels with the revenue field, so a firm that reports
    no revenue arrives with no date at all -- 485 operating firms here, many of
    them scored and reporting emissions. Discarding them would bias the sample
    towards firms with complete financial statements, which is precisely the
    selection the paper is about.

    Rows arrive newest first in blocks of one per fiscal year, so position
    within a firm identifies the year. Any dated row in a firm anchors the
    whole block; a firm with no dated row at all falls back to the year most
    commonly seen at that position elsewhere.
    """
    panel = panel.copy()
    panel["row_order"] = panel.groupby(COL["ric"], sort=False).cumcount()
    dated = pd.to_datetime(panel[COL["date"]], errors="coerce").dt.year

    # Year implied for position zero by each dated row.
    implied = dated + panel["row_order"]
    anchor = implied.groupby(panel[COL["ric"]]).transform("median")

    # Firms with no dated row anywhere: use the commonest first-position year.
    fallback = implied.dropna()
    fallback_year = int(fallback.mode().iloc[0]) if not fallback.empty else LAST_YEAR
    anchor = anchor.fillna(fallback_year)

    panel["year"] = (anchor - panel["row_order"]).round().astype(int)
    panel["year_imputed"] = dated.isna()
    # Where a date exists it is authoritative.
    panel.loc[dated.notna(), "year"] = dated[dated.notna()].astype(int)
    return panel


def screen(static: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply and record every screening step from the raw universe onward."""
    print("\nScreening")
    record("Raw universe", "all instruments in the extract",
           static[COL["ric"]].nunique(), len(panel))

    # Entities with no industry classification are funds, exchange-traded
    # products and similar instruments rather than operating firms. They carry
    # no operations and therefore no environmental outcome to measure.
    operating = static[static[COL["sector"]].notna()].copy()
    dropped = static[COL["ric"]].nunique() - operating[COL["ric"]].nunique()
    record("Drop non-operating instruments",
           "%d funds and index products, no industry classification" % dropped,
           operating[COL["ric"]].nunique())

    panel = panel[panel[COL["ric"]].isin(operating[COL["ric"]])].copy()
    panel = assign_fiscal_year(panel)
    imputed = int(panel["year_imputed"].sum())
    record("Assign fiscal years",
           "%d of %d firm-years dated by position" % (imputed, len(panel)),
           panel[COL["ric"]].nunique(), len(panel))

    outside = int(((panel["year"] < FIRST_YEAR) | (panel["year"] > LAST_YEAR)).sum())
    panel = panel[(panel["year"] >= FIRST_YEAR) & (panel["year"] <= LAST_YEAR)].copy()
    record("Restrict to the analysis window",
           "%d rows outside %d-%d" % (outside, FIRST_YEAR, LAST_YEAR),
           panel[COL["ric"]].nunique(), len(panel))

    # A firm should appear once per fiscal year; keep the most complete row.
    before = len(panel)
    panel["completeness"] = panel.notna().sum(axis=1)
    panel = (panel.sort_values("completeness", ascending=False)
                  .drop_duplicates(subset=[COL["ric"], "year"])
                  .drop(columns="completeness")
                  .sort_values([COL["ric"], "year"]))
    if before != len(panel):
        record("Resolve duplicate firm-years",
               "%d duplicate rows removed" % (before - len(panel)),
               panel[COL["ric"]].nunique(), len(panel))

    return operating, panel


# --------------------------------------------------------------------------
# Variable construction
# --------------------------------------------------------------------------

def safe_log(series: pd.Series, label: str, zeros: dict) -> pd.Series:
    """Natural log defined only on strictly positive values.

    Zeros and negatives are recorded rather than silently coerced, because the
    count of dropped zero-emission and zero-waste firms is itself reportable.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    non_positive = int(((numeric <= 0) & numeric.notna()).sum())
    if non_positive:
        zeros[label] = non_positive
    return np.log(numeric.where(numeric > 0))


def leave_one_out_median(values: pd.Series) -> pd.Series:
    """Median of the firm's sector and year, computed without the firm itself.

    A sector median that includes the firm makes the moderator a function of
    the outcome it moderates. The leave-one-out version removes that channel;
    with a few hundred firms per sector-year the two are nearly identical, but
    the identity is then a property of the data rather than an assumption.
    """
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    present = np.flatnonzero(~np.isnan(arr))
    if len(present) < 3:
        return pd.Series(out, index=values.index)
    kept = arr[present]
    # A firm contributing no intensity of its own has nothing to leave out, so
    # it takes the ordinary sector median; the moderator is a property of the
    # sector and is defined for every firm in it.
    out[:] = np.median(kept)
    for position, location in enumerate(present):
        out[location] = np.median(np.delete(kept, position))
    return pd.Series(out, index=values.index)


def winsorise(series: pd.Series, limits=WINSOR_LIMITS) -> pd.Series:
    low, high = series.quantile(limits[0]), series.quantile(limits[1])
    return series.clip(lower=low, upper=high)


def first_principal_component(frame: pd.DataFrame) -> pd.Series:
    """First principal component of standardised columns, sign-fixed positive.

    Computed directly rather than through an extra dependency; the covariance
    matrix is two by two.
    """
    usable = frame.dropna()
    if usable.empty:
        return pd.Series(np.nan, index=frame.index)
    standardised = (usable - usable.mean()) / usable.std(ddof=0)
    values, vectors = np.linalg.eigh(np.cov(standardised.T, ddof=0))
    loading = vectors[:, int(np.argmax(values))]
    if loading.sum() < 0:                      # orient so higher means more
        loading = -loading
    scores = standardised.to_numpy() @ loading
    return pd.Series(scores, index=usable.index).reindex(frame.index)


def country_groups(series: pd.Series, firms: pd.Series, cutoff: int) -> pd.Series:
    """Pool countries with fewer than `cutoff` distinct firms into a residual group.

    Counted over firms rather than firm-years, so the cutoff means the same
    thing in the panel and in the cross-section.
    """
    counts = firms.groupby(series).nunique()
    keep = set(counts[counts >= cutoff].index)
    return series.where(series.isin(keep), other="Other")


def build(static: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merge the two sources and construct every analysis variable."""
    print("\nConstructing variables")
    static_cols = [COL[k] for k in ("ric", "isin", "name", "country_hq", "country_exch",
                                    "sector", "business_sector", "industry_group",
                                    "industry", "activity", "founded")]
    df = panel.merge(static[static_cols], on=COL["ric"], how="left")

    numeric_keys = ("revenue", "net_income", "assets", "debt", "ev", "mcap",
                    "employees", "free_float", "co2_total", "scope1", "scope2",
                    "scope3", "energy", "water", "waste_total", "waste_recycled",
                    "waste_hazardous", "waste_nonhazardous", "pillar",
                    "emissions_score", "resource_use", "innovation", "esg",
                    "esgc", "reporting_scope", "founded")
    for key in numeric_keys:
        df[COL[key]] = pd.to_numeric(df[COL[key]], errors="coerce")

    # --- the rating measure and its alternatives -------------------------
    ru, ei = df[COL["resource_use"]], df[COL["innovation"]]
    both = ru.notna() & ei.notna()
    df["rating"] = np.where(both, (ru + ei) / 2.0, np.nan)          # equal weight
    df["rating_resource_use"] = ru
    df["rating_innovation"] = ei
    df["rating_emissions_theme"] = df[COL["emissions_score"]]
    df["rating_pillar"] = df[COL["pillar"]]
    df["rating_esg"] = df[COL["esg"]]
    df["rating_esgc"] = df[COL["esgc"]]

    # Three-theme mean, and a principal component, as weighting alternatives.
    themes = df[[COL["resource_use"], COL["innovation"], COL["emissions_score"]]]
    df["rating_three_theme"] = themes.mean(axis=1).where(themes.notna().all(axis=1))
    df["rating_pca"] = first_principal_component(df[[COL["resource_use"], COL["innovation"]]])

    df["scored"] = both

    # --- outcomes ---------------------------------------------------------
    df["scope12"] = df[[COL["scope1"], COL["scope2"]]].sum(axis=1, min_count=2)
    df["scope3"] = df[COL["scope3"]]
    method = df[COL["co2_method"]].astype(str).str.strip().str.lower()
    df["emissions_reported"] = np.where(
        df[COL["co2_method"]].isna(), np.nan, (method == "reported").astype(float))

    # --- denominators -----------------------------------------------------
    denominators = {"assets": COL["assets"], "revenue": COL["revenue"],
                    "employees": COL["employees"], "ev": COL["ev"],
                    "energy": COL["energy"]}

    zeros: dict[str, int] = {}
    df["ln_scope12"] = safe_log(df["scope12"], "Scope 1+2 total", zeros)
    df["ln_scope3"] = safe_log(df["scope3"], "Scope 3 total", zeros)

    for label, column in denominators.items():
        base = df[column].where(df[column] > 0)
        df["scope12_per_%s" % label] = df["scope12"] / base
        df["scope3_per_%s" % label] = df["scope3"] / base
        df["ln_scope12_per_%s" % label] = safe_log(
            df["scope12_per_%s" % label], "Scope 1+2 per %s" % label, zeros)
        df["ln_scope3_per_%s" % label] = safe_log(
            df["scope3_per_%s" % label], "Scope 3 per %s" % label, zeros)

    # --- waste ------------------------------------------------------------
    waste_base = df[COL["waste_total"]].where(df[COL["waste_total"]] > 0)
    df["waste_recycled_rate"] = df[COL["waste_recycled"]] / waste_base
    df.loc[df["waste_recycled_rate"] > 1, "waste_recycled_rate"] = np.nan
    df["hazardous_share"] = df[COL["waste_hazardous"]] / waste_base
    df["ln_waste_recycled"] = safe_log(df[COL["waste_recycled"]], "Waste recycled", zeros)
    df["ln_waste_total"] = safe_log(df[COL["waste_total"]], "Waste total", zeros)
    df["waste_per_revenue"] = df[COL["waste_total"]] / df[COL["revenue"]].where(
        df[COL["revenue"]] > 0)

    # --- controls ---------------------------------------------------------
    assets = df[COL["assets"]].where(df[COL["assets"]] > 0)
    df["ln_assets"] = safe_log(df[COL["assets"]], "Total assets", zeros)
    df["ln_revenue"] = safe_log(df[COL["revenue"]], "Revenue", zeros)
    df["ln_mcap"] = safe_log(df[COL["mcap"]], "Market capitalisation", zeros)
    df["ln_employees"] = safe_log(df[COL["employees"]], "Employees", zeros)
    df["ln_ev"] = safe_log(df[COL["ev"]], "Enterprise value", zeros)
    df["ln_energy"] = safe_log(df[COL["energy"]], "Energy use", zeros)
    df["roa"] = df[COL["net_income"]] / assets
    df["leverage"] = df[COL["debt"]] / assets
    df["free_float"] = df[COL["free_float"]]
    # A missing founding year arrives as zero, which would make a firm older
    # than the calendar. Anything outside a plausible range is treated as absent.
    founded = df[COL["founded"]].where(
        (df[COL["founded"]] > 1500) & (df[COL["founded"]] <= df["year"]))
    df["firm_age"] = df["year"] - founded
    df["reporting_scope"] = df[COL["reporting_scope"]]
    df["gri_reporting"] = df[COL["gri"]].astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0})

    # --- availability indicators, for the selection models ----------------
    # What is observed is whether a figure exists in the provider's database,
    # which is not the same thing as the firm having published one: where the
    # estimation-method flag is present it separates the two, and it is absent
    # for about half the cross-section. The indicators are therefore named for
    # availability, and two stricter variables carry the reporting distinction:
    # a firm-reported figure against everything else, and, among firms that
    # carry both a figure and a flag, whether that figure is the firm's own.
    df["scope12_available"] = df["scope12"].notna().astype(int)
    df["scope3_available"] = df["scope3"].notna().astype(int)
    df["waste_available"] = df[COL["waste_total"]].notna().astype(int)
    df["scope12_firm_reported"] = (
        df["scope12"].notna() & (df["emissions_reported"] == 1)).astype(int)
    # The unflagged figures are the difficulty: coding them as not firm-reported,
    # as the indicator above does, understates disclosure, so a second strict
    # variable sets them missing and compares firm-reported figures against
    # vendor estimates and absent figures alone.
    unclassified = df["scope12"].notna() & df["emissions_reported"].isna()
    df["scope12_reported_strict"] = df["scope12_firm_reported"].astype(float).where(
        ~unclassified)
    df["reported_given_available"] = df["emissions_reported"].where(
        df["scope12"].notna())

    # --- industry and country ---------------------------------------------
    df["sector"] = df[COL["sector"]]
    df["business_sector"] = df[COL["business_sector"]]
    df["industry_group"] = df[COL["industry_group"]]
    df["carbon_intensive"] = df["sector"].isin(CARBON_INTENSIVE_SECTORS).astype(int)

    df["country"] = df[COL["country_hq"]].fillna(df[COL["country_exch"]])
    df["country_group"] = country_groups(df["country"], df[COL["ric"]], COUNTRY_CUTOFF)
    for alternative in COUNTRY_CUTOFF_ALTERNATIVES:
        df["country_group_%d" % alternative] = country_groups(
            df["country"], df[COL["ric"]], alternative)

    # Sector median intensity, and intensity expressed relative to it, so the
    # scale of a firm's sector is separated from its position within it.
    for label in denominators:
        column = "ln_scope12_per_%s" % label
        median = df.groupby(["sector", "year"])[column].transform("median")
        df["%s_sector_demeaned" % column] = df[column] - median
    df["sector_carbon_intensity"] = df.groupby(["sector", "year"])[
        "ln_scope12_per_revenue"].transform(leave_one_out_median)

    # --- winsorised variants ---------------------------------------------
    winsorised = {"%s_w" % column: winsorise(df[column])
                  for column in ("roa", "leverage", "waste_recycled_rate",
                                 "hazardous_share")}
    df = pd.concat([df, pd.DataFrame(winsorised, index=df.index)], axis=1)

    return df, zeros


# --------------------------------------------------------------------------
# Derived samples
# --------------------------------------------------------------------------

def cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """One row per firm: the most recent fiscal year in which it was scored.

    This reproduces the fiscal-year-zero convention of the original extract
    while keeping the year of each observation explicit.
    """
    scored = df[df["scored"]].copy()
    latest = (scored.sort_values("year")
                    .drop_duplicates(subset=COL["ric"], keep="last")
                    .reset_index(drop=True))
    record("Cross-section: scored firms, latest year",
           "year range %d-%d" % (latest["year"].min(), latest["year"].max()),
           len(latest))
    record("  with an industry classification", "",
           int(latest["sector"].notna().sum()))
    record("  with Scope 1+2 emissions", "",
           int(latest["scope12"].notna().sum()))
    for label in ("revenue", "employees", "ev", "assets", "energy"):
        record("  with Scope 1+2 and %s" % label, "",
               int((latest["ln_scope12_per_%s" % label]).notna().sum()))
    return latest


def lagged(df: pd.DataFrame) -> pd.DataFrame:
    """Base-year rating against later-year outcomes, for the dynamic test."""
    base_cols = [COL["ric"], "rating", "rating_resource_use", "rating_innovation",
                 "ln_assets", "ln_revenue", "ln_mcap", "roa", "leverage",
                 "sector", "country_group", "firm_age", "year_imputed"]
    base = df[(df["year"] == LAG_BASE_YEAR) & df["scored"]][base_cols]

    outcome_cols = [COL["ric"], "scope12", "scope3", "ln_scope12", "ln_scope3",
                    "ln_scope12_per_revenue", "ln_scope12_per_assets",
                    "ln_scope12_per_employees", "ln_scope12_per_energy",
                    "waste_recycled_rate", "rating", "year_imputed"]
    outcome = df[df["year"] == LAG_OUTCOME_YEAR][outcome_cols]

    merged = base.merge(outcome, on=COL["ric"], how="inner",
                        suffixes=("_base", "_outcome"))
    record("Lagged sample: %d rating to %d outcome" % (LAG_BASE_YEAR, LAG_OUTCOME_YEAR),
           "", len(merged))
    record("  with a later Scope 1+2 figure", "",
           int(merged["ln_scope12"].notna().sum()))
    return merged


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_log(zeros: dict, panel: pd.DataFrame, latest: pd.DataFrame) -> None:
    lines = ["Sample construction", "=" * 70, ""]
    for entry in FLOW:
        lines.append("%-48s %6s  %s" % (entry["step"], entry["firms"], entry["detail"]))

    lines += ["", "Non-positive values excluded by log transformation", "-" * 70]
    if zeros:
        for label, count in sorted(zeros.items(), key=lambda item: -item[1]):
            lines.append("  %-44s %6d firm-years" % (label, count))
    else:
        lines.append("  none")

    lines += ["", "Emissions figures: reported by the firm or estimated", "-" * 70]
    method = panel[COL["co2_method"]].value_counts(dropna=False)
    for name, count in method.items():
        lines.append("  %-44s %6d firm-years" % (str(name), count))

    lines += ["", "Cross-section by sector", "-" * 70]
    for name, count in latest["sector"].value_counts().items():
        lines.append("  %-44s %6d firms" % (name, count))

    lines += ["", "Cross-section by observation year", "-" * 70]
    for name, count in latest["year"].value_counts().sort_index().items():
        lines.append("  %-44s %6d firms" % (name, count))

    lines += ["", "Country groups (cutoff %d firms)" % COUNTRY_CUTOFF, "-" * 70,
              "  %d groups including the residual category"
              % latest["country_group"].nunique()]
    for alternative in COUNTRY_CUTOFF_ALTERNATIVES:
        lines.append("  cutoff %2d gives %d groups"
                     % (alternative, latest["country_group_%d" % alternative].nunique()))

    (DATA / "screening_log.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    global DATA, STATIC_FILE, PANEL_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="directory holding the extracts")
    arguments = parser.parse_args()
    DATA = arguments.data_dir
    STATIC_FILE, PANEL_FILE = DATA / "lseg_static.csv", DATA / "lseg_panel.csv"

    static, panel = load()
    static, panel = screen(static, panel)
    df, zeros = build(static, panel)

    latest = cross_section(df)
    lag = lagged(df)

    df.to_csv(DATA / "analysis_panel.csv", index=False)
    latest.to_csv(DATA / "analysis_cross_section.csv", index=False)
    lag.to_csv(DATA / "analysis_lagged.csv", index=False)
    pd.DataFrame(FLOW).to_csv(DATA / "sample_flow.csv", index=False)
    write_log(zeros, panel, latest)

    print("\nWritten to %s" % DATA)
    for name in ("analysis_panel.csv", "analysis_cross_section.csv",
                 "analysis_lagged.csv", "sample_flow.csv", "screening_log.txt"):
        print("  %s" % name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
