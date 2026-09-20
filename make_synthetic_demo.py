"""Generate synthetic extracts with the schema of the licensed LSEG files.

The raw data are proprietary and cannot be redistributed, so this script
writes two files that carry the same column names, the same missingness
structure and the same broad distributions as the real extracts:

    lseg_static.csv   one row per instrument
    lseg_panel.csv    one row per firm-year, five fiscal years, newest first

build_analysis_data.py, run_analysis.py and make_figures.py then run against
these files unchanged, so a reader without a licence can execute the whole
pipeline and inspect what every step does.

Nothing proprietary is used to produce them. The firm-level draw is a
multivariate normal whose means, standard deviations and correlations are the
published descriptive statistics of the paper -- Table "Descriptive
statistics" for the moments, Table "Pairwise correlations" for the
dependence, and the sample-construction table for the screening counts. The
correlation matrix is projected onto the nearest positive semi-definite
matrix before it is used, since a matrix assembled cell by cell from a
published table need not be a valid one. Rated and unrated firms are drawn
with different means, reproducing the size, age and float gaps reported for
the rating gate, so the selection models have something to find.

The output is a demonstration of the code, not of the findings. It contains
no company and no real value, and no number computed from it should be cited.

    python make_synthetic_demo.py --data-dir synthetic_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260918

# Sample sizes, from the sample-construction table.
N_INSTRUMENTS = 4329
N_NON_OPERATING = 747
N_RATED = 2151
YEARS = 5

# Share of firm-years arriving without a reporting date, which in the real
# extract is the share carrying no revenue figure.
UNDATED_SHARE = 0.135

# Most recent fiscal year, in the proportions observed in the cross-section.
LATEST_YEAR = {2025: 1879, 2026: 253, 2024: 15, 2023: 3, 2021: 1}

# Sector shares of the rated cross-section.
SECTORS = {
    "Financials": 455, "Industrials": 396, "Consumer Cyclicals": 284,
    "Technology": 262, "Basic Materials": 165, "Healthcare": 161,
    "Real Estate": 144, "Consumer Non-Cyclicals": 127, "Energy": 82,
    "Utilities": 71, "Academic & Educational Services": 4,
}

# Country shares. The five named markets are the largest in the cross-section;
# the remainder are spread over a long tail, which is what makes the residual
# country group non-empty at the 25-firm cutoff.
LARGE_COUNTRIES = {"United Kingdom": 573, "Germany": 229, "Sweden": 225,
                   "France": 172, "Switzerland": 162}
MEDIUM_COUNTRIES = ["Italy", "Spain", "Netherlands", "Norway", "Denmark",
                    "Finland", "Belgium", "Poland", "Austria", "Portugal",
                    "Ireland", "Greece", "Turkey", "Russia", "Luxembourg",
                    "Israel", "Hungary", "Czechia", "Romania", "Iceland"]
SMALL_COUNTRIES = ["Malta", "Slovenia", "Croatia", "Cyprus", "Estonia",
                   "Latvia", "Lithuania", "Bulgaria", "Serbia", "Monaco",
                   "Liechtenstein", "South Africa", "Canada", "Australia"]

# Firm-level latent variables. Means and standard deviations are those of the
# rated cross-section; the unrated column holds the means reported for the
# firms that carry no theme score.
LATENTS = {
    #  name              mean     sd    unrated mean
    "ln_assets":       (21.842, 2.006, 19.890),
    "ln_mcap":         (21.327, 1.741, 19.540),
    "ln_revenue":      (21.070, 2.300, 18.700),
    "ln_employees":    (8.800,  1.900, 7.100),
    "ln_ev":           (21.600, 1.900, 19.800),
    "ln_energy":       (13.000, 2.400, 11.500),
    "ln_scope12":      (9.959,  3.028, 8.600),
    "ln_scope3":       (12.978, 3.133, 11.600),
    "theme_factor":    (0.000,  1.000, -0.900),
    "roa":             (0.035,  0.122, 0.020),
    "leverage":        (0.248,  0.194, 0.240),
    "free_float":      (66.663, 27.732, 45.100),
    "firm_age":        (37.940, 32.871, 27.000),
}

# Correlations between the latents. Entries taken from the published
# correlation table where it reports them; the remainder are set to values
# consistent with it. The theme factor is the common component of the two
# theme scores, so its correlations are the published ones divided by the
# loading of the composite on that factor.
CORRELATIONS = {
    ("ln_assets", "ln_mcap"): 0.846,
    ("ln_assets", "ln_revenue"): 0.820,
    ("ln_assets", "ln_employees"): 0.700,
    ("ln_assets", "ln_ev"): 0.830,
    ("ln_assets", "ln_energy"): 0.520,
    ("ln_assets", "ln_scope12"): 0.502,
    ("ln_assets", "ln_scope3"): 0.470,
    ("ln_assets", "theme_factor"): 0.726,
    ("ln_assets", "roa"): -0.018,
    ("ln_assets", "leverage"): 0.139,
    ("ln_mcap", "ln_revenue"): 0.800,
    ("ln_mcap", "ln_employees"): 0.690,
    ("ln_mcap", "ln_ev"): 0.950,
    ("ln_mcap", "ln_energy"): 0.480,
    ("ln_mcap", "ln_scope12"): 0.468,
    ("ln_mcap", "ln_scope3"): 0.440,
    ("ln_mcap", "theme_factor"): 0.688,
    ("ln_mcap", "roa"): 0.143,
    ("ln_mcap", "leverage"): 0.046,
    ("ln_revenue", "ln_employees"): 0.850,
    ("ln_revenue", "ln_ev"): 0.790,
    ("ln_revenue", "ln_energy"): 0.750,
    ("ln_revenue", "ln_scope12"): 0.705,
    ("ln_revenue", "ln_scope3"): 0.660,
    ("ln_revenue", "theme_factor"): 0.640,
    ("ln_employees", "ln_energy"): 0.640,
    ("ln_employees", "ln_scope12"): 0.650,
    ("ln_employees", "ln_scope3"): 0.600,
    ("ln_employees", "ln_ev"): 0.680,
    ("ln_employees", "theme_factor"): 0.560,
    ("ln_ev", "ln_energy"): 0.470,
    ("ln_ev", "ln_scope12"): 0.450,
    ("ln_ev", "ln_scope3"): 0.430,
    ("ln_ev", "theme_factor"): 0.650,
    ("ln_energy", "ln_scope12"): 0.880,
    ("ln_energy", "ln_scope3"): 0.620,
    ("ln_energy", "theme_factor"): 0.440,
    ("ln_scope12", "ln_scope3"): 0.700,
    ("ln_scope12", "theme_factor"): 0.442,
    ("ln_scope12", "roa"): -0.043,
    ("ln_scope12", "leverage"): 0.136,
    ("ln_scope3", "theme_factor"): 0.400,
    ("theme_factor", "roa"): -0.013,
    ("theme_factor", "leverage"): 0.224,
    ("theme_factor", "free_float"): 0.150,
    ("theme_factor", "firm_age"): 0.120,
    ("roa", "leverage"): -0.192,
    ("free_float", "ln_mcap"): 0.180,
    ("firm_age", "ln_assets"): 0.150,
}

STATIC_COLUMNS = [
    "RIC", "ISIN", "Company Common Name", "Country of Headquarters",
    "Country of Exchange", "TRBC Economic Sector Name",
    "TRBC Economic Sector Code", "TRBC Business Sector Name",
    "TRBC Business Sector Code", "TRBC Industry Group Name",
    "TRBC Industry Group Code", "TRBC Industry Name", "TRBC Industry Code",
    "TRBC Activity Name", "TRBC Activity Code", "Organization Founded Year",
]

PANEL_COLUMNS = [
    "RIC", "Date", "Revenue", "Net Income Incl Extra Before Distributions",
    "Total Assets", "Total Debt", "Enterprise Value (Daily Time Series)",
    "Company Market Cap", "Full-Time Employees", "Free Float (Percent)",
    "CO2 Equivalent Emissions Total", "CO2 Equivalent Emissions Direct, Scope 1",
    "CO2 Equivalent Emissions Indirect, Scope 2",
    "CO2 Equivalent Emissions Indirect, Scope 3", "CO2 Estimation Method",
    "Total CO2 Equivalent Emissions To Revenues USD in million",
    "Energy Use Total", "Water Withdrawal Total", "Waste Total",
    "Waste Recycled Total", "Hazardous Waste", "Non-Hazardous Waste",
    "Environmental Pillar Score", "Emissions Score", "Resource Use Score",
    "Environmental Innovation Score", "ESG Score", "ESG Combined Score",
    "ESG Reporting Scope", "GRI Report Guidelines", "ESG Period Last Update Date",
]


# --------------------------------------------------------------------------
# The firm-level draw
# --------------------------------------------------------------------------

def correlation_matrix() -> np.ndarray:
    """Assemble the target correlations and project them onto a valid matrix.

    A correlation matrix filled in cell by cell from a published table is not
    guaranteed to be positive semi-definite, so the negative eigenvalues are
    clipped away and the result rescaled to unit diagonal. The distance moved
    is reported, since a large move would mean the target was far from being a
    correlation matrix at all.
    """
    names = list(LATENTS)
    size = len(names)
    matrix = np.eye(size)
    for (first, second), value in CORRELATIONS.items():
        i, j = names.index(first), names.index(second)
        matrix[i, j] = matrix[j, i] = value

    values, vectors = np.linalg.eigh(matrix)
    if values.min() < 1e-8:
        repaired = vectors @ np.diag(np.clip(values, 1e-8, None)) @ vectors.T
        scale = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(scale, scale)
        print("  correlation matrix repaired, largest change %.4f"
              % np.abs(repaired - matrix).max())
        return repaired
    return matrix


def draw_firms(rng: np.random.Generator, count: int, rated: bool) -> pd.DataFrame:
    """Draw `count` firms from the latent distribution for their rating status."""
    names = list(LATENTS)
    standardised = rng.multivariate_normal(np.zeros(len(names)),
                                           correlation_matrix(), size=count)
    frame = pd.DataFrame(standardised, columns=names)
    for position, name in enumerate(names):
        mean, sd, unrated_mean = LATENTS[name]
        centre = mean if rated else unrated_mean
        frame[name] = centre + sd * standardised[:, position]

    frame["free_float"] = frame["free_float"].clip(0.0, 100.0)
    # Age is bounded below and long-tailed; the normal draw is folded rather
    # than truncated so the mean of the published distribution survives.
    frame["firm_age"] = np.abs(frame["firm_age"]).clip(0.0, 220.0).round()
    frame["rated"] = rated
    return frame


def theme_scores(rng: np.random.Generator, frame: pd.DataFrame) -> pd.DataFrame:
    """Turn the common theme factor into the provider's score fields.

    The two themes load equally on the common factor at the correlation the
    paper reports between them, are placed on the published 0-100 means and
    standard deviations, and are then clipped to the scale. The clipping is
    what produces the mass at zero in the innovation theme, which the real
    distribution also shows.
    """
    common = frame["theme_factor"].to_numpy()
    loading = np.sqrt(0.40)                       # themes correlate at 0.40
    residual = np.sqrt(1.0 - 0.40)
    resource = loading * common + residual * rng.standard_normal(len(frame))
    innovation = loading * common + residual * rng.standard_normal(len(frame))

    frame = frame.copy()
    frame["resource_use"] = np.clip(55.928 + 31.040 * resource, 0.0, 100.0)
    frame["innovation"] = np.clip(36.273 + 32.720 * innovation, 0.0, 100.0)
    composite = (frame["resource_use"] + frame["innovation"]) / 2.0

    # The remaining scores are built around the composite at the correlations
    # the paper reports with it: pillar 0.920, emissions theme 0.585.
    noise = rng.standard_normal((len(frame), 3))
    centred = (composite - composite.mean()) / composite.std(ddof=0)
    frame["pillar"] = np.clip(
        55.0 + 25.0 * (0.920 * centred + np.sqrt(1 - 0.920 ** 2) * noise[:, 0]),
        0.0, 100.0)
    frame["emissions_score"] = np.clip(
        60.0 + 30.0 * (0.585 * centred + np.sqrt(1 - 0.585 ** 2) * noise[:, 1]),
        0.0, 100.0)
    frame["esg"] = np.clip(
        55.0 + 18.0 * (0.700 * centred + np.sqrt(1 - 0.700 ** 2) * noise[:, 2]),
        0.0, 100.0)
    # The combined score penalises controversies, so it is never the higher.
    frame["esgc"] = frame["esg"] - rng.gamma(1.2, 4.0, len(frame)) * (
        rng.random(len(frame)) < 0.35)
    frame["esgc"] = frame["esgc"].clip(0.0, 100.0)
    return frame


def availability(rng: np.random.Generator, frame: pd.DataFrame) -> pd.DataFrame:
    """Decide which firms report each physical quantity.

    Disclosure rises with scale and with the score, which is the relationship
    the availability gate estimates; the intercepts are set so that the share
    reporting each item matches the sample-construction table.
    """
    frame = frame.copy()
    size = (frame["ln_mcap"] - frame["ln_mcap"].mean()) / frame["ln_mcap"].std(ddof=0)
    score = (frame["resource_use"] + frame["innovation"]) / 2.0
    score = (score - score.mean()) / score.std(ddof=0)

    def gate(intercept: float, on_size: float, on_score: float) -> np.ndarray:
        odds = intercept + on_size * size + on_score * score
        return rng.random(len(frame)) < 1.0 / (1.0 + np.exp(-odds))

    frame["reports_scope12"] = gate(2.4, 0.55, 0.85)      # 1,898 of 2,151
    frame["reports_scope3"] = gate(1.6, 0.50, 0.90)       # 1,716 of 2,151
    frame["reports_waste"] = gate(0.6, 0.35, 0.80)
    frame["reports_energy"] = gate(1.5, 0.40, 0.80)
    return frame


# --------------------------------------------------------------------------
# Assembling the two extracts
# --------------------------------------------------------------------------

def classification(rng: np.random.Generator, count: int) -> pd.DataFrame:
    """Sector and country labels in the observed proportions."""
    sectors = list(SECTORS)
    sector_weights = np.array(list(SECTORS.values()), dtype=float)
    sector = rng.choice(sectors, size=count, p=sector_weights / sector_weights.sum())

    countries = list(LARGE_COUNTRIES) + MEDIUM_COUNTRIES + SMALL_COUNTRIES
    weights = np.array(list(LARGE_COUNTRIES.values())
                       + [40.0] * len(MEDIUM_COUNTRIES)
                       + [4.0] * len(SMALL_COUNTRIES))
    country = rng.choice(countries, size=count, p=weights / weights.sum())

    return pd.DataFrame({"sector": sector, "country": country})


def build_static(rng: np.random.Generator, firms: pd.DataFrame) -> pd.DataFrame:
    """One row per instrument, with the non-operating rows left unclassified."""
    static = pd.DataFrame({
        "RIC": firms["RIC"],
        "ISIN": ["SY%010d" % n for n in range(len(firms))],
        "Company Common Name": ["Synthetic Firm %04d" % n for n in range(len(firms))],
        "Country of Headquarters": firms["country"],
        "Country of Exchange": firms["country"],
        "TRBC Economic Sector Name": firms["sector"],
        "TRBC Business Sector Name": firms["sector"] + " Services",
        "TRBC Industry Group Name": firms["sector"] + " Group",
        "TRBC Industry Name": firms["sector"] + " Industry",
        "TRBC Activity Name": firms["sector"] + " Activity",
        "Organization Founded Year": firms["founded"],
    })
    for level in ("Economic Sector", "Business Sector", "Industry Group",
                  "Industry", "Activity"):
        codes = pd.factorize(static["TRBC %s Name" % level])[0] + 50
        static["TRBC %s Code" % level] = np.where(
            static["TRBC %s Name" % level].isna(), np.nan, codes)

    # Funds and exchange-traded products carry no classification at any level,
    # which is the rule the screening step uses to remove them.
    non_operating = firms["operating"].eq(False).to_numpy()
    for column in static.columns:
        if column.startswith("TRBC") or column == "Organization Founded Year":
            static.loc[non_operating, column] = np.nan
    return static[STATIC_COLUMNS]


def build_panel(rng: np.random.Generator, firms: pd.DataFrame) -> pd.DataFrame:
    """Five fiscal years per instrument, newest first.

    Earlier years are the latest year moved backwards: financial scale shrinks,
    emissions rise slightly, scores fall, and coverage thins, so that the
    lagged design has an earlier score to work with and the pooled
    specification has within-firm variation.
    """
    month = rng.choice(["12-31", "09-30", "06-30", "03-31"], size=len(firms),
                       p=[0.72, 0.10, 0.10, 0.08])
    rows = []
    for offset in range(YEARS):                       # 0 is the latest year
        year = firms["latest_year"] - offset
        drift_financial = np.exp(-0.05 * offset + 0.02 * rng.standard_normal(len(firms)))
        drift_emissions = np.exp(0.03 * offset + 0.08 * rng.standard_normal(len(firms)))
        keeps_score = (rng.random(len(firms)) > 0.04 * offset) & firms["rated"]

        revenue = np.exp(firms["ln_revenue"]) * drift_financial
        assets = np.exp(firms["ln_assets"]) * drift_financial
        scope12 = np.exp(firms["ln_scope12"]) * drift_emissions
        share1 = rng.beta(2.0, 2.0, len(firms))
        # The firms reporting exactly zero operational emissions, which the log
        # transformation then excludes and the screening record counts.
        zero = rng.random(len(firms)) < 0.038
        scope12 = np.where(zero, 0.0, scope12)
        scope3 = np.where(rng.random(len(firms)) < 0.001, 0.0,
                          np.exp(firms["ln_scope3"]) * drift_emissions)
        waste_total = np.exp(firms["ln_scope12"] * 0.55 + 2.0) * drift_emissions
        recycled_rate = rng.beta(2.4, 1.1, len(firms))
        energy = np.exp(firms["ln_energy"]) * drift_emissions

        undated = rng.random(len(firms)) < UNDATED_SHARE
        frame = pd.DataFrame({
            "RIC": firms["RIC"],
            "Date": np.where(undated, "",
                             year.astype(str) + "-" + pd.Series(month, index=firms.index)),
            "Revenue": np.where(undated, np.nan, revenue),
            "Net Income Incl Extra Before Distributions": firms["roa"] * assets,
            "Total Assets": assets,
            "Total Debt": firms["leverage"].clip(0.0, 3.0) * assets,
            "Enterprise Value (Daily Time Series)": np.exp(firms["ln_ev"]) * drift_financial,
            "Company Market Cap": np.exp(firms["ln_mcap"]) * drift_financial,
            "Full-Time Employees": np.exp(firms["ln_employees"]).round(),
            "Free Float (Percent)": firms["free_float"],
            "CO2 Equivalent Emissions Direct, Scope 1": scope12 * share1,
            "CO2 Equivalent Emissions Indirect, Scope 2": scope12 * (1.0 - share1),
            "CO2 Equivalent Emissions Indirect, Scope 3": scope3,
            "Energy Use Total": energy,
            "Water Withdrawal Total": energy * rng.lognormal(0.0, 0.6, len(firms)),
            "Waste Total": waste_total,
            "Waste Recycled Total": waste_total * recycled_rate,
            "Hazardous Waste": waste_total * rng.beta(1.2, 9.0, len(firms)),
            "Environmental Pillar Score": firms["pillar"],
            "Emissions Score": firms["emissions_score"],
            "Resource Use Score": firms["resource_use"],
            "Environmental Innovation Score": firms["innovation"],
            "ESG Score": firms["esg"],
            "ESG Combined Score": firms["esgc"],
            "ESG Reporting Scope": firms["reporting_scope"],
            "GRI Report Guidelines": firms["gri"],
            "year": year,
        })
        frame["CO2 Equivalent Emissions Total"] = (
            frame["CO2 Equivalent Emissions Direct, Scope 1"]
            + frame["CO2 Equivalent Emissions Indirect, Scope 2"])
        frame["Non-Hazardous Waste"] = frame["Waste Total"] - frame["Hazardous Waste"]
        frame["Total CO2 Equivalent Emissions To Revenues USD in million"] = (
            frame["CO2 Equivalent Emissions Total"] / (frame["Revenue"] / 1e6))

        # The estimation method flag is carried for part of the sample only,
        # and distinguishes a firm-reported figure from a vendor estimate.
        method = rng.choice(["Reported", "Median", "CO2", "Energy", ""],
                            size=len(firms), p=[0.39, 0.06, 0.03, 0.03, 0.49])
        frame["CO2 Estimation Method"] = method
        frame["ESG Period Last Update Date"] = np.where(
            keeps_score, year.astype(str) + "-11-30", "")

        emissions_columns = ["CO2 Equivalent Emissions Total",
                             "CO2 Equivalent Emissions Direct, Scope 1",
                             "CO2 Equivalent Emissions Indirect, Scope 2",
                             "CO2 Estimation Method",
                             "Total CO2 Equivalent Emissions To Revenues USD in million"]
        hidden = ~(firms["reports_scope12"].to_numpy() & firms["rated"].to_numpy())
        frame.loc[hidden, emissions_columns] = np.nan
        frame.loc[~(firms["reports_scope3"].to_numpy() & firms["rated"].to_numpy()),
                  "CO2 Equivalent Emissions Indirect, Scope 3"] = np.nan
        frame.loc[~(firms["reports_waste"].to_numpy() & firms["rated"].to_numpy()),
                  ["Waste Total", "Waste Recycled Total", "Hazardous Waste",
                   "Non-Hazardous Waste"]] = np.nan
        frame.loc[~(firms["reports_energy"].to_numpy() & firms["rated"].to_numpy()),
                  ["Energy Use Total", "Water Withdrawal Total"]] = np.nan

        score_columns = ["Environmental Pillar Score", "Emissions Score",
                         "Resource Use Score", "Environmental Innovation Score",
                         "ESG Score", "ESG Combined Score", "ESG Reporting Scope",
                         "GRI Report Guidelines"]
        frame.loc[~keeps_score.to_numpy(), score_columns] = np.nan

        # An instrument with no operations reports nothing beyond its listing.
        frame.loc[~firms["operating"].to_numpy(),
                  [c for c in frame.columns if c not in
                   ("RIC", "Date", "Company Market Cap", "Free Float (Percent)",
                    "year")]] = np.nan
        rows.append(frame)

    panel = pd.concat(rows, ignore_index=False).sort_index(kind="stable")
    panel = panel.sort_values(["RIC", "year"], ascending=[True, False],
                              kind="stable")
    return duplicate_a_few_rows(rng, panel)[PANEL_COLUMNS]


def duplicate_a_few_rows(rng: np.random.Generator,
                         panel: pd.DataFrame) -> pd.DataFrame:
    """Repeat a handful of dated firm-years, one of the two less complete.

    The real extract returns a small number of firms twice in the same fiscal
    year, and the construction code resolves them by keeping the more complete
    row. Including them here means that step is exercised rather than skipped.
    """
    dated = panel[panel["Date"].astype(str).str.len() > 0]
    if dated.empty:
        return panel
    chosen = rng.choice(dated.index.to_numpy(), size=min(6, len(dated)),
                        replace=False)
    repeats = panel.loc[chosen].copy()
    repeats[["Water Withdrawal Total", "Free Float (Percent)",
             "ESG Reporting Scope"]] = np.nan
    combined = pd.concat([panel, repeats])
    return combined.sort_values(["RIC", "year"], ascending=[True, False],
                                kind="stable")


def reporting_fields(rng: np.random.Generator, firms: pd.DataFrame) -> pd.DataFrame:
    """Reporting-scope and GRI fields, both populated for part of the sample."""
    firms = firms.copy()
    scope = np.where(rng.random(len(firms)) < 0.78, 100.0,
                     np.clip(100.0 - rng.gamma(2.0, 12.0, len(firms)), 5.0, 100.0))
    firms["reporting_scope"] = np.where(rng.random(len(firms)) < 0.87, scope, np.nan)
    # The GRI field is populated for a minority of firms and is almost
    # constant among them, which is why the paper uses it descriptively.
    gri = np.where(rng.random(len(firms)) < 0.99, "TRUE", "FALSE")
    firms["gri"] = np.where(rng.random(len(firms)) < 0.33, gri, "")
    return firms


def generate(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    operating_count = N_INSTRUMENTS - N_NON_OPERATING
    unrated_count = operating_count - N_RATED

    rated = theme_scores(rng, draw_firms(rng, N_RATED, rated=True))
    unrated = theme_scores(rng, draw_firms(rng, unrated_count, rated=False))
    non_operating = theme_scores(rng, draw_firms(rng, N_NON_OPERATING, rated=False))

    firms = pd.concat([rated, unrated, non_operating], ignore_index=True)
    firms["operating"] = np.arange(len(firms)) < operating_count
    firms = availability(rng, firms)
    firms = reporting_fields(rng, firms)

    labels = classification(rng, len(firms))
    firms["sector"] = labels["sector"]
    firms["country"] = labels["country"]
    firms["RIC"] = ["SYN%04d.X" % n for n in range(len(firms))]

    years = np.array(list(LATEST_YEAR))
    weights = np.array(list(LATEST_YEAR.values()), dtype=float)
    firms["latest_year"] = rng.choice(years, size=len(firms),
                                      p=weights / weights.sum())
    # A founding year is missing for a few firms, and arrives as zero rather
    # than as a blank, which is the case the construction code has to handle.
    founded = firms["latest_year"] - firms["firm_age"]
    firms["founded"] = np.where(rng.random(len(firms)) < 0.05, 0, founded).astype(int)

    return build_static(rng, firms), build_panel(rng, firms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("synthetic_data"),
                        help="directory to write the synthetic extracts to")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="random seed, so the demonstration is reproducible")
    arguments = parser.parse_args()
    arguments.data_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic extracts")
    rng = np.random.default_rng(arguments.seed)
    static, panel = generate(rng)

    static.to_csv(arguments.data_dir / "lseg_static.csv", index=False)
    panel.to_csv(arguments.data_dir / "lseg_panel.csv", index=False)
    print("  lseg_static.csv  %5d instruments" % len(static))
    print("  lseg_panel.csv   %5d firm-years" % len(panel))
    print("\nWritten to %s. These values are synthetic and must not be cited."
          % arguments.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
