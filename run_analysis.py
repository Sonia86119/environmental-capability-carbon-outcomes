"""Estimate every model reported in the paper and write the tables to file.

Input:
    data/analysis_panel.csv
    data/analysis_cross_section.csv
    data/analysis_lagged.csv

Output (results/):
    t01_descriptives.csv            pooled and by sector
    t02_selection_score.csv         who receives a rating at all
    t03_correlations.csv            correlation matrix and variance inflation
    t04_main.csv                    absolute outcomes and intensities
    t05_denominators.csv            the same relationship under six denominators
    t06_components.csv              themes separately, principal component, weights
    t07_industry.csv                fixed effects, intensive split, interaction
    t08_availability.csv            availability selection and reweighted estimates
    t09_lagged.csv                  earlier rating against later outcomes
    t10_waste.csv                   recycling rate and tonnage
    t11_robustness.csv              alternative specifications
    summary.txt                     every table rendered for reading

Standard errors are clustered by country throughout, since country fixed
effects leave residual dependence within country.

    python run_analysis.py
    python run_analysis.py --data-dir synthetic_data --results-dir synthetic_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize

DATA = Path("data")
RESULTS = Path("results")

RATING = "rating"
CLUSTER = "country_group"

DENOMINATORS = ["revenue", "assets", "employees", "ev", "energy"]

TABLES: dict[str, pd.DataFrame] = {}


# --------------------------------------------------------------------------
# Estimation helpers
# --------------------------------------------------------------------------

def estimate(data: pd.DataFrame, outcome: str, regressors: list[str],
             fixed_effects: list[str] | None = None, weights: str | None = None,
             cluster: str | None = CLUSTER):
    """Least squares with cluster-robust errors, returned as the fitted model.

    Kept separate from ``fit`` because a few models are read for more than
    their leading coefficient: a model carrying both themes needs each theme's
    estimate and a contrast between the two.
    """
    fixed_effects = fixed_effects or []
    terms = list(regressors) + ["C(%s)" % effect for effect in fixed_effects]
    formula = "%s ~ %s" % (outcome, " + ".join(terms))

    needed = [outcome] + regressors + fixed_effects + ([cluster] if cluster else [])
    if weights:
        needed.append(weights)
    frame = data[[c for c in dict.fromkeys(needed) if c in data.columns]].dropna()
    if len(frame) < 30:
        return None, None

    model = (smf.wls(formula, frame, weights=frame[weights]) if weights
             else smf.ols(formula, frame))
    if cluster and frame[cluster].nunique() > 1:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster]})
    else:
        result = model.fit(cov_type="HC1")
    return result, frame


def fit(data: pd.DataFrame, outcome: str, regressors: list[str],
        fixed_effects: list[str] | None = None, weights: str | None = None,
        cluster: str | None = CLUSTER) -> dict | None:
    """Least squares with cluster-robust errors, returned as one tidy row."""
    result, _ = estimate(data, outcome, regressors, fixed_effects, weights, cluster)
    if result is None:
        return None

    key = regressors[0]
    return {"outcome": outcome, "predictor": key,
            "coef": result.params.get(key, np.nan),
            "se": result.bse.get(key, np.nan),
            "t": result.tvalues.get(key, np.nan),
            "p": result.pvalues.get(key, np.nan),
            "n": int(result.nobs),
            "r2": result.rsquared,
            "controls": ", ".join(regressors[1:]) or "none",
            "fixed_effects": ", ".join(fixed_effects) or "none",
            "weighted": weights or "no"}


def pool_rare_levels(series: pd.Series, outcome: pd.Series,
                     minimum: int = 20) -> pd.Series:
    """Collapse categories that are small or perfectly predict the outcome.

    A handful of sectors and countries here contain only a few firms, all of
    which carry a figure. Left alone they separate the likelihood perfectly and the
    model will not identify; pooling them keeps their firms in the sample.
    """
    counts = series.value_counts()
    keep = set(counts[counts >= minimum].index)
    varies = {level for level, group in outcome.groupby(series) if group.nunique() > 1}
    return series.where(series.isin(keep & varies), other="Pooled")


def logit(data: pd.DataFrame, outcome: str, regressors: list[str],
          fixed_effects: list[str] | None = None):
    """Binary outcome model, used for both selection stages."""
    fixed_effects = fixed_effects or []
    frame = data[[c for c in dict.fromkeys([outcome] + regressors + fixed_effects)
                  if c in data.columns]].dropna().copy()
    if len(frame) < 50 or frame[outcome].nunique() < 2:
        return None, None

    for effect in fixed_effects:
        frame[effect] = pool_rare_levels(frame[effect], frame[outcome])
    fixed_effects = [e for e in fixed_effects if frame[e].nunique() > 1]

    terms = list(regressors) + ["C(%s)" % effect for effect in fixed_effects]
    formula = "%s ~ %s" % (outcome, " + ".join(terms))
    model = smf.logit(formula, frame)
    try:
        return model.fit(disp=False, maxiter=200), frame
    except Exception:
        pass
    try:                                  # ridge penalty when the Hessian is singular
        return model.fit_regularized(disp=False, alpha=1e-4, maxiter=400), frame
    except Exception:
        return None, None


def variance_inflation(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Variance inflation from the auxiliary regression of each column on the rest."""
    frame = data[columns].dropna()
    rows = []
    for column in columns:
        others = [c for c in columns if c != column]
        model = sm.OLS(frame[column], sm.add_constant(frame[others])).fit()
        rows.append({"variable": column,
                     "vif": np.inf if model.rsquared >= 1 else 1.0 / (1.0 - model.rsquared)})
    return pd.DataFrame(rows)


def entropy_balance(treated: pd.DataFrame, control: pd.DataFrame,
                    covariates: list[str]) -> np.ndarray | None:
    """Weights making the control group's covariate means match the treated group.

    Reweighting on the moments themselves rather than on a fitted propensity,
    so balance holds by construction instead of being checked after the fact.
    """
    target = treated[covariates].mean().to_numpy()
    matrix = control[covariates].to_numpy()
    scale = matrix.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    matrix = (matrix - target) / scale

    def dual(lam):
        exponent = matrix @ lam
        shift = exponent.max()
        return shift + np.log(np.exp(exponent - shift).sum())

    def gradient(lam):
        exponent = matrix @ lam
        exponent -= exponent.max()
        share = np.exp(exponent)
        share /= share.sum()
        return matrix.T @ share

    try:
        solution = optimize.minimize(dual, np.zeros(matrix.shape[1]), jac=gradient,
                                     method="BFGS", options={"maxiter": 2000})
    except Exception:
        return None
    if np.abs(gradient(solution.x)).max() > 1e-4:      # moments did not balance
        return None
    exponent = matrix @ solution.x
    exponent -= exponent.max()
    weights = np.exp(exponent)
    return weights / weights.mean()


def table(name: str, rows: list[dict] | pd.DataFrame, note: str = "") -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame([r for r in rows if r])
    TABLES[name] = frame
    frame.to_csv(RESULTS / ("%s.csv" % name), index=False)
    print("  %-26s %3d rows   %s" % (name, len(frame), note))


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def t01_descriptives(cs: pd.DataFrame) -> None:
    variables = [RATING, "rating_resource_use", "rating_innovation",
                 "ln_scope12", "ln_scope3", "ln_scope12_per_revenue",
                 "ln_scope12_per_assets", "ln_mcap", "ln_assets", "roa",
                 "leverage", "firm_age", "free_float", "reporting_scope",
                 "waste_recycled_rate"]
    pooled = cs[variables].describe().T.reset_index().rename(columns={"index": "variable"})
    pooled.insert(0, "sector", "All")

    by_sector = []
    for sector, group in cs.groupby("sector"):
        summary = group[[RATING, "ln_scope12", "ln_scope12_per_revenue",
                         "ln_assets"]].describe().T.reset_index()
        summary = summary.rename(columns={"index": "variable"})
        summary.insert(0, "sector", sector)
        by_sector.append(summary)

    table("t01_descriptives", pd.concat([pooled] + by_sector, ignore_index=True),
          "pooled and by sector")


def t02_selection_score(panel: pd.DataFrame) -> None:
    """Who is rated at all: the first of the two selection gates."""
    latest = (panel.sort_values("year")
                   .drop_duplicates(subset="RIC", keep="last").copy())
    latest["is_scored"] = latest["scored"].astype(int)

    rows = []
    comparison = []
    for variable in ["ln_mcap", "ln_assets", "ln_revenue", "roa", "leverage",
                     "firm_age", "free_float"]:
        if variable not in latest.columns:
            continue
        scored = latest.loc[latest["is_scored"] == 1, variable].dropna()
        unscored = latest.loc[latest["is_scored"] == 0, variable].dropna()
        if len(scored) < 10 or len(unscored) < 10:
            continue
        pooled_sd = np.sqrt((scored.var(ddof=1) + unscored.var(ddof=1)) / 2.0)
        comparison.append({
            "variable": variable,
            "scored_mean": scored.mean(), "scored_n": len(scored),
            "unscored_mean": unscored.mean(), "unscored_n": len(unscored),
            "difference": scored.mean() - unscored.mean(),
            "standardised_difference": ((scored.mean() - unscored.mean()) / pooled_sd
                                        if pooled_sd else np.nan)})
    rows.extend(comparison)

    model, frame = logit(latest, "is_scored",
                         ["ln_mcap", "ln_assets", "roa", "leverage", "firm_age",
                          "free_float"], ["sector", "country_group"])
    if model is not None:
        for name in model.params.index:
            if name.startswith("C(") or name == "Intercept":
                continue
            # Raw logit coefficients sit on the units of their own covariate, so a
            # ratio and a log cannot be ranked against each other. The per-standard-
            # deviation figure puts them on one footing.
            rows.append({"variable": "logit: %s" % name,
                         "difference": model.params[name],
                         "se": model.bse[name],
                         "standardised_difference": model.pvalues[name],
                         "per_sd": model.params[name] * frame[name].std(ddof=1),
                         "scored_n": int(model.nobs)})
    table("t02_selection_score", rows, "scored against unscored firms")


def t03_correlations(cs: pd.DataFrame) -> None:
    variables = [RATING, "rating_resource_use", "rating_innovation",
                 "rating_emissions_theme", "rating_pillar", "ln_mcap", "ln_assets",
                 "ln_revenue", "roa", "leverage", "firm_age", "free_float",
                 "ln_scope12", "ln_scope12_per_revenue"]
    variables = [v for v in variables if v in cs.columns]
    correlations = cs[variables].corr().round(3).reset_index().rename(
        columns={"index": "variable"})
    correlations.insert(0, "panel", "correlation")

    predictors = [RATING, "ln_mcap", "ln_assets", "roa", "leverage", "firm_age",
                  "free_float"]
    vif = variance_inflation(cs, [p for p in predictors if p in cs.columns])
    vif.insert(0, "panel", "variance inflation")
    table("t03_correlations", pd.concat([correlations, vif], ignore_index=True),
          "correlations and variance inflation")


def t04_main(cs: pd.DataFrame) -> None:
    """The central question, with and without industry fixed effects."""
    controls = ["ln_mcap"]
    rows = []
    for outcome in ["ln_scope12", "ln_scope3",
                    "ln_scope12_per_revenue", "ln_scope3_per_revenue",
                    "ln_scope12_per_assets", "ln_scope3_per_assets"]:
        if outcome not in cs.columns:
            continue
        rows.append(fit(cs, outcome, [RATING] + controls, ["country_group"]))
        rows.append(fit(cs, outcome, [RATING] + controls, ["country_group", "sector"]))
    table("t04_main", rows, "absolute outcomes and intensities")


def t05_denominators(cs: pd.DataFrame) -> None:
    """The same relationship under every available scaling variable."""
    rows = []
    for denominator in DENOMINATORS:
        for prefix in ("ln_scope12_per_%s", "ln_scope12_per_%s_sector_demeaned"):
            outcome = prefix % denominator
            if outcome in cs.columns:
                rows.append(fit(cs, outcome, [RATING, "ln_mcap"],
                                ["country_group", "sector"]))

    # The same five denominators with country effects only, so whether an
    # estimate survives the removal of industry controls can be read off for
    # each of them rather than for revenue and assets alone.
    for denominator in DENOMINATORS:
        outcome = "ln_scope12_per_%s" % denominator
        if outcome in cs.columns:
            rows.append(fit(cs, outcome, [RATING, "ln_mcap"], ["country_group"]))
    table("t05_denominators", rows, "six denominators")


def t06_components(cs: pd.DataFrame) -> None:
    """Themes separately and alternative aggregations of them."""
    measures = [RATING, "rating_resource_use", "rating_innovation",
                "rating_emissions_theme", "rating_three_theme", "rating_pca",
                "rating_pillar", "rating_esg"]
    rows = []
    for measure in measures:
        if measure not in cs.columns:
            continue
        for outcome in ["ln_scope12", "ln_scope12_per_revenue"]:
            rows.append(fit(cs, outcome, [measure, "ln_mcap"],
                            ["country_group", "sector"]))
    # Both themes in one model. Entered separately they are two univariate
    # readings of correlated measures; entered together the question of whether
    # they differ from each other can be put to a test instead of asserted.
    themes = ["rating_resource_use", "rating_innovation"]
    if all(theme in cs.columns for theme in themes):
        for outcome in ["ln_scope12", "ln_scope12_per_revenue"]:
            result, _ = estimate(cs, outcome, themes + ["ln_mcap"],
                                 ["country_group", "sector"])
            if result is None:
                continue
            for theme in themes:
                rows.append({"outcome": outcome, "predictor": theme,
                             "coef": result.params[theme], "se": result.bse[theme],
                             "t": result.tvalues[theme], "p": result.pvalues[theme],
                             "n": int(result.nobs), "r2": result.rsquared,
                             "controls": "ln_mcap | both themes entered jointly",
                             "fixed_effects": "country_group, sector",
                             "weighted": "no"})
            contrast = np.zeros(len(result.params))
            names = list(result.params.index)
            contrast[names.index(themes[0])] = 1.0
            contrast[names.index(themes[1])] = -1.0
            test = result.t_test(contrast)
            rows.append({"outcome": outcome,
                         "predictor": "equality test: resource use = innovation",
                         "coef": float(np.squeeze(test.effect)),
                         "se": float(np.squeeze(test.sd)),
                         "t": float(np.squeeze(test.tvalue)),
                         "p": float(np.squeeze(test.pvalue)),
                         "n": int(result.nobs), "r2": result.rsquared,
                         "controls": "difference between the two theme coefficients",
                         "fixed_effects": "country_group, sector",
                         "weighted": "no"})

    both = cs[["rating_resource_use", "rating_innovation"]].dropna()
    rows.append({"outcome": "correlation between themes", "predictor": "resource use, innovation",
                 "coef": both.corr().iloc[0, 1], "n": len(both)})
    table("t06_components", rows, "themes and weighting alternatives")


def t07_industry(cs: pd.DataFrame) -> None:
    """Fixed effects, the intensive split, and the interaction."""
    rows = []
    for outcome in ["ln_scope12", "ln_scope12_per_revenue"]:
        if outcome not in cs.columns:
            continue
        rows.append(fit(cs, outcome, [RATING, "ln_mcap"], ["country_group"]))
        rows.append(fit(cs, outcome, [RATING, "ln_mcap"], ["country_group", "sector"]))
        rows.append(fit(cs, outcome, [RATING, "ln_mcap"],
                        ["country_group", "business_sector"]))

        for label, subset in (("carbon intensive", cs[cs["carbon_intensive"] == 1]),
                              ("other sectors", cs[cs["carbon_intensive"] == 0])):
            row = fit(subset, outcome, [RATING, "ln_mcap"], ["country_group", "sector"])
            if row:
                row["controls"] += " | %s" % label
                rows.append(row)

        # Sector fixed effects absorb the level of a sector-level moderator but
        # not its interaction with a score that varies within sectors, so each
        # interaction is estimated both ways: with country effects alone, where
        # the level is identified, and with sector effects retained, where it
        # is absorbed and only the interaction is read.
        interaction = cs.copy()
        interaction["rating_x_intensive"] = interaction[RATING] * interaction["carbon_intensive"]
        row = fit(interaction, outcome,
                  ["rating_x_intensive", RATING, "carbon_intensive", "ln_mcap"],
                  ["country_group"])
        if row:
            row["controls"] = "interaction with carbon-intensive sector"
            rows.append(row)
        row = fit(interaction, outcome, ["rating_x_intensive", RATING, "ln_mcap"],
                  ["country_group", "sector"])
        if row:
            row["controls"] = "interaction with carbon-intensive sector | sector effects"
            rows.append(row)

        if "sector_carbon_intensity" in cs.columns:
            continuous = cs.copy()
            continuous["rating_x_sector_intensity"] = (
                continuous[RATING] * continuous["sector_carbon_intensity"])
            row = fit(continuous, outcome,
                      ["rating_x_sector_intensity", RATING, "sector_carbon_intensity",
                       "ln_mcap"], ["country_group"])
            if row:
                row["controls"] = "interaction with sector carbon intensity"
                rows.append(row)
            row = fit(continuous, outcome,
                      ["rating_x_sector_intensity", RATING, "ln_mcap"],
                      ["country_group", "sector"])
            if row:
                row["controls"] = "interaction with sector carbon intensity | sector effects"
                rows.append(row)
    table("t07_industry", rows, "fixed effects, split, interaction")


def t08_availability(cs: pd.DataFrame) -> None:
    """The second selection gate, and estimates reweighted for it."""
    rows = []
    covariates = ["rating", "ln_mcap", "ln_assets", "roa", "leverage", "firm_age",
                  "free_float"]
    covariates = [c for c in covariates if c in cs.columns]

    def coefficients(model, outcome: str, note: str = "sector, country") -> None:
        for name in model.params.index:
            if name.startswith("C(") or name == "Intercept":
                continue
            rows.append({"outcome": outcome, "predictor": name,
                         "coef": model.params[name], "se": model.bse[name],
                         "p": model.pvalues[name], "n": int(model.nobs),
                         "r2": model.prsquared, "controls": note,
                         "fixed_effects": "sector, country", "weighted": "logit"})

    for outcome in ["scope12_available", "scope3_available"]:
        model, frame = logit(cs, outcome, covariates, ["sector", "country_group"])
        if model is None:
            continue
        coefficients(model, outcome)

        # Inverse probability weights, with the overlap actually reported.
        propensity = model.predict(frame).clip(0.02, 0.98)
        weighted = cs.loc[frame.index].copy()
        weighted["ipw"] = np.where(frame[outcome] == 1, 1.0 / propensity,
                                   1.0 / (1.0 - propensity))
        weighted["ipw"] = weighted["ipw"] / weighted["ipw"].mean()
        effective = weighted["ipw"].sum() ** 2 / (weighted["ipw"] ** 2).sum()
        rows.append({"outcome": outcome, "predictor": "propensity overlap",
                     "coef": propensity.min(), "se": propensity.max(),
                     "n": len(propensity), "r2": effective,
                     "controls": "min and max propensity; r2 column is effective sample size",
                     "weighted": "diagnostic"})

        target = "ln_scope12" if outcome == "scope12_available" else "ln_scope3"
        intensity = target + "_per_revenue"
        for response in (target, intensity):
            if response not in weighted.columns:
                continue
            rows.append(fit(weighted, response, [RATING, "ln_mcap"],
                            ["country_group", "sector"], weights="ipw"))
            # The same firms, unweighted. Without this the weighted estimate is
            # compared with a main model estimated on a different complete-case
            # sample, and weighting and sample cannot be told apart.
            row = fit(weighted, response, [RATING, "ln_mcap"],
                      ["country_group", "sector"])
            if row:
                row["controls"] += " | unweighted on the weighting sample"
                rows.append(row)

        # Entropy balancing. The estimation sample is the firms whose figure is
        # observed, so they are reweighted until their covariate means match the
        # whole rated population, observed and unobserved together. Balance holds
        # by construction rather than being hoped for after weighting.
        balance_covariates = [c for c in ["ln_mcap", "ln_assets", "roa", "leverage",
                                          "firm_age"] if c in cs.columns]
        complete = cs.dropna(subset=balance_covariates + [outcome]).copy()
        observed = complete[complete[outcome] == 1]
        if len(observed) > 50 and len(complete) > len(observed) + 20:
            weights = entropy_balance(complete, observed, balance_covariates)
            if weights is not None:
                balanced = observed.copy()
                balanced["eb"] = weights
                row = fit(balanced, target, [RATING, "ln_mcap"],
                          ["country_group", "sector"], weights="eb")
                baseline = fit(balanced, target, [RATING, "ln_mcap"],
                               ["country_group", "sector"])
                if baseline:
                    baseline["controls"] += " | unweighted on the balancing sample"
                    rows.append(baseline)
                if row:
                    row["weighted"] = "entropy balanced"
                    rows.append(row)
                for covariate in balance_covariates:
                    rows.append({
                        "outcome": outcome,
                        "predictor": "balance: %s" % covariate,
                        "coef": complete[covariate].mean(),
                        "se": observed[covariate].mean(),
                        "t": np.average(observed[covariate], weights=weights),
                        "n": len(observed),
                        "controls": "population mean | observed mean | reweighted mean",
                        "weighted": "entropy balanced"})
    # The stricter reading of the gate. The models above take a figure's
    # existence as the outcome; these two require the firm's own number, first
    # against the whole rated cross-section and then among the firms carrying
    # both a figure and an estimation-method flag, where the flag itself is
    # the outcome. If the score only predicted the presence of a vendor
    # estimate, it would lose its association here.
    for outcome in ("scope12_firm_reported", "scope12_reported_strict",
                    "reported_given_available"):
        if outcome not in cs.columns:
            continue
        model, _ = logit(cs, outcome, covariates, ["sector", "country_group"])
        if model is not None:
            coefficients(model, outcome, "sector, country | firm-reported figure")

    table("t08_availability", rows, "selection and reweighting")


def t09_lagged(lag: pd.DataFrame) -> None:
    """Does an earlier rating predict later outcomes."""
    rows = []
    predictors = [c for c in ("rating_base", "rating_resource_use", "rating_innovation")
                  if c in lag.columns]
    outcomes = [c for c in ("ln_scope12", "ln_scope3", "ln_scope12_per_revenue",
                            "ln_scope12_per_assets", "ln_scope12_per_employees",
                            "ln_scope12_per_energy") if c in lag.columns]
    for predictor in predictors:
        for outcome in outcomes:
            rows.append(fit(lag, outcome, [predictor, "ln_mcap"],
                            ["country_group", "sector"]))
    if {"rating_base", "rating_outcome"}.issubset(lag.columns):
        change = lag.dropna(subset=["rating_base", "rating_outcome"]).copy()
        change["rating_change"] = change["rating_outcome"] - change["rating_base"]
        for outcome in outcomes:
            row = fit(change, outcome, ["rating_change", "rating_base", "ln_mcap"],
                      ["country_group", "sector"])
            if row:
                row["controls"] = "change in rating over the window"
                rows.append(row)

    # Firms whose 2021 and 2025 fiscal years both carry an authoritative date.
    # The lagged design was added to answer a temporal criticism, so it should
    # not rest on years assigned by position within the extract.
    flags = ["year_imputed_base", "year_imputed_outcome"]
    if set(flags).issubset(lag.columns):
        dated = lag[~(lag[flags[0]].astype(bool) | lag[flags[1]].astype(bool))]
        for outcome in outcomes:
            row = fit(dated, outcome, ["rating_base", "ln_mcap"],
                      ["country_group", "sector"])
            if row:
                row["controls"] += " | dated observations only"
                rows.append(row)
    table("t09_lagged", rows, "earlier rating, later outcome")


def t10_waste(cs: pd.DataFrame) -> None:
    rows = []
    for outcome in ["ln_waste_recycled", "waste_recycled_rate", "ln_waste_total",
                    "hazardous_share"]:
        if outcome not in cs.columns:
            continue
        rows.append(fit(cs, outcome, [RATING, "ln_mcap"], ["country_group"]))
        rows.append(fit(cs, outcome, [RATING, "ln_mcap"], ["country_group", "sector"]))
    table("t10_waste", rows, "recycling rate and tonnage")


def t11_robustness(cs: pd.DataFrame, panel: pd.DataFrame) -> None:
    rows = []

    # Firm-reported emissions only, excluding vendor estimates.
    if "emissions_reported" in cs.columns:
        reported = cs[cs["emissions_reported"] == 1]
        for outcome in ("ln_scope12", "ln_scope12_per_revenue"):
            row = fit(reported, outcome, [RATING, "ln_mcap"], ["country_group", "sector"])
            if row:
                row["controls"] += " | firm-reported emissions only"
                rows.append(row)

    # Whether the reported and vendor-estimated firms actually differ, rather
    # than one subsample reaching significance and the other not. Estimated on
    # the firms carrying either flag, so both groups are the same population.
    if "emissions_reported" in cs.columns:
        flagged = cs[cs["emissions_reported"].notna()].copy()
        flagged["rating_x_reported"] = flagged[RATING] * flagged["emissions_reported"]
        for outcome in ("ln_scope12", "ln_scope12_per_revenue"):
            row = fit(flagged, outcome,
                      ["rating_x_reported", RATING, "emissions_reported", "ln_mcap"],
                      ["country_group", "sector"])
            if row:
                row["predictor"] = "rating × firm-reported"
                row["controls"] = "interaction with the firm-reported flag"
                rows.append(row)

    # Excluding firm-years whose fiscal year was assigned by position.
    if "year_imputed" in cs.columns:
        dated = cs[~cs["year_imputed"].astype(bool)]
        for outcome in ("ln_scope12", "ln_scope12_per_revenue"):
            row = fit(dated, outcome, [RATING, "ln_mcap"], ["country_group", "sector"])
            if row:
                row["controls"] += " | dated observations only"
                rows.append(row)

    # Alternative country groupings.
    for cutoff in (10, 50):
        column = "country_group_%d" % cutoff
        if column not in cs.columns:
            continue
        for outcome in ("ln_scope12", "ln_scope12_per_revenue"):
            row = fit(cs, outcome, [RATING, "ln_mcap"], [column, "sector"], cluster=column)
            if row:
                row["controls"] += " | country cutoff %d" % cutoff
                rows.append(row)

    # Winsorised ratios.
    for outcome in ("roa_w", "leverage_w"):
        if outcome in cs.columns:
            rows.append(fit(cs, "ln_scope12", [RATING, "ln_mcap", outcome],
                            ["country_group", "sector"]))

    # Pooled firm-years with year effects, as an alternative to one cross-section.
    pooled = panel[panel["scored"]].copy()
    for outcome in ("ln_scope12", "ln_scope12_per_revenue"):
        if outcome in pooled.columns:
            row = fit(pooled, outcome, [RATING, "ln_mcap"],
                      ["country_group", "sector", "year"])
            if row:
                row["controls"] += " | pooled firm-years"
                rows.append(row)
    table("t11_robustness", rows, "alternative specifications")


# --------------------------------------------------------------------------

def write_summary() -> None:
    lines = ["Results", "=" * 78, ""]
    for name, frame in TABLES.items():
        lines += [name, "-" * 78, frame.to_string(index=False, max_colwidth=46), ""]
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    global DATA, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="directory holding the constructed analysis files")
    parser.add_argument("--results-dir", type=Path, default=RESULTS,
                        help="directory to write the tables to")
    arguments = parser.parse_args()
    DATA, RESULTS = arguments.data_dir, arguments.results_dir

    RESULTS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(DATA / "analysis_panel.csv", low_memory=False)
    cs = pd.read_csv(DATA / "analysis_cross_section.csv", low_memory=False)
    lag = pd.read_csv(DATA / "analysis_lagged.csv", low_memory=False)
    print("Cross-section %d firms, panel %d firm-years, lagged %d firms\n"
          % (len(cs), len(panel), len(lag)))

    t01_descriptives(cs)
    t02_selection_score(panel)
    t03_correlations(cs)
    t04_main(cs)
    t05_denominators(cs)
    t06_components(cs)
    t07_industry(cs)
    t08_availability(cs)
    t09_lagged(lag)
    t10_waste(cs)
    t11_robustness(cs, panel)

    write_summary()
    print("\nWritten to %s" % RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
