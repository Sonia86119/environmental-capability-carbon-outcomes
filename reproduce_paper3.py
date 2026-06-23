#!/usr/bin/env python3
"""
Reproduction script for the empirical paper
"Do Corporate Environmental Capability Scores Track Carbon Outcomes?"
(Chalak, Keshtkar & Bidollahkhany).

Reproduces every estimate in the paper from the cleaned LSEG/Refinitiv cross-section,
plus the waste-recycled robustness (Table 5). Designed to be shipped as supplementary
material in place of the raw data, which is proprietary and cannot be redistributed.

INPUTS (place in --data-dir):
  refinitiv_clean_fy0_cross_section.csv   main cross-section (one row per firm)
  waste_recycled.csv                      optional: Identifier, waste_recycled  (Table 5)
  identifiers_trbc.csv                    optional: Identifier, TRBC_Economic_Sector (industry-FE)

USAGE:
  pip install pandas numpy statsmodels openpyxl
  python reproduce_paper3.py --data-dir .

DESIGN NOTES (match the manuscript):
  * Capability score = equal-weight mean of Resource Use and Environmental Innovation theme scores.
  * Continuous variables winsorised at the 1st/99th percentiles.
  * Country-group fixed effects: HQ countries with < 25 firms collapsed to "Other".
  * HC1 robust standard errors throughout; emissions/outcomes enter in natural logs (zeros dropped).
  * IPW: stabilised weights from the disclosure model, truncated at the 1st/99th percentiles.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

RU = "Resource Use Score (FY0)"
EI = "Environmental Innovation Score (FY0)"
TA = "Total Assets - Actual (FY0, USD)"
MC = "Company Market Cap (USD)"
S3 = "CO2 Equivalent Emissions Indirect, Scope 3 (FY0)"


def winz(s, lo=0.01, hi=0.99):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def build(data_dir: str, cross_section: str = "refinitiv_clean_fy0_cross_section.csv") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, cross_section))
    df["capability"] = df[[RU, EI]].mean(axis=1)
    df["RU"], df["EI"] = df[RU], df[EI]
    df["ln_mcap"] = np.log(df[MC])
    # logs already present for total/intensity; (re)build Scope 3 for completeness
    if S3 in df.columns:
        df["ln_co2_s3"] = np.log(pd.to_numeric(df[S3], errors="coerce").where(lambda x: x > 0))
        if TA in df.columns:
            den = pd.to_numeric(df[S3], errors="coerce") / df[TA]
            df["ln_co2_s3_assets"] = np.log(den.where(den > 0))
    # optional merges
    wr = os.path.join(data_dir, "waste_recycled.csv")
    if os.path.exists(wr):
        w = pd.read_csv(wr)
        df = df.merge(w, on="Identifier", how="left")
        df["waste_recycled"] = pd.to_numeric(df["waste_recycled"], errors="coerce")
        df["has_wr"] = (df["waste_recycled"].notna() & (df["waste_recycled"] > 0)).astype(int)
        df["ln_wr"] = np.log(df["waste_recycled"].where(df["waste_recycled"] > 0))
        if TA in df.columns:
            d = df["waste_recycled"] / df[TA]
            df["ln_wr_assets"] = np.log(d.where((df["waste_recycled"] > 0) & (df[TA] > 0)))
    trbc = os.path.join(data_dir, "identifiers_trbc.csv")
    if os.path.exists(trbc):
        t = pd.read_csv(trbc)[["Identifier", "TRBC_Economic_Sector"]].rename(
            columns={"TRBC_Economic_Sector": "sector"})
        df = df.merge(t, on="Identifier", how="left")
    else:
        # fall back to the public Wikidata-derived industry classification
        ind = os.path.join(os.path.dirname(__file__), "industry_wikidata.csv")
        if os.path.exists(ind):
            t = pd.read_csv(ind)[["Identifier", "sector"]]
            df = df.merge(t, on="Identifier", how="left")
    df = df.replace([np.inf, -np.inf], np.nan)
    cc = "Country of Headquarters"
    counts = df[cc].value_counts()
    df["cgroup"] = np.where(df[cc].isin(counts[counts < 25].index), "Other", df[cc])
    return df


def ols(df, dep, preds, weights=None, extra=""):
    cols = [dep] + preds + ["cgroup"] + ([weights] if weights else [])
    d = df.dropna(subset=cols).copy()
    for v in [dep] + preds:
        d[v] = winz(d[v])
    f = f"{dep} ~ " + " + ".join(preds) + " + C(cgroup)" + extra
    if weights:
        m = smf.wls(f, d, weights=d[weights]).fit(cov_type="HC1")
    else:
        m = smf.ols(f, d).fit(cov_type="HC1")
    return m, len(d)


def line(tag, m, n, focal="capability"):
    b, se, p = m.params[focal], m.bse[focal], m.pvalues[focal]
    r2 = getattr(m, "rsquared_adj", float("nan"))
    print(f"  {tag:42s} N={n:5d}  {focal}={b:+.4f} (SE {se:.4f}, p={p:.3g})  R2adj={r2:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--cross-section", default="refinitiv_clean_fy0_cross_section.csv",
                    help="cross-section filename inside --data-dir (use synthetic_demo.csv for the demo)")
    a = ap.parse_args()
    df = build(a.data_dir, a.cross_section)

    print("\n== TABLE 2: descriptive statistics (Scope 1+2 analytical sample) ==")
    samp = df.dropna(subset=["ln_co2_total", "capability", "ln_mcap", "cgroup"])
    desc_vars = [("capability", "Environmental capability score"), ("RU", "Resource-use score"),
                 ("EI", "Environmental-innovation score"), ("ln_co2_total", "Log Scope 1+2 emissions"),
                 ("ln_co2_s3", "Log Scope 3 emissions"), ("ln_co2_per_assets", "Log Scope 1+2 / assets"),
                 ("ln_mcap", "Log market capitalisation")]
    for v, lab in desc_vars:
        if v in samp:
            s = samp[v].dropna()
            print(f"  {lab:34s} N={len(s):5d} mean={s.mean():8.3f} SD={s.std():7.3f} "
                  f"P25={s.quantile(.25):8.3f} Md={s.median():8.3f} P75={s.quantile(.75):8.3f}")

    print("\n== APPENDIX A: disclosure and selection diagnostics (binomial GLM) ==")
    base = df.dropna(subset=["capability", "ln_mcap", "cgroup"]).copy()
    # disclosure = the emissions field is reported (raw field present), not the log
    TOTAL_RAW = "CO2 Equivalent Emissions Total (FY0)"
    base["disc_s12"] = base[TOTAL_RAW].notna().astype(int) if TOTAL_RAW in base else base["ln_co2_total"].notna().astype(int)
    if S3 in base:
        base["disc_s3"] = pd.to_numeric(base[S3], errors="coerce").notna().astype(int)
    elif "ln_co2_s3" in base:
        base["disc_s3"] = base["ln_co2_s3"].notna().astype(int)
    for dv, lab in [("disc_s12", "Scope 1+2 disclosure"), ("disc_s3", "Scope 3 disclosure")]:
        if dv not in base:
            continue
        gm = smf.glm(f"{dv} ~ capability + ln_mcap + C(cgroup)", base, family=sm.families.Binomial()).fit()
        for v in ["capability", "ln_mcap"]:
            print(f"  {lab:22s} {v:24s} b={gm.params[v]:+.4f} SE={gm.bse[v]:.4f} p={gm.pvalues[v]:.3g}")

    print("\n== TABLE 3: carbon-outcome models (capability coefficient) ==")
    line("M1 ln Scope1+2 total", *ols(df, "ln_co2_total", ["capability", "ln_mcap"]))
    line("M2 ln Scope1+2 / assets", *ols(df, "ln_co2_per_assets", ["capability", "ln_mcap"]))
    if "ln_co2_s3" in df:
        line("M3 ln Scope3 total", *ols(df, "ln_co2_s3", ["capability", "ln_mcap"]))
        line("M4 ln Scope3 / assets", *ols(df, "ln_co2_s3_assets", ["capability", "ln_mcap"]))

    print("\n== TABLE 4: IPW selection-weighted (Scope 1+2) ==")
    df_d = df.dropna(subset=["capability", "ln_mcap", "cgroup"]).copy()
    df_d["disc"] = df_d["ln_co2_total"].notna().astype(int)
    g = smf.glm("disc ~ capability + ln_mcap + C(cgroup)", df_d, family=sm.families.Binomial()).fit()
    df_d["p"] = g.predict(df_d)
    pD = df_d["disc"].mean()
    df_d["sw"] = np.where(df_d["disc"] == 1, pD / df_d["p"], (1 - pD) / (1 - df_d["p"]))
    df_d["sw"] = df_d["sw"].clip(df_d["sw"].quantile(.01), df_d["sw"].quantile(.99))
    dfw = df.merge(df_d[["Identifier", "sw"]], on="Identifier", how="left")
    line("IPW1 ln Scope1+2 total", *ols(dfw, "ln_co2_total", ["capability", "ln_mcap"], weights="sw"))
    line("IPW2 ln Scope1+2 / assets", *ols(dfw, "ln_co2_per_assets", ["capability", "ln_mcap"], weights="sw"))

    if "has_wr" in df:
        print("\n== TABLE 5: waste recycled (second physical outcome) ==")
        ds = df.dropna(subset=["capability", "ln_mcap", "cgroup"]).copy()
        gw = smf.glm("has_wr ~ capability + ln_mcap + C(cgroup)", ds, family=sm.families.Binomial()).fit()
        print(f"  W1 disclosure GLM                          N={int(gw.nobs):5d}  "
              f"capability={gw.params['capability']:+.4f} (SE {gw.bse['capability']:.4f}, p={gw.pvalues['capability']:.3g})")
        line("W2 ln total recycled", *ols(df, "ln_wr", ["capability", "ln_mcap"]))
        line("W3 ln recycled / assets", *ols(df, "ln_wr_assets", ["capability", "ln_mcap"]))
        ds["p"] = gw.predict(ds); pD = ds["has_wr"].mean()
        ds["sw"] = np.where(ds["has_wr"] == 1, pD / ds["p"], (1 - pD) / (1 - ds["p"]))
        ds["sw"] = ds["sw"].clip(ds["sw"].quantile(.01), ds["sw"].quantile(.99))
        dfw2 = df.merge(ds[["Identifier", "sw"]], on="Identifier", how="left")
        line("W4 IPW total recycled", *ols(dfw2, "ln_wr", ["capability", "ln_mcap"], weights="sw"))
        line("W5 IPW recycled / assets", *ols(dfw2, "ln_wr_assets", ["capability", "ln_mcap"], weights="sw"))

    if "sector" in df:
        n_cov = df.dropna(subset=["capability", "sector"]).shape[0]
        print(f"\n== ROBUSTNESS: industry (sector) fixed effects "
              f"[{n_cov} firms classified] ==")
        print("  outcome              baseline beta (p)        + industry-FE beta (p)")
        for dep, tag in [("ln_co2_total", "S1+2 total"), ("ln_co2_per_assets", "S1+2 intensity"),
                         ("ln_co2_s3", "S3 total"), ("ln_co2_s3_assets", "S3 intensity")]:
            if dep not in df:
                continue
            sub = df.dropna(subset=[dep, "capability", "ln_mcap", "cgroup", "sector"])
            b0 = ols(sub, dep, ["capability", "ln_mcap"])[0]
            b1 = ols(sub, dep, ["capability", "ln_mcap"], extra=" + C(sector)")[0]
            print(f"  {tag:<20} {b0.params['capability']:+.4f} (p={b0.pvalues['capability']:.3g})"
                  f"       {b1.params['capability']:+.4f} (p={b1.pvalues['capability']:.3g})  N={int(b1.nobs)}")
    else:
        print("\n(industry-FE skipped: industry_wikidata.csv not found)")


if __name__ == "__main__":
    main()