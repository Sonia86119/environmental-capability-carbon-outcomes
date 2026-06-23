#!/usr/bin/env python3
"""
Generate a SYNTHETIC demonstration dataset with the same schema as the proprietary
LSEG/Refinitiv cross-section, so reviewers can run reproduce_paper3.py without the
licensed data. The synthetic data reproduces the *structure and relationships*
(disclosure selection, scale, positive total / null intensity) but contains NO real
company values and must not be interpreted as results.

Usage:
  python make_synthetic_demo.py --n 2000 --out data/synthetic_demo.csv
Then:
  python reproduce_paper3.py --data-dir data   # pointing at the synthetic file
"""
import argparse
import numpy as np
import pandas as pd

COUNTRIES = (["DE", "FR", "GB", "IT", "ES", "SE", "NL", "CH", "JP", "US"] * 30
             + ["AT", "BE", "FI", "NO", "DK", "PL"] * 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260623)
    ap.add_argument("--out", default="data/synthetic_demo.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = a.n

    ln_mcap = rng.normal(21.6, 1.7, n)
    cap = np.clip(rng.normal(51, 24, n) + 6 * (ln_mcap - 21.6), 0, 100)   # bigger firms score higher
    ln_assets = ln_mcap + rng.normal(0.5, 0.6, n)

    # disclosure selection: higher cap + size -> more likely to disclose
    z = -2.0 + 0.06 * cap + 0.3 * (ln_mcap - 21.6)
    disc = rng.random(n) < 1 / (1 + np.exp(-z))

    # total emissions scale with size, weakly with capability; intensity ~ flat in capability
    ln_co2 = 2.0 + 0.66 * ln_mcap + 0.015 * cap + rng.normal(0, 2.4, n)
    ln_co2_pa = ln_co2 - ln_assets
    co2_total = np.exp(ln_co2)
    co2_total[~disc] = np.nan

    # waste recycled: same selection logic, positive total, null intensity
    zw = -2.5 + 0.029 * cap + 0.15 * (ln_mcap - 21.6)
    discw = rng.random(n) < 1 / (1 + np.exp(-zw))
    ln_wr = 3.0 + 0.45 * ln_mcap + 0.022 * cap + rng.normal(0, 2.8, n)
    wr = np.exp(ln_wr); wr[~discw] = np.nan

    df = pd.DataFrame({
        "Identifier": [f"SYN{ i:05d}.XX" for i in range(n)],
        "Country of Headquarters": rng.choice(COUNTRIES, n),
        "Company Market Cap (USD)": np.exp(ln_mcap),
        "CO2 Equivalent Emissions Total (FY0)": co2_total,
        "CO2 Equivalent Emissions Indirect, Scope 3 (FY0)": np.where(disc, np.exp(ln_co2 + 2.8), np.nan),
        "Total Assets - Actual (FY0, USD)": np.exp(ln_assets),
        "Resource Use Score (FY0)": np.clip(cap + rng.normal(0, 10, n), 0, 100),
        "Environmental Innovation Score (FY0)": np.clip(cap + rng.normal(0, 12, n), 0, 100),
        "ln_co2_total": np.where(disc, ln_co2, np.nan),
        "ln_co2_per_assets": np.where(disc, ln_co2_pa, np.nan),
    })
    df.to_csv(a.out, index=False)
    # companion waste file
    pd.DataFrame({"Identifier": df["Identifier"], "waste_recycled": wr}).dropna() \
        .to_csv(a.out.replace("synthetic_demo.csv", "waste_recycled.csv"), index=False)
    print(f"wrote {a.out} ({n} rows) + waste_recycled.csv  [SYNTHETIC - not real data]")


if __name__ == "__main__":
    main()
