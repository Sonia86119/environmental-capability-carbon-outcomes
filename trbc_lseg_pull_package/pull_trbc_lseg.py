#!/usr/bin/env python3
"""
Pull TRBC Economic Sector and Business Sector from LSEG Workspace/Data Library.

Input:
  CSV with a RIC column, for example:
      Identifier,RIC
      firm_001,BARC.L
      firm_002,TRI.N
      firm_003,TSLA.O

Output:
  CSV with:
      Identifier, RIC, TRBC_Economic_Sector, TRBC_Economic_Sector_Code,
      TRBC_Business_Sector, TRBC_Business_Sector_Code, plus optional lower TRBC levels.

Requirements:
  pip install lseg-data pandas

Before running:
  1. Open LSEG Workspace on the same machine.
  2. Make sure you are logged in.
  3. Run this script in the same user session.

Usage:
  python pull_trbc_lseg.py --input identifiers.csv --ric-column RIC --id-column Identifier --output identifiers_trbc.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


TRBC_FIELDS = [
    "TR.TRBCEconomicSector",
    "TR.TRBCEconSectorCode",
    "TR.TRBCBusinessSector",
    "TR.TRBCBusinessSectorCode",
    # Optional but useful for robustness / reviewer appendix:
    "TR.TRBCIndustryGroup",
    "TR.TRBCIndustryGroupCode",
    "TR.TRBCIndustry",
    "TR.TRBCIndustryCode",
    "TR.TRBCActivity",
    "TR.TRBCActivityCode",
]

OUTPUT_RENAME = {
    "Instrument": "RIC",
    "TRBC Economic Sector Name": "TRBC_Economic_Sector",
    "TRBC Economic Sector Code": "TRBC_Economic_Sector_Code",
    "TRBC Business Sector Name": "TRBC_Business_Sector",
    "TRBC Business Sector Code": "TRBC_Business_Sector_Code",
    "TRBC Industry Group Name": "TRBC_Industry_Group",
    "TRBC Industry Group Code": "TRBC_Industry_Group_Code",
    "TRBC Industry Name": "TRBC_Industry",
    "TRBC Industry Code": "TRBC_Industry_Code",
    "TRBC Activity Name": "TRBC_Activity",
    "TRBC Activity Code": "TRBC_Activity_Code",
}


def read_identifiers(path: Path, ric_column: str, id_column: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ric_column not in df.columns:
        raise ValueError(f"RIC column '{ric_column}' not found. Available columns: {list(df.columns)}")
    if id_column and id_column not in df.columns:
        raise ValueError(f"Identifier column '{id_column}' not found. Available columns: {list(df.columns)}")

    keep_cols = [ric_column] + ([id_column] if id_column else [])
    df = df[keep_cols].copy()
    df[ric_column] = df[ric_column].astype(str).str.strip()
    df = df[df[ric_column].ne("") & df[ric_column].ne("nan")]
    df = df.drop_duplicates(subset=[ric_column])
    return df


def pull_trbc_with_lseg_data(rics: list[str]) -> pd.DataFrame:
    import lseg.data as ld

    # Uses the desktop Workspace session if Workspace is open and logged in.
    ld.open_session()
    try:
        out = ld.get_data(
            universe=rics,
            fields=TRBC_FIELDS,
        )
    finally:
        ld.close_session()

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Input CSV containing RICs.")
    parser.add_argument("--ric-column", default="RIC", help="Column containing RICs.")
    parser.add_argument("--id-column", default="Identifier", help="Optional identifier column to preserve.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    args = parser.parse_args()

    id_col = args.id_column if args.id_column else None
    identifiers = read_identifiers(args.input, args.ric_column, id_col)
    rics = identifiers[args.ric_column].tolist()

    if not rics:
        raise ValueError("No valid RICs found in input file.")

    raw = pull_trbc_with_lseg_data(rics)
    raw = raw.rename(columns=OUTPUT_RENAME)

    if "RIC" not in raw.columns and "Instrument" in raw.columns:
        raw = raw.rename(columns={"Instrument": "RIC"})

    # Merge back to preserve original Identifier values and ordering.
    left = identifiers.rename(columns={args.ric_column: "RIC"})
    merged = left.merge(raw, on="RIC", how="left")

    # Basic QA columns
    merged["TRBC_Economic_Sector_missing"] = merged["TRBC_Economic_Sector"].isna() if "TRBC_Economic_Sector" in merged.columns else True
    merged["TRBC_Business_Sector_missing"] = merged["TRBC_Business_Sector"].isna() if "TRBC_Business_Sector" in merged.columns else True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    missing_econ = int(merged["TRBC_Economic_Sector_missing"].sum())
    missing_bus = int(merged["TRBC_Business_Sector_missing"].sum())
    print(f"Wrote: {args.output}")
    print(f"Rows: {len(merged)}")
    print(f"Missing Economic Sector: {missing_econ}")
    print(f"Missing Business Sector: {missing_bus}")

    if missing_econ or missing_bus:
        print("Check whether those RICs are valid, active, correctly mapped, or covered by your LSEG licence.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
