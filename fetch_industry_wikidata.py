#!/usr/bin/env python3
"""
Rebuild industry_wikidata.csv: an industry/sector classification for the sample firms,
sourced from the free, openly licensed Wikidata knowledge base (no commercial licence required).

For each firm it queries Wikidata by company name, reads the "industry" property (P452),
and maps the granular industry label to one of eleven broad sectors (plus "Other").

USAGE:
  pip install pandas requests
  python fetch_industry_wikidata.py \
      --cross-section data/refinitiv_clean_fy0_cross_section.csv \
      --output industry_wikidata.csv

Notes:
  * Coverage is partial (~55-60%): not all firms have an industry recorded in Wikidata,
    and coverage skews to larger / better-known firms.
  * The script is polite to the public API (a short delay between requests); a full run
    takes roughly an hour for ~2,000 firms. It checkpoints and resumes.
"""
from __future__ import annotations
import argparse, os, re, time
import pandas as pd
import requests

RU = "Resource Use Score (FY0)"
EI = "Environmental Innovation Score (FY0)"
API = "https://www.wikidata.org/w/api.php"
SUFFIX = re.compile(r'\b(SA|AG|SE|NV|PLC|SpA|Ltd|Limited|Inc|Corp|Corporation|Co|Company|'
                    r'Group|Holding|Holdings|AB|ASA|Oyj|SAS|GmbH|KGaA|Bhd|PCL|Tbk|PT|Oy)\b\.?', re.I)

S = requests.Session()
S.headers.update({"User-Agent": "academic-research/1.0 (industry classification)"})


def _get(params, tries=4):
    for t in range(tries):
        try:
            r = S.get(API, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            time.sleep(2 * (t + 1))
        except Exception:
            time.sleep(2 * (t + 1))
    return None


def _label(qid):
    j = _get({"action": "wbgetentities", "ids": qid, "props": "labels",
              "languages": "en", "format": "json"})
    try:
        return j["entities"][qid]["labels"]["en"]["value"]
    except Exception:
        return None


def industry(name):
    clean = SUFFIX.sub("", str(name)).strip().strip(",").strip()
    for q in ([clean, name] if clean and clean != name else [name]):
        j = _get({"action": "wbsearchentities", "search": q, "language": "en",
                  "format": "json", "type": "item", "limit": 5})
        if not j:
            continue
        for hit in j.get("search", []):
            c = _get({"action": "wbgetclaims", "entity": hit["id"],
                      "property": "P452", "format": "json"})
            cl = (c or {}).get("claims", {}).get("P452", [])
            if cl:
                try:
                    return _label(cl[0]["mainsnak"]["datavalue"]["value"]["id"])
                except Exception:
                    continue
    return None


def to_sector(ind):
    if not isinstance(ind, str):
        return None
    s = ind.lower()
    if re.search(r'petrol|oil|gas(?!e)|coal|fossil|upstream', s): return 'Energy'
    if re.search(r'electric|electricity|utility|utilit|water supply|power', s): return 'Utilities'
    if re.search(r'chemic|steel|mining|metal|cement|paper|pulp|materials|forestry|plastic|glass|mineral', s): return 'Materials'
    if re.search(r'construction|machin|aerospace|defen|weapon|engineer|transport|logistic|railway|airline|industrial|manufactur|shipbuild|aviation|aircraft', s): return 'Industrials'
    if re.search(r'pharmac|health|biotech|medical|medicine|hospital|life science', s): return 'Health Care'
    if re.search(r'bank|insurance|financ|investment|asset manage|holding company|capital market|credit', s): return 'Financials'
    if re.search(r'real estate|property|reit', s): return 'Real Estate'
    if re.search(r'software|semiconductor|information technology|computer|internet|electronics|technology|data', s): return 'Information Technology'
    if re.search(r'telecom|communication|broadcast|publish|media', s): return 'Communication'
    if re.search(r'food|beverage|tobacco|agricultur|household|brewery|dairy|consumer goods', s): return 'Consumer Staples'
    if re.search(r'automotiv|retail|tourism|apparel|hospitality|hotel|restaurant|leisure|gaming|consumer|fashion|luxury|e-commerce', s): return 'Consumer Discretionary'
    return 'Other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-section", default="data/refinitiv_clean_fy0_cross_section.csv")
    ap.add_argument("--output", default="industry_wikidata.csv")
    ap.add_argument("--delay", type=float, default=0.35)
    a = ap.parse_args()

    df = pd.read_csv(a.cross_section)
    firms = df[df[RU].notna() & df[EI].notna()][["Identifier", "Company Name"]]

    done = {}
    if os.path.exists(a.output):
        prev = pd.read_csv(a.output)
        done = {r.Identifier: (r.industry_raw, r.sector) for r in prev.itertuples()
                if isinstance(getattr(r, "industry_raw", None), str)}

    rows = []
    for i, (rid, name) in enumerate(firms.itertuples(index=False)):
        if rid in done:
            ind, sec = done[rid]
        else:
            ind = industry(name)
            sec = to_sector(ind)
            time.sleep(a.delay)
        if isinstance(ind, str) and ind:
            rows.append({"Identifier": rid, "industry_raw": ind, "sector": sec})
        if i % 50 == 0:
            pd.DataFrame(rows).to_csv(a.output, index=False)
            print(f"[{i}/{len(firms)}] classified {len(rows)}")
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print(f"done: {len(rows)} firms classified -> {a.output}")


if __name__ == "__main__":
    main()
