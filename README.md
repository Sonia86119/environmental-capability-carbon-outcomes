# Replication package — *Do Corporate Environmental Capability Scores Track Carbon Outcomes?*

Chalak, Keshtkar & Bidollahkhany. This package reproduces every table in the paper.

**The raw LSEG/Refinitiv data is proprietary and is NOT included** (licence prohibits
redistribution). The package ships the analysis code, a variable dictionary, and a synthetic
demonstration dataset so the pipeline can be run end-to-end without the licensed data.

## Data source
The industry/sector classification used in the industry-fixed-effects robustness check is from the openly available **Wikidata** knowledge base (`industry_wikidata.csv`, rebuilt by `fetch_industry_wikidata.py`); it is not proprietary and is included in this repository.

All other firm-level variables are drawn from **LSEG / Refinitiv** (London Stock Exchange Group, Data &
Analytics): ESG theme and pillar scores, CO2-equivalent emissions (Scope 1, 2 and 3), waste
recycled, total assets, market capitalisation, and the TRBC industry classification, exported as a
fiscal-year-0 (FY0) cross-section of listed firms. The data are accessed under an institutional
LSEG/Refinitiv licence and are cited in the paper as LSEG (2024). They cannot be redistributed;
see the data-access protocol below for how a licensed user regenerates the inputs.

## Contents
| File | Purpose |
|---|---|
| `reproduce_paper3.py` | Regenerates Tables 3, 4, 5 (and industry-FE robustness when available) |
| `variable_dictionary.csv` | Definition, operationalisation, source field and role of every variable |
| `industry_wikidata.csv` | Industry/sector per firm (public, from Wikidata) for the industry-FE robustness |
| `fetch_industry_wikidata.py` | Rebuilds the industry classification from Wikidata (free, no licence) |
| `make_figures.py` | Regenerates Figures 1-3 (PNG + vector PDF) from the source data |
| `make_synthetic_demo.py` | Generates a synthetic dataset with the same schema and relationships |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes all data files from version control |

## Quick start (no licensed data needed)
```bash
pip install -r requirements.txt
mkdir -p data
python make_synthetic_demo.py --out data/synthetic_demo.csv
python reproduce_paper3.py --data-dir data --cross-section synthetic_demo.csv
```
The synthetic output reproduces the *structure* of the findings (disclosure selection, positive
total, weak/null intensity) but contains NO real company values and must not be cited as results.

## Reproducing the published numbers (licensed users)
A user with an LSEG/Refinitiv licence regenerates the inputs, places them in `data/`, then runs:
```bash
python reproduce_paper3.py --data-dir data
```
Expected `data/` files:
- `refinitiv_clean_fy0_cross_section.csv` — main cross-section (see data-access protocol below)
- `waste_recycled.csv` — `Identifier, waste_recycled` (Table 5)
- `identifiers_trbc.csv` — `Identifier, TRBC_Economic_Sector` (industry-FE robustness)

## Data-access protocol (how to regenerate the proprietary inputs)
The inputs are built from an LSEG/Refinitiv screener export of listed firms with the following
fields at FY0: Resource Use Score, Environmental Innovation Score, Environmental Pillar Score,
CO2 Equivalent Emissions (Total/Scope 1/Scope 2/Scope 3), Total Assets, Company Market Cap,
Waste Recycled Total, plus Country of Headquarters. The TRBC sector classification is pulled with
the helper in `trbc_lseg_pull_package/` (run inside LSEG Workspace). Field-to-variable mapping
is documented in `variable_dictionary.csv`.

## Method summary (matches the manuscript)
- Capability score = equal-weight mean of Resource Use and Environmental Innovation theme scores
  (Environmental Pillar excluded to avoid double counting).
- Continuous variables winsorised at the 1st/99th percentiles; outcomes in natural logs (zeros dropped).
- Country-group fixed effects (HQ countries with < 25 firms collapsed to "Other"); HC1 robust SEs.
- IPW: stabilised weights from the disclosure model, truncated at the 1st/99th percentiles.
