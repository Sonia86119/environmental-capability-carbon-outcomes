# Replication package

**What Do Corporate Environmental Ratings Measure? Scale, Disclosure and Carbon Outcomes in
European Listed Firms**

The package holds everything needed to reproduce the paper: the data-access protocol that
regenerates the licensed extracts, the code that builds the analysis files from them, the code
that estimates every model and writes every table, the figure generator, and a synthetic
demonstration dataset so that the whole pipeline can be run and inspected without a licence.

**The raw LSEG data are proprietary and are not included.** The licence does not permit
redistribution. A reader with an LSEG Workspace licence regenerates them with the protocol below;
a reader without one runs the same code on the synthetic extracts.

## Data source and licence

All firm-level variables come from LSEG Workspace (London Stock Exchange Group, 2024) under an
institutional licence: the TRBC classification at five levels, country of headquarters and of
exchange, founding year, the environmental theme, pillar and ESG scores, Scope 1, Scope 2 and
Scope 3 CO2-equivalent emissions with the estimation method flag, energy, water and waste
quantities, eight financial and employment items, and two reporting fields. Each field is mapped
to its analysis variable in `variable_dictionary.csv`, which also records the transformation, the
role the variable plays in the analysis, and the source cited for the construct where the
literature supplies one.

## Data-access protocol

Both scripts run inside LSEG Workspace CodeBook, where the session is already authenticated. They
cannot be run outside a licensed Workspace installation.

1. `lseg_probe_fields.py` tests every candidate field on ten firms and writes
   `lseg_probe_report.csv`: which field names the service returns, whether the five-year fiscal
   parameters return five years, what the provider's catalogue holds for any missing concept, and
   which fields expand into several rows per firm. It pulls no analysis data.
2. `lseg_pull.py` requests the two extracts for the instrument list it carries and writes
   `lseg_static.csv` (one row per instrument), `lseg_panel.csv` (one row per firm-year, five
   fiscal years, newest first) and `lseg_pull_log.csv` (field completeness and any instrument the
   service failed to return). It takes roughly twenty minutes and saves partial results as it
   goes.
3. Download the two extracts into `data/`.

The panel is requested on the fiscal-year frequency from the most recent fiscal year backwards
(SDate 0 to EDate -4, Frq FY), with all amounts in US dollars. Fiscal year zero is therefore each
firm's own latest fiscal year and not a common calendar year, which is why every observation
carries its year explicitly and the pooled specification carries year fixed effects. The
reporting date travels with the revenue field, so a firm-year with no revenue arrives undated;
`build_analysis_data.py` dates those rows by their position within the firm's block and flags
them, and a robustness check drops them.

## Pipeline

```bash
pip install -r requirements.txt
python build_analysis_data.py      # data/lseg_*.csv      -> data/analysis_*.csv
python run_analysis.py             # data/analysis_*.csv  -> results/
python make_figures.py             # results/             -> the manuscript figures
```

| Script | Purpose |
|---|---|
| `build_analysis_data.py` | Screens the extracts, constructs every analysis variable, and writes the panel, the cross-section, the lagged sample, `sample_flow.csv` and `screening_log.txt` |
| `run_analysis.py` | Estimates every model and writes the eleven results tables and `summary.txt` |
| `make_figures.py` | Draws the figures from the screening record and the results tables, as vector PDF, 300 dpi PNG and a caption file |
| `make_synthetic_demo.py` | Generates synthetic extracts with the same schema |
| `lseg_probe_fields.py`, `lseg_pull.py` | The data-access protocol above |
| `variable_dictionary.csv` | Definition, source field, transformation, role and reference for every variable |

Every screening decision is recorded rather than applied silently, so the sample flow reported in
the paper is generated rather than transcribed. Standard errors are clustered by country group
throughout.

## What each results file corresponds to in the paper

| File | Table or figure in the paper |
|---|---|
| `data/sample_flow.csv`, `data/screening_log.txt` | Table "Sample construction"; Figure "Sample construction" |
| `results/t01_descriptives.csv` | Table "Descriptive statistics"; Table "Descriptive statistics by sector" |
| `results/t02_selection_score.csv` | Table "The two selection gates", Panels A and B (rating gate) |
| `results/t03_correlations.csv` | Table "Pairwise correlations", and the variance inflation factors in its note |
| `results/t04_main.csv` | Table "Main models" |
| `results/t05_denominators.csv` | Table "Denominator battery"; Figure "Five intensity denominators" |
| `results/t06_components.csv` | Table "Measure components and alternative constructions"; Figure "Alternative constructions of the measure" |
| `results/t07_industry.csv` | Table "Industry heterogeneity"; Figure "Industry heterogeneity" |
| `results/t08_disclosure.csv` | Table "The two selection gates", Panel C (disclosure gate); Table "Reweighting and robustness", Panel A |
| `results/t09_lagged.csv` | Table "Lagged and change specifications"; Figure "The 2021 score against 2025 outcomes" |
| `results/t10_waste.csv` | Table "Waste outcomes" |
| `results/t11_robustness.csv` | Table "Reweighting and robustness", Panel B |
| `results/summary.txt` | Every table above rendered for reading |

## Synthetic demonstration

`make_synthetic_demo.py` writes `lseg_static.csv` and `lseg_panel.csv` with the schema, the
missingness structure and the broad distributions of the real extracts. The firm-level draw is a
multivariate normal whose means, standard deviations and correlations are the published
descriptive statistics of the paper; rated and unrated firms are drawn with different means, so
the selection models have the same kind of gap to estimate. No proprietary value is used and none
is contained in the output.

```bash
python make_synthetic_demo.py --data-dir synthetic_data
python build_analysis_data.py --data-dir synthetic_data
python run_analysis.py --data-dir synthetic_data --results-dir synthetic_results
python make_figures.py --data-dir synthetic_data --results-dir synthetic_results \
    --figures-dir synthetic_results/figures
```

The pipeline runs to completion on these inputs and produces a table for every table in the
paper. It reproduces the structure of the analysis, not its findings: the numbers are synthetic
and must not be cited or read as results.

## Requirements

Python 3.10 or later with pandas, numpy, statsmodels, scipy and matplotlib, as listed in
`requirements.txt`. The two data-access scripts additionally require the `lseg-data` package,
which is already installed in LSEG Workspace CodeBook and is not needed for anything else.

## Licence

Code released under the MIT licence; see `LICENSE`. The licence covers the code only. The LSEG
data are the property of London Stock Exchange Group and are not redistributed here.
