# LSEG / Refinitiv TRBC pull

This package pulls native TRBC classification fields from LSEG Workspace/Data Library using existing RICs.

## Fields pulled

Core reviewer-requested fields:

- `TR.TRBCEconomicSector` → TRBC Economic Sector Name
- `TR.TRBCEconSectorCode` → TRBC Economic Sector Code
- `TR.TRBCBusinessSector` → TRBC Business Sector Name
- `TR.TRBCBusinessSectorCode` → TRBC Business Sector Code

Optional lower-level TRBC fields included for auditability:

- `TR.TRBCIndustryGroup`
- `TR.TRBCIndustryGroupCode`
- `TR.TRBCIndustry`
- `TR.TRBCIndustryCode`
- `TR.TRBCActivity`
- `TR.TRBCActivityCode`

## Input format

Create a CSV like:

```csv
Identifier,RIC
firm_001,BARC.L
firm_002,TRI.N
firm_003,TSLA.O
```

The `Identifier` column is preserved. The `RIC` column is used for the LSEG query.

## Run

```bash
pip install lseg-data pandas
python pull_trbc_lseg.py --input identifiers.csv --ric-column RIC --id-column Identifier --output identifiers_trbc.csv
```

Open LSEG Workspace first and make sure you are logged in before running the script.

## Excel formula alternative

In LSEG Workspace Excel, use your RIC range and add the same fields:

```excel
=@RDP.Data(A2:A1000,"TR.TRBCEconomicSector;TR.TRBCEconSectorCode;TR.TRBCBusinessSector;TR.TRBCBusinessSectorCode")
```

Depending on your Workspace Excel add-in version, the function name may appear as `=@RDP.Data(...)`, `=RDP.Data(...)`, or be inserted through the Formula Builder.
