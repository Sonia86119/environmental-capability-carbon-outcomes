"""Verify the LSEG field names and parameters before the extract is requested.

The script runs inside LSEG Workspace CodeBook and pulls ten firms only, so
it costs little and returns no analysis data. It establishes four things that
the extract in lseg_pull.py then relies on:

  1. which of the candidate field names the data service actually returns,
     and which are silently empty;
  2. whether the five-year fiscal-year parameters return five fiscal years
     per firm rather than a single year repeated;
  3. what the provider's own catalogue holds for the concepts still missing,
     so that a field is looked up rather than assumed;
  4. which fields return one row per firm and which expand into several rows,
     since the two cannot be requested in the same call.

It writes lseg_probe_report.csv, one row per field tested, with the verdict
and a sample value. Fields that fail here are removed from lseg_pull.py or
replaced by the catalogue name the probe reports.
"""

import pandas as pd

try:
    import lseg.data as ld
except ImportError:
    import refinitiv.data as ld

FIRMS = ["AAPL.O", "MSFT.O", "SHEL.L", "BP.L", "NESN.S",
         "TTE.PA", "RIO.L", "7203.T", "VOW3.DE", "JPM.N"]

FY0 = {"Period": "FY0", "Curn": "USD"}
PANEL = {"SDate": 0, "EDate": -4, "Frq": "FY", "Curn": "USD"}

GROUPS = {
    "identity": ["TR.ISIN", "TR.CommonName", "TR.HeadquartersCountry", "TR.ExchangeCountry"],
    "industry": ["TR.TRBCEconomicSector", "TR.TRBCEconSectorCode", "TR.TRBCBusinessSector",
                 "TR.TRBCIndustryGroup", "TR.TRBCIndustry", "TR.TRBCActivity"],
    "financials": ["TR.Revenue", "TR.TotalRevenue", "TR.NetIncome", "TR.TotalAssets",
                   "TR.TotalDebt", "TR.EV", "TR.CompanyMarketCap", "TR.OrgFoundedYear"],
    "employees": ["TR.Employees", "TR.NumberOfEmployees", "TR.CompanyNumEmployees",
                  "TR.EmployeeTotalNumber", "TR.F.EmpTotal", "TR.AnalyticTotalEmployees"],
    "emissions": ["TR.CO2EmissionTotal", "TR.CO2DirectScope1", "TR.CO2IndirectScope2",
                  "TR.CO2IndirectScope3", "TR.CO2EstimationMethod", "TR.AnalyticCO2"],
    "waste": ["TR.WasteTotal", "TR.TotalWaste", "TR.WasteRecycledTotal",
              "TR.WasteRecycledPct", "TR.HazardousWaste", "TR.NonHazardousWaste"],
    "scores": ["TR.EnvironmentPillarScore", "TR.TRESGEmissionsScore", "TR.TRESGResourceUseScore",
               "TR.TRESGInnovationScore", "TR.TRESGScore", "TR.TRESGCScore"],
    "reporting": ["TR.CSRReportingScope", "TR.ESGReportingScope", "TR.ESGPeriodLastUpdateDate"],
    "assurance": ["TR.ESGAuditedExternally", "TR.CSRSustainabilityExternalAudit",
                  "TR.AnalyticCSRAudit", "TR.CSRAuditing", "TR.CSRSustainabilityReporting",
                  "TR.CSRSustainabilityCommittee", "TR.ESGReportingAuditorName",
                  "TR.GlobalCompactSignatory", "TR.CSRSustainabilityReportGlobalActivities",
                  "TR.CSRExternalAudit", "TR.ExternalAuditCSR", "TR.CSRVerification",
                  "TR.AnalyticGRIReport", "TR.CSRReportingGRI", "TR.IntegratedStrategyMDA"],
    "ownership": ["TR.InvestorSumPctOfSharesOutHeld", "TR.InstitutionalOwnershipPct",
                  "TR.InstOwnPctOfShrsOut", "TR.SharesHeldByInstitutionsPct",
                  "TR.FreeFloatPct", "TR.FreeFloat", "TR.SharesFreeFloat",
                  "TR.StrategicEntitiesPctOfSharesOut", "TR.PctOfSharesOutHeld"],
    "output": ["TR.ProductionVolume", "TR.TotalProduction", "TR.OutputVolume",
               "TR.UnitsProduced", "TR.EnergyUseTotal", "TR.TotalEnergyUse",
               "TR.WaterWithdrawalTotal", "TR.RenewableEnergyUse"],
}

KEYWORDS = ["audit", "assurance", "verification", "institutional", "ownership",
            "free float", "production", "output", "energy use", "employee"]

rows = []


def check(field, params, label):
    """Report firms answering, rows returned, and a sample value."""
    try:
        frame = ld.get_data(universe=FIRMS, fields=[field], parameters=params)
    except Exception as exc:
        rows.append({"group": label, "field": field, "params": "FY0", "firms": 0,
                     "rows": 0, "sample": str(exc)[:60], "verdict": "error"})
        return 0
    if frame is None or len(frame) == 0:
        rows.append({"group": label, "field": field, "params": "FY0", "firms": 0,
                     "rows": 0, "sample": "", "verdict": "empty"})
        return 0
    cols = [c for c in frame.columns if c not in ("Instrument", "Date")]
    if not cols:
        rows.append({"group": label, "field": field, "params": "FY0", "firms": 0,
                     "rows": len(frame), "sample": "", "verdict": "no value column"})
        return 0
    series = frame[cols[0]]
    firms = int(series.notna().sum())
    sample = "" if firms == 0 else str(series.dropna().iloc[0])[:40]
    verdict = "OK" if 0 < len(frame) <= len(FIRMS) and firms > 0 else (
        "TOO MANY ROWS" if len(frame) > len(FIRMS) else "no data")
    rows.append({"group": label, "field": field, "params": "FY0", "firms": firms,
                 "rows": len(frame), "sample": sample, "verdict": verdict})
    return firms


def test_panel():
    """Confirm the five-year parameters really return five rows per firm."""
    print("\n" + "=" * 68)
    print("PANEL TEST: do five-year parameters work?")
    print("=" * 68)
    for field in ["TR.TRESGResourceUseScore", "TR.CO2DirectScope1", "TR.Revenue"]:
        try:
            frame = ld.get_data(universe=FIRMS[:3], fields=[field], parameters=PANEL)
        except Exception as exc:
            print("  %-32s ERROR %s" % (field, str(exc)[:50]))
            continue
        if frame is None or len(frame) == 0:
            print("  %-32s returned nothing" % field)
            continue
        per_firm = len(frame) / 3.0
        dates = ""
        for col in frame.columns:
            if "date" in col.lower():
                dates = ", ".join(sorted(set(str(v)[:10] for v in frame[col].dropna()))[:6])
                break
        print("  %-32s %d rows = %.1f per firm   %s" % (field, len(frame), per_firm, dates))
    print("=" * 68)


def discover():
    """Ask LSEG which fields exist, rather than guessing names."""
    print("\n" + "=" * 68)
    print("CATALOGUE SEARCH")
    print("=" * 68)
    any_hit = False
    for word in KEYWORDS:
        hits = []
        try:
            res = ld.discovery.search(query=word, view=ld.discovery.Views.DATA_ITEMS, top=20)
            frame = getattr(getattr(res, "data", res), "df", res)
            if frame is not None and len(frame) > 0:
                for col in frame.columns:
                    for value in frame[col].astype(str).tolist():
                        if value.startswith("TR."):
                            hits.append(value)
        except Exception:
            pass
        if not hits:
            try:
                d = ld.delivery.endpoint_request.Definition(
                    url="/data/datagrid/beta1/fields", query_parameters={"query": word})
                text = str(d.get_data().data.raw)
                cursor = 0
                while True:
                    cursor = text.find('"TR.', cursor)
                    if cursor == -1:
                        break
                    end = text.find('"', cursor + 1)
                    hits.append(text[cursor + 1:end])
                    cursor = end
            except Exception:
                pass
        hits = list(dict.fromkeys(hits))[:12]
        if hits:
            any_hit = True
            print("\n  %s:" % word)
            for h in hits:
                print("      %s" % h)
                rows.append({"group": "catalogue", "field": h, "params": word,
                             "firms": "", "rows": "", "sample": "", "verdict": "candidate"})
    if not any_hit:
        print("\n  Catalogue not reachable from this session.")
        print("  Open the Data Item Browser instead: type DIB in the Workspace search bar,")
        print("  search those same words and read off the exact field names it shows.")
    print("=" * 68)


print("Opening session ...")
session = ld.open_session()
if session is None or not str(session.open_state).endswith("Opened"):
    raise SystemExit("Session did not open.")
print("Session open. Testing %d fields on %d firms.\n" % (
    sum(len(v) for v in GROUPS.values()), len(FIRMS)))

for label, fields in GROUPS.items():
    print("\n--- %s ---" % label.upper())
    for field in fields:
        firms = check(field, FY0, label)
        last = rows[-1]
        mark = "OK  " if last["verdict"] == "OK" else "... "
        print("  %s %-42s %2s/%d firms  %s  %s" % (
            mark, field, firms, len(FIRMS), last["verdict"], last["sample"][:28]))

test_panel()
discover()

report = pd.DataFrame(rows)
report.to_csv("lseg_probe_report.csv", index=False)
print("\nWrote lseg_probe_report.csv")
print("\nWORKING FIELDS BY GROUP:")
ok = report[report["verdict"] == "OK"]
for label in GROUPS:
    names = ok[ok["group"] == label]["field"].tolist()
    print("  %-12s %s" % (label, ", ".join(names) if names else "NONE"))
