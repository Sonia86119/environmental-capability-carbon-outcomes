"""Extract the licensed LSEG data on which the analysis is built.

The script runs inside LSEG Workspace CodeBook, where the session is already
authenticated; it cannot be run outside a licensed Workspace installation.
Every field name below was verified against the data service by
lseg_probe_fields.py before it was used, so the extract is specified in terms
of fields the service actually returns.

Two extracts are requested for the same list of instruments and merged later
on the Refinitiv Instrument Code:

    lseg_static.csv    one row per firm: the TRBC classification at all five
                       levels, country of headquarters, country of exchange,
                       ISIN, name and founding year
    lseg_panel.csv     one row per firm-year for five fiscal years: scores,
                       emissions, resource use, waste, financial and
                       employment items, and reporting fields
    lseg_pull_log.csv  the completeness of every field in both extracts, and
                       any instrument the service failed to return

The panel is requested on the fiscal-year frequency from the most recent
fiscal year backwards (SDate 0 to EDate -4, Frq FY), so year zero is each
firm's own latest fiscal year rather than a common calendar year, and all
monetary amounts are converted to US dollars. The reporting date travels with
the revenue field, which is why build_analysis_data.py dates the firm-years
that arrive without one.

The pull takes roughly twenty minutes for the instrument list below. It is
chunked, retries a failing request and then halves the chunk, so that one
unavailable instrument costs only itself; partial results are written every
fifth chunk. The three output files are then downloaded from CodeBook into
the data/ directory of this package.
"""

import pandas as pd

try:
    import lseg.data as ld
except ImportError:
    import refinitiv.data as ld

YEARS = 5
STATIC_PARAMS = {"Curn": "USD"}
PANEL_PARAMS = {"SDate": 0, "EDate": -(YEARS - 1), "Frq": "FY", "Curn": "USD"}
CHUNK_STATIC = 200
CHUNK_PANEL = 100

# Additional mnemonics, if the extract is ever extended. Field names are
# taken from the Data Item Browser (type DIB in the Workspace search bar).
USER_EXTRA_FIELDS = []

STATIC_FIELDS = [
    "TR.ISIN",
    "TR.CommonName",
    "TR.HeadquartersCountry",
    "TR.ExchangeCountry",
    "TR.TRBCEconomicSector",
    "TR.TRBCEconSectorCode",
    "TR.TRBCBusinessSector",
    "TR.TRBCBusinessSectorCode",
    "TR.TRBCIndustryGroup",
    "TR.TRBCIndustryGroupCode",
    "TR.TRBCIndustry",
    "TR.TRBCIndustryCode",
    "TR.TRBCActivity",
    "TR.TRBCActivityCode",
    "TR.OrgFoundedYear",
]

PANEL_FIELDS = [
    "TR.Revenue.Date",
    "TR.Revenue",
    "TR.NetIncome",
    "TR.TotalAssets",
    "TR.TotalDebt",
    "TR.EV",
    "TR.CompanyMarketCap",
    "TR.Employees",
    "TR.FreeFloatPct",
    "TR.CO2EmissionTotal",
    "TR.CO2DirectScope1",
    "TR.CO2IndirectScope2",
    "TR.CO2IndirectScope3",
    "TR.CO2EstimationMethod",
    "TR.AnalyticCO2",
    "TR.EnergyUseTotal",
    "TR.WaterWithdrawalTotal",
    "TR.WasteTotal",
    "TR.WasteRecycledTotal",
    "TR.HazardousWaste",
    "TR.NonHazardousWaste",
    "TR.EnvironmentPillarScore",
    "TR.TRESGEmissionsScore",
    "TR.TRESGResourceUseScore",
    "TR.TRESGInnovationScore",
    "TR.TRESGScore",
    "TR.TRESGCScore",
    "TR.CSRReportingScope",
    "TR.CSRReportingGRI",
    "TR.ESGPeriodLastUpdateDate",
]

RICS = [
    "EVNV.VI", "BHAV.VI", "OMVV.VI", "STRV.VI", "CAMB.BR", "TUB.BR", "USJE.MKE", "CBOS.BR", "WHATS.BR", "RGLDS.S",
    "INEA.PA", "UCB.BR", "ALVG.DE", "BYWGnx.DE", "DBKGn.DE", "DBANn.DE", "EIN_p.DE", "BBHKk.H", "BSV.L", "APTP.WA",
    "KRNG.DE", "LEIG.DE", "G1AG.DE", "PSHG_p.DE", "FREETR.TE", "TUI1n.DE", "RWEG.DE", "JPEM.SJ", "FFARMS.CO", "RLVG.D",
    "SWAG.DE", "TKAG.DE", "CIEA.MC", "APE.MC", "IDR.MC", "ECR.MC", "ANA.MC", "RYA.I", "ELE.MC", "FAE.MC",
    "CBOX.L", "IBE.MC", "YTRM.MC", "IGET.L", "MAP.MC", "SCYR.MC", "TPT.L", "SBOE.VI", "PROFb.ST", "AIRP.PA",
    "CHBE.PA", "AM.PA", "EDSP.PA", "FOUG.PA", "GRBT.PA", "ALLHB.PA", "VTU.L", "ALBON.PA", "OREP.PA", "MANP.PA",
    "MWDP.PA", "LAGA.PA", "ALEXA.PA", "IBGX.L", "GIMV.BR", "JEN.BR", "ICEAIR.IC", "RCOP.PA", "ATLD.PA", "ALBKI.RTS",
    "ADEN.S", "KGH.WA", "RSU1L.VL", "ALWA.L", "AHT.L", "PNL.L", "BWY.L", "FOUR.L", "ROE1L.VL", "COA.L",
    "ANSY.L", "CLDN.L", "YHTI.MC", "CGS.L", "CBRO.L", "CPG.L", "CRDA.L", "RS1R.L", "MONT.L", "JEGI.L",
    "ZIG.L", "GDWN.L", "DGE.L", "A7An.DE", "IBST.L", "INCH.L", "BGFD.L", "KGF.L", "MMIT.L", "LWDB.L",
    "HHFGn.DE", "LSC.L", "MACF.L", "MAJE.L", "MNKS.L", "DYVOX.ST", "PAGPA.L", "NXT.L", "OXIG.L", "PZC.L",
    "SPL1.WA", "XSFR.DE", "PRU.L", "SJP.L", "NWG.L", "SVS.L", "SFR.L", "PNN.L", "SP5USY.S", "TW.L",
    "TSCO.L", "AFGA.OL", "TFW.L", "VLX.L", "ARBNO.S", "ABBN.S", "BCJ.S", "BELL.S", "BOS.S", "MSE.PA",
    "VELD.BR", "BUCN.S", "APGN.S", "GURN.S", "JFN.S", "HUBN.S", "BEKN.S", "LUKN.S", "OFN.S", "BMA.OL",
    "ROTLV.BX", "IFRB.AS", "SNBN.S", "REHN.S", "SOFF.OL", "ZEHN.S", "CPS.WA", "HYSG.BR", "PARKEN.CO", "FTV.L",
    "TFG.AS", "KNOW.ST", "ALR.BX", "SPIE.PA", "AAF.L", "XDBC.DE", "XSDX.DE", "ZMP1L.VL", "X13E.DE", "MNL.L",
    "PHNX.L", "SRET.L", "SYSR.ST", "ADGR.ZA", "EXA.PA", "PATC.PA", "BIFR.BU", "ARS.BX", "IVL1L.VL", "BNOR.OL",
    "FTF.L", "STGN.S", "OGN.I", "SBSG.DE", "EGLA.MI", "IJPH.L", "AG1G.DE", "ADDVa.ST", "IZER.MC", "VLS.PA",
    "LIMET.ST", "BARV.L", "KIPO.BR", "MNCP.WA", "HKFOODS.HE", "ADVG.DE", "GETP.PA", "YHSP.MC", "DIDA.MC", "BRAI.L",
    "VWS.CO", "ACPP.WA", "FYB.DE", "UNTP.WA", "KRU.WA", "SYNCS.L", "LIFCOb.ST", "BIOGb.ST", "37B.MI", "KNE1L.VL",
    "JENGn.DE", "MVEU.L", "CHMF.MM", "TECG.MOT", "IBAB.BR", "WPP.L", "PHAR.AS", "CHIPM.PA", "YAKG.MM", "MT.AS",
    "CKML.ZA", "COMO.PA", "HICL.L", "CMBN.S", "KRKN.MM", "INISSb.ST", "ENVI.AS", "BGSFA.BB", "VPABb.TE", "XMWO.DE",
    "XMJP.DE", "LEGn.DE", "XASX.L", "ARBONa.NGM", "ATBN.MOT", "A1OS.DE", "BMED.MI", "GLTN.PA", "PEMB.L", "UNPR.SJ",
    "FER.MC", "CRAP.PA", "XEMB.DE", "WOSG.L", "IQTr.AT", "DNP.WA", "MAERSKb.CO", "ALKb.CO", "SPGP.CO", "LOLB.CO",
    "NOVOb.CO", "PAALb.CO", "SCHO.CO", "AALB.AS", "BAMN.AS", "NCAB.ST", "CORB.AS", "SLIGR.AS", "TWKNc.AS", "IS3B.DE",
    "ALSTW.PA", "TEM.L", "ABDP.L", "PSRF.MI", "AMDG.DE", "MLPS.L", "TRNT.L", "SECTb.ST", "BSIF.L", "CTPE.L",
    "BRE2.ST", "YADR.MC", "SAGAb.ST", "NRC.OL", "ILM1k.DE", "AAL.L", "NTV.L", "VAGE.DE", "MDBI.MI", "EMBI.MI",
    "RECI.MI", "SPMI.MI", "LDOF.MI", "TNSE.MM", "FDEV.L", "PLAKR.AT", "WIIT.MI", "NOEJ.DE", "ROVI.MC", "VGP1.BR",
    "KER.WA", "ATCOa.ST", "BOL.ST", "PVAC.PA", "ELUXb.ST", "MSONb.ST", "ERICb.ST", "FAG.ST", "HUFVa.ST", "LUNDb.ST",
    "NCCb.ST", "RATOb.ST", "GOTLb.ST", "ISOS.PA", "GLINT.LS", "SAND.ST", "ALBAV.HE", "FSKRS.HE", "HUH1V.HE", "ISWD.L",
    "METSB.HE", "PACT.ST", "IPRP.L", "LEVMIB.MI", "XESX.DE", "PRO.MI", "VOLCARb.ST", "PREMr.AT", "AUTr.AT", "BST1.DE",
    "ULTIMA.BN", "ODLO.OL", "CRASTI.HT", "KMK.L", "NOHOP.HE", "ASY.PA", "ASMI.AS", "XBAG.DE", "AC5.MI", "ERNE.L",
    "IGSD.L", "LONG.BX", "MLVIN.EUA", "M9S4.F", "SW2CHB.S", "TRACb.ST", "SPRP.WA", "XLPE.DE", "XSPS.DE", "GAZS.MM",
    "ETFSTOXX50E.DE", "ZPR1.DE", "RQFI.DE", "ITH.L", "MSGL.H", "RAAG.DE", "IFXGn.DE", "NAVF.L", "UNIP.WA", "PSPN.S",
    "AOFG.DE", "ENPG.MM", "NESF.L", "LYSGQI.PA", "EUXG.F", "ROKOb.ST", "CMPG.L", "RWS.L", "DEM.CY", "NESI.L",
    "EWRK.ST", "XFAB.PA", "DIGI.BX", "LDSV.PA", "SPECI.L", "DR0G.DE", "SOLPW.MC", "LSG.OL", "DEBS.L", "IRAO.MM",
    "SFLG.MI", "N91.L", "CRSL.L", "NBA.LS", "KVIKA.IC", "PETSP.L", "PRE.L", "ASRNL.AS", "IVUG.DE", "HOC.H",
    "IGG.L", "IMPsdba.ST", "EURBr.AT", "WIX.L", "STOG_p.DE", "ROCRC.BX", "HIAG.S", "MLHK.EUA", "AMEGOV.PA", "GMAB.CO",
    "IBTS.L", "CCL.L", "WRDCHA.S", "PUST.PA", "IFCN.S", "IUKP.L", "CGL.L", "SELG.MM", "SLPb.ST", "CAPMAN.HE",
    "YAI1.MC", "MD4X.DE", "EQTAB.ST", "WIZZ.L", "OPGP.WA", "SCLP.L", "FAN.L", "MOEX.MM", "FEOIb.ST", "BPOST.BR",
    "HWWA.L", "NEOBO.ST", "SGIL.L", "VZUG.S", "ELIX.L", "TARP.WA", "ADKO.VI", "CAIROMEZr.AT", "XDEM.DE", "XAXJ.DE",
    "XSX6.DE", "BESQAB.ST", "XPLRA.OL", "GDXJ.L", "PHOR.MM", "VDEV.L", "TC1n.H", "GFM.L", "BPTB.L", "UKUZ.MM",
    "LPPP.WA", "MBHJB.BU", "IS3R.DE", "SPY5.DE", "ANRP.WA", "FRAG.DE", "PGOO.L", "TLGG.H", "ETFGS51.DE", "FG.ST",
    "JCDX.PA", "TLXGn.DE", "JIGI.L", "ASL.L", "ROBE.L", "EMBLA.CO", "SMSUSA.DE", "SR2000.DE", "SFTS.L", "CBU0.L",
    "TPG0n.DE", "ETFEUAC.DE", "ROSNP.BX", "BOGr.AT", "HPBZ.ZA", "SPSTS.PA", "SPSTZ.PA", "STW.PA", "ERO.PA", "EUN6.DE",
    "XUT3.L", "CMTM.SCT", "UAVP.L", "DOCS.L", "LKD.WA", "ALVU.PA", "BAER.S", "CAPE.PA", "HUKX.L", "ALPDX.PA",
    "CASP.PA", "IBTL.L", "ZPRU.DE", "ZPRX.DE", "TRACT.PA", "INOV.L", "ANNEb.ST", "XRS2.DE", "COL.MC", "DIG.L",
    "GTT.PA", "XMEU.DE", "IDLA.PA", "ALSO.PA", "GYL.OL", "AUTOA.L", "BANE.MM", "PEABb.ST", "BOPr.AT", "ARTO.PA",
    "BYIT.L", "FXC.L", "INFR.L", "ORY.MC", "DETEC.HE", "BMAX.ST", "ISACI.L", "MIPS.ST", "EKF.L", "IWDA.L",
    "ATRAV.HE", "SPH.BB", "DOBH.BB", "HRZ.BB", "LUMI.OL", "ENAG.MC", "RKKE.MM", "ENENTO.HE", "BRLA.L", "LAEP.PA",
    "LECN.S", "KITW.L", "SKAGI.IC", "BCXP.WA", "NORSE.OL", "HIMR.ZA", "AIR.PA", "WAGA.PA", "CSSX5E.S", "EXCr.AT",
    "ENGP.WA", "PLAP.EUA", "RAIVV.HE", "EDNn.MI", "JY0n.H", "ANCR.L", "SCT.LS", "FRCJ.DE", "SATG.DE", "SWICHA.S",
    "WSGR.ST", "YSO.LS", "GFTG.DE", "IBCZ.DE", "XDGU.DE", "HLIP.MT", "WISEa.L", "HHI.L", "URW.PA", "HBH.DE",
    "PREN.MOT", "MONET.PR", "BCG.L", "SYBY.DE", "RBAV.VI", "ENQ.L", "HSW.L", "FEV.L", "NORION.ST", "IUIT.L",
    "AEI.L", "DIOR.PA", "DNB.OL", "LAUU.L", "ED4.DE", "SUSS.L", "ISWING.L", "IPO.L", "BRIOX.NGM", "ES50.DE",
    "HELr.AT", "GPGR.ST", "FUGR.AS", "RACE.MI", "NELLY.ST", "RCVR.F", "WAWI.OL", "JUP.L", "ARN.MI", "REVOI.MI",
    "SYBR.DE", "SGKN.S", "EEUE.PA", "XDWF.DE", "XDWH.DE", "EPEJ.PA", "OCDO.L", "JEMI.L", "AIRF.PA", "YU.L",
    "CLRE.MC", "CEA1.L", "TDSA.LS", "CTEK.ST", "PINT.L", "FORT.L", "EPIRa.ST", "WFIN.AS", "WKB.S", "MFEB.MI",
    "FLATb.ST", "WIND.AS", "SSGWNRG.AS", "RORRC.BX", "NEKG.DE", "HMXJ.L", "UINC.L", "GTCP.WA", "ROCFH.BX", "CARP.WA",
    "ALLEI.ST", "MEDI.OL", "TDIV.AS", "PNOR.OL", "INT.S", "IVSO.ST", "GVS.MI", "HDEM.L", "LOCAL.PA", "ATAS.L",
    "DHH.MI", "MULTI.OL", "BGTBD.BB", "KLIN.S", "ESFL.MI", "CBOM.MM", "NCYF.L", "TBCG.L", "TRESTATESr.AT", "HEIJ.AS",
    "LMTD.PA", "IGWD.L", "MUMG.DE", "ANE.MC", "ADML.L", "JMT.LS", "INGR.MM", "NEXI.PA", "RSTP.L", "GJFG.OL",
    "DGIT.L", "GBFG.DE", "SWEDa.ST", "CWR.L", "EGL.L", "FF.MI", "CCCP.WA", "TIP1S.S", "BORG.ST", "2HR.DE",
    "DOVA.MI", "EMMUKD.S", "BRWM.L", "INSTA.OL", "AMB.WA", "PUBLI.OL", "GILI.PA", "NATURLAND.BU", "DS.MI", "YIBI.MC",
    "SAVE.ST", "HMWO.L", "PUIGb.MC", "ENE.WA", "HSTC.L", "TRE.MC", "BRFI.L", "CHIM.BB", "EMBRACb.ST", "W7L.L",
    "GLPG.AS", "TIP.MI", "REITIR.IC", "CENER.BR", "TPV.L", "CBESG.S", "ALVDM.PA", "PJS1.DE", "IUVF.L", "VEMT.L",
    "DNOS.BEL", "GOODWILLPHRM.BU", "TAL1T.TL", "OM3X.DE", "FPARa.ST", "INAI.ZA", "DKSH.S", "ESSITYb.ST", "MTLS.BR", "GEA1.WA",
    "SALM.OL", "TGYM.MI", "SPOL.L", "MAB1.L", "MGTP.WA", "ROSNN.BX", "TGN.BX", "KOSKI.HE", "NWRN.S", "ETFLCDNF.DE",
    "RBW.WA", "HBMG.H", "FNTL.L", "ERRES.S", "KIST.L", "CMP.WA", "DARD.L", "V3Vn.DE", "BPCR.L", "SHIP.L",
    "AVOL.S", "MTRO.L", "CRJ.WA", "STM1.DE", "FGEQ.DE", "DOFG.OL", "HKOR.L", "SHUR.BR", "GSUKEX.DE", "ICBU.L",
    "TRET.AS", "TPLR.BB", "GEAP.PA", "SRSr.AT", "MTRS.ST", "ECN.PA", "RUSI.MM", "USAG.L", "SPYIG.DE", "ABOG.DE",
    "BSTH.L", "PGLC.MT", "ALFAAL.L", "SWMP.WA", "SDZ.S", "STORYb.ST", "PREr.AT", "NXTGE.MI", "BETCO.CO", "GENL.L",
    "MMSA.MT", "IDIA.S", "QUID.L", "CYNL.L", "ADMr.AT", "TAMT.L", "SOON.S", "COMM.L", "IQEE.PA", "ETS.VI",
    "IQJP.PA", "B4B.H", "YTST.MC", "CAG0n.H", "CSBGC0.S", "USACHA.S", "MANTA.HE", "SCF.L", "DNR.MI", "STOXXIEX.DE",
    "SXPPEX.DE", "KETL.L", "XACTSVERIGE.ST", "ROM.BX", "DFV.H", "ZTF.L", "MFGS.MM", "HYDPA.AS", "C030.DE", "ARPL.ST",
    "ALEX.HE", "ASML.AS", "WSREUA.DE", "PSREUA.DE", "POLNP.L", "SGCG.DE", "ALCGM.PA", "KOA.OL", "GALD.S", "AUUSI.S",
    "H4ZE.DE", "VAR.OL", "SBC.MI", "FARM.PFT", "ASMDEEb.ST", "NYKD.OL", "IQQG.DE", "SVEAF.ST", "SMCP.PA", "DEMANT.CO",
    "BET.BU", "1VUB02AE.BV", "GEP.L", "ABI.BR", "OPT.L", "IVAT.MM", "ROBIO.BX", "STNA.MC", "LEAL.ST", "INWI.ST",
    "OSEC.L", "EONGn.DE", "RECIV.L", "ANYB.BU", "GCHE.MM", "AAIF.L", "EG7.ST", "NBLB.BJ", "KP2.L", "EMLB.L",
    "MTE.PA", "SOL.S", "CYPC.CY", "VCW.CY", "CARAS.ST", "MMBANK.NFF", "HUSQb.ST", "SB1G.DE", "USDV.DE", "TIWn.DE",
    "GAZP.MM", "LSNG.MM", "ID34.L", "SYBL.DE", "AKBM.OL", "ALHG.PA", "HRX5.VI", "GENDER.S", "SHFG.DE", "AIAA.PA",
    "PANR.L", "ELIOR.PA", "COFFEEb.ST", "DIOS.ST", "EIMS.IC", "YTAN.MC", "ICLU.S", "MHA.L", "MEWD.DE", "ELIS.PA",
    "WACGn.DE", "TENP.WA", "VGGF.DE", "CCCG.ST", "RLIA.MC", "CG9.PA", "ONWD.BR", "MTLN.L", "MRIC.DE", "IAPD.L",
    "ELMRA.OL", "AMV0n.DE", "YTEM.MC", "ALAT.PA", "CMCOM.AS", "HAUTO.OL", "YATO.MC", "ADSGn.DE", "IB83EX.DE", "ROGS.OL",
    "QCEU.PA", "LYY8.DE", "UQB2.DE", "LKOH.MM", "CI2.PA", "MCTAa.L", "WIL.L", "UPRO.MM", "O4BG.DE", "CW8.PA",
    "DWNG.DE", "VIEV.VI", "SJJG.DE", "WVIA.L", "MEDA.S", "GR6.BB", "SPOE.PA", "VBKG.DE", "EATP.WA", "SREF.S",
    "URKZ.MM", "ALDEL.PA", "TGKB.MM", "KCRA.HE", "MONTE.BR", "VPLAYb.ST", "BX4.PA", "BASS.PA", "CHMK.MM", "NOD.OL",
    "JEST.DE", "TEL2b.ST", "DAST.PA", "FAS.L", "RING.OL", "SOLI.L", "CNCT.L", "HBANC.S", "GMMG.DE", "SPT6.F",
    "LKPG.LJ", "TKBP.S", "HEX.OL", "BPLAZ.HT", "MBWS.PA", "IMHE.S", "CETG.LJ", "RCSM.MI", "UNIQ.VI", "LENV.VI",
    "SMPV.VI", "WBSV.VI", "UMI.BR", "ACKB.BR", "GBLB.BR", "FLOB.BR", "TESB.BR", "TEXB.BR", "GAZT.MM", "BEIG.DE",
    "BIOG.H", "MBGn.DE", "DAMG.DE", "DRWG.DE", "EISG.F", "EUKG_p.F", "FREG.DE", "GWKGn_p.H", "LECG.F", "ICP1V.HE",
    "INPP.L", "TLKM.BJ", "NLVGn.DE", "RHMG.DE", "BATD.PA", "STGG.DE", "TEGG.DE", "HVBG.F", "WCMk.H", "ZILGn.DE",
    "BKT.MC", "NTGY.MC", "UNITED.HE", "CAF.MC", "ENOR.MC", "INRN.S", "PRIM.MC", "PSG.MC", "ACCP.PA", "SAF.PA",
    "BAIN.PA", "ANDF.OL", "KOMN.S", "BICP.PA", "DANO.PA", "NIBEb.ST", "SPY4.L", "FFLY.ST", "KINVb.ST", "CAPP.PA",
    "CRIP.PA", "VIV.PA", "BRUN.AS", "EXAI.MI", "SOGN.PA", "RUBF.PA", "ISTB.L", "JCQ.PA", "LEGD.PA", "GNFT.PA",
    "MFBP.PA", "TEPRF.PA", "FSDV.PA", "SCHN.PA", "SPYF.DE", "AXFO.ST", "KRSB.MM", "TTEF.PA", "TEIF.PA", "NAIT.L",
    "ABF.L", "TTRGn.DE", "GL9.I", "AVON.L", "BTRW.L", "GRG1L.VL", "BKGH.L", "AFLA.L", "BP.L", "CNE.L",
    "CGT.L", "LGRE.PA", "EDIN.L", "ELCO.L", "BGCG.L", "FSJ.L", "FCIT.L", "GRI.L", "GPEG.L", "MTO.L",
    "IMI.L", "GHC.MI", "AJOT.L", "JSG.L", "MRCH.L", "MUT.L", "PAGE.L", "PORV.L", "VOD.L", "RSW.L",
    "SGE.L", "SBRY.L", "SNR.L", "SGRO.L", "AML.L", "TRY.L", "WEIR.L", "GAMH.S", "BCVN.S", "PARG19.LU",
    "ARM.MC", "FORN.S", "ISN.S", "VATN.S", "LLBN.S", "METG.S", "METN.S", "VTBR.MM", "MCHN.S", "PXMP.WA",
    "SLHN.S", "UHR.S", "VAHN.S", "YJSS.MC", "VPBN.S", "ZUBN.S", "VUKE.L", "AEIGn.DE", "PAZA.MM", "DXSE.DE",
    "X57E.DE", "X25E.DE", "XGIN.DE", "XEON.DE", "JOMA.ST", "BSTA.MI", "ROAROBS.BX", "VTYV.L", "B7C.MI", "RABA.BU",
    "DUKE.L", "DABA.CO", "ZUGN.S", "KRAR.ZA", "TRIA.PA", "RIVP.ZA", "ZPAL.S", "CPEI.NGM", "TRAY.BEL", "NORAM.OL",
    "DVEC.MM", "UTDI.DE", "ZV.MI", "FXPO.L", "GEST.MC", "ARTC.MC", "CRPL.L", "FLUI.MC", "TES.MI", "PEXIP.OL",
    "BRGB.OL", "DVLP.WA", "RBO.PA", "ANV.MI", "PPH.L", "PTNL.AS", "ALCRB.PA", "COSH.OL", "IGHY.L", "EMG.L",
    "GFRN.MI", "SPSO.LS", "ISIUSC.S", "BARN.S", "CRADb.ST", "AUBT.PA", "UTAR.MM", "EKTAb.ST", "URKAI.RTS", "HP3An.DE",
    "APX.PA", "LYEE.PA", "INTE.ST", "HUW.L", "HNSA.ST", "AURG.OL", "BLMG.DE", "BTLS.BR", "TNXT.MI", "KIT.OL",
    "SMCRT.OL", "CRW.L", "XUKX.L", "XMAS.DE", "XGSD.DE", "EVSB.BR", "XMTW.DE", "AKRBP.OL", "EVTG.DE", "OPL.WA",
    "LYXEUC.PA", "8TRA.DE", "ZERS.BJ", "COKG.DE", "WEHB.BR", "SVN.IC", "TRCS.L", "PNEGn.DE", "FORTUM.HE", "SPBH.F",
    "XSTR.DE", "MVID.MM", "TINGSa.ST", "ALMB.CO", "BO.CO", "CBSM.PA", "DANSKE.CO", "RBREW.CO", "AUDK.LU", "SBMO.AS",
    "INGA.AS", "EAST9.ST", "NEDP.AS", "SVEL.AS", "PSDE.MI", "BITI.PA", "ZPRA.DE", "RTA4.MC", "SIMINN.IC", "MATAS.CO",
    "MBKG.DE", "LMPL.L", "OS.MI", "ELK.OL", "PCFT.L", "FQT.DE", "ODF.OL", "CIRI.MI", "BZU.MI", "KZOS.MM",
    "EUZG.DE", "UNPI.MI", "SASY.PA", "ACT1.DE", "LPER.PA", "SCAb.ST", "HOLMb.ST", "HMb.ST", "BRSC.L", "ALDLT.PA",
    "WDPP.BR", "PEY.L", "JM.ST", "ACADE.ST", "XLS.S", "ISDE.L", "ASTr.AT", "TFI.MI", "SLV1.WA", "IEUX.L",
    "TIG.L", "XDN0.DE", "ETZ.PA", "ATXEX.DE", "STOXXEEX.DE", "LCXPEX.DE", "ROONE.BX", "FSFL.L", "GUBRA.CO", "CWCG.DE",
    "TMV.DE", "NBS.L", "OTV2.L", "HSPX.L", "ISF.L", "ERND.L", "ROEVER.BX", "ROTRANSI.BX", "INFINITY.BX", "ERNS.L",
    "WITH.HE", "PLT.OL", "W5.ST", "MRCP.WA", "NOS.LS", "MSABb.ST", "NFGN.L", "MZXG.DE", "XSGI.DE", "ETFSG2P.DE",
    "LOGM.CY", "TPFG.L", "WAVE.PA", "ABDN.L", "FILA.MI", "MUXG.DE", "EYDr.AT", "AEME.PA", "ORIT.L", "VLG.L",
    "ALLN.S", "OSPGk.H", "BKP.LJ", "ALMT.ST", "SFPI.PA", "GLJn.DE", "TRP.BX", "SPLUS.BU", "THULE.ST", "FGENF.L",
    "QBYn.DE", "EDPR.LS", "CRLI.MI", "EKTG.DE", "EMILpref.ST", "ALHEX.PA", "ISJP.L", "PHO.OL", "GIVN.S", "DIMANDr.AT",
    "EPEND.ST", "ADNGk.DE", "SONO.PA", "BNP1.WA", "REG1V.HE", "PRIMOF.CO", "MTSS.MM", "COFA.PA", "SAGS.L", "ZGLD.S",
    "OSBO.L", "ZPRR.DE", "ORILINAr.AT", "CREI.L", "PND.CY", "NREST.ST", "HYQGn.DE", "LMN.S", "KOG.OL", "CHDVD.S",
    "IBGS.L", "IHI.MT", "ATIN.CY", "ITAMID.MI", "IQQ9.DE", "IUSA.L", "IH2O.L", "INRG.L", "SPIR.OL", "PIAP.LU",
    "CG1.PA", "SFIN.MM", "KTN.DE", "XDEW.DE", "LQDE.L", "PGH.L", "IREE.MI", "BMEB.L", "ELEN.MI", "UEF0.DE",
    "AKTIA.HE", "OP4E.PA", "PSRW.L", "ETFES11.DE", "ICOS.MI", "HBR.L", "PPEU.MI", "ROBRD.BX", "AKSOA.OL", "BREL.LU",
    "MRVM.L", "TRBG.L", "ISB.IC", "SNI.OL", "CTPNV.AS", "HMUG.DE", "VTWRn.H", "IS3U.DE", "SPLTN.NGM", "USBN.MM",
    "BANQ.BR", "EXS.ST", "MUBG.DE", "OPAr.AT", "SVIK.ST", "CCNP.PA", "IS3Q.DE", "DBPG.DE", "TCM.CO", "UBKG.DE",
    "BUYB.L", "SLXX.L", "ZPRI.DE", "AGRV.VI", "EXI2.DE", "SBBb.ST", "BEIAb.ST", "ACSO.L", "ANTO.L", "FBK.MI",
    "OLTr.AT", "CBU3.L", "SKEL.IC", "LINK.OL", "UEF7.DE", "CBU7.L", "EL45.DE", "AJAb.ST", "MERCM.L", "SMC.PA",
    "BGBSE.BB", "STQ.PA", "CAIF.PA", "BILIa.ST", "CTY1S.HE", "JET2.L", "BILL.ST", "XUTD.L", "XHYG.DE", "CINT.ST",
    "DEK.WA", "ENRGV.L", "CMO.PA", "CRTO.PA", "KABEb.ST", "CIV.PA", "YFID.MC", "1INN.DE", "IEFQ.L", "JDAN.CO",
    "SPOG.OL", "ENEA.ST", "HRPKk.DE", "NSKOG.OL", "BLEE.PA", "JEDT.L", "JMGI.L", "PEUG.PA", "AFISH.OL", "ENGIE.PA",
    "EVOG.ST", "VH2.DE", "ALGIL.PA", "XD5D.L", "EMEIS.PA", "KREATE.HE", "OHBG.DE", "CNYA.L", "IRKT.MM", "IF.MI",
    "ALFA.ST", "ISUR.MC", "INTRUM.ST", "IR5B_u.I", "IQQS.DE", "IPRV.L", "IUKD.L", "WG.L", "AOJb.CO", "UWGN.MM",
    "EEXF.L", "CCB.BB", "MSH.BB", "KCOGn.DE", "AA4A.L", "AMEZ.MM", "WTCM.MM", "GTLY.L", "AZRN.AS", "ACAST.ST",
    "PSDL.L", "LUOTEA.HE", "ATG.L", "ATSI.MC", "NTEA_p.L", "ALIG.ST", "NCH2.DE", "BGSO.BB", "OHLA.MC", "ACS.MC",
    "OLVAS.HE", "NXFIL.AS", "KNOS.L", "ZPRP.DE", "XLES.L", "XLYS.L", "PENR.OL", "CBSUS.MI", "PCR.WA", "CU9.PA",
    "TNIE.DE", "JDG.L", "SDGI.PA", "EAPI.PA", "MIA.MT", "SOLN.BR", "SOPR.PA", "LHV1T.TL", "PPGN.S", "ORSO.MI",
    "ERNT.ZA", "DEF.DE", "ESCT.L", "KOFOL.PR", "SHCS.L", "D5BL.DE", "YRM.MI", "YZBL.MC", "RXP5EX.DE", "ELMT.SBX",
    "ROSMTL.BX", "VOES.VI", "EDP.LS", "FDJU.PA", "CPIE.VI", "VIC.L", "COMF.L", "IUHC.L", "ZPDJ.DE", "ICGIN.L",
    "PDXI.ST", "ATTE.ST", "LRND.DE", "NMTP.MM", "BVAL.HT", "ZAYM.MM", "MURP.WA", "GMI.S", "EUHD.L", "TRIANb.ST",
    "XCS5.DE", "XCS4.DE", "EEEE.L", "CAPD.L", "ETFEMMA.DE", "TRUEb.ST", "ENEI.MI", "GIGA.OL", "MBR.WA", "SPP3.DE",
    "XACTOBLIG.ST", "FDEL.PA", "LABS.L", "IVG.MI", "EJAP.PA", "EESM.PA", "EVROr.AT", "3SUG.DE", "XDWI.DE", "VETY.L",
    "LGQM.DE", "FSGF.L", "LIAB.ST", "SAXG.DE", "ALIFb.ST", "SIVI.ST", "YDOA.MC", "7V0.DE", "AOM.L", "PSALr.AT",
    "WTCH.AS", "WTEL.AS", "AMBEA.ST", "TOKMAN.HE", "BEZG.L", "MLGP.WA", "MBHG_p.F", "INFU.PA", "HOME.MC", "ADVNT.MI",
    "LYPE.DE", "CSN.L", "EXENS.PA", "MBH.L", "ITAB.ST", "SNOR.OL", "IMBS.L", "CTT.LS", "ANFO.S", "ZPHR.L",
    "LIGHT.AS", "TTALO.HE", "THEON.AS", "ADB.MI", "COPN.S", "INTEAb.ST", "SVOLb.ST", "SCANFL.HE", "COPX.L", "BOWL.L",
    "RENI.MM", "PFD.L", "SKP.MKE", "UN0k.DE", "MAHAa.ST", "DIG1.WA", "NORCO.OL", "MAA.PA", "ORTHEX.HE", "ENRGY.BR",
    "LNSX.F", "IUSEA.L", "IGUS.L", "CARDC.L", "GIL5.L", "ADP.PA", "FRLr.AT", "INRr.AT", "NBGr.AT", "BRGE.L",
    "RAFP.WA", "MLMTP.EUA", "JZCP.L", "ZAB.WA", "ALTS.BU", "2B78.DE", "SHEL.L", "VNHq.L", "CMBT.BR", "RWLP.WA",
    "MING.OL", "FEES.MM", "HAUTE.BN", "ASCI.MI", "ESNT.L", "GGC.MC", "CTEC.L", "PHMR.MC", "SENSI.S", "SFSN.S",
    "RVU.WA", "UKRGAZB_p.EP", "UPR.I", "ICAG.L", "PATRI.BN", "SABT.PA", "FLOT.MM", "ASKER.ST", "ALFPC.PA", "HT.ZA",
    "SNG.BX", "HMCH.L", "STEFb.ST", "DSFIR.AS", "ROEL.BX", "NIIS.BEL", "NONG.OL", "HCAN.L", "QQ.L", "F3CG.DE",
    "ASTOR.NGM", "ENAE.WA", "ASTH.WA", "V3RE.AS", "XPS.L", "VLAN.AS", "XMME.DE", "METSO.HE", "ISMVEA.L", "HEBAb.ST",
    "EISP.NFF", "CZG.PR", "SBOS.OL", "MAIS.ZA", "SLTG.H", "ORX.ST", "GLEN.L", "SYENS.BR", "DUNI.ST", "DOMETIC.ST",
    "ATRLJb.ST", "IJPN.L", "SYBA.DE", "SYBB.DE", "VOLO.ST", "SYBC.DE", "SYBTG.DE", "SYBS.DE", "III.L", "EMRG.DE",
    "PAGRD.LU", "MPCC.OL", "WIHL.ST", "IMC.WA", "ONT.L", "AMIF.L", "TKMS.DE", "INSTAL.ST", "USFM.L", "HITV.NFF",
    "INCLU.BR", "AGROUP.ST", "PAX.ST", "ALBFR.PA", "MOZN.S", "STXPP.WA", "FSV.L", "SPXS.L", "VKCO.MM", "RKH.L",
    "SDIPpref.ST", "SBNOR.OL", "TIBN.S", "CSBGC7.S", "DJIEX.DE", "STX50EEX.DE", "BRIQr.AT", "ENITY.ST", "GPW.WA", "SOHO.L",
    "SYSD.MI", "OIG.L", "HYPEG.DE", "C006.DE", "EQUI.MI", "RTKM.MM", "ETLNDR.MM", "CA1.DE", "HACK.ST", "ENOG.L",
    "UIMP.DE", "ESREUA.DE", "ELFV.MM", "YIRG.MC", "HBX.MC", "SEL.PA", "CAMX.ST", "SVUSA.S", "AUEUAH.S", "MERY.PA",
    "ABNX.PA", "OSE.PA", "SEDANA.ST", "MLPG.DE", "IWRD.L", "RPI.L", "PON1V.HE", "DESI.MI", "CS.ST", "BIOAb.ST",
    "IMCO.MC", "CHEF.ST", "GENOb.ST", "30IG.DE", "TYRES.HE", "IMPN.S", "NLFSK.CO", "IKOR.L", "IAEX.L", "IBCI.L",
    "VJGZ.MM", "PRISMA.ST", "BALYO.PA", "100H.PA", "ROSFG.BX", "HPHA.DE", "AGVI.L", "BERI.L", "YVIV.MC", "SEAA.DE",
    "GXIG.DE", "IOGP.L", "INFRAC.NGM", "DBV.PA", "KEYS.L", "CLN.S", "U03A.L", "VALS.MI", "OP5H.PA", "BKSB.L",
    "IFAG.H", "AIBG.I", "CHRT.L", "MRWN.L", "EDVD.L", "BGSYN.BB", "PATGn.DE", "NLMK.MM", "CBF.WA", "HRIG.ZA",
    "MOTA.LS", "ATYM.L", "C4D.PA", "ISFEL.IC", "VEGE.DE", "SP3C.DE", "YVCP.MC", "IES.L", "O5G.DE", "DMPG.DE",
    "CCH.L", "ABIO.MM", "CLTN.S", "IQGA.S", "PGE.WA", "NDAFI.HE", "KARNELb.ST", "GSP.MI", "C50.PA", "SREA.S",
    "MBHB.BU", "S92G.DE", "BPGEG.DE", "PRIUA.PR", "NTHV.L", "OIT.L", "X1G.PA", "LYY7.DE", "MCEU.PA", "ACT.WA",
    "LAMr.AT", "TIN.L", "USAB.L", "ALFEN.AS", "CCC.L", "BESI.AS", "LOUP.PA", "CV9.PA", "VCTX.L", "AEEM.PA",
    "AMCr.AT", "SNWS.L", "ASBP.WA", "MAGN.MM", "PHP.L", "Z29.DE", "DOMP.WA", "FLXr.AT", "SPGG.DE", "BOOK.L",
    "FFARM.AS", "HOCM.L", "MGTS.MM", "235H.BB", "BORY.WA", "NTGNT.CO", "ALBOU.PA", "CRSU.PA", "MEL.MC", "CECG.DE",
    "FTEP.WA", "CEZP.PR", "MERG.L", "NLBR.LJ", "DPTP.PA", "OXB.L", "ALLAN.PA", "NETC.L", "SFC.MT", "BALTO.HT",
    "PROX.BR", "CNA.L", "CAIV.VI", "ABGV.VI", "UBMV.VI", "BAR.BR", "IMMO.BR", "IWDP.L", "RECT.BR", "SIFB.BR",
    "AGRG.F", "EBKG.DE", "SURG.DE", "BAYGn.DE", "KWGG.DE", "BBVAE.MC", "WWGG.F", "HEIG.DE", "DEZG.DE", "ATB.BX",
    "ISCEUG.L", "SKBG.DE", "NEAG.S", "SZGG.DE", "SZUG.DE", "CAST.ST", "PEHN.S", "SAN.MC", "PHARP.L", "AIE.L",
    "TUBA.MC", "NEA.MC", "TEF.MC", "CHEMM.CO", "VIS.MC", "CARR.PA", "AXAF.PA", "TCFP.PA", "RXL.PA", "ICAD.PA",
    "NAE.PA", "EEPC.PA", "KSBG.DE", "BETSb.ST", "VRCP.WA", "FRVIA.PA", "SGEF.PA", "MGCI.L", "IRCP.L", "MAUP.PA",
    "PERP.PA", "AREIT.PA", "RENA.PA", "MOWI.OL", "SAMS.PA", "SEBF.PA", "EXHO.PA", "IMAF.PA", "TFFP.PA", "THHG.PA",
    "AAA.PA", "DSCV.L", "SDY.L", "ALUG.L", "BNKR.L", "BAES.L", "ADIG.L", "AGT.L", "BLND.L", "BUT.L",
    "NTAS.L", "CWK.L", "POLR.L", "PRO.BN", "DLN.L", "BGEU.L", "JCH.L", "JUSC.L", "FSTA.L", "GSK.L",
    "JHD.L", "HMSO.L", "HLCL.L", "UPFE.S", "JMAT.L", "KSP.I", "LAND.L", "LISN.S", "STHY.L", "RNWH.L",
    "APTD.L", "MPAC.L", "MGAMM.L", "STP.WA", "NXR.L", "RSL2.DE", "UU.L", "PACA.L", "ASCE.BR", "TOR.WA",
    "PSON.L", "PSN.L", "IMP.BX", "PIRC.MI", "REL.L", "ATTP.WA", "EUBG.BB", "XSMC.S", "RTO.L", "MPE.L",
    "INS2.DE", "AET.L", "RIO.L", "NOVN.S", "NIVIb.ST", "COME.MI", "SAIN.L", "SRP.L", "AREN.ZA", "SN.L",
    "MTGb.ST", "WEW.DE", "SPX.L", "FOBANK.CO", "TMPL.L", "TLW.L", "KONL.ZA", "ULVR.L", "MARS.L", "ZURN.S",
    "NESN.S", "JPNH.PA", "ROG.S", "BVZN.S", "ALSEN.PA", "HLEE.S", "CFR.S", "XAR.L", "NN.AS", "CFT.S",
    "SGSN.S", "LEHN.S", "LOGN.S", "ESOE.L", "SCHP.S", "SWTQ.S", "SFZN.S", "SIKA.S", "ALVELg.LU", "BYS.S",
    "BOUV.OL", "AIXGn.DE", "MTEL.BU", "SIGNC.S", "NP3.ST", "KGX.DE", "AAEV.L", "XGLE.DE", "SMOP.OL", "SMLT.MM",
    "X710.DE", "GHH.L", "ERB.WA", "SEIT.L", "GSFG.OL", "ZPLA.S", "SWOR.PA", "ARGAN.PA", "HAE1T.TL", "OOA.L",
    "WPPL.WA", "SAFE.L", "SPDY.BB", "CICG.LJ", "POMRY.PA", "ZEEP.WA", "1U1.DE", "HYDR.MM", "SLRS.MC", "ACR.OL",
    "VERTb.ST", "CRPS.L", "DKUPL.PA", "HAWG.DE", "MONY.L", "BEFG.F", "MNTN.L", "BEAN.S", "ISEMMV.L", "VSVS.L",
    "XDUK.DE", "ESGE.PA", "NAFG.DE", "NKNC.MM", "SESFd.PA", "SYNSAM.ST", "DOCO.VI", "SOLS.MI", "DUST.ST", "KRW.PA",
    "SUSW.ST", "ACIR.ZA", "PSAGn.DE", "FTON.S", "ALEP.WA", "BITTI.HE", "DBX9.DE", "WARB.BR", "UKWG.L", "XD3E.DE",
    "XNIF.DE", "XMKO.DE", "DFCH.L", "YGMP.MC", "CATHP.MI", "CABK.MC", "LPKG.DE", "RLRT.BR", "IKBZ.SJ", "ROLO.MM",
    "SOFb.ST", "MHV.CY", "COLOb.CO", "GN.CO", "TIV.CO", "TAALA.HE", "AERS.L", "FLS.CO", "ARBN.AS", "LTEN.PA",
    "VOPA.AS", "RLIN.AS", "WEHA.AS", "WLSNc.AS", "NOVAM.MI", "EVKn.DE", "QDT.PA", "GOPD.BU", "NORBT.OL", "AGOP.WA",
    "MEVr.AT", "AKER.OL", "EPROb.ST", "ATEA.OL", "ORK.OL", "ESYGn.H", "MRKS.MM", "ISP.MI", "CRDI.MI", "IBCQ.DE",
    "SGFI.MI", "GWI.L", "UTG.L", "KALDA.IC", "PALF.VI", "RECL.L", "INDUa.ST", "AMG1L.VL", "XYP1.DE", "LATOb.ST",
    "SHBa.ST", "HINV.BR", "SAMPO.HE", "LINDEX.HE", "ISUS.L", "IBCX.L", "SYBD.DE", "SYBF.DE", "LIO.L", "DJDVPEX.DE",
    "TKOO.PA", "FOXT.L", "EMUL.MI", "EMUM.S", "DIGIA.HE", "CLASb.ST", "ZPRS.DE", "VIE.PA", "RKW.L", "LONN.S",
    "HTWS.L", "ROLION.BX", "BYLOTr.AT", "YADV.MC", "BFSA.DE", "CEMI.MI", "RTW.L", "QRF.BR", "KBUG.H", "VASTB.BR",
    "ETFGDAXI.DE", "ALFLE.PA", "KOD.L", "KOF.PA", "MRKP.MM", "FREY.PA", "ENTRA.OL", "SELP.WA", "BYG.L", "AGLX.OL",
    "MUSTI.HE", "SQN.S", "KGN.WA", "TXTP.WA", "FRP.L", "CLOUD.OL", "VIEI.NFF", "RTX.CO", "SCHLR.MC", "NOVALr.AT",
    "NEL.OL", "DATA.L", "HTRO.ST", "SANION.ST", "JDEP.AS", "GNS.L", "QRS.WA", "CAI.MI", "STLL.ST", "TCAPI.L",
    "MRL.MC", "CEU2.PA", "ROAQ.BX", "JAREN.OL", "BAVL.PFT", "XD9U.L", "TNRC.EUA", "EQSE.PA", "DHLn.DE", "DIB.MI",
    "IDVY.L", "PSP5.PA", "AWAT.PA", "BVB.DE", "EIMI.L", "QDVD.DE", "EBMMEX.DE", "ETFUSLC.DE", "EDRE.MC", "REY.MI",
    "ONTEX.BR", "PREFr.AT", "PHL.CY", "CADLR.OL", "SKDR.LJ", "EVDG.DE", "AATG.L", "TNB.MKE", "XDWD.DE", "LOG.MC",
    "CPRI.MI", "ATRP.WA", "SDIS.L", "CP9.PA", "CD8.PA", "TINCC.BR", "TPRO.MI", "XDEV.DE", "UEF5.DE", "CSSMI.S",
    "DXS6.DE", "BIM.S", "XDNY.DE", "BEWI.OL", "IUS6.DE", "SOIT.PA", "KBCA.BR", "GDX.L", "ZITO.ZA", "ZGBM.SJ",
    "VNRT.L", "VERX.L", "SGZH.MM", "IWVL.L", "PPET.L", "HOMJ.EUA", "CLSH.L", "ETFGS11.DE", "ZAP.OL", "ETFGSMM.DE",
    "BINV.ST", "KRE.CO", "AGFEb.CO", "SBIO.L", "PRT.MI", "SMSWLD.DE", "EUN8.DE", "SEGA.L", "IPXJ.L", "IBGY.L",
    "EMMN.S", "WASP.WA", "AFK.OL", "INF.L", "TIMAn.DE", "CSUSS.S", "TUNE.L", "ADDTb.ST", "LAGRb.ST", "9HF.D",
    "BCP.LS", "QLT.L", "BAWG.VI", "STU.PA", "IEAG.AS", "FIM.MT", "ASOS.L", "IEAC.L", "CAMBI.OL", "CFEB.BR",
    "CHMS.EUA", "SFAST.ST", "XHY1.DE", "INAC.L", "ITWB.MI", "SRG.MI", "SPBE.MM", "NTN.L", "CIAK.ZA", "REALCONSr.AT",
    "IEFM.L", "MAAT.PA", "ENI.MI", "ERST.VI", "SEQI.L", "PLNW.PA", "GFTU_u.L", "QB7G.DE", "NB2.DE", "EUXS.L",
    "IDVP.PA", "CORD.L", "ILKKA2.HE", "ARGX.BR", "RUG.ST", "MOONM.L", "IGLO.L", "GNC.L", "VQTG.H", "NAPA.OL",
    "HTGG.DE", "ISIJPA.L", "ITWN.L", "JUNG_p.DE", "ELFA.DE", "TTS.BX", "HVAR.BB", "BGSHB.BB", "MILDEF.ST", "H50E.L",
    "TLGO.MC", "CSGLDC.S", "SDGPEX.DE", "SLEN.MM", "CSGLDE.S", "KGRG.F", "WRT1V.HE", "RCH.L", "HKDU.S", "JLP.L",
    "SHO1.WA", "RKET.H", "MCM.MC", "AQUA.MM", "COOR.ST", "BULTEN.ST", "SDJE5D.DE", "RSGN.S", "YUBICO.ST", "IHPI.L",
    "KURN.S", "BACB.BB", "DVD.OL", "RJFE.MC", "EQQQ.L", "FIMI.MI", "JUGI.L", "XLFS.L", "XLKS.L", "ELFB.DE",
    "SECUb.ST", "XBLC.DE", "BEIJb.ST", "CRI1.WA", "PCOP.WA", "RGLDOU.S", "EQNR.OL", "UBUS.DE", "UBUT.DE", "ABNd.AS",
    "ETFLCD.DE", "ELFC.DE", "SVEDb.ST", "VIMIAN.ST", "1COVG.F", "TSM1T.TL", "FCSS.L", "PTSB.I", "ADXR.MC", "OTB.L",
    "OKEA.OL", "NORDH.OL", "HRA.MI", "VBGb.ST", "CALI.MI", "RXP1EX.DE", "DUNA.BU", "CORA.LS", "ALAFY.PA", "MDKT.ZA",
    "GYM.L", "KRIr.AT", "OLPr.AT", "ICDU.L", "CAML.L", "SHOTE.ST", "NWGP.WA", "APNA.L", "E907.DE", "BSRT.L",
    "AMBUb.CO", "XIOR.BR", "MMGRb.ST", "GROW.L", "FTKn.DE", "HLE.DE", "A3M.MC", "PRFr.AT", "BIKE.DE", "MRFr.AT",
    "NVM.DE", "SKISb.ST", "THG.L", "DEEZR.PA", "APR.WA", "FHZN.S", "ROSN.MM", "PCGH.L", "XQUA.DE", "ALVAZ.PA",
    "OMR.MI", "VEND.OL", "APPB.BU", "EWGW.L", "IDUNb.ST", "MOHM.L", "AVANZ.ST", "GCC.L", "XDWS.DE", "MCG.L",
    "XWTS.DE", "LEMON.HE", "VACN.S", "ALTPC.PA", "TRMDa.CO", "PLWP.WA", "CMXC0.L", "MFONI.RTS", "IHYG.L", "DTGGe.DE",
    "MEDX.S", "5NB.D", "CRNOSb.ST", "IS15.L", "LYPD.DE", "F701.DE", "XB4F.DE", "AUTO.OL", "MIDWM.L", "TRN.MI",
    "ITM.L", "GETIb.ST", "RAUTE.HE", "BIOX.PA", "SPAN.ZA", "NOTE.ST", "MEDCAP.ST", "IREN.S", "MERM.MKE", "ZWCG.BU",
    "MASIA.MI", "GRLS.MC", "ATGR.ZA", "SUBC.OL", "LMTC.PA", "BURE.ST", "FAHY.L", "DEME.BR", "OTL.OL", "EMREA.CY",
    "PAYP.L", "EEE.PA", "E40.PA", "NKHP.MM", "GIBG.H", "BMPS.MI", "RWY.MI", "HIAB.HE", "GEO.MI", "SAVS.L",
    "EPAr.AT", "URNG.L", "SPKSJF.CO", "TIP10D.S", "AKE.PA", "BSNL.SJ", "EELT.MM", "LUCEL.L", "GILS.PA", "BRKN.S",
    "US13.PA", "R3NK.DE", "EEFG.PA", "PZU.WA", "BBH.L", "IG.MI", "ORSTED.CO", "ARDS.AS", "1SXP.DE", "KOJAMO.HE",
    "FOPE.MI", "ELC.MI", "AIAr.AT", "LDA.MC", "KOGK.MM", "D6HG.DE", "MLATR.EUL", "VETO.PA", "ZELA.CO", "GGR.MC",
    "ALSMB.LS", "ENT.L", "APSA.MT", "EIOF.OL", "DNREM.BEL", "APAM.AS", "CASHP.MC", "MBK.WA", "HRI.L", "AUTON.S",
    "RANA.OL", "AERO.BEL", "SFER.MI", "S5USAS.S", "MUUSAS.S", "IPAR.PA", "RENE.LS", "AGNr.AT", "YEPSA.MC", "BGUK.L",
    "BFF.MI", "ZUMV.VI", "SRENH.S", "PIA.MI", "ALK.MKE", "TOM2.AS", "SANN.S", "BOCH.CY", "RFXR.L", "COXGA.MC",
    "WUSH.MM", "MMKV.VI", "KLR.L", "PRMB.MM", "KNIN.S", "PROT.OL", "SRFCHA.S", "DESN.S", "MCOVb.ST", "MVC.MC",
    "XESP.DE", "EJFI.L", "PANP.BU", "INDB.MI", "OTPB.BU", "SPYYG.DE", "SVAV.MM", "SD3PEX.DE", "EMAE.DE", "AAK.ST",
    "KNEBV.HE", "TVE1T.TL", "CHG1.DE", "HUSCO.CO", "LSEG.L", "FREN.S", "EVS.MI", "PHN.WA", "R1GR.AS", "MWOP.DE",
    "SECU.MC", "PRBU.BX", "STDM.PA", "VIGR.VI", "DOUn.DE", "HNRGn.DE", "ROPE.BX", "B26A.PA", "GRPG.I", "SEC.L",
    "BKOM.PR", "SEA1.OL", "LEAS.MM", "AMSU.L", "COFB.BR", "ASPO.HE", "KBHL.CO", "EUN.S", "100CHA.S", "MDAXIEX.DE",
    "GDAXIEX.DE", "STX50EX.DE", "TECDAXEX.DE", "SX7EEX.DE", "SX8PEX.DE", "SXIPEX.DE", "SX6PEX.DE", "SX3PEX.DE", "SXNPEX.DE", "CSSMIM.S",
    "SCAML.MC", "MLPRG.EUA", "FNOVAb.ST", "INTD.MI", "TRAINb.ST", "MTU.L", "DFND.PA", "EFGN.S", "SCWEL.MC", "AL2SI.PA",
    "ADMCM.HE", "CITYVA.HE", "EUREUA.S", "WWH.L", "TRYG.CO", "MEDCL.PA", "CANTA.ST", "LQAG.DE", "PRFD.L", "VOTR.DE",
    "NYXH.BR", "WPEA.PA", "POST.VI", "SUPPA.LU", "IFFF.L", "BRBI.MI", "HEIO.AS", "KALMAR.HE", "NYAB.ST", "HASE.L",
    "PCXP.WA", "MDMG.MM", "JDC.DE", "SKAN.S", "MTI.PA", "IGC.L", "PEBB.L", "DOM.L", "ATM.L", "RHIM.L",
    "STB.MKE", "ARL1.WA", "EIKF.IC", "ENH.OL", "B8AG.MU", "SBRE.L", "THRr.AT", "TEQ.ST", "MCB.L", "BPM.L",
    "CRLA.PA", "REGU.VI", "MSRS.MM", "EGV5.DE", "LYKOa.ST", "FIR.S", "SIC.S", "CEVI.ST", "ISLAX.OL", "AYV.PA",
    "OLEO.MC", "FESH.MM", "SYB3.DE", "PGHN.S", "SNGS.MM", "ZILL.MM", "SATSS.OL", "ID31.L", "ID32.L", "34GI.DE",
    "31IG.DE", "BKHTn.DE", "PUUILO.HE", "SEM.LS", "LASP.CO", "CATE.ST", "1TAT01DE.BV", "ESCEUA.DE", "KEO.CY", "MULT.DE",
    "CLIG.L", "XVIVO.ST", "XB4A.DE", "CEU.PA", "NWLF.MI", "INHG.DE", "CBAV.MC", "VLTSA.PA", "EVRE.L", "CBHB.BR",
    "PLZL.MM", "MORROW.ST", "VLP1L.VL", "LNA.PA", "D77n.DE", "FRAMERY.HE", "PCZ.DE", "S9I.H", "DIAS.MI", "SCW.WA",
    "IM.S", "KAMBI.ST", "JTC.L", "CCP.L", "AM3A.PA", "AGEB.PA", "SEDY.L", "SCMAS.MC", "CSW.PA", "MRCG.DE",
    "PARRO.PA", "MAST.BU", "VSURE.ST", "RS2U.PA", "JPNK.PA", "CIBUS.ST", "SUP.L", "MIVA.DE", "GREENL.ST", "NG.L",
    "N225EX.DE", "IDEr.AT", "MXHNn.DE", "DWSG.DE", "ALMP.PA", "CHKZ.MM", "LNZL.MM", "ALNSE.PA", "BSPB.MM", "NYFO.ST",
    "EVPL.L", "DTEGn.DE", "LVC.PA", "SAGO.MM", "ELEVR.RI", "VNAn.DE", "UPM.HE", "NORTHM.CO", "BHW.WA", "QIA.DE",
    "FMEG.DE", "ELMN.S", "IPX.L", "WCHG.DE", "SPOLS.OL", "GLKBN.S", "TELIA.ST", "BOSP.WA", "MANV.VI", "OBER.VI",
    "BNAB.BR", "BEKB.BR", "COLR.BR", "ATEO.BR", "PHG.AS", "SOF.BR", "RASP.MM", "CVXC.PA", "VSYD.MM", "PETG.LJ",
    "SOAC.BR", "BASFn.DE", "PBBG.DE", "BOSSn.DE", "BIJG.DE", "NTGG.DE", "ALSOG.PA", "DUEG.DE", "DREGa.F", "HABAn.DE",
    "HLAG.DE", "PHHGn.F", "FDSS.SJ", "HNKG_p.DE", "OELG.F", "TCMK.SJ", "VBBB.BJ", "KGHK.L", "PUMG.DE", "KULG.MU",
    "HETR.BJ", "LSL.L", "SIEGn.DE", "NEPG.H", "SSHG.F", "WUG.H", "YCA.L", "KEFI.L", "LUVE.MI", "WUWGn.DE",
    "ACX.MC", "PQ.MI", "EBRO.MC", "ENC.MC", "ASAI.L", "BNPP.PA", "BOUY.PA", "BULY.PA", "SCOR.PA", "VCTP.PA",
    "ODET.PA", "ESLX.PA", "EURA.PA", "COVH.PA", "EXEP.PA", "FMONC.PA", "BABr.AT", "PANDXb.ST", "UEFF.DE", "BIOGW.L",
    "CVO.PA", "ALBI.PA", "GFII.PA", "IMTP.PA", "FINA.PA", "LVMH.PA", "MICP.PA", "ST5G.DE", "OPM.PA", "ALPM.PA",
    "PRTP.PA", "PUBP.PA", "ROBF.PA", "ATOS.PA", "VEIL.PA", "VIRB.PA", "FAISr.AT", "BAB.L", "BAG.L", "BATS.L",
    "BSTP.WA", "ALCLS.PA", "BOY.L", "BNZL.L", "CKN.L", "DHER.DE", "COSG.L", "DPLM.L", "GSCT.L", "JAM.L",
    "JFJ.L", "HLMA.L", "ELM.L", "FCH.L", "HAYS.L", "HILS.L", "HSBA.L", "HTG.L", "WLTP.WA", "KYGa.I",
    "LTHM.L", "RNK.L", "MYI.L", "NWB_pa.L", "PHI.L", "SSON.L", "MEGPM.L", "RCP.L", "SDR.L", "FGT.L",
    "JANF.ZA", "PODR.ZA", "SMT.L", "ARBB.L", "SBI.HE", "SVT.L", "BGS.L", "JAGIJ.L", "SMIN.L", "SWC.L",
    "STAN.L", "UBU3.DE", "UBU7.DE", "WLN.PA", "V3SG.H", "SUS.L", "MRCM.L", "THRG.L", "PKO.WA", "TPK.L",
    "FRAS.L", "ATRAS.L", "HSL.L", "EUFI.PA", "AMCP.WA", "TTG.L", "AVG.L", "LBIRD.PA", "VIP.L", "WTB.L",
    "HELG.OL", "ASCN.S", "UZU.DE", "RIEN.S", "BSKP.S", "OERL.S", "KTCG.VI", "DAE.S", "EMSN.S", "GF.S",
    "BHMG.L", "MLXS.BR", "HOLN.S", "HBLN.S", "DGV.MI", "KARN.S", "MIKN.S", "IS0M.DE", "PMN.S", "TECN.S",
    "IS0P.DE", "VONN.S", "CLDP.WA", "VFEM.L", "ABSP.WA", "GYC.DE", "IBS.LS", "JNOS.MM", "XS7R.DE", "X35E.DE",
    "ELISA.HE", "TITC.BR", "NEWAb.ST", "ALECR.PA", "PRY.MI", "KBSB.MM", "NVGR.LS", "MLCHE.EUA", "TPE.WA", "KTAG.DE",
    "LOGIa.ST", "OBL.WA", "SFAB.ST", "CRST.L", "NSTEc.AS", "RETE.BR", "ALM.MC", "MNDI.L", "FAGRO.BR", "MARTI.LS",
    "MNG.L", "SPY1.DE", "CICN.S", "HPOLb.ST", "ETGG.DE", "OPUSG.BU", "LAZI.MI", "IVAA.PA", "AJAX.AS", "PLEJD.TE",
    "KLARAb.ST", "MVCT.L", "O2Dn.H", "PREVb.ST", "WWL.WA", "AMG.AS", "MRKZ.MM", "MLTAC.MT", "2GBG.DE", "XDDX.DE",
    "XDJP.DE", "BOND.PA", "ALCIS.PA", "EM.MI", "NBPE.L", "MRKU.MM", "VGOP.WA", "MTCM.MI", "KMB.MKE", "MPT.MKE",
    "EDLG.DE", "IGST.MM", "F35.D", "SWECb.ST", "CITT.PA", "OCI.AS", "SKUE.OL", "XMEM.DE", "YTRI.MC", "RKT.L",
    "BAVA.CO", "CEKG.DE", "SLICHA.S", "TFIF.L", "OVH.PA", "OSRn.H", "ARCHA.OL", "XFFE.DE", "NKT.CO", "DNORD.CO",
    "ALSYDB.CO", "SOFI.LU", "AD.AS", "AKZO.AS", "FNAC.PA", "TFF.PA", "HMSN.ST", "TSWE.AS", "HEIN.AS", "NL0000009645.BR",
    "ZPRG.DE", "MVVGn.DE", "CTTS.ST", "BSLG.DE", "ADE.DE", "FRGT.L", "BONHR.OL", "NHY.OL", "STB.OL", "TOM.OL",
    "KSL.HE", "THRLT.L", "CIG.WA", "SNOX.ASE", "BRAV.ST", "MOED.MI", "GASI.MI", "BTGI.MI", "ITMI.MI", "SIBN.MM",
    "TLIT.MI", "TRIG.L", "IQE.L", "HVPEa.L", "FLES.LU", "SFPN.S", "TPEG.DE", "BMAG.S", "ANODb.ST", "HLUNa.CO",
    "SEBa.ST", "ORES.ST", "SKAb.ST", "SKFb.ST", "CSLP.S", "TRELb.ST", "KESKOB.HE", "VITb.ST", "OUT1V.HE", "NOKIA.HE",
    "FUTR.L", "AEVS.S", "NEXTA.BR", "SNGS.MC", "TIETO.HE", "SISG.DE", "ATSV.VI", "UNAC.MM", "POSR.LJ", "DJAPSDEEX.DE",
    "SUBEEX.DE", "SCXPEX.DE", "ESDD.PA", "KTEn.H", "TTKG.DE", "STUDBO.ST", "EKOP.BR", "SDHY.L", "ALRS.MM", "XZEM.DE",
    "MCON.I", "VIOH.AT", "JUSTJ.L", "BBOXT.L", "SFT.BB", "XSLI.DE", "XFVT.DE", "HYL.BR", "XXSC.DE", "XDGM.DE",
    "DADP.WA", "SFRG.ST", "BANB.S", "LUS1n.DE", "HFEL.L", "IMSI.MI", "TRIFOR.CO", "SPRSP.L", "TGHN.DE", "MRK1T.TL",
    "S30.PA", "SUNDELL.BU", "SPSN.S", "LTMC.MI", "ETTE.HE", "XCAC.DE", "KOMPLK.OL", "YTME.S", "ALRIB.PA", "ALKEY.PA",
    "ZO1G.H", "PAFR.L", "MEKO.ST", "SNC.LS", "TISGR.MI", "YACHT.MI", "CEVG.H", "TXTS.MI", "ARYN.S", "STORb.ST",
    "ETFGDAXP.DE", "PST.MI", "SPPY.DE", "EYE.L", "MPCKk.DE", "IRES.I", "FC9.DE", "FESTI.IC", "TXGN.S", "IMIB.L",
    "LTAM.L", "GHV1.L", "FEW0n.TG", "SRAG.DE", "EZJ.L", "SEFER.PA", "MOBN.S", "PAEJ.PA", "KLDVK.OL", "MEDG.DE",
    "ENX.PA", "SMIF.L", "BFG.MI", "GAW.L", "CHAM.S", "ETFES57.DE", "IDOX.L", "SSH1V.HE", "HANZA.ST", "IGP.L",
    "AKO1L.VL", "STLAM.MI", "SPI.L", "BRMD.L", "LOOMIS.ST", "UBSGE.S", "SERK.MI", "P911_p.DE", "SFQ.DE", "DAL.MI",
    "ABL.OL", "CNAA.PA", "GPPP.WA", "ETFSTX5.DE", "BGGTH.BB", "SABE.MC", "VNV.ST", "NA9n.DE", "ZPRC.DE", "NEXS.PA",
    "BTSb.ST", "AQ.ST", "GDSF.PA", "HAGG.DE", "FNXF.L", "ECNER.MC", "HG1G.F", "BSS.MI", "DHS.L", "ETFGS13.DE",
    "RUAL.MM", "SDJ600.DE", "TEMN.S", "NXTMH.HE", "SFOODS.HE", "RMMC.L", "DLG.MI", "HUMBLE.ST", "WOEE.AS", "CM9.PA",
    "ANP.L", "DEMD.L", "SXRQ.DE", "ETFUSAC.DE", "ORRON.ST", "TWSr.AT", "SPSTP.PA", "EMII.MI", "STK.PA", "IEMS.L",
    "REACH.OL", "MLCFM.EUA", "BAKKA.OL", "MSTT.MM", "PRODr.AT", "X7PS.DE", "HHDC.CO", "ADVT.L", "CRLO.PA", "DJUR.CO",
    "93M1k.DE", "ALRR.WA", "DSV.CO", "IEFV.L", "ECORE.L", "IS05.DE", "IIG.L", "HEXAb.ST", "ZPRV.DE", "ALPUL.PA",
    "ELANb.ST", "DKG.DE", "VAF.LS", "FNMI.MI", "DFSD.L", "ALCJ.PA", "FIA1S.HE", "FLUGb.CO", "ALMDT.PA", "NIOX.L",
    "FEML.L", "GPIF.PA", "ZEG.L", "PIKK.MM", "NANOB.PA", "AVAN.MM", "IUAG.L", "IASP.L", "ITRK.L", "INPST.AS",
    "IBGL.L", "IBGM.L", "TE.PA", "SALTb.ST", "EOTE.L", "CHSB.BB", "ATC.WA", "CFIP.WA", "ODES.BB", "PARB.OL",
    "TORO.L", "CSGOLD.S", "C5E.PA", "DEX.MI", "THQM.BB", "CNDF.PA", "TNOM.HE", "INVP.L", "TEKNA.OL", "CRN.L",
    "LRHR.ZA", "PIHLIS.HE", "ENTP.WA", "NOLAb.ST", "SSE.L", "CSINDU.S", "EFT1T.TL", "FIB.BB", "PLANZ.S", "PSMGn.DE",
    "TENR.MI", "RAND.AS", "ALESE.PA", "XLIS.L", "TORGH.NFF", "JEMA.L", "NTI.OL", "GOBP.WA", "ACWIU.S", "ECMPA.AS",
    "GRANIT.BU", "APTK.MM", "SSABa.ST", "TFBANK.ST", "ARVEN.PA", "TMI.L", "M12.DE", "RAYb.ST", "REJLb.ST", "MSICH.PFT",
    "SINCH.ST", "AR4k.H", "HAYPP.ST", "D5BH.DE", "D5BI.DE", "ABS.PA", "GABIG.L", "EXMR.BR", "VOSG.DE", "FABG.ST",
    "ISPY.L", "LUDE.MOT", "CURY.L", "FOTH.LU", "FROP.WA", "AUHV.ZA", "LUKA.ZA", "RGLR.L", "BVXP.L", "VEFAB.ST",
    "BPSI.MI", "CCJI.L", "GLEG.L", "GDRB.BU", "AMA.MC", "PAMPALO.HE", "SAAS.L", "BIGB.L", "CIGP.BU", "1AST.VI",
    "QEV.AS", "BOOMA.L", "GOT.L", "WWI.OL", "NAS.OL", "RODN.BX", "EDEN.PA", "SPPX.DE", "XACTSMAB.ST", "XRES.L",
    "BRINb.ST", "LEON.S", "JDW.L", "CTUK.L", "QDV5.DE", "XDWC.DE", "VECP.L", "VUCP.L", "VUTY.L", "AZE.BR",
    "XDWU.DE", "SOSIL.NGM", "IGN1L.VL", "SLBEN.LS", "PEEL.L", "CER.L", "OTEC.OL", "WAFGn.DE", "AMCO.L", "VOF.L",
    "AKAST.OL", "CSKR.L", "RFG.MI", "PABV.DE", "KEMPOWR.HE", "WCOS.AS", "XY4P.DE", "ZZb.ST", "ONYXr.AT", "SRSA.L",
    "CANATU.HE", "FGA.PA", "SRAD.L", "ZVTG.LJ", "ALSEM.PA", "HFD.L", "GBGP.L", "HRMS.PA", "WABE.BU", "1SN.L",
    "FMMb.ST", "ATOM.BB", "NTH.MC", "AZMT.MI", "FTC.L", "IDNAG.CY", "RICA_p.L", "RUSTA.ST", "AUSC.L", "SIFG.AS",
    "YVIT.MC", "AO.L", "FASTN.AS", "MTF.PA", "IWDE.L", "ANRJ.PA", "CREDIAr.AT", "ENR1n.DE", "JCGI.L", "HRMr.AT",
    "AGP.MI", "XTPG.DE", "YPSN.S", "ORAN.PA", "IGNY.BU", "NEUP.WA", "BTG.L", "AGY.L", "CPA1T.TL", "INF1T.TL",
    "HHV.L", "AGED.L", "XTB.WA", "AUCO.L", "AFKS.MM", "RDC.DE", "MGN.OL", "LHYFE.PA", "YEUR.MC", "FGP.L",
    "PMIP.L", "NOVA.IC", "BNJ.AS", "IGLS.L", "HBMN.S", "ROH2O.BX", "FERGR.AS", "HSB.MT", "SSPG.L", "SY1G.DE",
    "TTLK.MM", "BION.S", "SCHST.MC", "SDSD.OL", "VARN.S", "HIK.L", "CMOD.L", "ALCPB.PA", "DELIA.OL", "SLYG.BB",
    "IUQF.L", "LSRG.MM", "IOT.MI", "REMEDY.HE", "TABK.PR", "EUTR.MM", "TRFT.L", "MILP.WA", "BLNG.MM", "NEXII.MI",
    "TLSG.LJ", "BCGE.S", "INLIFr.AT", "BOKUS.ST", "MTXGn.DE", "JSW.WA", "NF4.DE", "S7XE.DE", "EUR.WA", "IGSG.L",
    "KPL.WA", "SRAIL.S", "DOCM.S", "GSJ.MC", "RBWR.L", "ALCAF.PA", "FPIP.ST", "ELTEL.ST", "AAGG.DE", "CLNP.WA",
    "YORE.MC", "2B7B.DE", "SB1NO.OL", "BRK.L", "ALSS.LS", "HIDRI.L", "FUSG.DE", "RBIV.VI", "DIVI.L", "NESTE.HE",
    "OLYr.AT", "GALE.S", "SJG.L", "YOU.L", "AKW.PA", "PEPP.WA", "EMSD.DE", "SRB.L", "TSTL.L", "NOAP.OL",
    "AEDAS.MC", "BREE.L", "AVAr.AT", "BIOV.S", "R1VL.AS", "FIEG.DE", "MMTP.PA", "ALFRE.PA", "DCC.L", "NTK.CY",
    "SNTIA.OL", "SOCP.BX", "ID26.L", "28ID.PA", "GENG.L", "ASSAb.ST", "ISMIDD.L", "HPROP.L", "ASTR.MM", "DI27.PA",
    "CDAF.PA", "EKTr.AT", "STMPA.PA", "DBG.PA", "JST.DE", "IB27.PA", "B28A.PA", "KARNO.ST", "RTSB.MM", "YCPS.MC",
    "HUMAN.ST", "SES.MI", "AEGN.AS", "SIGC.L", "BONEX.ST", "EQL.ST", "JPNEUA.DE", "R1JKEX.DE", "GRK.HE", "SX7PEX.DE",
    "SXKPEX.DE", "SXDPEX.DE", "SXAPEX.DE", "FINT.BEL", "SXEPEX.DE", "IOSn.DE", "CLARI.PA", "LPSB.MM", "JAAG.L", "DBC.WA",
    "YGOP.MC", "COTE.BX", "VICOR.ST", "ABP.MI", "AZT.OL", "YMRE.MC", "MLUAV.EUA", "DVYSR.ST", "ALSTI.PA", "C005.DE",
    "KMAZ.MM", "SCORE.MC", "SCPP.WA", "YLFG.MC", "SOST.L", "DIAS.MM", "ABRD.MM", "EGUSAS.S", "DXRX.L", "SABFG.PR",
    "ACCS.L", "VRSB.MM", "AT1.DE", "LOGISTb.TE", "IWDS.AS", "STEMS.L", "POSTI.HE", "SORA.L", "PCTN.L", "BDXP.WA",
    "NNSB.MM", "DRX.L", "1SLP401E.BV", "SCBYT.MC", "MUTG.F", "IBZL.L", "WSDG.DE", "MONC.MI", "SPAGA.L", "SPGPG.L",
    "SNL.MI", "ASLI.L", "VESTUM.ST", "VU.PA", "LYQ2.PA", "MVPP.WA", "SMHNn.DE", "CAN.L", "MSNG.MM", "HAVAS.AS",
    "SUNN.S", "SYBJ.DE", "GMKN.MM", "IAD.L", "NOM.OL", "FP.BX", "IG32.DE", "VOXP.WA", "SYB5.DE", "HEALTH.HE",
    "SAGA.L", "GSPA.BU", "MORLD.OL", "POLV.VI", "MGNT.MM", "BSEL.HT", "ORIOLA.HE", "BELU.MM", "ISS.CO", "CMCX.L",
    "TAAA.DE", "XACTNORDEN.ST", "RELAIS.HE", "TRMED.OL", "ATRY.MC", "QLCOr.AT", "TRU.L", "SHLG.DE", "BPDEG.DE", "LMDr.AT",
    "MICCT.AS", "REDE.MC", "TBTG.L", "MOLB.BU", "TGAS.BEL", "AAS.L", "SMGC.S", "CN1.PA", "IPH.PA", "AASI.PA",
    "XYLD.DE", "LASTIK.HE", "LPHr.AT", "AFLE.PA", "HARVIA.HE", "SDP.L", "DOKA.S", "ATT.L", "KAR.AT", "GSF.L",
    "C6E.PA", "ANX.PA", "RSF.S", "ECH.WA", "SELER.PA", "ASHM.L", "FUMA.MI", "NETCG.CO", "AOO.BR", "SHC.PA",
    "KROT.MM", "VSMO.MM", "FVEN.L", "FVSH.DE", "HDDG.DE", "IMB.L", "BDTG.DE", "LBW.WA", "TEL.OL", "KIE.L",
    "ELHA.AT", "SNK.WA", "CDRDRS.HT", "OPTIMAr.AT", "AURT.L", "KAER.VI", "BTS.VI", "VERB.VI", "LOTB.BR", "OAP3.L",
    "ECONB.BR", "IETB.BR", "SOLB.BR", "TKM1T.TL", "COGP.WA", "SPAB.BR", "BMWG.DE", "GEBN.S", "CBKG.DE", "CONG.DE",
    "LHAG.DE", "DWBG.SG", "CAVP.WA", "BRNKn.DE", "GILG.DE", "HNDG.F", "HOTG.DE", "SDFGn.DE", "BHTS.SJ", "MUVGn.DE",
    "RHKG.DE", "VIBG_p.DE", "VOWG.DE", "AZK.MC", "BBVA.MC", "FCC.MC", "GPM.L", "REP.MC", "XDAX.DE", "BOIR.PA",
    "SAVEN.PA", "CCAP.PA", "CBLP.PA", "COREa.ST", "CREADa.ST", "CIRSA.MC", "INXG.L", "IGLT.L", "FCMC.PA", "CROS.PA",
    "DPAP.PA", "GAUM.PA", "VIRI.PA", "ALGEV.PA", "GFCP.PA", "LOIM.PA", "LECS.PA", "BOLL.PA", "OTEr.AT", "PU13.L",
    "SPYW.DE", "SPYG.DE", "IDTP.L", "PADI.PA", "SGOB.PA", "VTA.AS", "VLLP.PA", "VLOF.PA", "ANEA.L", "BARC.L",
    "BALF.L", "BOOT.L", "BT.L", "CAME.L", "CPI.L", "TGKA.MM", "CHG.L", "AV.L", "AMNX.L", "ICGT.L",
    "JGGI.L", "ITV.L", "EXPN.L", "GRG.L", "HGT.L", "PANI.L", "KMR.L", "LGEN.L", "LWI.L", "MKS.L",
    "MSLH.L", "MIDW.L", "MTVW.L", "ISEMCR.L", "OBCK.DE", "MSTL.L", "SCP.L", "UNI.MC", "BCU.MI", "NICL.L",
    "PINE.L", "VANQ.L", "RAT.L", "RIICi.L", "RR.L", "M9SD.DE", "LAGP.ZA", "ROR.L", "SESTS.L", "SHI.L",
    "SHRS.L", "VISC.ST", "MGNS.L", "SMWH.L", "VZN.S", "TATE.L", "TET.L", "LLOY.L", "VP.L", "YNGa.L",
    "FNTGn.DE", "CHESG.S", "KCCK.OL", "ALSN.S", "BLKB.S", "CHRY.L", "CALN.S", "GRKP.S", "3IN.L", "ERG.MI",
    "ALLIX.PA", "ELT.WA", "BGBSP.BB", "SUN.S", "VETN.S", "WARN.S", "ZUGER.S", "TGS.OL", "AFEN.L", "IS0L.DE",
    "GRID.L", "WSU.DE", "CCHE.PA", "IPF.L", "XEIN.DE", "IBG.MC", "CEMB.MI", "OPN.WA", "EGTX.ST", "WLD.PA",
    "ALNTA.MC", "NXTE.MC", "ALMr.AT", "NMAN.ST", "BAHNb.ST", "TRNG.PA", "APR.OL", "HFG.L", "DATP.WA", "SABF.MI",
    "TRIOd.AS", "OMZZ_p.MM", "MOL.MI", "PWS.MI", "GSC1n.DE", "UIQI.DE", "SRV1V.HE", "SPYJ.DE", "TRKP.WA", "PTEC.L",
    "SAAA.L", "AUTW.BU", "COLUM.CO", "CSSLI.S", "SQRL.MC", "ENDUR.OL", "TYA.MI", "MVOL.L", "ISMVUS.L", "STF.PA",
    "BRZL.MM", "APG1L.VL", "STMN.S", "EPCG.MOT", "MRKC.MM", "ALCC.S", "USF.L", "BSQR.ZA", "HEPr.AT", "EWI.L",
    "MOVE.S", "AEJ.PA", "A2.MI", "IGV.MI", "ZPRE.DE", "ELL_p.L", "SCMN.S", "ALPN.S", "OBEL.BR", "XMUS.DE",
    "XMBR.DE", "RCN.L", "DOTD.L", "CTNGk.F", "SEYE.ST", "ROGREEN.BX", "OMASP.HE", "KBC.BR", "CVSG.L", "TZLB.SJ",
    "UBXN.S", "DFDS.CO", "CARLb.CO", "GYLDb.CO", "JYSK.CO", "RILBA.CO", "CABP.L", "ETOF.PA", "WBAHk.F", "SKJE.CO",
    "PSHP.L", "CVCG.L", "MTRK.AS", "MEKKO.HE", "YODAP.CY", "DNO.OL", "VEI.OL", "WBD.MI", "KME.MI", "DANI.MI",
    "LYXLEM.PA", "NREN.S", "PKPP.WA", "4DSG.DE", "INVEb.ST", "SAABb.ST", "VOLVb.ST", "FPE3_p.DE", "STERV.HE", "KWSG.DE",
    "DJMC.L", "CROR.ZA", "SIXG.DE", "ACE.MI", "KOBR.NGM", "NDXEX.DE", "XACTOMXS30.ST", "AJBA.L", "MCXPEX.DE", "SSMIEX.DE",
    "UKUKD.S", "SHAW.L", "SCST.ST", "ROSIFI.BX", "BRCK.L", "ELGG.DE", "IS3F.DE", "FAAS.DE", "FDM.L", "ASEP.WA",
    "BALDb.ST", "CURN.S", "PRSO.OL", "CARM.PA", "XJSE.DE", "BDRP.PA", "CTXP.WA", "PHIL.MI", "YSNG.DE", "BBSN.L",
    "BCNT.MI", "PKN.WA", "CHCORP.S", "SYBQ.DE", "PLAZb.ST", "YENT.MC", "MLSCI.EUA", "NAT.MC", "GAZC.MM", "VALMT.HE",
    "ACAGr.AT", "GPI.MI", "TRNF_p.MM", "MYCR.ST", "GFRD.L", "AFXG.DE", "BC8G.DE", "XMAW.DE", "MRKK.MM", "MRKV.MM",
    "MRKY.MM", "ECEL.L", "PVN.L", "ALLDL.PA", "ABVX.PA", "MDBr.AT", "TUNT.L", "PRX.AS", "DAIr.AT", "NRO.PA",
    "AERO.S", "NOHAL.OL", "EPICN.S", "VERK.HE", "STBS.L", "TEL1L.VL", "SJOVA.IC", "CED.MI", "PRS.MC", "ETFSD3E.DE",
    "UIE.CO", "BLEKEDROSr.AT", "TEP.L", "RAMA.LS", "UGLD.MM", "ALC.MC", "FEVR.L", "ANDR.VI", "ELF1.DE", "DEQGn.DE",
    "CHSPI.S", "PINR.PA", "TELA.VI", "NSISb.CO", "CLAVI.ST", "IEEM.L", "UEF6.DE", "LVE.PA", "DOMI.MC", "XGSH.DE",
    "SQZ.L", "PNDORA.CO", "REIT.LU", "DSPW.PR", "POL.OL", "SAPG.DE", "LDIT.L", "SAVES.L", "FA17.L", "MCI.WA",
    "FCT.MI", "BERNERb.ST", "CLOEb.ST", "ARION.IC", "OVS.MI", "CLNX.MC", "ZSILEU.S", "XDEB.DE", "MRB.WA", "XDEQ.DE",
    "SALME.OL", "ANTIN.PA", "ABSO.ST", "UBUM.DE", "ZALG.DE", "LIC.ST", "NDXG.DE", "N400.L", "TEM1V.HE", "BSC.L",
    "GMVM.DE", "GINX.SJ", "SVCB.MM", "UKSR.L", "IS3T.DE", "KCHE.MM", "ITX.MC", "GMRG.L", "ERMT.PA", "KRLB.BJ",
    "GAMA.L", "SPICHA.S", "CPHN.S", "AMPF.MI", "VITR.ST", "ETFDXMD.DE", "QTX.L", "CURAS.TE", "FFP1.WA", "MORr.AT",
    "BRIMH.IC", "AFRY.ST", "PYRG.DE", "C3M.PA", "CL2.PA", "AMUN.PA", "KID.OL", "CUKS.L", "CSEMUS.S", "CSBGE3.S",
    "LDCE.DE", "CSBGE7.S", "HAPD.IC", "BERGb.ST", "STN.PA", "SUY1V.HE", "FYNBK.CO", "CEV.MC", "SCATC.OL", "CTY.L",
    "AKTRr.AT", "CAGR.PA", "CAT31.PA", "GENFIN.MI", "JUVE.MI", "AGES.BR", "XQUI.DE", "IDHC.L", "DECB.BR", "IEBB.MI",
    "ATEME.PA", "FBH.I", "ABGA.OL", "AENA.MC", "CDR.WA", "FKAV.VI", "XDPE.DE", "XRSM.DE", "IEM.L", "HEM.ST",
    "HYDRA.AS", "MBBG.DE", "MOLN.S", "INWT.MI", "KBX.DE", "TROAX.ST", "MTHH.CO", "ZAL.OL", "SWON.S", "ARAMI.PA",
    "AI.MC", "WOODT.L", "IEAM.L", "IDUP.L", "XANOb.ST", "CONSTI.HE", "SMEA.L", "EEX5.S", "WTEEI.L", "AUHEUA.S",
    "DUH.BB", "NOBI.ST", "KEMIRA.HE", "AEWU.L", "HJ2.H", "AKRN.MM", "LGQK.DE", "HDLG.L", "CALCi.PA", "BRBY.L",
    "APETIT.HE", "SHA0n.DE", "PCELL.ST", "TRST.L", "MNBA.CO", "RVRC.ST", "GREG.MC", "ALREW.PA", "EPR.OL", "FEDF.MI",
    "DLKV.ZA", "AMEN.MC", "VSSABb.ST", "PBEE.L", "OEMb.ST", "HOFI.ST", "IMCD.AS", "CAHGBD.S", "AVI.MI", "OSCAPU.L",
    "YIT.HE", "TDT.AS", "AUCHAH.S", "XBTR.DE", "XLUS.L", "BEMO.L", "ADOC.PA", "XACTBULL2.ST", "SAUS.L", "SIM0.F",
    "SANOMA.HE", "ACBr.AT", "MORGS.OL", "EQV1V.HE", "TEL.MKE", "STXS.L", "IBC0.DE", "UMG.AS", "DJE.PA", "MAB.L",
    "IHG.L", "EQNX.LJ", "IDX1R.RI", "G24n.DE", "ALMAC.HE", "CY4.MI", "PFSE.DE", "VAIAS.HE", "VID.MC", "RXP2EX.DE",
    "MASM.OL", "WALLb.ST", "ADPL.ZA", "VAM.BB", "D5BK.DE", "IESU.L", "IUFS.L", "SAGAS.OL", "SSIT.L", "MRON.L",
    "AVCT.L", "LSPU.L", "GALP.LS", "SERE.L", "LOM.MT", "MXFS.L", "NNIT.CO", "GCPI.L", "VIVA.ST", "TRMK.MM",
    "XCS6.DE", "SKA.WA", "4BB.L", "SBEME.MI", "SYBV.DE", "HWDN.L", "MTLR.MM", "ROADgb.ASE", "LENT.MM", "EEMU.PA",
    "JBEM.PA", "COP1n.H", "TOYA.WA", "XDW0.DE", "SADA.DE", "CHI.L", "XDWT.DE", "XDWM.DE", "ARHP.WA", "CATb.ST",
    "YAR.OL", "WJG.L", "ITTL.CY", "MCP.LS", "BSLN.S", "XDEP.DE", "ARIS.MI", "AFLT.MM", "BASC.L", "WHEA.AS",
    "WMAT.AS", "PRAM.DE", "ELI.BR", "INFL.PA", "QTCOM.HE", "BGN.MI", "AFAGR.HE", "MOTR.L", "ICEP.WA", "SCTS.L",
    "INDXA.MC", "BONAVb.ST", "SIMA.S", "EMSR.PA", "HWG.L", "KBCO.ST", "ALTNA.L", "BFIT.AS", "BGTIB.BB", "AZN.L",
    "1AT.WA", "B2I.OL", "GGRG.L", "SECARE.ST", "GOMX.ST", "SAA.L", "NCCG.L", "HLN.L", "FACC.VI", "ANII.L",
    "AMS.S", "RWAY.MI", "LBG.L", "SREI.L", "ANIM.MI", "ENAV.MI", "PCE.WA", "ZAPP.WA", "SOLARb.CO", "SYNACT.ST",
    "DINN.BEL", "YISC.MC", "FRAN.L", "IJPE.L", "LP2.D", "NRRT.L", "EVLI.HE", "MSZ.WA", "DIAP.WA", "LQQ.PA",
    "DIAL.L", "GHV2.L", "NVTK.MM", "HAGA.IC", "MTL.L", "KPN.AS", "PEAN.S", "RMV.L", "GMET.L", "VMEn.DE",
    "FLERIE.ST", "VOT.WA", "CVC.AS", "SILG.L", "ANORA.HE", "AVTX.AS", "LWBP.WA", "BICO.ST", "US37.PA", "WKP.L",
    "JPES.SJ", "ICESEA.IC", "HOSP.S", "FSECURE.HE", "DEHr.AT", "MLFIR.EUA", "LUXP.LU", "LGD1L.VL", "ACLN.S", "GEV.PR",
    "AURR.L", "IUMO.L", "ISIUSF.L", "B5A0n.H", "EXOR.AS", "IEG.MI", "BVI.PA", "BYGGP.ST", "FTSE.PA", "IWG.L",
    "SIVEH.ST", "INGP.WA", "VRGP.WA", "SOBIV.ST", "SFOR.L", "BAMI.MI", "HEIMpref.ST", "GKI.WA", "RNFT.MM", "ROTEL.BX",
    "ECNL.MI", "MINT.L", "RGSS.MM", "BUFAB.ST", "LXSG.DE", "ICOP.MI", "IGD.MI", "2B7D.DE", "IXXG.DE", "MAV4.L",
    "EPH.S", "IBT.L", "BFT.WA", "AMAV.VI", "AHYI.DE", "CGX.DE", "BLPU.L", "VERVE.ST", "FLOW.AS", "EUE.L",
    "CRBP2.PA", "SYBG.DE", "MARR.MI", "SD3EEX.DE", "ALCAT.PA", "GOFORE.HE", "SPA.L", "DCR.WA", "ENGCONb.ST", "MMTX.S",
    "DB1Gn.DE", "SMLP.L", "ALLIGOb.ST", "INDT.ST", "CMVX.BX", "RESIR.L", "YSIL.MC", "EQU.WA", "GPEP.PA", "NOBA.ST",
    "BLOP.WA", "GNSP.WA", "ISFLOT.L", "SUPR.L", "ENEQ.PR", "HNFG.MM", "EBROM.MC", "BIRG.I", "RM.L", "SRC.L",
    "WYWYN.L", "VBSN.S", "SMICHA.S", "RXRGEX.DE", "SX4PEX.DE", "SXOPEX.DE", "LANDI.S", "SXQPEX.DE", "YRSB.MM", "MISB.MM",
    "EUV.WA", "OBXEDNBN.OL", "SNTP.WA", "SOMA.OL", "SLGOMXH25.HE", "KFASTb.ST", "AEMr.AT", "SOFL.MM", "PLX.PA", "TOMA.PR",
    "ERAGn.DE", "RES.MI", "UD.MI", "GRANG.ST", "GEMC.MM", "PARP.PA", "IE0UZZ5SU.I", "ONDP.WA", "GLV.I", "HMUS.L",
    "TCSG.MM", "EVOK.L", "DMYDb.ST", "ETL.PA", "XSPRAY.ST", "PHNU.MI", "WINC.AS", "IPN.PA", "STOGR.CO", "ALERS.PA",
    "NKR.OL", "1TMR001E.BV", "VIH1n.DE", "29GI.DE", "LFFEP.L", "HFGG.DE", "BOOZT.ST", "ARJOb.ST", "ROSE.L", "EUEE.AS",
    "74SW.PA", "GTRK.MM", "RDF.MI", "CBRAIN.CO", "PMSB.MM", "KODT.ZA", "BHGF.ST", "4X0.DE", "VIK1V.HE", "SBER.MM",
    "LYXIB.MC", "ATER.BB", "YCSH.DE", "BCHN.S", "BREF.BB", "AW1R.AS", "YOUNI.AS", "SPMCHA.S", "BNRGn.DE", "BKWB.S",
    "CULKe.LU", "HEIMAR.IC", "CLOD.S", "ROWINE.BX", "PRN.L", "UEM.L", "ROEM.BX", "LIK.DE", "SRPG.PA", "OLGERD.IC",
    "VRLA.PA", "ROCKW.BR", "ADYEN.AS", "MHPCq.L", "GAMQ.MC", "AUGM.L", "500.PA", "BOV.MT", "OVZON.ST", "ALLFG.AS",
    "CGEO.L", "SGN.WA", "AHYE.PA", "CU2.PA", "ORNBV.HE", "CC1.PA", "CSG.AS", "AUSS.OL", "ELO.OL", "ROCKb.CO",
    "GCLq.L", "BBVAI.MC", "OGKB.MM", "CE8.PA", "TPHC.PA", "KAZT.MM", "CPINV.BR", "AFRN.PA", "KTY.WA", "SALR.LJ",
    "CE2G.H", "BGSFI.BB", "ABEP.WA", "5MVL.DE", "DNLM.L", "SOME.MI", "T2GG.F", "GAZA.MM", "TGKN.MM", "BGEO.L",
    "DJCOMEX.DE", "TATN.MM", "AKVA.OL", "BBN.S", "LIB.MC", "FLUX.BR", "UBIP.PA", "ZBB.ZA", "PV.DE", "RWA.L",
    "COTNE.S", "EUA.L", "PEO.WA", "JD.L", "PCT.L", "ITPG.MI", "SOAG.OL", "BDIS.HT", "KRKG.LJ",
]

LOG = []


def pull(rics, fields, params, chunk_size, tag):
    """Chunked pull. Retries, then halves a failing chunk so one bad firm costs nothing."""
    def one(chunk):
        for _ in range(3):
            try:
                frame = ld.get_data(universe=chunk, fields=fields, parameters=params)
                if frame is not None and len(frame) > 0:
                    return frame
            except Exception:
                pass
        if len(chunk) == 1:
            LOG.append({"stage": tag, "item": chunk[0], "status": "failed"})
            return None
        half = len(chunk) // 2
        parts = [p for p in (one(chunk[:half]), one(chunk[half:])) if p is not None]
        return pd.concat(parts, ignore_index=True) if parts else None

    frames = []
    total = (len(rics) + chunk_size - 1) // chunk_size
    for n, start in enumerate(range(0, len(rics), chunk_size), start=1):
        print("  %s chunk %d/%d" % (tag, n, total))
        got = one(rics[start:start + chunk_size])
        if got is not None:
            frames.append(got)
        if n % 5 == 0 and frames:
            pd.concat(frames, ignore_index=True).to_csv("lseg_%s.partial.csv" % tag, index=False)
            print("    saved so far")
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    if "Instrument" in out.columns:
        out = out.rename(columns={"Instrument": "RIC"})
    return out


def summarise(frame, name):
    print("\n%s   rows=%d   cols=%d" % (name, len(frame), len(frame.columns)))
    for col in frame.columns:
        filled = frame[col].astype(str).str.strip().ne("").sum()
        print("  %-44s %6d/%d" % (col[:44], filled, len(frame)))
        LOG.append({"stage": name, "item": col, "status": "%d/%d" % (filled, len(frame))})


print("Opening session ...")
session = ld.open_session()
if session is None or not str(session.open_state).endswith("Opened"):
    raise SystemExit("Session did not open.")
print("Session open.\n")

print("PULL 1 of 2: static firm facts (%d fields)" % len(STATIC_FIELDS))
static = pull(RICS, STATIC_FIELDS, STATIC_PARAMS, CHUNK_STATIC, "static")
if static is not None:
    static = static.drop_duplicates(subset="RIC")
    static.to_csv("lseg_static.csv", index=False)
    summarise(static, "lseg_static.csv")

panel_fields = list(PANEL_FIELDS) + list(USER_EXTRA_FIELDS)
print("\nPULL 2 of 2: %d years of panel data (%d fields). This is the long one."
      % (YEARS, len(panel_fields)))
panel = pull(RICS, panel_fields, PANEL_PARAMS, CHUNK_PANEL, "panel")
if panel is not None:
    panel.to_csv("lseg_panel.csv", index=False)
    summarise(panel, "lseg_panel.csv")
    if "RIC" in panel.columns:
        per_firm = len(panel) / float(panel["RIC"].nunique())
        print("\n  %.1f rows per firm (expect about %d)" % (per_firm, YEARS))

pd.DataFrame(LOG).to_csv("lseg_pull_log.csv", index=False)
ld.close_session()
print("\nDONE. Download lseg_static.csv, lseg_panel.csv, lseg_pull_log.csv")
