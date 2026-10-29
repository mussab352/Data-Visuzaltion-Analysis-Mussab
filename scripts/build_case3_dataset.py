"""Case 3 - The traveller satisfaction paradox.

Joins Regularite_TGV (real monthly punctuality) with NPS_Satisfaction
(SIMULATED satisfaction) on (departure_station, arrival_station) and writes:

  data/processed/case3_route_summary.csv      one row per directional route (130)
  data/processed/case3_monthly_joined.csv     monthly punctuality + NPS attributes
  data/processed/case3_nps_punctuality.xlsx   both tables + data dictionary + notes

Run from the repo root:  python scripts/build_case3_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/Donnees_SNCF_Projet_Dataviz_EN_columns.xlsx"
OUT = ROOT / "data/processed"
KEY = ["departure_station", "arrival_station"]
LAST_12M_START = "2025-04"  # data ends 2026-03
CAUSES = [
    "pct_cause_external",
    "pct_cause_infrastructure",
    "pct_cause_traffic_management",
    "pct_cause_rolling_stock",
    "pct_cause_station_operations",
    "pct_cause_passenger_handling",
]


def load():
    reg = pd.read_excel(SRC, sheet_name="Regularite_TGV")
    nps = pd.read_excel(SRC, sheet_name="NPS_Satisfaction")
    for df in (reg, nps):
        for c in KEY:
            df[c] = df[c].str.strip()
    return reg, nps


def add_weights(reg):
    """Helper columns so that averages can be re-weighted when summing months."""
    r = reg.copy()
    r["trains_run"] = r["trains_scheduled"] - r["trains_cancelled"]
    r["w_journey_time"] = r["avg_journey_time_min"] * r["trains_run"]
    r["w_arr_delay_all"] = r["avg_arrival_delay_all_trains_min"] * r["trains_run"]
    r["w_dep_delay_all"] = r["avg_departure_delay_all_trains_min"] * r["trains_run"]
    r["w_arr_delay_late"] = r["avg_arrival_delay_late_trains_min"] * r["trains_late_arrival"]
    # Cause shares describe delays >15 min; months with no cause split are excluded.
    has_causes = r[CAUSES].sum(axis=1) > 0
    r["cause_weight"] = np.where(has_causes, r["trains_late_over_15min"], 0)
    for c in CAUSES:
        r["w_" + c] = r[c] * r["cause_weight"]
    r["incident_flag"] = r["arrival_delay_comment"].notna().astype(int)
    return r


SUM_COLS = [
    "trains_scheduled", "trains_cancelled", "trains_run", "trains_late_departure",
    "trains_late_arrival", "trains_late_over_15min", "trains_late_over_30min",
    "trains_late_over_60min", "w_journey_time", "w_arr_delay_all", "w_dep_delay_all",
    "w_arr_delay_late", "cause_weight", "incident_flag",
] + ["w_" + c for c in CAUSES]


def kpis(g, prefix=""):
    """Turn summed counts into rates and weighted averages."""
    run = g["trains_run"].replace(0, np.nan)
    sched = g["trains_scheduled"].replace(0, np.nan)
    late = g["trains_late_arrival"].replace(0, np.nan)
    out = pd.DataFrame(index=g.index)
    out["trains_scheduled"] = g["trains_scheduled"]
    out["trains_cancelled"] = g["trains_cancelled"]
    out["trains_run"] = g["trains_run"]
    out["cancellation_rate_pct"] = 100 * g["trains_cancelled"] / sched
    out["late_arrival_rate_pct"] = 100 * g["trains_late_arrival"] / run
    out["on_time_arrival_rate_pct"] = 100 - out["late_arrival_rate_pct"]
    out["late_over_15min_rate_pct"] = 100 * g["trains_late_over_15min"] / run
    out["late_over_30min_rate_pct"] = 100 * g["trains_late_over_30min"] / run
    out["late_over_60min_rate_pct"] = 100 * g["trains_late_over_60min"] / run
    # Traveller-centric reliability: share of SCHEDULED trains that ran AND arrived on time.
    out["reliability_pct"] = 100 * (g["trains_run"] - g["trains_late_arrival"]) / sched
    out["avg_arrival_delay_all_trains_min"] = g["w_arr_delay_all"] / run
    out["avg_arrival_delay_late_trains_min"] = g["w_arr_delay_late"] / late
    out["avg_departure_delay_all_trains_min"] = g["w_dep_delay_all"] / run
    out["avg_journey_time_min"] = g["w_journey_time"] / run
    out = out.round(2)
    return out.add_prefix(prefix)


def route_summary(r):
    g = r.groupby(KEY)
    sums = g[SUM_COLS].sum()
    full = kpis(sums)

    meta = g.agg(
        reg_service=("service", lambda s: " / ".join(sorted(s.unique()))),
        first_month=("date", "min"),
        last_month=("date", "max"),
        months_of_data=("date", "nunique"),
    )
    full["incident_months"] = sums["incident_flag"]

    cw = sums["cause_weight"].replace(0, np.nan)
    causes = pd.DataFrame({c: (sums["w_" + c] / cw).round(1) for c in CAUSES})
    nice = {c: c.replace("pct_cause_", "") for c in CAUSES}
    causes["dominant_delay_cause"] = causes[CAUSES].idxmax(axis=1).map(nice)

    recent = r[r["date"] >= LAST_12M_START].groupby(KEY)[SUM_COLS].sum()
    last12 = kpis(recent, prefix="l12m_")[[
        "l12m_trains_scheduled", "l12m_cancellation_rate_pct", "l12m_on_time_arrival_rate_pct",
        "l12m_late_over_15min_rate_pct", "l12m_reliability_pct",
        "l12m_avg_arrival_delay_all_trains_min",
    ]]

    out = meta.join(full).join(causes).join(last12, how="left")
    out["active_last_12m"] = out["l12m_trains_scheduled"].fillna(0).gt(0)
    return out.reset_index()


def build():
    reg, nps = load()
    r = add_weights(reg)

    routes = route_summary(r)
    df = nps.merge(routes, on=KEY, how="outer", indicator=True, validate="one_to_one")
    unmatched = (df["_merge"] != "both").sum()
    assert unmatched == 0, f"{unmatched} routes did not match"
    df = df.drop(columns="_merge")

    # Identifiers
    df.insert(0, "route_id", df["departure_station"] + " > " + df["arrival_station"])
    pair = df[KEY].apply(lambda x: " <> ".join(sorted(x)), axis=1)
    df.insert(1, "route_pair", pair)
    df["paris_route"] = df[KEY].apply(lambda x: x.str.startswith("PARIS").any(), axis=1)

    # NPS-side derived fields
    df["nps_band"] = pd.cut(df["nps_score"], [-101, 0, 30, 50, 100],
                            labels=["Negative (<0)", "Low (0-30)", "Good (30-50)", "Excellent (>50)"])
    punct = "Retard / ponctualite"
    df["delay_is_top_complaint"] = df["complaint_reason_1"].eq(punct)
    df["delay_in_top2_complaints"] = df["delay_is_top_complaint"] | df["complaint_reason_2"].eq(punct)
    df["delay_complaint_pct"] = np.select(
        [df["complaint_reason_1"].eq(punct), df["complaint_reason_2"].eq(punct)],
        [df["complaint_reason_1_pct"], df["complaint_reason_2_pct"]], default=0)

    # Paradox metrics: compare rank on measured performance vs rank on perceived satisfaction.
    # Ranks are percentiles (0 = worst, 100 = best) so they sit on the same scale.
    perf = df["l12m_reliability_pct"].fillna(df["reliability_pct"])
    df["perf_metric_used"] = np.where(df["l12m_reliability_pct"].notna(), "last_12m", "full_period")
    df["perf_percentile"] = (perf.rank(pct=True) * 100).round(1)
    df["nps_percentile"] = (df["nps_score"].rank(pct=True) * 100).round(1)
    df["perception_gap"] = (df["nps_percentile"] - df["perf_percentile"]).round(1)
    df["paradox_quadrant"] = np.select(
        [(df["perf_percentile"] >= 50) & (df["nps_percentile"] >= 50),
         (df["perf_percentile"] >= 50) & (df["nps_percentile"] < 50),
         (df["perf_percentile"] < 50) & (df["nps_percentile"] >= 50)],
        ["Punctual & satisfied", "Punctual but dissatisfied", "Late but satisfied"],
        default="Late & dissatisfied")
    df["data_note"] = "NPS columns are SIMULATED; punctuality columns are real SNCF open data"

    # Monthly table for trend charts
    monthly = reg.merge(nps, on=KEY, how="left", suffixes=("", "_nps"), validate="many_to_one")
    monthly.insert(0, "route_id", monthly["departure_station"] + " > " + monthly["arrival_station"])
    monthly["year"] = monthly["date"].str[:4].astype(int)
    monthly["month_date"] = pd.to_datetime(monthly["date"] + "-01")
    return df, monthly


DICTIONARY = [
    ("route_id", "Directional route: departure > arrival", "Key"),
    ("route_pair", "Same route regardless of direction (to combine A>B and B>A)", "Key"),
    ("service / main_trip_purpose / nps_score / review_count / complaint_*", "Columns from NPS_Satisfaction", "SIMULATED"),
    ("reg_service", "Service label(s) in Regularite_TGV for the route", "Real"),
    ("first_month / last_month / months_of_data", "Coverage of punctuality data", "Real"),
    ("trains_scheduled / cancelled / run", "Totals over Jan 2018 - Mar 2026 (run = scheduled - cancelled)", "Real"),
    ("cancellation_rate_pct", "cancelled / scheduled", "Calculated"),
    ("late_arrival_rate_pct / on_time_arrival_rate_pct", "late arrivals / trains run (AQST definition: late = >5 min short trips, >10-15 min long trips)", "Calculated"),
    ("late_over_15/30/60min_rate_pct", "Trains arriving >15/30/60 min late / trains run", "Calculated"),
    ("reliability_pct", "(trains run - late arrivals) / trains scheduled: share of planned trains that ran AND arrived on time", "Calculated"),
    ("avg_*_delay_*_min / avg_journey_time_min", "Monthly averages re-weighted by trains run (or by late trains for 'late_trains' delay)", "Calculated"),
    ("incident_months", "Months with an incident comment on arrival delay", "Calculated"),
    ("pct_cause_*", "Share of >15 min delays by cause, weighted by trains late >15 min", "Calculated"),
    ("dominant_delay_cause", "Cause with highest weighted share", "Calculated"),
    ("l12m_*", "Same KPIs for the last 12 months (Apr 2025 - Mar 2026); blank if route no longer runs", "Calculated"),
    ("active_last_12m", "Route has scheduled trains in the last 12 months", "Calculated"),
    ("paris_route", "Paris station at either end", "Calculated"),
    ("nps_band", "NPS grouped: <0, 0-30, 30-50, >50", "Derived from SIMULATED"),
    ("delay_is_top_complaint / delay_in_top2_complaints / delay_complaint_pct", "Whether 'Retard / ponctualite' is complaint 1 or 2, and its %", "Derived from SIMULATED"),
    ("perf_percentile", "Percentile rank of reliability_pct (last 12m, or full period if route inactive); 100 = best", "Calculated"),
    ("nps_percentile", "Percentile rank of NPS; 100 = best", "Derived from SIMULATED"),
    ("perception_gap", "nps_percentile - perf_percentile. >0: travellers happier than punctuality suggests; <0: less happy", "Derived from SIMULATED"),
    ("paradox_quadrant", "Above/below median on performance x satisfaction", "Derived from SIMULATED"),
]

NOTES = [
    "Sources: Regularite_TGV (real, SNCF open data, Jan 2018 - Mar 2026) joined with NPS_Satisfaction (SIMULATED for teaching).",
    "Join key: (departure_station, arrival_station). 130/130 routes matched exactly; no name mapping needed for this join.",
    "Regularite_TGV has 84 rows where a route appears twice in the same month (split National / International, from Jul 2025). They are summed.",
    "Rates are rebuilt from summed counts, not averaged across months, so busy months weigh more.",
    "9 routes have no data in the last 12 months: 6 Paris Vaugirard routes (2018 only), 2 Madrid routes (to Dec 2022), Tourcoing > Marseille (to Aug 2022). Their l12m_* columns are blank.",
    "225 months have zero trains late >15 min and no cause split; they are excluded from the cause weighting.",
    "Any deliverable using NPS columns must state that they are simulated.",
]


def main():
    df, monthly = build()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "case3_route_summary.csv", index=False)
    monthly.drop(columns="month_date").to_csv(OUT / "case3_monthly_joined.csv", index=False)
    with pd.ExcelWriter(OUT / "case3_nps_punctuality.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Route_Summary", index=False)
        monthly.to_excel(xw, sheet_name="Monthly_Joined", index=False)
        pd.DataFrame(DICTIONARY, columns=["column", "definition", "origin"]).to_excel(xw, sheet_name="Data_Dictionary", index=False)
        pd.DataFrame({"notes": NOTES}).to_excel(xw, sheet_name="Notes", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    60, max(10, max(len(str(c.value or "")) for c in col[:200]) + 2))
    print(df.shape, monthly.shape)


if __name__ == "__main__":
    main()
