# Data-Visuzaltion-Analysis-Mussab

EDHEC MSc Marketing Analytics – Data Visualisation Project (SNCF open data).
**Case 3 – The traveller satisfaction paradox.**

## Layout
- `data/raw/` – project brief (PDF) and source workbook
- `scripts/build_case3_dataset.py` – joins `Regularite_TGV` (real punctuality) with `NPS_Satisfaction` (**simulated**) on `(departure_station, arrival_station)`
- `data/processed/`
  - `case3_route_summary.csv` – 130 rows, one per directional TGV route: NPS + complaints + aggregated punctuality KPIs (full period and last 12 months) + paradox metrics
  - `case3_monthly_joined.csv` – 12,181 monthly rows with NPS attributes attached (for trend charts)
  - `case3_nps_punctuality.xlsx` – both tables plus a data dictionary and method notes

Rebuild: `python scripts/build_case3_dataset.py`

> NPS, review counts, trip purpose and complaint reasons are SIMULATED for teaching purposes.
