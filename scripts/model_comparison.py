"""Compare linear models with tree-based models (XGBoost, LightGBM, CatBoost, Random Forest)
for predicting route NPS (SIMULATED) from punctuality.

Validation: 5-fold GroupKFold on route_pair (A>B and B>A always in the same fold), repeated
with 10 shuffles. Scores are out-of-fold R2 and RMSE.

Run from the repo root:  python scripts/model_comparison.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"
SEED_REPEATS = 10

d = pd.read_csv(OUT / "case3_route_summary.csv")
y = d["nps_score"].to_numpy(float)
groups = d["route_pair"].to_numpy()

CORE = ["cancellation_rate_pct", "late_over_15min_rate_pct", "avg_arrival_delay_late_trains_min"]
PERF = CORE + [
    "reliability_pct", "late_arrival_rate_pct", "late_over_30min_rate_pct", "late_over_60min_rate_pct",
    "avg_arrival_delay_all_trains_min", "avg_departure_delay_all_trains_min", "avg_journey_time_min",
    "incident_months", "pct_cause_external", "pct_cause_infrastructure", "pct_cause_traffic_management",
    "pct_cause_rolling_stock", "pct_cause_station_operations", "pct_cause_passenger_handling",
]
X_core = d[CORE].astype(float)
X_perf = pd.concat([d[PERF].astype(float),
                    d["service"].eq("International").astype(float).rename("international"),
                    d["paris_route"].astype(float)], axis=1)
# Adds the (simulated) trip purpose and top complaint reason as one-hot columns.
X_ctx = pd.concat([X_perf, pd.get_dummies(d[["main_trip_purpose", "complaint_reason_1"]], dtype=float)], axis=1)


def tree_models():
    # Small, heavily regularised trees: 130 rows cannot support deep ones.
    return {
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, reg_lambda=5, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=300, num_leaves=4, max_depth=2, learning_rate=0.03,
                                  min_child_samples=10, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=5, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=400, depth=3, learning_rate=0.03, l2_leaf_reg=5, verbose=0),
        "Random forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5, n_jobs=-1),
    }


def oof_scores(model_fn, X, repeats=SEED_REPEATS):
    r2s, rmses = [], []
    pairs = np.unique(groups)
    for rep in range(repeats):
        rng = np.random.default_rng(rep)
        # Shuffle which pairs land in which fold by relabelling groups.
        relabel = dict(zip(pairs, rng.permutation(len(pairs))))
        g = np.array([relabel[p] for p in groups])
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            m = model_fn(rep)
            m.fit(X.iloc[tr], y[tr])
            pred[te] = m.predict(X.iloc[te])
        r2s.append(r2_score(y, pred))
        rmses.append(np.sqrt(mean_squared_error(y, pred)))
    return np.mean(r2s), np.std(r2s), np.mean(rmses)


def with_seed(model, rep):
    for k in ("random_state", "random_seed"):
        if k in model.get_params():
            model.set_params(**{k: rep})
    return model


def main():
    rows = []
    feature_sets = {"Core 3 punctuality": X_core, "All performance (21)": X_perf,
                    "Performance + trip purpose + complaints": X_ctx}
    for fs_name, X in feature_sets.items():
        lin = (lambda rep: LinearRegression()) if X.shape[1] <= 3 else \
              (lambda rep: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))))
        lin_name = "Linear (OLS)" if X.shape[1] <= 3 else "Linear (Ridge)"
        candidates = {lin_name: lin}
        for name in tree_models():
            candidates[name] = (lambda rep, name=name: with_seed(tree_models()[name], rep))
        for name, fn in candidates.items():
            r2m, r2s, rmse = oof_scores(fn, X)
            rows.append({"features": fs_name, "n_features": X.shape[1], "model": name,
                         "cv_r2": round(r2m, 3), "cv_r2_sd": round(r2s, 3), "cv_rmse_nps": round(rmse, 2)})
            print(rows[-1])
    # Baseline: reliability only, linear.
    r2m, r2s, rmse = oof_scores(lambda rep: LinearRegression(), d[["reliability_pct"]])
    rows.append({"features": "Reliability only (current simulator)", "n_features": 1, "model": "Linear (OLS)",
                 "cv_r2": round(r2m, 3), "cv_r2_sd": round(r2s, 3), "cv_rmse_nps": round(rmse, 2)})
    res = pd.DataFrame(rows).sort_values("cv_r2", ascending=False)
    res.to_csv(OUT / "case3_model_comparison.csv", index=False)
    print(res.to_string(index=False))

    # Partial dependence of the boosted model on the 3 core features: does the effect flatten?
    m = CatBoostRegressor(iterations=400, depth=3, learning_rate=0.03, l2_leaf_reg=5, verbose=0, random_seed=0)
    m.fit(X_core, y)
    lin = LinearRegression().fit(X_core, y)
    pd_rows = []
    for f in CORE:
        grid = np.linspace(X_core[f].quantile(0.05), X_core[f].quantile(0.95), 15)
        for v in grid:
            Xg = X_core.copy(); Xg[f] = v
            pd_rows.append({"feature": f, "value": round(v, 2),
                            "catboost_avg_nps": round(m.predict(Xg).mean(), 2),
                            "linear_avg_nps": round(lin.predict(Xg).mean(), 2)})
    pdp = pd.DataFrame(pd_rows)
    pdp.to_csv(OUT / "case3_partial_dependence.csv", index=False)
    print(pdp.to_string(index=False))
    imp = pd.Series(m.get_feature_importance(), index=CORE).round(1)
    print("CatBoost importance:\n", imp)


if __name__ == "__main__":
    main()
