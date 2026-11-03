"""PSX weekly direction backtest.

Question: using only information available at the end of a trading week, can a
model predict whether a liquid PSX stock closes higher or lower at the end of
the following week?

Data: daily OHLCV for 105 KSE-100 constituents (2016-10 .. 2026-10-08), scraped
from PSX's Data Portal (dps.psx.com.pk) and published by Muhammad Wasif under the
MIT licence: https://github.com/Muhammad-Wasif/PSX-Stock-Market-Dataset
A snapshot is stored in data/raw/psx/. Prices are NOT adjusted for dividends,
bonus shares or splits (see the corporate-action filter below).

Method
  * One sample per stock per week: features at the week's last trading day,
    label = 1 if next week's last close > this week's last close.
  * "Significant volume" filter, point in time: the stock's median daily traded
    value (close x volume) over the previous 60 sessions must be >= MIN_VALUE_PKR,
    and it must have traded on >= 90% of those sessions.
  * Walk-forward test over the last 52 weeks of data. Models are refit every
    REFIT_WEEKS weeks on all earlier weeks (expanding window), with one week
    held back so no training label overlaps the test period.
  * Trading simulation: each week buy the TOP_K liquid stocks with the highest
    predicted probability of going up (only if prob > 0.5), equal weight, hold one
    week, pay COST_PER_SIDE on turnover. Long-only, as shorting on PSX is limited.

Run: python scripts/psx_weekly_direction_backtest.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "psx" / "PSX_KSE100_Full_Historical_Daily.csv.gz"
OUT = ROOT / "data" / "processed" / "psx"
FIG = ROOT / "reports" / "psx"

MIN_VALUE_PKR = 50_000_000  # median daily traded value over 60 sessions
TEST_WEEKS = 52  # "last year"
REFIT_WEEKS = 4
TRAIN_START = "2018-01-01"
TOP_K = 10
COST_PER_SIDE = 0.0015  # brokerage + taxes, approx. 0.15% per side
MAX_DAILY_MOVE = 0.11  # PSX circuit breaker is ~10%; bigger = corporate action
SEED = 42


# ---------------------------------------------------------------- features
def load_daily() -> pd.DataFrame:
    d = pd.read_csv(RAW, parse_dates=["date"])
    d = d[~d["is_anomaly"]].drop(columns="is_anomaly")
    d = d[(d["close"] > 0) & (d["high"] > 0) & (d["low"] > 0)]
    return d.sort_values(["symbol", "date"]).reset_index(drop=True)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def stock_features(g: pd.DataFrame) -> pd.DataFrame:
    c, v = g["close"], g["volume"]
    r = c.pct_change()
    value = c * v
    f = pd.DataFrame(index=g.index)
    f["date"], f["symbol"], f["close"] = g["date"], g["symbol"], c
    for n in (1, 5, 10, 20, 60, 120):
        f[f"ret_{n}d"] = c.pct_change(n)
    f["vol_20d"] = r.rolling(20).std()
    f["vol_60d"] = r.rolling(60).std()
    f["vol_ratio"] = f["vol_20d"] / f["vol_60d"]
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["rsi_14"] = rsi(c)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    f["range_5d"] = ((g["high"] - g["low"]) / c).rolling(5).mean()
    f["close_loc_5d"] = (c - g["low"].rolling(5).min()) / (
        g["high"].rolling(5).max() - g["low"].rolling(5).min()
    )
    f["hi_52w_dist"] = c / c.rolling(250, min_periods=120).max() - 1
    f["volu_5_60"] = v.rolling(5).mean() / v.rolling(60).mean()
    f["volu_1_20"] = v / v.rolling(20).mean()
    f["log_value_60d"] = np.log1p(value.rolling(60).median())
    f["med_value_60d"] = value.rolling(60).median()
    # share of the last 60 calendar-free sessions the stock actually traded
    f["trade_days_60"] = (v > 0).rolling(60).mean()
    f["big_move"] = r.abs() > MAX_DAILY_MOVE  # corporate-action marker
    return f


def build_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    feats = pd.concat(
        [stock_features(g) for _, g in daily.groupby("symbol", sort=False)]
    )
    feats["week"] = feats["date"].dt.to_period("W-FRI")
    # corporate action anywhere in the last 6 days -> features unreliable
    feats["ca_recent"] = feats.groupby("symbol")["big_move"].transform(
        lambda s: s.rolling(6, min_periods=1).max()
    ).astype(bool)

    wk = feats.groupby(["symbol", "week"]).tail(1).copy()  # last day of week
    wk = wk.sort_values(["symbol", "week"])
    nxt = wk.groupby("symbol")
    wk["next_close"] = nxt["close"].shift(-1)
    wk["next_week"] = nxt["week"].shift(-1)
    wk["fwd_ret"] = wk["next_close"] / wk["close"] - 1
    # corporate action during the holding week -> label unreliable
    big = feats.groupby(["symbol", "week"])["big_move"].max().rename("ca_next")
    wk = wk.join(
        big.groupby(level=0).shift(-1), on=["symbol", "week"]
    )
    consecutive = (wk["next_week"] - wk["week"]).apply(
        lambda x: getattr(x, "n", np.nan)
    ) == 1
    wk.loc[~consecutive, ["fwd_ret", "next_close"]] = np.nan

    # market context: equal-weight average of all stocks that week
    mkt = wk.groupby("week")[["ret_5d", "ret_20d", "ret_60d"]].mean()
    mkt.columns = ["mkt_ret_5d", "mkt_ret_20d", "mkt_ret_60d"]
    wk = wk.join(mkt, on="week")
    wk["breadth_sma50"] = wk.groupby("week")["dist_sma50"].transform(
        lambda s: (s > 0).mean()
    )
    for col in ("ret_5d", "ret_20d", "ret_60d"):
        wk[f"rel_{col}"] = wk[col] - wk[f"mkt_{col}"]
        wk[f"rank_{col}"] = wk.groupby("week")[col].rank(pct=True)
    wk["rank_volu_5_60"] = wk.groupby("week")["volu_5_60"].rank(pct=True)

    wk["liquid"] = (wk["med_value_60d"] >= MIN_VALUE_PKR) & (
        wk["trade_days_60"] >= 0.9
    )
    wk["label"] = (wk["fwd_ret"] > 0).astype(float)
    wk.loc[wk["fwd_ret"].isna(), "label"] = np.nan
    wk["clean"] = ~wk["ca_recent"] & ~wk["ca_next"].fillna(False).astype(bool)
    wk["signal_date"] = wk["date"]
    return wk.reset_index(drop=True)


FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    "vol_20d", "vol_60d", "vol_ratio", "dist_sma20", "dist_sma50", "dist_sma200",
    "rsi_14", "macd_hist", "range_5d", "close_loc_5d", "hi_52w_dist",
    "volu_5_60", "volu_1_20", "log_value_60d",
    "mkt_ret_5d", "mkt_ret_20d", "mkt_ret_60d", "breadth_sma50",
    "rel_ret_5d", "rel_ret_20d", "rel_ret_60d",
    "rank_ret_5d", "rank_ret_20d", "rank_ret_60d", "rank_volu_5_60",
]


# ------------------------------------------------------------------ models
def make_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)
        ),
        "KNN (k=50)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=50)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            n_jobs=-1, random_state=SEED,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            min_samples_leaf=100, random_state=SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            random_state=SEED, n_jobs=-1, eval_metric="logloss",
        ),
        "Neural Net (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2,
                          max_iter=300, early_stopping=True, random_state=SEED),
        ),
    }


def walk_forward(wk: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    data = wk[wk["liquid"] & wk["clean"] & (wk["signal_date"] >= TRAIN_START)]
    data = data.dropna(subset=FEATURES + ["label"])
    weeks = sorted(data["week"].unique())
    test_weeks = weeks[-TEST_WEEKS:]
    preds = []
    for i in range(0, len(test_weeks), REFIT_WEEKS):
        block = test_weeks[i : i + REFIT_WEEKS]
        # purge: the week before the block has its label inside the block
        train = data[data["week"] < block[0] - 1]
        test = data[data["week"].isin(block)].copy()
        Xtr, ytr = train[FEATURES].values, train["label"].values
        for name, model in make_models().items():
            model.fit(Xtr, ytr)
            test[name] = model.predict_proba(test[FEATURES].values)[:, 1]
        preds.append(test)
        print(f"  refit {i // REFIT_WEEKS + 1}: train={len(train):,} "
              f"test={len(test):,} weeks {block[0]}..{block[-1]}")
    out = pd.concat(preds)
    model_names = list(make_models())
    out["Ensemble (avg)"] = out[model_names].mean(axis=1)
    # rule-based baselines expressed as pseudo-probabilities
    out["Baseline: always up"] = 1.0
    out["Baseline: 4-wk momentum"] = (out["ret_20d"] > 0).astype(float) * 0.2 + 0.4
    out["Baseline: 1-wk reversal"] = (out["ret_5d"] < 0).astype(float) * 0.2 + 0.4
    return out, model_names + ["Ensemble (avg)"]


# ----------------------------------------------------------------- scoring
def score(pred: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    y = pred["label"].values
    for c in cols:
        p = pred[c].values
        hit = (p > 0.5) == (y == 1)
        weekly = pd.Series(hit, index=pred["week"]).groupby(level=0).mean()
        # stock-picking skill: AUC computed inside each week, so it ignores
        # whether the whole market went up or down that week
        xs = [roc_auc_score(g["label"], g[c]) for _, g in pred.groupby("week")
              if g["label"].nunique() == 2 and g[c].nunique() > 1]
        xs_p = (stats.ttest_1samp(np.array(xs) - 0.5, 0, alternative="greater").pvalue
                if len(xs) > 2 else np.nan)
        # t-test across weeks: is weekly accuracy above 50%?
        t50 = stats.ttest_1samp(weekly - 0.5, 0, alternative="greater")
        try:
            auc = roc_auc_score(y, p)
        except ValueError:
            auc = np.nan
        rows.append({
            "model": c,
            "accuracy": accuracy_score(y, p > 0.5),
            "balanced_accuracy": balanced_accuracy_score(y, p > 0.5),
            "auc": auc,
            "pct_predicted_up": (p > 0.5).mean(),
            "precision_up": y[p > 0.5].mean() if (p > 0.5).any() else np.nan,
            "weekly_acc_mean": weekly.mean(),
            "p_value_vs_50pct": t50.pvalue,
            "within_week_auc": np.mean(xs) if xs else np.nan,
            "p_value_within_week_auc": xs_p,
        })
    return pd.DataFrame(rows)


def simulate(pred: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves, rows = {}, []
    bench = pred.groupby("week")["fwd_ret"].mean()
    curves["Benchmark: equal-weight liquid universe"] = bench
    for c in cols:
        rets, prev = [], set()
        for wkk, g in pred.groupby("week"):
            pick = g[g[c] > 0.5].nlargest(TOP_K, c)
            hold = set(pick["symbol"])
            n_prev = max(len(prev), 1)
            if hold:
                gross = pick["fwd_ret"].mean()
                # names sold + names bought, as a fraction of the book;
                # 1.0 = one full side traded, 2.0 = full rotation
                turnover = len(hold ^ prev) / max(len(hold), n_prev)
            else:
                gross, turnover = 0.0, (1.0 if prev else 0.0)
            rets.append((wkk, gross - turnover * COST_PER_SIDE))
            prev = hold
        curves[c] = pd.Series(dict(rets))
    for name, s in curves.items():
        s = s.sort_index()
        eq = (1 + s).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        rows.append({
            "strategy": name,
            "total_return": eq.iloc[-1] - 1,
            "ann_vol": s.std() * np.sqrt(52),
            "sharpe_0rf": s.mean() / s.std() * np.sqrt(52) if s.std() else np.nan,
            "max_drawdown": dd,
            "win_weeks": (s > 0).mean(),
            "weeks_in_cash": int((s == 0).sum()),
            "excess_vs_bench_wk": (s - bench.reindex(s.index)).mean(),
        })
    eqs = pd.DataFrame({k: (1 + v.sort_index()).cumprod() for k, v in curves.items()})
    return pd.DataFrame(rows), eqs


# ------------------------------------------------------------------- plots
def plots(acc: pd.DataFrame, eqs: pd.DataFrame, imp: pd.Series) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    a = acc.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#9aa5b1" if m.startswith("Baseline") else "#2f6db5" for m in a["model"]]
    ax.barh(a["model"], a["accuracy"], color=colors)
    ax.axvline(0.5, color="#c0392b", ls="--", lw=1, label="coin flip")
    ax.set_xlim(0.4, max(0.62, a["accuracy"].max() + 0.02))
    for y_, v in enumerate(a["accuracy"]):
        ax.text(v + 0.002, y_, f"{v:.1%}", va="center", fontsize=8)
    ax.set_title("Next-week direction accuracy, last 52 weeks (walk-forward)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_by_model.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    keep = ["Benchmark: equal-weight liquid universe", "Ensemble (avg)",
            "Logistic Regression", "LightGBM", "Random Forest",
            "Baseline: 4-wk momentum"]
    for k in keep:
        if k in eqs:
            lw = 2.5 if k.startswith(("Benchmark", "Ensemble")) else 1.2
            ax.plot(eqs.index.to_timestamp(), eqs[k], label=k, lw=lw)
    ax.set_title(f"Growth of 1 PKR: top-{TOP_K} long-only weekly picks, after costs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "equity_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    imp.sort_values().tail(15).plot.barh(ax=ax, color="#2f6db5")
    ax.set_title("LightGBM feature importance (gain), final fit")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- main
def main() -> None:
    global OUT, FIG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test-end", help="cut the data at this date (YYYY-MM-DD) "
                    "to backtest an earlier year, e.g. 2025-10-08")
    args = ap.parse_args()
    if args.test_end:
        OUT = OUT / f"test_end_{args.test_end}"
        FIG = FIG / f"test_end_{args.test_end}"
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading daily data ...")
    daily = load_daily()
    if args.test_end:
        daily = daily[daily["date"] <= args.test_end]
    wk = build_weekly(daily)
    print(f"Weekly rows: {len(wk):,}; liquid & clean with label: "
          f"{(wk.liquid & wk.clean & wk.label.notna()).sum():,}")

    print("Walk-forward backtest ...")
    pred, model_cols = walk_forward(wk)
    all_cols = model_cols + [c for c in pred.columns if c.startswith("Baseline")]

    acc = score(pred, all_cols).sort_values("accuracy", ascending=False)
    perf, eqs = simulate(pred, all_cols)
    perf = perf.sort_values("total_return", ascending=False)

    # final fit on everything for next-week predictions + importance
    data = wk[wk.liquid & wk.clean & (wk.signal_date >= TRAIN_START)]
    train = data.dropna(subset=FEATURES + ["label"])
    latest_week = wk["week"].max()
    live = wk[(wk["week"] == latest_week) & wk["liquid"] & ~wk["ca_recent"]]
    live = live.dropna(subset=FEATURES).copy()
    models = make_models()
    for name, m in models.items():
        m.fit(train[FEATURES].values, train["label"].values)
        live[name] = m.predict_proba(live[FEATURES].values)[:, 1]
    live["Ensemble (avg)"] = live[list(models)].mean(axis=1)
    imp = pd.Series(
        models["LightGBM"].booster_.feature_importance("gain"), index=FEATURES
    )
    plots(acc, eqs, imp)

    cols = ["symbol", "signal_date", "close", "med_value_60d", "ret_5d", "ret_20d"]
    live = live[cols + list(models) + ["Ensemble (avg)"]].sort_values(
        "Ensemble (avg)", ascending=False
    )

    pred_cols = ["week", "signal_date", "symbol", "close", "fwd_ret", "label"]
    pred[pred_cols + all_cols].to_csv(OUT / "walkforward_predictions.csv", index=False)
    acc.to_csv(OUT / "model_accuracy.csv", index=False)
    perf.to_csv(OUT / "strategy_performance.csv", index=False)
    eqs.to_csv(OUT / "equity_curves.csv")
    live.to_csv(OUT / "next_week_predictions.csv", index=False)

    test = pred
    summary = {
        "data_last_date": str(daily["date"].max().date()),
        "test_weeks": f"{test['week'].min()} .. {test['week'].max()}",
        "test_samples": int(len(test)),
        "liquid_stocks_per_week_avg": float(test.groupby("week").size().mean()),
        "share_of_up_weeks_in_test": float(test["label"].mean()),
        "min_median_daily_value_pkr": MIN_VALUE_PKR,
        "prediction_for_week_after": str(latest_week),
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(json.dumps(summary, indent=2))
    print("\nAccuracy\n", acc.round(3).to_string(index=False))
    print("\nTrading simulation\n", perf.round(3).to_string(index=False))
    print("\nNext-week (top 15 by ensemble)\n",
          live.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
