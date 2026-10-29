"""Track B3: are the customers we see now different from the ones the model learned from?

Compares the prediction log with the training data, one column at a time, using the
Population Stability Index (PSI), and writes a short report.

    python -m churn.drift                    # every logged prediction
    python -m churn.drift --days 30          # only the last 30 days
    python -m churn.drift --fail-on-drift    # exit with an error on major drift (for GitHub Actions)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import CATEGORICAL_COLS, MODELS_DIR, NUMERIC_COLS, RANDOM_STATE, REPORTS_DIR
from churn.data import load_data, split_xy
from churn.prediction_log import get_engine, load_predictions

WATCH, MAJOR = 0.10, 0.25  # the usual PSI rules of thumb
MIN_ROWS = 200  # with fewer, random noise alone can look like drift (100 rows gave PSI 0.245)
EPS = 1e-4  # stops log(0) when a bin is empty
DRIFT_DIR = REPORTS_DIR / "drift"


def _psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    ref_share, cur_share = np.clip(ref_share, EPS, None), np.clip(cur_share, EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Cut the reference into 10 equal-sized groups, then compare how full each group is now."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, edges)[0] / len(ref)
    cur_share = np.histogram(cur, edges)[0] / len(cur)
    return _psi(ref_share, cur_share)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref) | set(cur))
    ref_share = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_share = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_share, cur_share)


def status(psi: float) -> str:
    return "major drift" if psi >= MAJOR else "watch" if psi >= WATCH else "stable"


def _describe(ref: pd.Series, cur: pd.Series, numeric: bool) -> tuple[str, str]:
    """A human-readable 'then vs now' for the report."""
    if numeric:
        return f"mean {ref.mean():.2f}", f"mean {cur.mean():.2f}"
    ref_share, cur_share = ref.value_counts(normalize=True), cur.value_counts(normalize=True)
    change = cur_share.sub(ref_share, fill_value=0).abs()
    cat = change.idxmax()
    return f"{cat} {ref_share.get(cat, 0):.0%}", f"{cat} {cur_share.get(cat, 0):.0%}"


def drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_COLS + CATEGORICAL_COLS + ["churn_probability"]:
        if col not in reference or col not in current:
            continue
        numeric = col not in CATEGORICAL_COLS
        ref, cur = reference[col], current[col]
        psi = psi_numeric(ref, cur) if numeric else psi_categorical(ref.astype(str), cur.astype(str))
        then, now = _describe(ref, cur, numeric)
        rows.append({"feature": col, "psi": round(psi, 3), "status": status(psi),
                     "training data": then, "recent traffic": now})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def reference_data() -> pd.DataFrame:
    """The training customers, plus what the current model says about them."""
    X, y = split_xy(load_data())
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model_file = MODELS_DIR / "best_model.joblib"
    if model_file.exists():
        X_train = X_train.assign(churn_probability=joblib.load(model_file).predict_proba(X_train)[:, 1])
    return X_train


def recent_predictions(days: int | None) -> pd.DataFrame:
    df = load_predictions()
    if days is not None and len(df):
        since = datetime.now(timezone.utc) - timedelta(days=days)
        df = df[pd.to_datetime(df["created_at"], utc=True) >= since]
    return df


def _markdown_table(df: pd.DataFrame) -> str:
    rows = [list(df.columns), ["---"] * len(df.columns), *df.astype(str).to_numpy().tolist()]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def to_markdown(table: pd.DataFrame | None, n_current: int, n_reference: int, days: int | None) -> str:
    window = f"last {days} days" if days else "all time"
    lines = [f"# Drift report, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", "",
             f"Recent traffic: **{n_current}** logged predictions ({window}). "
             f"Training data: **{n_reference}** customers.", ""]
    if table is None:
        lines.append(f"Not enough traffic to judge drift yet (need at least {MIN_ROWS}).")
        return "\n".join(lines) + "\n"
    worst = table.iloc[0]
    verdict = {"stable": "No drift: recent customers look like the training data.",
               "watch": f"Some movement, worth watching. Biggest: **{worst['feature']}**.",
               "major drift": f"**Major drift** in {', '.join(table.query('psi >= @MAJOR')['feature'])}. "
                              "Check the data feed, then consider retraining."}[worst["status"]]
    lines += [verdict, "", _markdown_table(table), "",
              f"PSI below {WATCH} = stable, {WATCH} to {MAJOR} = watch, {MAJOR}+ = major drift."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None, help="only look at the last N days")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args(argv)

    print(f"Prediction log: {get_engine().url.render_as_string(hide_password=True)}")
    current = recent_predictions(args.days)
    reference = reference_data()
    table = drift_table(reference, current) if len(current) >= MIN_ROWS else None
    report = to_markdown(table, len(current), len(reference), args.days)

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    (DRIFT_DIR / "latest.md").write_text(report)
    (DRIFT_DIR / "latest.json").write_text(json.dumps(
        {"n_current": len(current), "days": args.days,
         "features": [] if table is None else table.to_dict(orient="records")}, indent=2))
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):  # shows on the Actions run page
        with open(summary_file, "a") as f:
            f.write(report)

    print("\n" + report)
    print(f"Saved {DRIFT_DIR.relative_to(REPORTS_DIR.parent)}/latest.md and latest.json")
    major = table is not None and (table["status"] == "major drift").any()
    return 1 if (major and args.fail_on_drift) else 0


if __name__ == "__main__":
    sys.exit(main())
