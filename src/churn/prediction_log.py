"""Track B2: save every prediction the API makes, so the model can be monitored.

Where the log goes is set by PREDICTION_DB_URL:
- not set: a local SQLite file, predictions.db, in the project folder
- set:     any database SQLAlchemy understands, e.g. a Neon Postgres URL

See what's been logged:  python -m churn.prediction_log
"""

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    case,
    create_engine,
    func,
    insert,
    select,
)

from churn.config import CATEGORICAL_COLS, NUMERIC_COLS, ROOT

FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
logger = logging.getLogger(__name__)

metadata = MetaData()
predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("model_source", String(200), nullable=False),
    *[Column(c, Float, nullable=False) for c in NUMERIC_COLS],
    *[Column(c, String(50), nullable=False) for c in CATEGORICAL_COLS],
    Column("churn_probability", Float, nullable=False),
    Column("contact_customer", Boolean, nullable=False),
)


def database_url() -> str:
    url = os.environ.get("PREDICTION_DB_URL", f"sqlite:///{ROOT / 'predictions.db'}")
    # Neon gives postgresql://...; tell SQLAlchemy to use the psycopg 3 driver
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


@lru_cache
def _engine_for(url: str) -> Engine:
    engine = create_engine(url, pool_pre_ping=True)  # pre_ping: Neon closes idle connections
    metadata.create_all(engine)  # makes the table the first time only
    return engine


def get_engine() -> Engine:
    return _engine_for(database_url())


def log_prediction(customer: dict, probability: float, contact: bool, model_source: str) -> None:
    """Save one prediction. Runs after the response is sent, and never breaks the API."""
    row = {
        "created_at": datetime.now(timezone.utc),
        "model_source": model_source,
        **{c: customer[c] for c in FEATURES},
        "churn_probability": probability,
        "contact_customer": contact,
    }
    try:
        with get_engine().begin() as conn:
            conn.execute(insert(predictions).values(**row))
    except Exception:
        logger.exception("Could not log a prediction")


def load_predictions() -> pd.DataFrame:
    """Every logged prediction, oldest first."""
    return pd.read_sql(select(predictions).order_by(predictions.c.id), get_engine())


def summary() -> dict:
    """A few headline numbers about recent traffic, worked out inside the database."""
    query = select(
        func.count().label("predictions"),
        func.min(predictions.c.created_at).label("first"),
        func.max(predictions.c.created_at).label("last"),
        func.avg(predictions.c.churn_probability).label("avg_probability"),
        func.avg(case((predictions.c.contact_customer, 1.0), else_=0.0)).label("contact_rate"),
    )
    with get_engine().connect() as conn:
        row = conn.execute(query).mappings().one()
    return {
        "predictions": row["predictions"],
        "first": row["first"].isoformat() if row["first"] else None,
        "last": row["last"].isoformat() if row["last"] else None,
        "avg_probability": round(row["avg_probability"], 3) if row["predictions"] else None,
        "contact_rate": round(row["contact_rate"], 3) if row["predictions"] else None,
    }


def main() -> None:
    print(f"Database: {get_engine().url.render_as_string(hide_password=True)}\n")
    print("Summary:", summary(), "\n")
    df = load_predictions()
    cols = ["id", "created_at", "tenure", "Contract", "MonthlyCharges", "churn_probability",
            "contact_customer"]
    print("Latest 5 predictions:")
    print(df[cols].tail().to_string(index=False) if len(df) else "(none yet)")


if __name__ == "__main__":
    main()
