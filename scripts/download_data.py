"""Download the IBM Telco Customer Churn dataset into data/raw/."""

from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already downloaded: {DEST}")
        return
    urlretrieve(URL, DEST)
    print(f"Saved {DEST}")


if __name__ == "__main__":
    main()
