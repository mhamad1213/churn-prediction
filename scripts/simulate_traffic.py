"""Send real customers from the dataset to the API, as if a CRM were calling it.

    python scripts/simulate_traffic.py                       # your local API
    python scripts/simulate_traffic.py --url https://churn-prediction-hbcb.onrender.com --n 100
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.data import load_data, split_xy  # noqa: E402


def post(url: str, payload: dict) -> dict:
    req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=90) as resp:  # 90s: a sleeping Render app takes a while to wake
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200, help="how many customers to send")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, _ = split_xy(load_data())
    customers = X.sample(args.n, random_state=args.seed).to_dict(orient="records")

    start = time.time()
    contacted = 0
    for i, customer in enumerate(customers, 1):
        contacted += post(f"{args.url}/predict", customer)["contact_customer"]
        if i % 25 == 0 or i == len(customers):
            print(f"  sent {i}/{len(customers)}")
    print(f"\nDone in {time.time() - start:.0f}s. {contacted} of {len(customers)} flagged for contact.")

    with urlopen(f"{args.url}/monitoring/summary", timeout=90) as resp:
        print("Monitoring summary:", json.loads(resp.read()))


if __name__ == "__main__":
    main()
