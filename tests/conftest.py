import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def throwaway_prediction_log(tmp_path_factory):
    """Send the API's prediction log to a temporary file, never to your real database."""
    os.environ["PREDICTION_DB_URL"] = f"sqlite:///{tmp_path_factory.mktemp('log') / 'predictions.db'}"
