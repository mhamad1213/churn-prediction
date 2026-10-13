# Milestone 8: package the churn API so it runs the same anywhere.

# 1. Start from a small official Python image
FROM python:3.12-slim

# 2. Sensible Python settings for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# 3. Install dependencies first, so Docker can cache this slow step
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# 4. Copy the project code
COPY src/ src/
COPY scripts/ scripts/
COPY models/threshold.json models/
COPY reports/baseline_metrics.json reports/

# 5. Download the data and train the model inside the image
RUN python scripts/download_data.py \
    && python -m churn.tune \
    && rm -rf data

# 6. Run as a normal user, not root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# 7. Start the API. Render tells us which port to use via $PORT.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn churn.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
