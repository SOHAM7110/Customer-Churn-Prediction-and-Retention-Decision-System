# Deploying the demo (free)

The app needs no raw data and no retraining: it reads `app/data/customers.parquet` (scored test customers with
precomputed explanations), `models/`, `results/` and `config.yaml`. Build them with `make all` (or `make export`).

Commit these so the deployment has them: `models/`, `app/data/`, `results/` (png + csv), `config.yaml`, `src/`, `app/`.
Raw and interim data stay out of git.

## Option A: Streamlit Community Cloud
1. Push the repo to GitHub.
2. share.streamlit.io, New app, pick the repo, main file `app/streamlit_app.py`.
3. Dependencies come from `app/requirements.txt`. If the build does not pick it up, copy it to a root-level
   `requirements.txt` in the deploy branch (the heavy training libraries in the root file are not needed by the app).

## Option B: Hugging Face Spaces (Docker)
1. Create a Space with SDK = Docker.
2. In the Space repo, put this at the very top of `README.md`:
   ```
   ---
   title: Churn retention decisions
   sdk: docker
   app_port: 8501
   ---
   ```
3. Push the repo (the included `Dockerfile` runs Streamlit on 8501).

## Local Docker
```bash
docker build -t churn-retention .
docker run -p 8501:8501 churn-retention
```

## Optional scoring API
```bash
uvicorn app.api:app --port 8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"features": {"recency_days": 80, "frequency": 3, "monetary": 450, "orders_90": 1}}'
```
Missing features are treated as NaN (the models handle them).
