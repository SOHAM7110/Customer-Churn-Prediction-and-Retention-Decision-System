FROM python:3.11-slim
WORKDIR /app
# libgomp: OpenMP runtime needed by lightgbm / xgboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
COPY config.yaml .
COPY src src
COPY models models
COPY results results
COPY app app
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
