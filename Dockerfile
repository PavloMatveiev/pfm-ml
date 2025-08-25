FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends tini curl && \
    rm -rf /var/lib/apt/lists/*
RUN useradd -m appuser

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# COPY everything you need from train.py: settings, utils, ml, train.py itself, and the API code
COPY settings.py ./settings.py
COPY utils/ ./utils/
COPY ml/ ./ml/
COPY train.py ./train.py
COPY app/ ./app/

# Train the model during assembly (creates /app/model.pkl)
RUN python train.py

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model.pkl \
    PORT=8000
EXPOSE 8000

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
