FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python (version allégée pour Docker)
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Code source
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY models/ ./models/
COPY .env.example .env

EXPOSE 8000

CMD ["uvicorn", "src.api_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
