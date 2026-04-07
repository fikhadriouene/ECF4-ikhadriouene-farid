FROM python:3.11-slim

WORKDIR /app

# Dépendances système utiles
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie du fichier des dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter uvicorn \
    python -m spacy download en_core_web_sm 

# Copie du projet
COPY . .

EXPOSE 8000
EXPOSE 8888
