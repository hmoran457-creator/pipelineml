FROM python:3.11-slim

WORKDIR /app

# cron ejecuta el reentrenamiento diario del modelo.
RUN apt-get update && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/assets && chmod +x /app/start.sh

CMD ["sh", "./start.sh"]
