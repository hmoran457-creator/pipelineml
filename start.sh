#!/bin/bash
set -e

CRON_FILE="/app/src/tasks/crontab.$CRON_FILE_NAME"
echo " ⏰  Loading crontab file: $CRON_FILE"
crontab "$CRON_FILE"

echo " 📏 Rules:"
cat "$CRON_FILE"

# Entrenamiento inicial: garantiza que exista un modelo sin esperar al cron.
echo " 🏋️  Entrenamiento inicial..."
/usr/local/bin/python3 /app/app.py -app TrainModel -r "startup" \
  >> /app/logs/output.txt 2>> /app/logs/errors.txt \
  || echo " ⚠️  El entrenamiento inicial fallo; el cron reintentara a la 01:00"

echo " 🎸 Starting cron..."
cron -f
