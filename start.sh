#!/bin/bash
set -e

# Sin CRON_FILE_NAME la ruta quedaria en "crontab." y `crontab` fallaria: con
# `set -e` eso mata el contenedor antes de entrenar y, bajo restart:
# unless-stopped, produce un bucle de reinicio. Se falla con un mensaje claro.
if [ -z "$CRON_FILE_NAME" ]; then
  echo " ❌ CRON_FILE_NAME no esta definido (esperado: TrainModel)." >&2
  echo "    Ejecute este contenedor vía docker compose, o pase -e CRON_FILE_NAME=TrainModel." >&2
  exit 1
fi

CRON_FILE="/app/src/tasks/crontab.$CRON_FILE_NAME"
if [ ! -f "$CRON_FILE" ]; then
  echo " ❌ No existe el archivo de crontab: $CRON_FILE" >&2
  exit 1
fi

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
