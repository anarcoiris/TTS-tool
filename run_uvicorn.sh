#!/usr/bin/env bash
# run_uvicorn.sh - arranca la API FastAPI con uvicorn
set -euo pipefail

# Opcionales: sobreescribe por entorno
: "${TTS_MODEL_NAME:=tts_models/multilingual/multi-dataset/xtts_v2}"
: "${VOICES_DIR:=C:/Users/soyko/Documents/git2/tts-tool/wavs}"  # mantiene tu ruta por defecto (útil para Windows/host montado)
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

export TTS_MODEL_NAME
export VOICES_DIR

echo "Iniciando API Coqui TTS"
echo "  TTS_MODEL_NAME=${TTS_MODEL_NAME}"
echo "  VOICES_DIR=${VOICES_DIR}"
echo "  host=${HOST} port=${PORT}"

# Recomiendo workers=1 si usas GPU (torch + forking pueden dar problemas).
# Usa --reload solo en desarrollo (no recomendado en producción con GPU).
uvicorn tts_es:app --host "${HOST}" --port "${PORT}" --workers 1
