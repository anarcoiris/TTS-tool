#!/usr/bin/env python3
# tts_api_es.py - Versión en español y adaptada a ruta Windows para entradas
import os
import json
import uuid
from typing import List, Annotated

from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
from TTS.api import TTS
from pathlib import Path

# Dispositivo: cuda si está disponible, sino cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

voices = None

# Ruta adaptada a Windows: carpeta donde buscar/guardar WAVs y voces
voices_dir = Path("C:/Users/soyko/Documents/git2/tts-tool/wavs")

# Asegurar existencia del directorio (crea padres si es necesario)
voices_dir.mkdir(parents=True, exist_ok=True)

voices_config = voices_dir / "voices.json"
if not voices_config.exists():
    with open(voices_config, "w", encoding="utf-8") as f:
        f.write("{}")
with open(voices_config, encoding="utf-8") as f:
    voices = json.loads(f.read())

# Inicializar TTS con la variable de entorno (si no existe, lanzará KeyError)
tts = TTS(os.environ["TTS_MODEL_NAME"]).to(device)
model_name = os.environ["TTS_MODEL_NAME"]


class SetModelRequest(BaseModel):
    model: str = Field(description="Nombre del modelo a utilizar.")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "tts_models/multilingual/multi-dataset/xtts_v2"
                }
            ]
        }
    }


class TTSRequest(BaseModel):
    voice: str = Field(description="Nombre de la voz a utilizar.")
    text: str = Field(description="Texto que se desea sintetizar.")
    language: str = Field(description="Idioma del texto (código ISO, ej. 'es', 'en', 'zh').")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "voice": "demo_voice",
                    "text": "Hola, esto es un ejemplo de texto sintetizado.",
                    "language": "es"
                }
            ]
        }
    }


# Orígenes permitidos (CORS)
origins = [
    "*",
    "http://localhost",
    "https://p5js.org",
    "https://editor.p5js.org",
    "https://preview.p5js.org",
]

app = FastAPI(
    title="Coqui TTS API",
    description="Usa la librería Coqui TTS para síntesis de voz y clonación de voces."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, summary="Página principal de la API")
def read_root():
    return """
    <html>
        <head>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/purecss@3.0.0/build/pure-min.css" integrity="sha384-X38yfunGUhNzHpBaEBsWLO+A0HDYOQi8ufWDkZ0k9e0eXz/tH3II7uKZ9msv++Ls" crossorigin="anonymous">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
            <h1>Coqui TTS API está en funcionamiento</h1>
            <a class="pure-button pure-button-primary" href="/docs">Ver documentación</a>
        </body>
    </html>
    """


@app.get("/models", summary="Obtener lista de modelos de síntesis disponibles")
def get_models():
    return TTS().list_models().models_dict


@app.get("/model", summary="Obtener el nombre del modelo actualmente seleccionado")
def get_current_model():
    global model_name
    return model_name if model_name is not None else ""


@app.get(
    "/voices",
    summary="Obtener lista de voces disponibles",
    description="Devuelve la lista de voces que el usuario ha subido actualmente."
)
def get_voices():
    global voices
    return voices


@app.post(
    "/voice/upload",
    summary="Subir una grabación para clonar una voz",
    description=(
        "El usuario puede subir una grabación en formato WAV junto con un nombre para la voz. "
        "Si ya existe una voz con el mismo nombre, el nuevo archivo sobrescribirá al anterior. "
        "Nota: este endpoint espera multipart/form-data (no JSON)."
    )
)
def upload_voice(name: Annotated[str, Form(description="Nombre de la voz.")],
                 audio: Annotated[UploadFile, File(description="Archivo WAV con la voz.")]):
    filename = voices[name] if name in voices else str(uuid.uuid4()) + ".wav"
    audio_path = voices_dir / filename

    # Guardar archivo subido
    with open(audio_path, "wb") as f:
        f.write(audio.file.read())
    voices[name] = str(audio_path.name)

    # Actualizar JSON de voces
    with open(voices_config, "w", encoding="utf-8") as f:
        f.write(json.dumps(voices))

    return {"name": name, "audio": str(audio_path)}


@app.post(
    "/voice/generate",
    summary="Generar audio a partir de texto",
    description="Proporciona texto, lenguaje y nombre de voz; el API generará un WAV y lo guardará temporalmente."
)
def generate(request: TTSRequest):
    global voices_dir
    # Buscar archivo de voz de referencia
    if request.voice not in voices:
        return {"error": f"Voz '{request.voice}' no encontrada."}
    voice_file = voices[request.voice]
    output_file = voices_dir / "output.wav"

    # Generar usando TTS y guardar en output_file
    tts.tts_to_file(
        text=request.text,
        speaker_wav=str(voices_dir / voice_file),
        language=request.language,
        file_path=str(output_file)
    )
    return {"output": str(output_file)}


@app.get(
    "/voice/result",
    summary="Obtener resultado de la última generación",
    description="Devuelve el último archivo WAV generado."
)
def result():
    output_path = voices_dir / "output.wav"
    if not output_path.exists():
        return {"error": "No hay audio generado aún."}
    return FileResponse(str(output_path))
