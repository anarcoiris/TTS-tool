# server_entry.py
import os
import uvicorn

if __name__ == "__main__":
    # Valores por defecto (puedes cambiarlos aquí)
    os.environ.setdefault("TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
    # Si ejecutas en contenedor y montaste el host en /wavs, pon '/wavs' o la ruta adecuada
    os.environ.setdefault("VOICES_DIR", os.environ.get("VOICES_DIR", "C:/Users/soyko/Documents/git2/tts-tool/wavs"))

    host = "0.0.0.0"
    port = 8000

    # workers=1 recomendado si usas torch con GPU
    uvicorn.run("tts_es:app", host=host, port=port, workers=1)
