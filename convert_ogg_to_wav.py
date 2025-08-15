#!/usr/bin/env python3
# """
# convert_ogg_to_wav.py
#
# Convierte archivos .ogg (o cualquier formato soportado) a WAV PCM 16-bit, mono, 22050 Hz
# (preconfigurado) adecuado para entrenamiento de TTS.
#
# Características:
#  - usa ffmpeg si está disponible (rápido y fiable)
#  - fallback a librosa+soundfile si ffmpeg no está presente
#  - puede procesar directorios de forma recursiva
# - preserva la estructura de carpetas relativa
#  - soporte para multithreading (paraleliza llamadas a ffmpeg)
#  - control de overwrite / dry-run
#
# Ejemplos:
#   python convert_ogg_to_wav.py --input C:\datasets\mi_dataset\wavs --output C:\datasets\mi_dataset_converted
#
# """

from pathlib import Path
import argparse
import shutil
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Dependencias opcionales para fallback
try:
    import librosa
    import soundfile as sf
except Exception:
    librosa = None
    sf = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("ogg2wav")


def is_ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def convert_with_ffmpeg(in_path: Path, out_path: Path, sample_rate: int = 22050, channels: int = 1, overwrite: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        logger.info(f"Omitido (ya existe): {out_path}")
        return out_path

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(in_path),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(out_path),
    ]

    # Si usamos -n (no overwrite) y fichero existe, ffmpeg devuelve código 1; pero ya lo controlamos arriba.
    try:
        logger.debug("Ejecutando: %s", " ".join(cmd))
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"Convertido (ffmpeg): {in_path} -> {out_path}")
        return out_path
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg falló para {in_path}: {e}")
        raise


def convert_with_librosa(in_path: Path, out_path: Path, sample_rate: int = 22050, channels: int = 1, overwrite: bool = False):
    if librosa is None or sf is None:
        raise RuntimeError("librosa/soundfile no están instalados; instala librosa and soundfile o instala ffmpeg.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        logger.info(f"Omitido (ya existe): {out_path}")
        return out_path

    logger.debug(f"Cargando {in_path} con librosa (sr={sample_rate})")
    # sr=None para preservar sample rate original y luego remuestrear a target
    audio, sr_src = librosa.load(str(in_path), sr=sample_rate, mono=(channels==1))

    # librosa.load ya remuestrea si se pide sr=sample_rate
    # Garantizar forma mono/estéreo: si se pidió channels=1, librosa devuelve mono por defecto cuando mono=True
    # soundfile.write acepta float y convertirá a PCM_16 con subtype
    sf.write(str(out_path), audio, samplerate=sample_rate, subtype='PCM_16')
    logger.info(f"Convertido (librosa): {in_path} -> {out_path}")
    return out_path


def find_input_files(input_path: Path, pattern: str = "*.ogg", recursive: bool = True):
    if input_path.is_file():
        return [input_path]
    if recursive:
        return list(input_path.rglob(pattern))
    else:
        return list(input_path.glob(pattern))


def build_output_path(in_file: Path, input_root: Path, output_root: Path):
    try:
        rel = in_file.relative_to(input_root)
    except Exception:
        # Si no es relativo (por ejemplo cuando input is a single file), usar solo el nombre
        rel = in_file.name
    out_rel = Path(rel).with_suffix('.wav')
    return output_root.joinpath(out_rel)


def process_file(in_file: Path, input_root: Path, output_root: Path, cfg, ffmpeg_available: bool):
    out_file = build_output_path(in_file, input_root, output_root)
    try:
        if ffmpeg_available:
            return convert_with_ffmpeg(in_file, out_file, sample_rate=cfg.sample_rate, channels=cfg.channels, overwrite=cfg.overwrite)
        else:
            return convert_with_librosa(in_file, out_file, sample_rate=cfg.sample_rate, channels=cfg.channels, overwrite=cfg.overwrite)
    except Exception as e:
        logger.error(f"Error convirtiendo {in_file}: {e}")
        return None


class Config:
    def __init__(self, sample_rate=22050, channels=1, overwrite=False, jobs=4, pattern="*.ogg", recursive=True):
        self.sample_rate = sample_rate
        self.channels = channels
        self.overwrite = overwrite
        self.jobs = jobs
        self.pattern = pattern
        self.recursive = recursive


def main():
    parser = argparse.ArgumentParser(description="Convertir .ogg (u otros) a WAV PCM16 mono 22050 Hz para entrenamiento TTS")
    parser.add_argument('--input', '-i', default='./ogg', help='Archivo .ogg o directorio que contiene .ogg (ej: C:\\datasets\\wavs)')
    parser.add_argument('--output', '-o', default='./wavs', help='Directorio de salida para los WAV convertidos')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate objetivo (por defecto 22050)')
    parser.add_argument('--channels', type=int, choices=[1, 2], default=1, help='Canales de salida (1=mono, 2=stereo)')
    parser.add_argument('--jobs', type=int, default=4, help='Número de hilos para conversión en paralelo')
    parser.add_argument('--pattern', default='*.ogg', help='Patrón de búsqueda (por defecto *.ogg)')
    parser.add_argument('--no-recursive', action='store_true', help='No buscar recursivamente en subdirectorios')
    parser.add_argument('--overwrite', action='store_true', help='Sobrescribir archivos de salida si ya existen')
    parser.add_argument('--dry-run', action='store_true', help='Mostrar qué archivos se convertirían, sin ejecutar la conversión')

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_root = Path(args.output).expanduser()

    if not input_path.exists():
        logger.error(f"La ruta de entrada no existe: {input_path}")
        sys.exit(1)

    cfg = Config(sample_rate=args.sr, channels=args.channels, overwrite=args.overwrite, jobs=args.jobs, pattern=args.pattern, recursive=(not args.no_recursive))

    ffmpeg_ok = is_ffmpeg_available()
    if ffmpeg_ok:
        logger.info("ffmpeg detectado: se usará ffmpeg para las conversiones (recomendado)")
    else:
        if librosa is None or sf is None:
            logger.error("No se encontró ffmpeg y librosa/soundfile no están instalados. Instala ffmpeg o 'librosa'+'soundfile'.")
            sys.exit(1)
        else:
            logger.info("ffmpeg no detectado: se usará librosa+soundfile como fallback.")

    # Buscar archivos
    files = find_input_files(input_path, pattern=cfg.pattern, recursive=cfg.recursive)
    files = [f for f in files if f.is_file()]
    if not files:
        logger.error("No se encontraron archivos a convertir.")
        sys.exit(1)

    logger.info(f"Archivos encontrados: {len(files)}. Inicio de conversion (jobs={cfg.jobs}).")

    if args.dry_run:
        for f in files:
            out = build_output_path(f, input_path, output_root)
            print(f"{f} -> {out}")
        sys.exit(0)

    # Ejecutar en paralelo
    failed = []
    converted = []
    with ThreadPoolExecutor(max_workers=cfg.jobs) as exe:
        futures = {exe.submit(process_file, f, input_path, output_root, cfg, ffmpeg_ok): f for f in files}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                res = fut.result()
                if res is None:
                    failed.append(src)
                else:
                    converted.append(res)
            except Exception as e:
                logger.error(f"Excepción procesando {src}: {e}")
                failed.append(src)

    logger.info(f"Conversión finalizada. Convertidos: {len(converted)}; Fallidos: {len(failed)}")
    if failed:
        logger.info("Lista de fallidos:")
        for f in failed:
            logger.info(str(f))


if __name__ == '__main__':
    main()
