#!/usr/bin/env python3
# """
# tts_tool.py
#
# Script para:
#  - entrenar (fine-tune) un modelo clonable con Coqui TTS
#  - generar/sintetizar audio desde texto (con clonación por wav de referencia opcional)
#
# Características:
#  - logging informativo
#  - detecta y avisa sobre compatibilidad CUDA / PyTorch (verifica versión CUDA usada por PyTorch)
#  - intenta usar la API Python para entrenar si está disponible, si no, hace fallback a la CLI `tts`
#  - comprobaciones básicas de paths y argumentos
#  - extensible (hooks para callbacks / config extern)
# """
#
import argparse
import logging
import subprocess
from pathlib import Path
import sys
import shutil
# --- DEBUG / logging robusto: pega esto tras los imports ---

# Dependencias opcionales
try:
    import torch
except Exception:
    torch = None

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tts_tool")
# --- DEBUG / logging robusto: pega esto justo después de `logger = logging.getLogger("tts_tool")` ---
import faulthandler
faulthandler.enable()

# Mensaje inmediato para comprobar arranque
print("DEBUG: arrancando tts_tool.py", flush=True)

# Archivo de log en el cwd
log_file = Path.cwd() / "tts_tool_debug.log"

# FileHandler para traza completa
fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s - %(message)s")
fh.setFormatter(formatter)

# Añadir handler al logger raíz (propagará a todos los loggers)
root_logger = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_file) for h in root_logger.handlers):
    root_logger.addHandler(fh)

# Asegurarnos de que el nivel mínimo permite DEBUG (se puede bajar después con --verbose)
root_logger.setLevel(logging.DEBUG)
# --- fin debug/logging ---


# ---------- Utilities ----------
def check_pytorch_cuda(required_cuda="11.8"):
    """Comprueba si torch está instalado y la versión CUDA que usa PyTorch.
    No garantiza que el driver del sistema soporte esa versión, solo informa.
    """
    if torch is None:
        logger.warning("PyTorch no está instalado (torch import falló). No se puede comprobar CUDA desde PyTorch.")
        return {"installed": False, "cuda_available": False, "torch_cuda_version": None, "ok_for_required": False}

    installed = True
    cuda_available = torch.cuda.is_available()
    torch_cuda = torch.version.cuda  # ejemplo: "11.8" o None
    ok_for_required = (torch_cuda is not None and required_cuda in torch_cuda)
    logger.info(f"PyTorch instalado: {installed}, CUDA disponible: {cuda_available}, PyTorch built with CUDA: {torch_cuda}")
    if not ok_for_required:
        logger.warning(f"PyTorch reporta CUDA '{torch_cuda}'. Si necesitas exactamente CUDA {required_cuda}, asegúrate de instalar la versión correcta de PyTorch / CUDA.")
    return {"installed": installed, "cuda_available": cuda_available, "torch_cuda_version": torch_cuda, "ok_for_required": ok_for_required}


def ensure_path(path_str, must_exist=True, create=False):
    p = Path(path_str)
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Ruta no encontrada: {p}")
    if create:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Entrenamiento ----------
def train_model_pyapi(
    dataset_dir,
    model_out,
    config_path=None,
    pretrained_model=None,
    epochs=50,
    batch_size=None,
    use_cuda=True,
    extra_args=None,
):
    """
    Intento de entrenamiento usando la API Python de Coqui TTS (si la instalación la provee).
    Si las importaciones fallan, lanzar ImportError para que el caller use fallback a la CLI.
    NOTA: La API interna cambia entre versiones de TTS; este método intenta ser lo más genérico posible.
    """
    logger.info("Intentando entrenar usando la API Python de Coqui TTS...")
    try:
        # imports dinámicos para que el script no falle si no se instaló coqui-tts
        from TTS.configs.shared_configs import BaseDatasetConfig
        # API de training puede cambiar entre versiones:
        # intentamos importar clases de Trainer (nombres históricos: Trainer o TTSTrainer)
        try:
            from TTS.trainers import Trainer, TrainerArgs
        except Exception:
            # versión anterior/alternativa
            from TTS.tts.trainers import TTSTrainer as Trainer
            TrainerArgs = None

        # construimos argumentos mínimos
        output_path = str(Path(model_out).expanduser())
        dataset_path = str(Path(dataset_dir).expanduser())

        # Ejemplo simple: si la API es Trainer(TrainerArgs(), config, output_path,...)
        # Necesitarás adaptar mucho según la versión exacta de coqui-tts instalada.
        # Aquí dejamos un ejemplo instructivo; el usuario probablemente necesite ajustar config.json.
        if TrainerArgs is not None:
            trainer_args = TrainerArgs()
            logger.debug("TrainerArgs creado.")
            # Cargar/configurar config: si el usuario proporciona config_path, usarlo
            if config_path:
                logger.info(f"Cargando config personalizada: {config_path}")
                import json
                cfg = json.load(open(config_path, "r", encoding="utf-8"))
            else:
                cfg = None

            trainer = Trainer(trainer_args, cfg, output_path)
            logger.info("Trainer inicializado. Cargando dataset y modelo (si procede)...")
            # Métodos concretos dependen de la versión de la librería:
            # Los siguientes son heurísticos; si fallan, abandonamos al fallback CLI.
            try:
                if pretrained_model:
                    logger.info(f"Cargando checkpoint base: {pretrained_model}")
                    trainer.load_checkpoint(pretrained_model)
            except Exception:
                logger.debug("No se pudo cargar checkpoint con trainer.load_checkpoint (puede no existir en esta versión).")

            # cargar dataset: método variará; intentamos con nombres comunes
            try:
                trainer.load_dataset(dataset_path)
            except Exception:
                logger.debug("trainer.load_dataset no disponible o falló — continuar igualmente.")

            logger.info("Lanzando fit() (puede tardar).")
            trainer.fit()
            trainer.save_model()
            logger.info("Entrenamiento finalizado (API Python).")
        else:
            # Si no tenemos TrainerArgs (API antigua TTSTrainer)
            trainer = Trainer()
            if pretrained_model:
                try:
                    trainer.load_model(pretrained_model)
                except Exception:
                    logger.debug("trainer.load_model no disponible o falló.")
            trainer.load_dataset(dataset_dir)
            trainer.fit()
            trainer.save_model(output_path)
            logger.info("Entrenamiento finalizado (TTSTrainer).")

    except Exception as e:
        logger.exception("Entrenamiento mediante API Python falló o no soportado por la versión instalada.")
        raise ImportError("API de entrenamiento Python de Coqui TTS no disponible o falló: " + str(e))


def train_model_cli(dataset_dir, model_out, config_path=None, epochs=50, additional_args=None):
    """
    Fallback: usa la CLI 'tts' o 'python -m TTS' si 'tts' no está en PATH.
    """
    logger.info("Usando fallback: entrenamiento mediante la CLI `tts` (o python -m TTS).")
    tts_cmd = shutil.which("tts")
    if tts_cmd:
        base_cmd = [tts_cmd]
    else:
        # usa el mismo python que ejecuta el script para evitar problemas de entorno
        base_cmd = [sys.executable, "-m", "TTS"]

    cmd = base_cmd + ["--train", "--train_data", str(dataset_dir), "--output_path", str(model_out), "--epochs", str(epochs)]
    if config_path:
        cmd += ["--config", str(config_path)]
    if additional_args:
        cmd += additional_args

    logger.info("Ejecutando comando: " + " ".join(cmd))
    subprocess.check_call(cmd)
    logger.info("Entrenamiento finalizado (CLI).")


def train_model(
    dataset_dir,
    model_out,
    config_path=None,
    pretrained_model=None,
    epochs=50,
    use_python_api=True,
):
    """
    Función wrapper que primero intenta la API Python y si falla usa la CLI.
    """
    # DEBUG informativo al entrar en la función
    print(f"DEBUG: train_model called with dataset_dir={dataset_dir}, model_out={model_out}, use_python_api={use_python_api}", flush=True)
    logger.info(f"train_model: dataset_dir={dataset_dir}, model_out={model_out}, use_python_api={use_python_api}")

    dataset_dir = ensure_path(dataset_dir, must_exist=True)
    Path(model_out).mkdir(parents=True, exist_ok=True)
    try:
        if use_python_api:
            train_model_pyapi(dataset_dir, model_out, config_path=config_path, pretrained_model=pretrained_model, epochs=epochs)
        else:
            raise ImportError("forzado fallback a CLI (use_python_api=False)")
    except ImportError as e:
        logger.warning("Fallo la API Python de entrenamiento: " + str(e))
        logger.info("Intentando fallback a la CLI `tts`...")
        train_model_cli(dataset_dir, model_out, config_path=config_path, epochs=epochs)


# ---------- Síntesis ----------
def generate_speech(model_spec, text, speaker_wav=None, language="en", out_wav="output.wav", use_cuda=True, progress=True):
    """
    Genera audio usando la API de Coqui TTS (TTS.api.TTS).
    model_spec puede ser:
     - nombre de modelo público ("tts_models/...") -> la librería descargará/usa el release preentrenado
     - ruta a carpeta con model/config -> cargará localmente
    """
    logger.info("Generando audio...")
    try:
        from TTS.api import TTS
    except Exception as e:
        logger.exception("No se pudo importar TTS.api. ¿Instalaste coqui-tts?")
        raise

    # Selección de dispositivo informativa
    device = "cuda" if (torch is not None and torch.cuda.is_available() and use_cuda) else "cpu"
    if torch is not None:
        logger.info(f"Dispositivo elegido: {device} (torch CUDA disponible: {torch.cuda.is_available()})")
    else:
        logger.info(f"Dispositivo elegido (torch no disponible): {device}")

    logger.info(f"Cargando modelo '{model_spec}'. Esto puede tardar la primera vez si hay que descargar pesos.")
    tts = TTS(model_spec)

    # Intenta mover a device si la clase TTS tiene .to()
    if hasattr(tts, "to"):
        try:
            tts = tts.to(device)
        except Exception:
            logger.debug("El objeto TTS no admite .to(device) o ha fallado mover a device; seguir con configuración por defecto.")

    # Preparar kwargs
    kwargs = {}
    if speaker_wav:
        kw_sp = str(Path(speaker_wav).expanduser())
        kwargs["speaker_wav"] = kw_sp
        logger.info(f"Usando WAV de referencia para clonación: {kw_sp}")

    # Llamada para generar
    try:
        tts.tts_to_file(text=text, file_path=out_wav, language=language, **kwargs)
        logger.info(f"Audio generado: {out_wav}")
    except TypeError as e:
        logger.exception("tts_to_file recibió argumentos inesperados (API puede ser distinta). Intentando llamada alternativa...")
        # Intento alternativo muy simple:
        try:
            wav = tts.tts(text, **kwargs)
            # Guardar wav a archivo si retorna ndarray
            import soundfile as sf
            sf.write(out_wav, wav, samplerate=22050)
            logger.info(f"Audio generado (ruta alternativa): {out_wav}")
        except Exception:
            logger.exception("Fallo la generación de audio con la API alternativa.")
            raise


# ---------- CLI ----------
def build_cli_parser():
    # parser padre para opciones que queremos disponibles en el subcomando (sin help duplicado)
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-v', '--verbose', action='store_true', help='Modo verbose (DEBUG)')

    parser = argparse.ArgumentParser(description="Entrenamiento o síntesis con Coqui TTS (soporta clonación)")
    # No añadir aquí -v global para evitar posición forzada; lo pasamos a subparsers via parents.
    sub = parser.add_subparsers(dest="mode", required=True)

    # Pasamos `parents=[parent]` para que train/gen acepten -v en cualquier posición
    train = sub.add_parser("train", parents=[parent], help="Entrenar/fine-tune un modelo con datos locales")
    train.add_argument("--data", required=True, help="Directorio con dataset (ej: metadata + wavs) para entrenamiento")
    train.add_argument("--output", required=True, help="Directorio para guardar el modelo entrenado")
    train.add_argument("--config", help="Archivo config.json opcional para el entrenamiento")
    train.add_argument("--pretrained", help="Checkpoint / modelo base para fine-tuning (opcional)")
    train.add_argument("--epochs", type=int, default=50, help="Número de epochs")
    train.add_argument("--no-api", action="store_true", help="Forzar uso de la CLI en lugar de la API Python")

    gen = sub.add_parser("generate", parents=[parent], help="Generar audio a partir de texto")
    gen.add_argument("--model", required=True, help="Nombre de modelo Coqui o ruta a modelo entrenado")
    gen.add_argument("--text", required=True, help="Texto a convertir a voz")
    gen.add_argument("--speaker", help="Archivo .wav de referencia para clonación (opcional)")
    gen.add_argument("--lang", default="en", help="Código ISO del idioma")
    gen.add_argument("--out", default="out.wav", help="Archivo de salida .wav")
    gen.add_argument("--no-cuda", action="store_true", help="Forzar uso de CPU (ignora CUDA aunque esté disponible)")

    parser.add_argument("--required-cuda", default="11.8", help="Versión CUDA objetivo para comprobar (informativo)")

    return parser




def main():
    parser = build_cli_parser()
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse imprime ayuda/errores y lanza SystemExit; queremos que quede registrado en el log
        logger.exception("Argument parsing falló: SystemExit")
        raise

    # activar DEBUG si se solicitó
    if getattr(args, "verbose", False):
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Logger en nivel DEBUG")

    logger.info(f"Argumentos recibidos: {args!r}")
    print(f"DEBUG: argumentos recibidos: {args!r}", flush=True)

    # chequeo rápido PyTorch/CUDA
    try:
        cuda_report = check_pytorch_cuda(required_cuda=args.required_cuda)
    except Exception as e:
        logger.exception("Error comprobando PyTorch/CUDA")
        print("ERROR: fallo comprobación PyTorch/CUDA, ver tts_tool_debug.log", flush=True)
        raise

    try:
        if args.mode == "train":
            use_api = not getattr(args, "no_api", False)
            logger.info("Modo: train")
            print("DEBUG: entrando en train_model()", flush=True)
            train_model(
                dataset_dir=args.data,
                model_out=args.output,
                config_path=args.config,
                pretrained_model=getattr(args, "pretrained", None),
                epochs=getattr(args, "epochs", 50),
                use_python_api=use_api,
            )
        elif args.mode == "generate":
            logger.info("Modo: generate")
            generate_speech(
                model_spec=args.model,
                text=args.text,
                speaker_wav=args.speaker,
                language=args.lang,
                out_wav=args.out,
                use_cuda=(not args.no_cuda),
            )
        else:
            logger.error("Modo desconocido.")
    except Exception as e:
        logger.exception("Error en la ejecución principal:")
        print("ERROR: Ocurrió una excepción. Revisa tts_tool_debug.log para más detalles.", flush=True)
        sys.exit(2)

if __name__ == "__main__":
    main()
