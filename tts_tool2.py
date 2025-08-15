#!/usr/bin/env python3
# tts_tool.py - versión arreglada para evitar colisiones entre "tts" y "TTS"
import argparse
import logging
import subprocess
from pathlib import Path
import sys
import shutil
import os
import traceback

# NOTA: no importamos torch de forma global para poder controlar el orden (y permitir forzar CPU antes de importar)
# Dependencias se importan de forma perezosa dentro de funciones.

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tts_tool")

# --- DEBUG / logging robusto ---
import faulthandler
faulthandler.enable()

print("DEBUG: arrancando tts_tool.py", flush=True)

log_file = Path.cwd() / "tts_tool_debug.log"
fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s - %(message)s")
fh.setFormatter(formatter)

root_logger = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_file) for h in root_logger.handlers):
    root_logger.addHandler(fh)
root_logger.setLevel(logging.DEBUG)
# --- fin debug/logging ---


# ---------- Utilities ----------
def import_torch():
    """Importa torch de forma segura. No imprime trace de excepción arriba, solo lo registra."""
    try:
        import torch
        return torch
    except Exception as e:
        logger.debug(f"Import torch falló: {e}")
        return None


def check_pytorch_cuda(required_cuda="11.8"):
    """Comprueba estado de PyTorch/CUDA, importando torch cuando haga falta."""
    torch = import_torch()
    if torch is None:
        logger.warning("PyTorch no está instalado o falló al importar (ver debug).")
        return {"installed": False, "cuda_available": False, "torch_cuda_version": None, "ok_for_required": False}

    installed = True
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        logger.debug("torch.cuda.is_available() lanzó excepción: %s", e)
        cuda_available = False

    torch_cuda = getattr(torch.version, "cuda", None)
    ok_for_required = (torch_cuda is not None and required_cuda in str(torch_cuda))
    logger.info(f"PyTorch instalado: {installed}, CUDA disponible: {cuda_available}, PyTorch built with CUDA: {torch_cuda}")
    if not ok_for_required:
        logger.warning(f"PyTorch reporta CUDA '{torch_cuda}'. Si necesitas exactamente CUDA {required_cuda}, instala la build adecuada.")
    return {"installed": installed, "cuda_available": cuda_available, "torch_cuda_version": torch_cuda, "ok_for_required": ok_for_required}


def ensure_path(path_str, must_exist=True, create=False):
    p = Path(path_str)
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Ruta no encontrada: {p}")
    if create:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def detect_conflicting_tts_cli():
    """
    Detecta si hay un ejecutable 'tts' en PATH que no corresponda a 'python -m TTS'.
    Devuelve la ruta si existe; solo para advertencias.
    """
    tts_cmd = shutil.which("tts")
    if not tts_cmd:
        return None
    # Intentemos inspeccionar la versión de ese binario (silencioso)
    try:
        out = subprocess.check_output([tts_cmd, "--help"], stderr=subprocess.STDOUT, text=True, timeout=3)
        # Si el help muestra opciones de inferencia solamente, avisamos. No podemos inferir paquete exacto, solo avisar.
        return tts_cmd
    except Exception:
        return tts_cmd


# ---------- Entrenamiento (API Python) ----------
def train_model_pyapi(
    dataset_dir,
    model_out,
    config_path=None,
    pretrained_model=None,
    epochs=50,
    use_cuda=True,
    extra_args=None,
):
    """
    Entrena usando la API Python de Coqui TTS (intenta importar módulos internos).
    Si use_cuda==False forzamos el entrenamiento en CPU *antes* de importar torch/TTS.
    """
    logger.info("Intentando entrenar usando la API Python de Coqui TTS...")
    # Si se quiere forzar CPU, escondemos todas las GPUs antes de cualquier import que cargue torch/CUDA.
    if not use_cuda:
        logger.info("Forzando CPU: deshabilitando visibilidad de GPUs (CUDA_VISIBLE_DEVICES='').")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Ahora intentamos importar TTS internamente.
    try:
        # Import perezoso de torch para poder informar mejor si la import falla
        torch = import_torch()
        # Intentamos cargar la API de Coqui TTS
        from TTS.configs.shared_configs import BaseDatasetConfig  # comprobación rápida de disponibilidad
        # Importar entrenador (nombre puede variar según versión)
        try:
            # interfaz moderna
            from TTS.trainers import Trainer, TrainerArgs
        except Exception:
            # interfaz antigua
            try:
                from TTS.tts.trainers import TTSTrainer as Trainer
                TrainerArgs = None
            except Exception as e_inner:
                logger.debug("No se ha podido localizar clase Trainer en ninguno de los módulos esperados: %s", e_inner)
                raise

        output_path = str(Path(model_out).expanduser())
        dataset_path = str(Path(dataset_dir).expanduser())

        device = "cuda" if (torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available() and use_cuda) else "cpu"
        logger.info(f"Entrenamiento en dispositivo: {device}")

        # Esta parte es heurística: la API puede cambiar entre versiones de TTS.
        if TrainerArgs is not None:
            trainer_args = TrainerArgs()
            logger.debug("TrainerArgs creado.")
            if config_path:
                logger.info(f"Cargando config personalizada: {config_path}")
                import json
                cfg = json.load(open(config_path, "r", encoding="utf-8"))
            else:
                cfg = None

            trainer = Trainer(trainer_args, cfg, output_path)
            logger.info("Trainer inicializado.")

            # Intentar mover trainer/model al device si ofrece .to()
            if hasattr(trainer, "to"):
                try:
                    trainer = trainer.to(device)
                except Exception:
                    logger.debug("No se pudo mover trainer al dispositivo con .to() (no crítico).")

            try:
                if pretrained_model:
                    logger.info("Intentando cargar checkpoint base desde %s", pretrained_model)
                    if hasattr(trainer, "load_checkpoint"):
                        trainer.load_checkpoint(pretrained_model)
                    elif hasattr(trainer, "load_model"):
                        trainer.load_model(pretrained_model)
            except Exception:
                logger.debug("No se pudo cargar checkpoint base con los métodos conocidos; continuar.")

            # Intentar cargar dataset (método puede variar)
            try:
                if hasattr(trainer, "load_dataset"):
                    trainer.load_dataset(dataset_path)
            except Exception:
                logger.debug("trainer.load_dataset no disponible o falló — continuar igualmente.")

            logger.info("Lanzando fit() (esto puede tardar).")
            trainer.fit()
            # Guardar si existe el método
            try:
                if hasattr(trainer, "save_model"):
                    trainer.save_model()
                elif hasattr(trainer, "save"):
                    trainer.save(output_path)
            except Exception:
                logger.debug("No se pudo invocar método de guardado estándar; revisa la API de la versión instalada.")
            logger.info("Entrenamiento finalizado (API Python).")
        else:
            # API antigua (TTSTrainer)
            trainer = Trainer()
            if pretrained_model:
                try:
                    trainer.load_model(pretrained_model)
                except Exception:
                    logger.debug("trainer.load_model no disponible o falló.")
            trainer.load_dataset(dataset_dir)
            trainer.fit()
            try:
                trainer.save_model(output_path)
            except Exception:
                logger.debug("trainer.save_model no disponible o falló.")
            logger.info("Entrenamiento finalizado (TTSTrainer).")

    except Exception as e:
        # Capturamos y reportamos la excepción original: suele ser ImportError o errores de torch.
        logger.exception("Entrenamiento mediante API Python falló o no soportado por la versión instalada.")
        # Si la causa fue un fallo en torch (ej. dll faltante), detectarlo y dar diagnóstico más útil:
        tb = traceback.format_exc()
        if "shm.dll" in tb or "WinError" in tb or "cannot find" in tb:
            logger.error("Parece que PyTorch no carga correctamente (DLL faltante o build incompatible). "
                         "Revisa que la versión de torch coincida con tu CUDA/drivers (ej: torch==2.3.0+cu118 para CUDA 11.8).")
        raise ImportError("API de entrenamiento Python de Coqui TTS no disponible o falló: " + str(e))


# ---------- Entrenamiento (CLI fallback seguro) ----------
def train_model_cli(dataset_dir, model_out, config_path=None, epochs=50, use_cuda=True, additional_args=None):
    """
    Fallback seguro usando 'python -m TTS.bin.train_tts' (más fiable que confiar en un 'tts' binario)
    Nota: las versiones modernas de Coqui TTS esperan un config.json/hydra; por eso requerimos config_path.
    """
    logger.info("Usando fallback: entrenamiento mediante 'python -m TTS.bin.train_tts' (si está disponible).")

    # Localizar el módulo TTS disponible para ejecución con el mismo intérprete:
    python_module_cmd = [sys.executable, "-m", "TTS.bin.train_tts"]

    if not config_path:
        raise RuntimeError("El fallback CLI requiere --config (config.json) para entrenar con 'python -m TTS.bin.train_tts'. "
                           "Si no tienes config, usa la API Python o crea un config.json válido para tu dataset.")

    cmd = python_module_cmd + ["--config_path", str(config_path)]
    # epoch/otros suelen estar en el config; si el usuario pasó epochs intentamos inyectarlo (no todas las versiones aceptan).
    if epochs:
        cmd += ["--epochs", str(epochs)]
    if additional_args:
        cmd += additional_args

    env = os.environ.copy()
    if not use_cuda:
        # forzamos CPU para el proceso hijo ocultando GPUs
        env["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Forzando CPU para CLI fallback (CUDA_VISIBLE_DEVICES='').")

    logger.info("Ejecutando comando CLI fallback: %s", " ".join(cmd))
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        logger.exception("El fallback CLI (python -m TTS.bin.train_tts) devolvió error: %s", e)
        raise


# ---------- Wrapper ----------
def train_model(
    dataset_dir,
    model_out,
    config_path=None,
    pretrained_model=None,
    epochs=50,
    use_python_api=True,
    use_cuda=True,
):
    print(f"DEBUG: train_model called with dataset_dir={dataset_dir}, model_out={model_out}, use_python_api={use_python_api}, use_cuda={use_cuda}", flush=True)
    logger.info(f"train_model: dataset_dir={dataset_dir}, model_out={model_out}, use_python_api={use_python_api}, use_cuda={use_cuda}")

    dataset_dir = ensure_path(dataset_dir, must_exist=True)
    Path(model_out).mkdir(parents=True, exist_ok=True)

    # Comprobación simple de posibles conflictos con un 'tts' binario diferente
    tts_bin = detect_conflicting_tts_cli()
    if tts_bin:
        logger.debug(f"Detectado ejecutable 'tts' en PATH: {tts_bin}. Esto puede pertenecer a otra librería. "
                     "Preferimos usar 'python -m TTS' o la API Python para evitar colisiones.")

    try:
        if use_python_api:
            train_model_pyapi(dataset_dir, model_out, config_path=config_path, pretrained_model=pretrained_model, epochs=epochs, use_cuda=use_cuda)
        else:
            raise ImportError("forzado fallback a CLI (use_python_api=False)")
    except ImportError as e:
        logger.warning("Falló la API Python de entrenamiento: %s", e)
        logger.info("Intentando fallback a la CLI (python -m TTS.bin.train_tts)...")
        # CLI fallback exige config_path en este script (por las versiones modernas):
        train_model_cli(dataset_dir, model_out, config_path=config_path, epochs=epochs, use_cuda=use_cuda)


# ---------- Síntesis ----------
def generate_speech(model_spec, text, speaker_wav=None, language="en", out_wav="output.wav", use_cuda=True, progress=True):
    logger.info("Generando audio...")
    # Si queremos forzar CPU para generación, ocultamos GPU antes de importar TTS
    if not use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Forzando CPU para generación (CUDA_VISIBLE_DEVICES='').")

    try:
        from TTS.api import TTS
    except Exception as e:
        logger.exception("No se pudo importar TTS.api. Asegúrate de tener instalado 'TTS' (Coqui TTS) y una versión compatible con tu torch.")
        raise

    # Intentar import de torch para decidir dispositivo sólo si está disponible
    torch = import_torch()
    device = "cuda" if (torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available() and use_cuda) else "cpu"
    logger.info(f"Dispositivo elegido para generación: {device}")

    logger.info(f"Cargando modelo '{model_spec}'.")
    tts = TTS(model_spec)

    if hasattr(tts, "to"):
        try:
            tts = tts.to(device)
        except Exception:
            logger.debug("El objeto TTS no admite .to(device) o ha fallado mover a device; continuar con configuración por defecto.")

    kwargs = {}
    if speaker_wav:
        kw_sp = str(Path(speaker_wav).expanduser())
        kwargs["speaker_wav"] = kw_sp
        logger.info(f"Usando WAV de referencia: {kw_sp}")

    try:
        tts.tts_to_file(text=text, file_path=out_wav, language=language, **kwargs)
        logger.info(f"Audio generado: {out_wav}")
    except TypeError:
        logger.exception("tts_to_file recibió argumentos inesperados. Intentando llamada alternativa...")
        try:
            wav = tts.tts(text, **kwargs)
            import soundfile as sf
            sf.write(out_wav, wav, samplerate=22050)
            logger.info(f"Audio generado (ruta alternativa): {out_wav}")
        except Exception:
            logger.exception("Fallo la generación alternativa.")
            raise


# ---------- CLI ----------
def build_cli_parser():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-v', '--verbose', action='store_true', help='Modo verbose (DEBUG)')

    parser = argparse.ArgumentParser(description="Entrenamiento o síntesis con Coqui TTS (soporta clonación)")
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", parents=[parent], help="Entrenar/fine-tune un modelo con datos locales")
    train.add_argument("--data", required=True, help="Directorio con dataset (ej: metadata + wavs) para entrenamiento")
    train.add_argument("--output", required=True, help="Directorio para guardar el modelo entrenado")
    train.add_argument("--config", help="Archivo config.json opcional para el entrenamiento (requerido para el fallback CLI)")
    train.add_argument("--pretrained", help="Checkpoint / modelo base para fine-tuning (opcional)")
    train.add_argument("--epochs", type=int, default=50, help="Número de epochs")
    train.add_argument("--no-api", action="store_true", help="Forzar uso de la CLI (python -m TTS.bin.train_tts) en lugar de la API Python")
    train.add_argument("--no-cuda", action="store_true", help="Forzar uso de CPU (ignora CUDA aunque esté disponible)")

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
        logger.exception("Argument parsing falló: SystemExit")
        raise

    if getattr(args, "verbose", False):
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Logger en nivel DEBUG")

    logger.info(f"Argumentos recibidos: {args!r}")
    print(f"DEBUG: argumentos recibidos: {args!r}", flush=True)

    try:
        cuda_report = check_pytorch_cuda(required_cuda=args.required_cuda)
    except Exception as e:
        logger.exception("Error comprobando PyTorch/CUDA")
        print("ERROR: fallo comprobación PyTorch/CUDA, ver tts_tool_debug.log", flush=True)
        raise

    try:
        if args.mode == "train":
            use_api = not getattr(args, "no_api", False)
            use_cuda = not getattr(args, "no_cuda", False)
            logger.info("Modo: train")
            print("DEBUG: entrando en train_model()", flush=True)
            train_model(
                dataset_dir=args.data,
                model_out=args.output,
                config_path=args.config,
                pretrained_model=getattr(args, "pretrained", None),
                epochs=getattr(args, "epochs", 50),
                use_python_api=use_api,
                use_cuda=use_cuda,
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
