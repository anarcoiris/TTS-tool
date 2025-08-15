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
def check_pytorch_cuda(required_cuda="11.8"):
    if torch is None:
        logger.warning("PyTorch no está instalado (torch import falló). No se puede comprobar CUDA desde PyTorch.")
        return {"installed": False, "cuda_available": False, "torch_cuda_version": None, "ok_for_required": False}

    installed = True
    cuda_available = torch.cuda.is_available()
    torch_cuda = torch.version.cuda
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
    logger.info("Intentando entrenar usando la API Python de Coqui TTS...")
    try:
        from TTS.configs.shared_configs import BaseDatasetConfig
        try:
            from TTS.trainers import Trainer, TrainerArgs
        except Exception:
            from TTS.tts.trainers import TTSTrainer as Trainer
            TrainerArgs = None

        output_path = str(Path(model_out).expanduser())
        dataset_path = str(Path(dataset_dir).expanduser())

        device = "cuda" if (torch is not None and torch.cuda.is_available() and use_cuda) else "cpu"
        logger.info(f"Entrenando en dispositivo: {device}")

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
            if hasattr(trainer, "to"):
                try:
                    trainer = trainer.to(device)
                except Exception:
                    logger.debug("No se pudo mover trainer al dispositivo elegido.")

            try:
                if pretrained_model:
                    logger.info(f"Cargando checkpoint base: {pretrained_model}")
                    trainer.load_checkpoint(pretrained_model)
            except Exception:
                logger.debug("No se pudo cargar checkpoint con trainer.load_checkpoint.")

            try:
                trainer.load_dataset(dataset_path)
            except Exception:
                logger.debug("trainer.load_dataset no disponible o falló.")

            logger.info("Lanzando fit() (puede tardar).")
            trainer.fit()
            trainer.save_model()
            logger.info("Entrenamiento finalizado (API Python).")
        else:
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
    logger.info("Usando fallback: entrenamiento mediante la CLI `tts` (o python -m TTS).")
    tts_cmd = shutil.which("tts")
    if tts_cmd:
        base_cmd = [tts_cmd]
    else:
        base_cmd = [sys.executable, "-m", "TTS"]

    cmd = base_cmd + ["train", "--data", str(dataset_dir), "--output", str(model_out), "--epochs", str(epochs)]
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
    use_cuda=True,
):
    print(f"DEBUG: train_model called with dataset_dir={dataset_dir}, model_out={model_out}, use_python_api={use_python_api}, use_cuda={use_cuda}", flush=True)
    logger.info(f"train_model: dataset_dir={dataset_dir}, model_out={model_out}, use_python_api={use_python_api}, use_cuda={use_cuda}")

    dataset_dir = ensure_path(dataset_dir, must_exist=True)
    Path(model_out).mkdir(parents=True, exist_ok=True)
    try:
        if use_python_api:
            train_model_pyapi(dataset_dir, model_out, config_path=config_path, pretrained_model=pretrained_model, epochs=epochs, use_cuda=use_cuda)
        else:
            raise ImportError("forzado fallback a CLI (use_python_api=False)")
    except ImportError as e:
        logger.warning("Falló la API Python de entrenamiento: " + str(e))
        logger.info("Intentando fallback a la CLI `tts`...")
        extra_cli_args = ["--use_cpu"] if not use_cuda else None
        train_model_cli(dataset_dir, model_out, config_path=config_path, epochs=epochs, additional_args=extra_cli_args)


# ---------- Síntesis ----------
def generate_speech(model_spec, text, speaker_wav=None, language="en", out_wav="output.wav", use_cuda=True, progress=True):
    logger.info("Generando audio...")
    try:
        from TTS.api import TTS
    except Exception as e:
        logger.exception("No se pudo importar TTS.api. ¿Instalaste coqui-tts?")
        raise

    device = "cuda" if (torch is not None and torch.cuda.is_available() and use_cuda) else "cpu"
    if torch is not None:
        logger.info(f"Dispositivo elegido: {device} (torch CUDA disponible: {torch.cuda.is_available()})")
    else:
        logger.info(f"Dispositivo elegido (torch no disponible): {device}")

    logger.info(f"Cargando modelo '{model_spec}'.")
    tts = TTS(model_spec)

    if hasattr(tts, "to"):
        try:
            tts = tts.to(device)
        except Exception:
            logger.debug("El objeto TTS no admite .to(device) o falló moverlo.")

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
            logger.exception("Fallo la generación de audio alternativa.")
            raise


# ---------- CLI ----------
def build_cli_parser():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-v', '--verbose', action='store_true', help='Modo verbose (DEBUG)')

    parser = argparse.ArgumentParser(description="Entrenamiento o síntesis con Coqui TTS (soporta clonación)")
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", parents=[parent], help="Entrenar/fine-tune un modelo con datos locales")
    train.add_argument("--data", required=True, help="Directorio con dataset")
    train.add_argument("--output", required=True, help="Directorio para guardar el modelo entrenado")
    train.add_argument("--config", help="Archivo config.json opcional para el entrenamiento")
    train.add_argument("--pretrained", help="Checkpoint / modelo base para fine-tuning (opcional)")
    train.add_argument("--epochs", type=int, default=50, help="Número de epochs")
    train.add_argument("--no-api", action="store_true", help="Forzar uso de la CLI en lugar de la API Python")
    train.add_argument("--no-cuda", action="store_true", help="Forzar uso de CPU (ignora CUDA aunque esté disponible)")

    gen = sub.add_parser("generate", parents=[parent], help="Generar audio a partir de texto")
    gen.add_argument("--model", required=True, help="Nombre de modelo o ruta a modelo entrenado")
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
    except SystemExit:
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
    except Exception:
        logger.exception("Error comprobando PyTorch/CUDA")
        print("ERROR: fallo comprobación PyTorch/CUDA", flush=True)
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
                use_cuda=(not args.no_cuda),
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
    except Exception:
        logger.exception("Error en la ejecución principal:")
        print("ERROR: Ocurrió una excepción. Revisa tts_tool_debug.log", flush=True)
        sys.exit(2)

if __name__ == "__main__":
    main()
