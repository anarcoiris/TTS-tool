from pydub import AudioSegment
import glob

# Ruta donde están los wav
ruta = "./wavs"

# Buscar todos los wav (puedes cambiar el patrón)
archivos = sorted(glob.glob(f"{ruta}/*.wav"))

# Cargar y concatenar
if not archivos:
    raise ValueError("No se encontraron archivos WAV en la ruta especificada.")

audio_final = AudioSegment.empty()
for archivo in archivos:
    print(f"Añadiendo: {archivo}")
    audio = AudioSegment.from_wav(archivo)
    audio_final += audio

# Guardar el resultado
audio_final.export("combinado.wav", format="wav")
print("✅ Archivo combinado guardado como 'combinado.wav'")
