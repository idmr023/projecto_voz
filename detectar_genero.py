import sounddevice as sd
from scipy.io.wavfile import write
from voice_gender_classification import GenderClassificationPipeline
import speech_recognition as sr
import pyttsx3

# Grabar audio
def grabar(nombre="voz.wav", dur=5, fs=16000):
    print(f"Grabando durante {dur} segundos...")
    audio = sd.rec(int(dur * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(nombre, fs, audio)
    return nombre

# Detectar género
def detectar_genero(path):
    clf = GenderClassificationPipeline.from_pretrained(
        "griko/gender_cls_svm_ecapa_voxceleb"
    )
    res = clf(path)
    return res[0]

# Reconocer texto
def reconocer_texto(path):
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)
    try:
        texto = r.recognize_google(audio, language="es-ES")
        return texto
    except sr.UnknownValueError:
        return "No se entendió lo que dijiste"
    except sr.RequestError as e:
        return f"Error en el servicio de reconocimiento: {e}"

# Reproducir voz
def hablar(texto, genero):
    engine = pyttsx3.init()
    voces = engine.getProperty('voices')
    # Cambiar voz según género detectado
    if genero == 'female':
        for v in voces:
            if "female" in v.name.lower():
                engine.setProperty('voice', v.id)
                break
    else:
        for v in voces:
            if "male" in v.name.lower():
                engine.setProperty('voice', v.id)
                break
    engine.say(texto)
    engine.runAndWait()

# Ejecución principal
if __name__ == "__main__":
    archivo = grabar()
    genero = detectar_genero(archivo)
    print("Género detectado:", genero)

    texto = reconocer_texto(archivo)
    print("Texto reconocido:", texto)

    print("Reproduciendo texto...")
    hablar(texto, genero)