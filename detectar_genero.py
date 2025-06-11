import sounddevice as sd
from scipy.io.wavfile import write
from voice_gender_classification import GenderClassificationPipeline

def grabar(nombre="voz.wav", dur=5, fs=16000):
    print(f"Grabando durante {dur} seg…")
    audio = sd.rec(int(dur*fs), samplerate=fs, channels=1)
    sd.wait()
    write(nombre, fs, audio)
    return nombre

def detectar(path):
    clf = GenderClassificationPipeline.from_pretrained(
        "griko/gender_cls_svm_ecapa_voxceleb"
    )
    res = clf(path)
    return res[0]

if __name__ == "__main__":
    archivo = grabar()
    genero = detectar(archivo)
    print("Género detectado:", genero)