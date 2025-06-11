# Detector de Género por Voz (Python + ML)

Este proyecto permite grabar la voz del usuario desde el micrófono, analizarla con un modelo de aprendizaje automático preentrenado, y determinar si la voz es masculina o femenina.

El modelo utilizado es `ECAPA-TDNN` + `SVM`, desarrollado por el repositorio de HuggingFace [griko/voice-gender-classification].

---

## 🚀 ¿Qué hace este programa?

1. Graba 5 segundos de audio con tu micrófono.
2. Analiza la voz usando un modelo de Machine Learning.
3. Imprime el género detectado: `male` o `female`.

---

## 🔧 Requisitos

- Python 3.8 o superior
- Micrófono conectado
- Acceso a internet para descargar el modelo

---

## 🛠️ Instalación

1. Abre una terminal o consola de comandos.

2. Instala las dependencias necesarias:

```bash
pip install git+https://github.com/griko/voice-gender-classification.git

pip install sounddevice scipy git+https://github.com/griko/voice-gender-classification.git

pip install SpeechRecognition

pip install pyaudio

pip install pyttsx3