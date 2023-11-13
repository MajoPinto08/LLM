from openai import OpenAI
from config import api_key
from pathlib import Path
import playsound
import re
import time
from config import speech_subscription_key
#from Voice_analysis import analysis
import azure.cognitiveservices.speech as speechsdk

region = "eastus"
languages = ["en-US", "es-ES"]
client = OpenAI(
  api_key=api_key,  # this is also the default, it can be omitted
)


# Configurar la configuración del servicio de reconocimiento de voz
def speech_recognize_continuous_from_microphone():
    # Configuración de la suscripción y la región
    speech_config = speechsdk.SpeechConfig(subscription=speech_subscription_key, region=region)

    # Habilitar la identificación automática del idioma
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)

    # Crear un reconocedor de voz con la configuración y el idioma automático
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config)

    # Conectar eventos para recibir resultados de STT
    speech_recognizer.recognized.connect(lambda evt: print(f"RECOGNIZED: {evt.result.text}"))

    # Comenzar la transcripción continua
    speech_recognizer.start_continuous_recognition()

    # Mantener el programa en ejecución mientras se realiza la transcripción
    try:
        input("Presiona Enter para terminar...\n")
    except KeyboardInterrupt:
        pass

    # Detener la transcripción
    speech_recognizer.stop_continuous_recognition()


def speech_recognize_continuous_async_from_microphone():
  """performs continuous speech recognition asynchronously with input from microphone"""
  silence_threshold = 200
  speech_config = speechsdk.SpeechConfig(subscription=speech_subscription_key, region='eastus')
  speech_config.speech_recognition_language = "es-ES" ## Exploreee (?).
  speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
  speech_recognition_result = ""
  last_audio_timestamp = time.time()

  def recognizing_cb(*args):
    """
    *args means "any number of arguments". Required for Recognizer.recognizing.connect(...)
    :return:
    """
    nonlocal last_audio_timestamp
    last_audio_timestamp = time.time()

  def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    nonlocal speech_recognition_result
    nonlocal silence_threshold
    silence_threshold = 0.8  ## Elderly (1.8-1.5)
    speech_recognition_result += evt.result.text + " "

  speech_recognizer.recognized.connect(recognized_cb)
  speech_recognizer.recognizing.connect(recognizing_cb)

  result_future = speech_recognizer.start_continuous_recognition_async()
  result_future.get()  # wait for voidfuture, so we know engine initialization is done.
  print('Speak in your microphone.')

  while True:
    current_timestamp = time.time()
    if current_timestamp - last_audio_timestamp > silence_threshold:
      print('Detected silence.')
      speech_recognizer.stop_continuous_recognition_async()
      if not speech_recognition_result.strip():
        print("Silence")
        return "Silence is detected. Pepper needs to ask the response again because he could not hear it", speech_recognizer
      print("User:", speech_recognition_result)
      return speech_recognition_result, speech_recognizer


def LLM_module():
  buffer = ""
  pause_characters = r"[.!?;:]+"
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Zeg: achter grootmoeders schuurtje hangen drie grote kroten te drogen. Herhaal mijn woorden"}
    ],
    stream=True
  )
  for chunk in completion:
    event_text = chunk.choices[0].delta
    if event_text.content is not None: buffer += event_text.content
    while True:
      sentence_end = re.search(pause_characters, buffer)
      if sentence_end:
        sentence = buffer[:sentence_end.start() + 1]
        buffer = buffer[sentence_end.end():]
        sentence = sentence.strip()
        audio(sentence)
      else: break
  if buffer:  # After the loop, send any remaining text in the buffer to the TTS system
    # agent_voice(buffer)
    audio(buffer)
    print("end")


def audio(text):
  speech_file_path = Path(__file__).parent / "speech.mp3"
  response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=f"{text}"
  )
  response.stream_to_file(speech_file_path)
  playsound.playsound('./speech.mp3', True)


def main():
  speech_recognize_continuous_from_microphone()
  #message, fml = speech_recognize_continuous_async_from_microphone()  # VAD - Speech recognition
  #LLM_module()

if __name__ == "__main__":
  main()


