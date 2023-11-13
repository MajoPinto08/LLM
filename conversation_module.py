from transformers import GPT2TokenizerFast
from datetime import datetime
from pydub import AudioSegment
from config import api_key
#from Voice_analysis import analysis
import azure.cognitiveservices.speech as speechsdk
import threading as th
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=api_key)
import json
import tiktoken
import socket
import atexit
import time
import pytz
import librosa
import requests
import pyaudio
import wave

# URL Request
url = 'https://echoweb.hri.idlab.ugent.be/api/1'


#Tokens.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
MAX_TOKENS = 3800 #7800 gpt4  #3800 gpt-3.5-turbo
API = "gpt-4"
conversation_going = True
recording = True
start_conversation = ""
user_filename = "audio_conversation/user_voice.wav"
pepper_filename = "audio_conversation/agent_voice.wav"
# General information for the agent.
timezone = pytz.timezone('Europe/Brussels')
now= datetime.now(timezone)
time_str = now.strftime('%H:%M')
location = 'Ghent, Belgium'
language = ""
language_id = ""
last_conversation = ""
voice_agent = ""
user_name = ""

def load_user_information():
    global language, language_id, last_conversation, voice_agent, user_name, interaction
    with open("users/information.json", "r") as json_file:
        data = json.load(json_file)

    user_name = input("Enter the user's name: ")
    for user in data.values():
        if user["name"] == user_name:
            language = user["language"]
            language_id = user["language_id"]
            last_conversation = user["info_conversation"]
            voice_agent = user["voice_agent"]
            interaction = user["interaction"]
            break
    else:
        print(f"No user found with the name {user_name}.")
        last_name = input("Enter the user's last name: ")
        df = pd.read_excel('./language-options.ods', engine='odf')
        language = input("Enter the user's language: ")
        language_id = df.loc[df['Language'] == language, 'Locale'].iloc[0]
        voice_agent = df.loc[df['Language'] == language, 'Text-to-speech-voices'].iloc[0]
        interaction = 1
        last_conversation = "First Conversation"
        new_user = dict(name=user_name, last_name=last_name, language=language, language_id=language_id,
                        voice_agent=voice_agent, interaction=interaction, info_conversation=last_conversation)
        user_num = len(data) + 1
        data[f"user{user_num}"] = new_user
        with open("users/information.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

def initialize_dialog():
    global dialog
    now = datetime.now(pytz.timezone('Europe/Brussels'))
    location = 'Ghent, Belgium'
    dialog = [
        {"role": "system",
         "content": f"You are Pepper. Pepper is creative, clever, empathic, friendly, and emotional. Pepper always keep his responses brief, short and chatty. Keep all the answers below 50 words during the entire conversation"
                    f"You will have a spoken, natural and friendly conversation between your friend called {user_name}. You role is being a human friend, so you have emotions and feelings."
                    f"The next information is about the last conversation and the information that you already know for your friend. Information:  ###{last_conversation}###."
                    f"The current date and time are {now.strftime('%m/%d/%Y %H:%M')} and the location where you are speaking is in {location}. The conversation will be in {language}."
                    f"Let's keep our responses filled with warmth, excitement, and friendliness, just like human friend!"}
    ]

def key_capture_thread():
    global conversation_going
    input()
    conversation_going = False

def program():
    global recording, dialog
    # # Socket connection
    # host = socket.gethostname()  # get the hostname
    # port = 5000  # initiate port no above 1024
    # server_socket = socket.socket()  # get instance (creation)
    # atexit.register(server_socket.close)
    # server_socket.bind((host, port))  # bind host address and port together
    # print("Listening")
    # server_socket.listen(5)  # configure how many client the server can listen simultaneously
    # client_socket, address = server_socket.accept()  # accept new connection
    # print("Connection from: " + str(address))

    # Open AI (Chat completion) using gpt-3.5-turbo model
    token_count = 0
    key_capture = th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True)
    key_capture.start()
    while conversation_going:
        # recording_thread = th.Thread(target=record_audio) # Separate flow of execution to record the audio.
        # recording_thread.start()
        message = recognize_from_microphone() #VAD - Speech recognition.
    #    recording = False # End the audio when the user finished to speak.
    #     recording_thread.join()
    #     save_audio()  #Save the user's audio
        token_count += len(tokenizer.encode(message))  # Tokens
        if token_count >= MAX_TOKENS:  #When the tokens are over but the conversation still going, it is essential to make a summary about the current conversation.
            summary = summarize_dialog()
            dialog = [
                {"role": "system",
                 "content": f"You are Pepper. Pepper is a helpful, creative, clever, empathic, friendly, and emotional."
                            f"Now you are having a natural conversation between your friend called {user_name}, and the following information is a summary of the current conversation {summary}."
                            f"Additionally, the next information is about the last conversation and the information that you already know for your friend: {last_conversation}. "
                            f"The current date and time are {now.strftime('%m/%d/%Y %H:%M')} and the location where we are speaking is in {location}."}
            ]
            token_count = len(tokenizer.encode(summary))  # Tokens
        answer = ask(message)
        requests.put(url, json=answer)  # Sending the message and the request to the webapp
        agent_voice(answer)  # Generate the .wav file with the agent voice
        client_socket.send(answer.encode())  # send message
        audio_time = librosa.get_duration(filename=pepper_filename)
        time.sleep(audio_time)  # Pepper is talking ...
        combine_audio_files()  #Combine audio files to generate the conversation file.
        token_count += len(tokenizer.encode(answer))  # Tokens
        # recording = True  # To start the audio recording
    key_capture.join()
    summary = final(f"Summarize in English our conversation, including the date {now.strftime('%m/%d/%Y')} and location {location} where we are speaking. "
                    f"Update relevant information based on what we've discussed and the information that you already know. Please include as a bullet point list of "
                    "the most important points of our conversation, any important details mentioned, and any actions we've agreed upon.")

    with open("users/information.json", "r") as json_file_1:
        data1 = json.load(json_file_1)
    for user1 in data1.values():
        if user1["name"] == user_name:
            user1["interaction"] = interaction + 1
            user1["info_conversation"] = summary
            break
    with open("users/information.json", "w") as archivo_json:
        json.dump(data1, archivo_json, indent=4)
    print(summary)
    final_text()
    client_socket.send("final".encode())  # send message
    client_socket.close()  # close the connection

def recognize_from_microphone():
    # This function requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription='fae9922ffa8146ae82ca74722dcbf955', region='eastus')
    speech_config.speech_recognition_language = language_id
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        return "Silence for part of the user. Pepper needs to ask the response again because he could not hear it"
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    return str(speech_recognition_result.text)

def ask(question):
    new_user_message = {"role": "user", "content": f"{question}"}
    dialog.append(new_user_message)
    res = client.chat.completions.create(model=API,
    messages = dialog,
    max_tokens= 150, #To control the length of the response generated by the model. around 20-30 words. 3 or 4 sentences.
    temperature=0.9,
    frequency_penalty=0,
    presence_penalty=0.6)
    response = res.choices[0].message.content.strip()
    response.replace('""', '')
    new_assistant_message = {"role": "assistant", "content":  f"{response}"}
    dialog.append(new_assistant_message)
    return response

def final(question):
    new_user_message = {"role": "user", "content": f"{question}"}
    dialog.append(new_user_message)
    res = client.chat.completions.create(model=API,
    messages = dialog,
    max_tokens= 500,
    temperature=0.9,
    frequency_penalty=0,
    presence_penalty=0.6)
    response = res.choices[0].message.content.strip()
    response.replace('""', '')
    new_assistant_message = {"role": "assistant", "content":  f"{response}"}
    dialog.append(new_assistant_message)
    return response

def final_text():
    conversation_file = f"text_conversation/{user_name}_conversation_interaction_{interaction}.txt"
    dialog_str_list = []
    user_name_role = f'{user_name}: '

    for item in dialog:
        if isinstance(item, dict):
            if item.get('role') == 'assistant':
                item_str = 'Pepper: ' + item.get('content')
            elif item.get('role') == 'user':
                item_str = user_name_role + item.get('content')
            else:
                item_str = str(item)  # Convert the dictionary to a string
        else:
            item_str = str(item)  # Convert the element to a string

        dialog_str_list.append(item_str)

    dialog_str = '\n'.join(dialog_str_list)
    dialog_str = dialog_str.replace('".}', '".')
    dialog_str = dialog_str.replace('\\n', '\n')  # Replace '\\n' with '\n' for line breaks
    archivo = open(conversation_file, "w")
    archivo.write(dialog_str)
    archivo.close()

def num_tokens(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def agent_voice(text):
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription='fae9922ffa8146ae82ca74722dcbf955',
                                           region='eastus')
    #audio_config = speechsdk.audio.AudioOutputConfig(filename=pepper_filename)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True) #Reproduce the audio
    speech_config.speech_synthesis_voice_name = f"{voice_agent}"   #The language of the agent voice.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

def record_audio():
    global p, channels, frames, fs, sample_format, stream
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True,
                    input_device_index=None,
                    start=False)
    stream.start_stream()
    frames = []
    while recording:
        audio_recording = stream.read(chunk)
        frames.append(audio_recording)


def save_audio():
    wf = wave.open(user_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    stream.stop_stream()
    stream.close()
    p.terminate()

def combine_audio_files():
    combine_filename = f"audio_conversation/{user_name}_conversation_interaction_{interaction}.wav"
    try:
        combined_audio = AudioSegment.from_wav(combine_filename)
    except:
        combined_audio = AudioSegment.empty()
    user_audio = AudioSegment.from_wav(user_filename)
    assistant_audio = AudioSegment.from_wav(pepper_filename)
    combined_audio = combined_audio + user_audio + assistant_audio
    combined_audio.export(combine_filename, format='wav')

def summarize_dialog():
    # Concatenate messages from the user and the robot into a single text
    text = ""
    turn = 1
    for message in dialog:
        if message["role"] in ["user", "assistant"]:
            if turn == 1:
                text += f"{user_name}:" + message["content"] + f"\n"
                turn = 2
            else:
                text += "Pepper:" + message["content"] + f"\n"
                turn = 1
    prompt = f'Summarize the following conversation. Conversation: ""{text}"".' #Generate a summary of the following conversation:
    summary = client.completions.create(model="gpt-4",
    prompt=prompt,
    temperature=0.9,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6)
    summary_text = summary.choices[0].text.strip()
    return summary_text

def main():
    load_user_information()
    initialize_dialog()
    program()

if __name__ == "__main__":
    main()