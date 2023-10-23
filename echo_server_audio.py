import os
from Speech_recognition import detect_intent_audio
import socket
import threading, pyaudio, pickle, struct

PROJECT_ID = 'pepper-try-fpqf'
LANGUAGE_CODE = 'en'
SESSION_ID = 'me'
WAVE_OUTPUT_FILENAME = "test.wav"
SAMPLE_RATE = 44100

host_name = socket.gethostname()
host_ip = '127.0.0.1'
port = 5000
server_socket = socket.socket()
server_socket.bind((host_ip, port))
server_socket.listen(5)
client_socket, address = server_socket.accept()  # accept new connection
print("Connection from: " + str(address))

def audio_stream():
    CHUNK = 1024
    CHANNELS = 1
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2),
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=CHUNK)
    data = b""
    payload_size = struct.calcsize("Q")
   # message = 'starting'
    while True:
        # while len(data) < payload_size:
        #     packet = client_socket.recv(4 * 1024)  # 4K
        #     if not packet: break
        #     data += packet
        # packed_msg_size = data[:payload_size]
        # data = data[payload_size:]
        # msg_size = struct.unpack("Q", packed_msg_size)[0]
        # while len(data) < msg_size:
        #     data += client_socket.recv(4 * 1024)
        # frame_data = data[:msg_size]
        # data = data[msg_size:]
        # frame = pickle.loads(frame_data)
        # stream.write(frame)
      #  print('Audio closed')
        data = client_socket.recv(1024 * 4).decode()
        if not data:
            # if data is not received break
            break
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/maria/Downloads/pepper-try-fpqf-9530bd049090.json"
        message = detect_intent_audio(PROJECT_ID, SESSION_ID, "/home/maria/pepper/audio.wav", LANGUAGE_CODE, SAMPLE_RATE)
        client_socket.send(message.encode())  # send message
    os._exit(1)
    client_socket.close()

    #print(a)

t1 = threading.Thread(target=audio_stream, args=())
t1.start()