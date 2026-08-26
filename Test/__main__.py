import json, socket
from Trackmania.AI.GameInfoProcessing.FrameProcessor import takePhoto

takePhoto("Trackmania", 1)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 9000))
data = client_socket.recv(512)
data = client_socket.recv(512).replace(data, b'').decode('utf8')
try:
    print(json.loads(data))
except:
    print("Err")
