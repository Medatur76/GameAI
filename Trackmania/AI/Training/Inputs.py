from GameInfoProcessing.FrameProcessor import takePhoto
import socket, json

def getInputs():
    output: list[float] = takePhoto("Trackmania")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 9000))
    data = client_socket.recv(512)
    data = client_socket.recv(512).replace(data, b'')
    output.append(json.loads(data.decode('utf8'))["speed"])
    return output