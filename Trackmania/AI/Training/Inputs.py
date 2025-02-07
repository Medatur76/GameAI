from GameInfoProcessing.FrameProcessor import takePhoto
import socket, json, time

def getInputs():
    output: list[float] = takePhoto("Trackmania")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 9000))
    data = client_socket.recv(512)
    data = client_socket.recv(512).replace(data, b'')
    jsonData = json.loads(data.decode('utf8'))
    output.append(float(jsonData["speed"]))
    gameData: list = []
    gameData.append(jsonData["position"])
    if (jsonData["end"] == "true"): gameData.append(True)
    else: gameData.append(False)
    if (jsonData["running"] == "true"): gameData.append(True)
    else: gameData.append(False)
    return output, gameData